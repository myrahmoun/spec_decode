"""Expert predictor: shapes, loss, metric, windows, tree annotation, learning."""
import math

import numpy as np
import pytest
import torch

from ecospec.predictor import (
    ExpertPredictor, annotate_tree, collate, load_predictor, loss_fn, make_windows,
    save_predictor, topk_overlap,
)
from ecospec.tiny import tiny_backbone_config, tiny_model
from ecospec.tree import DraftTree, Node

LAYERS, EXPERTS, K = 2, 8, 2


@pytest.fixture
def embedding():
    return tiny_model(seed=0).get_input_embeddings()


@pytest.fixture
def predictor(embedding):
    torch.manual_seed(0)
    return ExpertPredictor(embedding, tiny_backbone_config(), LAYERS, EXPERTS).eval()


def test_forward_shape_and_frozen_embedding(predictor):
    out = predictor(torch.randint(0, 512, (3, 7)))
    assert out.shape == (3, 7, LAYERS, EXPERTS)
    assert not predictor.target_embedding.weight.requires_grad
    assert not any(k.startswith("target_embedding.") for k in predictor.trainable_state_dict())
    assert predictor.backbone.embed_tokens is None


def test_loss_minimum_is_log_k():
    # Optimal logits put equal mass on the k true experts -> CE = log k.
    true = torch.tensor([[[[1, 5], [0, 2]]]])               # [1, 1, L, k]
    logits = torch.full((1, 1, LAYERS, EXPERTS), -1e4)
    logits.scatter_(-1, true, 0.0)
    assert loss_fn(logits, true, K).item() == pytest.approx(math.log(K), abs=1e-4)


def test_loss_ignores_masked_positions():
    true = torch.zeros(1, 2, LAYERS, K, dtype=torch.long)
    true[..., 1] = 1
    logits = torch.zeros(1, 2, LAYERS, EXPERTS)
    logits[0, 1] = torch.randn(LAYERS, EXPERTS) * 10       # garbage at a padded position
    mask = torch.tensor([[True, False]])
    expected = loss_fn(logits[:, :1], true[:, :1], K)
    assert loss_fn(logits, true, K, mask).item() == pytest.approx(expected.item())


def test_topk_overlap_by_hand():
    true = torch.tensor([[[[0, 1], [2, 3]]]])
    logits = torch.zeros(1, 1, LAYERS, EXPERTS)
    logits[0, 0, 0, [0, 1]] = 1.0   # layer 0: both right
    logits[0, 0, 1, [2, 7]] = 1.0   # layer 1: one of two right
    assert topk_overlap(logits, true).item() == pytest.approx(0.75)


def test_windows_drop_unrouted_last_token_and_collate():
    tokens = np.arange(11, dtype=np.int32)                  # 10 routed + final token
    experts = np.zeros((10, LAYERS, K), dtype=np.int16)
    windows = make_windows(tokens, experts, max_len=4)
    assert [len(t) for t, _ in windows] == [4, 4, 2]
    assert all(len(t) == len(e) for t, e in windows)
    assert windows[-1][0].tolist() == [8, 9]
    ids, ex, mask = collate(windows)
    assert ids.shape == (3, 4) and ex.shape == (3, 4, LAYERS, K)
    assert mask.sum().item() == 10


def _small_tree() -> DraftTree:
    return DraftTree(nodes={
        0: Node(0, None, 10, 1.0),
        1: Node(1, 0, 11, 0.6),
        2: Node(2, 0, 12, 0.3),
        3: Node(3, 1, 13, 0.4),
    })


def test_annotate_tree_matches_unbatched(predictor):
    context = [5, 6, 7, 8, 9]
    tree = _small_tree()
    annotate_tree(predictor, context, tree, K)
    for node_id, node in tree.nodes.items():
        assert len(node.experts) == LAYERS * K
        seq = torch.tensor([context + [n.token for n in tree.path(node_id)]])
        with torch.no_grad():
            last = predictor(seq)[0, -1]
        expected = {(l, int(e)) for l in range(LAYERS) for e in last[l].topk(K).indices}
        assert node.experts == expected, node_id


def test_annotate_tree_truncates_context(predictor):
    tree_a, tree_b = _small_tree(), _small_tree()
    annotate_tree(predictor, list(range(100)), tree_a, K, max_context=8)
    annotate_tree(predictor, list(range(95, 100)), tree_b, K, max_context=8)
    # Depth 3 -> 5 context tokens kept: the last five, either way.
    assert {i: n.experts for i, n in tree_a.nodes.items()} == \
           {i: n.experts for i, n in tree_b.nodes.items()}


def test_learns_per_token_rule(embedding):
    # Experts of token t are a fixed function of t itself. Learnable only if
    # labels are aligned with inputs (no off-by-one).
    torch.manual_seed(0)
    predictor = ExpertPredictor(embedding, tiny_backbone_config(), LAYERS, EXPERTS)
    rule = torch.stack([torch.stack([(torch.arange(32) + l) % EXPERTS,
                                     (torch.arange(32) * 3 + l + 1) % EXPERTS], -1)
                        for l in range(LAYERS)], 1)          # [32, L, k]
    assert (rule[..., 0] != rule[..., 1]).all()
    opt = torch.optim.Adam([p for p in predictor.parameters() if p.requires_grad], lr=3e-3)
    for _ in range(150):
        ids = torch.randint(0, 32, (8, 16))
        loss = loss_fn(predictor(ids), rule[ids], K)
        opt.zero_grad()
        loss.backward()
        opt.step()
    ids = torch.randint(0, 32, (8, 16))
    with torch.no_grad():
        assert topk_overlap(predictor(ids), rule[ids]).item() > 0.9


def test_save_load_roundtrip(tmp_path, predictor, embedding):
    path = tmp_path / "p.pt"
    save_predictor(predictor, path, {"backbone": "tiny", "top_k": K})
    loaded, meta = load_predictor(path, embedding)
    assert meta["top_k"] == K and meta["num_layers"] == LAYERS
    ids = torch.randint(0, 512, (2, 5))
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids), predictor(ids))
