"""RouterTraceLogger records the experts the routers really pick."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ecospec.routing_trace import RouterTraceLogger, find_router_modules
from ecospec.tiny import tiny_model


@pytest.fixture
def model():
    return tiny_model(seed=0)


def random_ids(model, length: int, batch: int = 1, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, model.config.vocab_size, (batch, length), generator=g)


def router_own_topk(model, input_ids) -> np.ndarray:
    """Ground truth: hook router.forward itself and read the indices it returns."""
    captured = []
    handles = [
        router.register_forward_hook(lambda m, a, out, i=i: captured.append((i, out[2])))
        for i, router in find_router_modules(model)
    ]
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in handles:
            h.remove()
    per_layer = [idx for _, idx in sorted(captured, key=lambda x: x[0])]
    return torch.stack(per_layer, dim=1).numpy()


def test_finds_one_router_per_layer(model):
    routers = find_router_modules(model)
    assert [i for i, _ in routers] == list(range(model.config.num_hidden_layers))


def test_shape_and_dtype(model):
    config = model.config
    with RouterTraceLogger(model) as log, torch.no_grad():
        model(random_ids(model, 10))
    assert log.experts.shape == (10, config.num_hidden_layers, config.num_experts_per_tok)
    assert log.experts.dtype == np.int16
    assert ((log.experts >= 0) & (log.experts < config.num_local_experts)).all()


def test_matches_router_own_topk(model):
    input_ids = random_ids(model, 12)
    with RouterTraceLogger(model) as log, torch.no_grad():
        model(input_ids)
    np.testing.assert_array_equal(log.experts, router_own_topk(model, input_ids))


def test_experts_in_a_row_are_distinct(model):
    with RouterTraceLogger(model) as log, torch.no_grad():
        model(random_ids(model, 20))
    sorted_rows = np.sort(log.experts, axis=-1)
    assert (np.diff(sorted_rows, axis=-1) != 0).all()


def test_batch_is_flattened_batch_major(model):
    input_ids = random_ids(model, 6, batch=3)
    with RouterTraceLogger(model) as log, torch.no_grad():
        model(input_ids)
    assert log.experts.shape[0] == 18
    with RouterTraceLogger(model) as single, torch.no_grad():
        model(input_ids[1:2])
    np.testing.assert_array_equal(log.experts[6:12], single.experts)


def test_calls_concatenate_and_match_full_pass(model):
    # Prefill + one cached decode step should trace the same experts as one full pass.
    input_ids = random_ids(model, 9)
    with RouterTraceLogger(model) as stepwise, torch.no_grad():
        out = model(input_ids[:, :8], use_cache=True)
        model(input_ids[:, 8:], past_key_values=out.past_key_values, use_cache=True)
    with RouterTraceLogger(model) as full, torch.no_grad():
        model(input_ids)
    assert stepwise.experts.shape[0] == 9
    np.testing.assert_array_equal(stepwise.experts, full.experts)


def test_hooks_removed_on_exit(model):
    with RouterTraceLogger(model) as log, torch.no_grad():
        model(random_ids(model, 4))
    with torch.no_grad():
        model(random_ids(model, 4))
    assert log.experts.shape[0] == 4


def test_empty_log_has_correct_shape(model):
    with RouterTraceLogger(model) as log:
        pass
    assert log.experts.shape == (0, model.config.num_hidden_layers, model.config.num_experts_per_tok)


class InlineRoutingMoE(nn.Module):
    """Imitates transformers' Mxfp4GptOssMLP: computes routing inline, never calls router.forward."""

    def __init__(self, router, experts):
        super().__init__()
        self.router = router
        self.experts = experts

    def forward(self, hidden_states):
        b, s, d = hidden_states.shape
        h = hidden_states.reshape(-1, d)
        logits = F.linear(h, self.router.weight, self.router.bias)
        top_value, top_idx = torch.topk(logits, self.router.top_k, dim=-1)
        scores = torch.softmax(top_value, dim=1)
        return self.experts(h, top_idx, scores).reshape(b, s, d), logits


def test_works_when_router_forward_is_never_called(model):
    input_ids = random_ids(model, 10)
    expected = router_own_topk(model, input_ids)

    for layer in model.model.layers:
        layer.mlp = InlineRoutingMoE(layer.mlp.router, layer.mlp.experts)
        layer.mlp.router.forward = lambda *a: pytest.fail("router.forward was called")

    with RouterTraceLogger(model) as log, torch.no_grad():
        model(input_ids)
    np.testing.assert_array_equal(log.experts, expected)
