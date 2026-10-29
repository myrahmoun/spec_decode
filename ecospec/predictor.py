"""Expert predictor (paper Section 4.1).

Architecture: target-token ids -> TARGET's embedding layer -> linear projection
-> Qwen3-0.6B decoder layers -> head producing L x E logits (one softmax per layer).
Using the target's embeddings sidesteps the tokenizer mismatch: gpt-oss and
Qwen use different vocabularies, so Qwen's own embedding layer is not used.

Loss: per-layer cross-entropy against a uniform distribution over the true top-k.

Alignment: the output at position i predicts the experts of token i itself
(paper Eq. 2: z_ti = Π(t_i | t_<i)), so no shift between inputs and labels.
"""
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PretrainedConfig

from ecospec.tree import DraftTree

# Longest token window the predictor sees, in training and at inference.
MAX_CONTEXT = 512


class ExpertPredictor(nn.Module):
    def __init__(self, target_embedding: nn.Embedding, backbone_id: str | PretrainedConfig,
                 num_layers: int, num_experts: int):
        """`backbone_id` may also be a config, for a randomly initialised backbone (tests)."""
        super().__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts
        # Frozen and excluded from checkpoints: it is the target's, reloaded from the target.
        self.target_embedding = target_embedding.requires_grad_(False)
        if isinstance(backbone_id, PretrainedConfig):
            self.backbone = AutoModel.from_config(backbone_id)
        else:
            self.backbone = AutoModel.from_pretrained(backbone_id, dtype=torch.float32)
        # Unused (we pass inputs_embeds); dropping it saves ~155M params for Qwen3-0.6B.
        self.backbone.embed_tokens = None
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(target_embedding.embedding_dim, hidden)
        self.head = nn.Linear(hidden, num_layers * num_experts)

    def forward(self, input_ids):
        """Returns logits of shape [batch, seq, num_layers, num_experts]."""
        with torch.no_grad():
            embeds = self.target_embedding(input_ids)
        x = self.proj(embeds.to(self.proj.weight.dtype))
        h = self.backbone(inputs_embeds=x).last_hidden_state
        logits = self.head(h)
        return logits.view(*input_ids.shape, self.num_layers, self.num_experts)

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items()
                if not k.startswith("target_embedding.")}


def loss_fn(logits, true_experts, top_k: int, mask=None):
    """true_experts: [batch, seq, num_layers, top_k] int tensor.

    Cross-entropy against q = uniform over the true top-k (Eq. 5-6), averaged
    (rather than summed) over valid tokens and layers. `mask`: [batch, seq] bool.
    """
    log_p = F.log_softmax(logits.float(), dim=-1)
    # Sum of log-probs of the k true experts, times 1/k.
    loss = -log_p.gather(-1, true_experts.long()).sum(-1) / top_k  # [batch, seq, layers]
    if mask is None:
        return loss.mean()
    loss = loss.mean(-1)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def topk_overlap(logits, true_experts, mask=None) -> torch.Tensor:
    """Paper's predictor accuracy: mean |predicted top-k ∩ true top-k| / k."""
    k = true_experts.shape[-1]
    pred = logits.topk(k, dim=-1).indices                          # [b, s, L, k]
    hits = (pred.unsqueeze(-1) == true_experts.unsqueeze(-2).long()).any(-1).sum(-1)
    overlap = hits.float() / k                                     # [b, s, L]
    if mask is None:
        return overlap.mean()
    return (overlap.mean(-1) * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def annotate_tree(predictor, context_ids, tree: DraftTree, top_k: int,
                  max_context: int = MAX_CONTEXT) -> None:
    """Fill node.experts with predicted top-k per layer for every node.

    Open decision (paper is silent): how the predictor sees a node's context.
    Simplest: run it on context + root-to-node path, one sequence per node.
    `context_ids` are the tokens before the root. The context is truncated so
    each sequence fits in `max_context`, matching training windows.
    """
    node_ids = list(tree.nodes)
    paths = [[n.token for n in tree.path(i)] for i in node_ids]
    depth = max(len(p) for p in paths)
    context = list(context_ids)[-(max_context - depth):] if max_context > depth else []
    seqs = [context + p for p in paths]
    lengths = torch.tensor([len(s) for s in seqs])
    device = next(predictor.parameters()).device
    # Right padding is safe without an attention mask: attention is causal, so
    # real tokens never see the padding that follows them.
    batch = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long)
    for row, s in enumerate(seqs):
        batch[row, : len(s)] = torch.tensor(s)
    logits = predictor(batch.to(device))                            # [n, seq, L, E]
    last = logits[torch.arange(len(seqs)), (lengths - 1).to(device)]  # [n, L, E]
    top = last.topk(top_k, dim=-1).indices.cpu()                   # [n, L, k]
    for row, node_id in enumerate(node_ids):
        tree.nodes[node_id].experts = {
            (layer, int(e)) for layer in range(top.shape[1]) for e in top[row, layer]
        }


def load_target_embedding(model_id: str, revision: str | None = None) -> nn.Embedding:
    """Load only the target's input embedding from its safetensors shards.

    Avoids loading the full 20B model just to train the predictor.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    index_path = hf_hub_download(model_id, "model.safetensors.index.json", revision=revision)
    key = "model.embed_tokens.weight"
    shard = json.load(open(index_path))["weight_map"][key]
    with safe_open(hf_hub_download(model_id, shard, revision=revision), framework="pt") as f:
        weight = f.get_tensor(key)
    return nn.Embedding.from_pretrained(weight, freeze=True)


def make_windows(tokens, experts, max_len: int = MAX_CONTEXT):
    """Cut one traced sample into non-overlapping windows of at most max_len tokens.

    Only tokens that have routing (the first len(experts)) are used.
    Returns a list of (tokens [n], experts [n, L, k]) array pairs.
    """
    n = len(experts)
    tokens = tokens[:n]
    return [(tokens[s:s + max_len], experts[s:s + max_len]) for s in range(0, n, max_len)]


def collate(windows) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad windows into (input_ids, true_experts, mask) tensors."""
    length = max(len(t) for t, _ in windows)
    _, layers, k = windows[0][1].shape
    ids = torch.zeros(len(windows), length, dtype=torch.long)
    experts = torch.zeros(len(windows), length, layers, k, dtype=torch.long)
    mask = torch.zeros(len(windows), length, dtype=torch.bool)
    for row, (t, e) in enumerate(windows):
        ids[row, : len(t)] = torch.as_tensor(t, dtype=torch.long)
        experts[row, : len(t)] = torch.as_tensor(e, dtype=torch.long)
        mask[row, : len(t)] = True
    return ids, experts, mask


def target_embedding_for(run_info: dict) -> nn.Embedding:
    """The target embedding matching a trace run (run.json written by collect_traces.py)."""
    if run_info["model"] == "tiny":
        from ecospec.tiny import tiny_model
        return tiny_model(vocab_size=run_info["vocab_size"]).get_input_embeddings()
    return load_target_embedding(run_info["model"])


def save_predictor(predictor: ExpertPredictor, path, meta: dict) -> None:
    """`meta` must include 'backbone' (id or 'tiny') and 'top_k'."""
    torch.save({"state_dict": predictor.trainable_state_dict(),
                "num_layers": predictor.num_layers,
                "num_experts": predictor.num_experts, **meta}, path)


def load_predictor(path, target_embedding: nn.Embedding) -> tuple[ExpertPredictor, dict]:
    """Rebuild a saved predictor around the given (shared) target embedding."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    backbone = ckpt["backbone"]
    if backbone == "tiny":
        from ecospec.tiny import tiny_backbone_config
        backbone = tiny_backbone_config()
    predictor = ExpertPredictor(target_embedding, backbone, ckpt["num_layers"], ckpt["num_experts"])
    missing, unexpected = predictor.load_state_dict(ckpt["state_dict"], strict=False)
    if unexpected or any(not k.startswith("target_embedding.") for k in missing):
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return predictor.eval(), meta
