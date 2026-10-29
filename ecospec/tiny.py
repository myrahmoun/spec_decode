"""A tiny, randomly initialised gpt-oss for CPU testing.

Same architecture code as the real gpt-oss-20b (the `transformers` GptOss classes),
just with small dimensions, so routing hooks, tracing and decoding can be tested
on a laptop. Its outputs are meaningless: weights are random.
"""
import torch
from transformers import GptOssConfig, GptOssForCausalLM, Qwen3Config


def tiny_config(**overrides) -> GptOssConfig:
    """Small GptOssConfig. Any field can be overridden, e.g. num_local_experts=16.

    Two layers so we get one sliding-window and one full-attention layer, like
    the real model's alternating pattern. Top-k < num_experts so routing is a
    real choice.
    """
    defaults = dict(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=8,
        num_experts_per_tok=2,
        sliding_window=8,
        # Kept at the real value: RoPE has no learned table (so no cost), and a
        # smaller value contradicts the inherited YaRN scaling and triggers a warning.
        max_position_embeddings=131072,
        layer_types=["sliding_attention", "full_attention"],
    )
    defaults.update(overrides)
    if "num_hidden_layers" in overrides and "layer_types" not in overrides:
        n = defaults["num_hidden_layers"]
        defaults["layer_types"] = [
            "sliding_attention" if i % 2 == 0 else "full_attention" for i in range(n)
        ]
    return GptOssConfig(**defaults)


def tiny_model(seed: int = 0, device: str | torch.device = "cpu", **overrides) -> GptOssForCausalLM:
    """Build a tiny gpt-oss with deterministic random weights, in eval mode.

    `eager` attention: gpt-oss uses attention sinks, which the eager path
    implements on every backend; it is also the easiest to reason about when
    we later pass custom tree attention masks.
    """
    torch.manual_seed(seed)
    config = tiny_config(**overrides)
    config._attn_implementation = "eager"
    model = GptOssForCausalLM(config)
    return model.to(device).eval()


def tiny_backbone_config(**overrides) -> Qwen3Config:
    """Small random Qwen3 config standing in for Qwen3-0.6B as the predictor backbone."""
    defaults = dict(
        vocab_size=16,  # unused: the predictor feeds inputs_embeds
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=1024,
    )
    defaults.update(overrides)
    return Qwen3Config(**defaults)
