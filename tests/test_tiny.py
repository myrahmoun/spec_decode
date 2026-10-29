"""The tiny gpt-oss builds and runs a forward pass quickly on CPU."""
import time

import torch

from ecospec.tiny import tiny_config, tiny_model


def test_forward_pass_runs_fast_on_cpu():
    start = time.perf_counter()
    model = tiny_model()
    config = model.config
    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        out = model(input_ids)
    elapsed = time.perf_counter() - start

    assert out.logits.shape == (1, 16, config.vocab_size)
    assert torch.isfinite(out.logits).all()
    assert elapsed < 5.0, f"tiny model took {elapsed:.1f}s"


def test_same_seed_gives_same_weights():
    a, b = tiny_model(seed=0), tiny_model(seed=0)
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), name


def test_overrides_and_layer_types():
    config = tiny_config(num_hidden_layers=4, num_local_experts=16)
    assert config.num_local_experts == 16
    assert config.layer_types == [
        "sliding_attention", "full_attention", "sliding_attention", "full_attention"
    ]


def test_router_logits_have_expected_shape():
    # Sanity check that the model really is a MoE with the configured expert count.
    model = tiny_model()
    config = model.config
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        out = model(input_ids, output_router_logits=True)
    assert len(out.router_logits) == config.num_hidden_layers
    for logits in out.router_logits:
        assert logits.shape[-1] == config.num_local_experts
