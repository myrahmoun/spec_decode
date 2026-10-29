"""Trace collection on the tiny model; no network needed."""
import numpy as np
import pytest
import torch

from ecospec.benchmarks import Sample, split_of
from ecospec.collect import (
    FIXED_DATE, TraceWriter, load_trace, prompt_ids, read_manifest, trace_generation,
)
from ecospec.routing_trace import RouterTraceLogger
from ecospec.tiny import tiny_model


@pytest.fixture(scope="module")
def model():
    return tiny_model(seed=0)


@pytest.fixture(scope="module")
def generation(model):
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, model.config.vocab_size, (12,), generator=g).tolist()
    return ids, trace_generation(model, ids, max_new_tokens=6)


def test_shapes_and_prompt_prefix(model, generation):
    ids, gen = generation
    config = model.config
    assert gen.prompt_len == 12
    assert len(gen.tokens) == 18
    assert gen.tokens[:12].tolist() == ids
    assert gen.experts.shape == (17, config.num_hidden_layers, config.num_experts_per_tok)
    assert gen.tokens.dtype == np.int32 and gen.experts.dtype == np.int16


def test_generation_is_greedy(model, generation):
    _, gen = generation
    with torch.no_grad():
        logits = model(torch.tensor([gen.tokens[:-1].tolist()])).logits[0]
    predicted = logits.argmax(-1)[gen.prompt_len - 1:]
    assert predicted.tolist() == gen.tokens[gen.prompt_len:].tolist()


def test_trace_matches_one_full_pass(model, generation):
    # Routing recorded during cached generation == routing of the whole sequence at once.
    _, gen = generation
    with RouterTraceLogger(model) as log, torch.no_grad():
        model(torch.tensor([gen.tokens[:-1].tolist()]))
    np.testing.assert_array_equal(gen.experts, log.experts)


def test_writer_roundtrip_and_resume(tmp_path, generation):
    _, gen = generation
    sample = Sample("humaneval", "HumanEval/7", "def f():")
    writer = TraceWriter(tmp_path)
    assert not writer.has(sample)
    writer.write(sample, "train", gen)

    resumed = TraceWriter(tmp_path)
    assert resumed.has(sample)
    [record] = read_manifest(tmp_path)
    assert record["file"] == "humaneval/HumanEval_7.npz"
    assert record["new_tokens"] == 6
    loaded = load_trace(tmp_path, record)
    np.testing.assert_array_equal(loaded.tokens, gen.tokens)
    np.testing.assert_array_equal(loaded.experts, gen.experts)
    assert loaded.prompt_len == gen.prompt_len


def test_split_is_stable_and_about_20_percent():
    assert split_of("gsm8k", "5") == split_of("gsm8k", "5")
    splits = [split_of("gsm8k", str(i)) for i in range(2000)]
    assert 0.17 < splits.count("test") / len(splits) < 0.23


class FakeTokenizer:
    """Mimics the gpt-oss template's date stamp."""

    def apply_chat_template(self, messages, **kwargs):
        return f"Current date: 2031-01-01\nReasoning: {kwargs['reasoning_effort']}|{messages[0]['content']}"

    def __call__(self, text, add_special_tokens):
        self.last_text = text
        return {"input_ids": [ord(c) for c in text]}


def test_prompt_date_is_pinned():
    tok = FakeTokenizer()
    ids = prompt_ids(tok, "hi", reasoning_effort="low")
    assert tok.last_text == f"Current date: {FIXED_DATE}\nReasoning: low|hi"
    assert ids == [ord(c) for c in tok.last_text]
