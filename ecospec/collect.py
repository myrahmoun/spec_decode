"""Routing-trace collection: greedy generation with RouterTraceLogger switched on.

One generate() call per prompt yields the predictor's training data (tokens +
true experts), the AR reference output (for the speculative == AR check), and
the token count. Storage is one .npz per sample plus a manifest.jsonl line; the
manifest line is written last, so a sample without one is treated as missing
and re-collected after an interruption.
"""
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ecospec.benchmarks import Sample
from ecospec.routing_trace import RouterTraceLogger

# The gpt-oss template stamps today's date into the system prompt; pin it so the
# same prompt always tokenizes the same way. (gpt-oss release date.)
FIXED_DATE = "2025-08-05"
_DATE_PATTERN = re.compile(r"Current date: \d{4}-\d{2}-\d{2}")


def prompt_ids(tokenizer, user_message: str, reasoning_effort: str = "low") -> list[int]:
    """Token ids of the chat-formatted prompt, ending where the assistant starts."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )
    text, n = _DATE_PATTERN.subn(f"Current date: {FIXED_DATE}", text)
    if n > 1:
        raise ValueError(f"expected at most one date in the chat template, found {n}")
    return tokenizer(text, add_special_tokens=False)["input_ids"]


@dataclass
class TracedGeneration:
    tokens: np.ndarray    # int32 [prompt_len + new_tokens]: prompt then response
    experts: np.ndarray   # int16 [len(tokens) - 1, num_layers, top_k]
    prompt_len: int
    seconds: float        # wall-clock of generate(), *with* tracing hooks


def trace_generation(model, input_ids: list[int], max_new_tokens: int) -> TracedGeneration:
    """Greedy-generate and record the experts of every token fed to the model.

    The last generated token is never fed back in, so it has no routing; hence
    experts has one row fewer than tokens.
    """
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], device=device)
    with RouterTraceLogger(model) as log, torch.no_grad():
        start = time.perf_counter()
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,  # gpt-oss's generation_config defaults to sampling
            temperature=None,
            top_p=None,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start

    tokens = out[0].cpu().numpy().astype(np.int32)
    experts = log.experts
    if experts.shape[0] != len(tokens) - 1:
        raise RuntimeError(
            f"traced {experts.shape[0]} tokens, expected {len(tokens) - 1}; "
            "generate() fed the model something unexpected"
        )
    return TracedGeneration(tokens, experts, len(input_ids), seconds)


def _file_name(sample: Sample) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", sample.sample_id) + ".npz"


class TraceWriter:
    """Writes `out_dir/<dataset>/<id>.npz` and appends `out_dir/manifest.jsonl`."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.out_dir / "manifest.jsonl"
        self.done = {(r["dataset"], r["sample_id"]) for r in read_manifest(self.out_dir)}

    def has(self, sample: Sample) -> bool:
        return (sample.dataset, sample.sample_id) in self.done

    def write(self, sample: Sample, split: str, gen: TracedGeneration) -> None:
        rel_path = Path(sample.dataset) / _file_name(sample)
        (self.out_dir / sample.dataset).mkdir(exist_ok=True)
        np.savez(self.out_dir / rel_path, tokens=gen.tokens, experts=gen.experts,
                 prompt_len=gen.prompt_len)
        record = {
            **asdict(sample),
            "split": split,
            "file": str(rel_path),
            "prompt_len": gen.prompt_len,
            "new_tokens": len(gen.tokens) - gen.prompt_len,
            "seconds_with_hooks": round(gen.seconds, 4),
        }
        with self.manifest.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self.done.add((sample.dataset, sample.sample_id))


def read_manifest(out_dir: Path) -> list[dict]:
    path = Path(out_dir) / "manifest.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_trace(out_dir: Path, record: dict) -> TracedGeneration:
    with np.load(Path(out_dir) / record["file"]) as data:
        return TracedGeneration(
            data["tokens"], data["experts"], int(data["prompt_len"]),
            record["seconds_with_hooks"],
        )
