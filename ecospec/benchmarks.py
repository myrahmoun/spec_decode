"""Benchmark prompts (paper Appendix E) and the stable train/test split.

Each loader returns the first `max_samples` items in dataset order, so runs are
reproducible and a bigger run is a superset of a smaller one.
"""
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass

from datasets import load_dataset

TEST_FRACTION = 0.2


@dataclass(frozen=True)
class Sample:
    dataset: str
    sample_id: str
    user_message: str  # becomes the single user turn of the chat template


def _gsm8k(max_samples: int | None) -> list[Sample]:
    """5-shot CoT: the first 5 train problems as exemplars (paper: "standard 5-shot CoT")."""
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    # Strip the calculator annotations like <<48/2=24>>; they aren't natural text.
    shots = "".join(
        f"Question: {ex['question']}\nAnswer: {re.sub(r'<<[^>]*>>', '', ex['answer'])}\n\n"
        for ex in train.select(range(5))
    )
    rows = test if max_samples is None else test.select(range(min(max_samples, len(test))))
    return [
        Sample("gsm8k", str(i), f"{shots}Question: {row['question']}\nAnswer:")
        for i, row in enumerate(rows)
    ]


def _humaneval(max_samples: int | None) -> list[Sample]:
    """The raw benchmark prompt (function signature + docstring)."""
    rows = load_dataset("openai/openai_humaneval", split="test")
    samples = [Sample("humaneval", row["task_id"], row["prompt"]) for row in rows]
    return samples[:max_samples]


def _mt_bench(max_samples: int | None) -> list[Sample]:
    """First turn only (scope decision: no multi-turn)."""
    rows = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    samples = [Sample("mt_bench", str(row["prompt_id"]), row["prompt"][0]) for row in rows]
    return samples[:max_samples]


LOADERS: dict[str, Callable[[int | None], list[Sample]]] = {
    "gsm8k": _gsm8k,
    "humaneval": _humaneval,
    "mt_bench": _mt_bench,
}


def load_samples(name: str, max_samples: int | None = None) -> list[Sample]:
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(LOADERS)}")
    return LOADERS[name](max_samples)


def split_of(dataset: str, sample_id: str, test_fraction: float = TEST_FRACTION) -> str:
    """'train' or 'test', from a hash of the id, so it never depends on run order or size."""
    digest = hashlib.sha256(f"{dataset}/{sample_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "test" if fraction < test_fraction else "train"
