"""Collect routing traces: greedy-generate on benchmark prompts with router hooks on.

Real run (EC2):
    python scripts/collect_traces.py --out traces/gpt-oss-20b
Local smoke test (CPU, tiny random model, real tokenizer):
    python scripts/collect_traces.py --model tiny --max-samples 2 --max-new-tokens 8 --out traces/tiny

Re-running with the same --out resumes: samples already in manifest.jsonl are skipped.
"""
import argparse
import json
import platform
import subprocess
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ecospec.benchmarks import LOADERS, load_samples, split_of
from ecospec.collect import FIXED_DATE, TraceWriter, prompt_ids, trace_generation
from ecospec.routing_trace import find_router_modules
from ecospec.tiny import tiny_model

DEFAULT_MODEL = "openai/gpt-oss-20b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id, or 'tiny' for the CPU test model")
    p.add_argument("--tokenizer", default=DEFAULT_MODEL)
    p.add_argument("--datasets", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    p.add_argument("--max-samples", type=int, default=160, help="per dataset")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--reasoning-effort", default="low", choices=["low", "medium", "high"])
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_model(name: str, tokenizer):
    if name == "tiny":
        # Vocab must cover the real tokenizer's ids, since prompts are real text.
        return tiny_model(vocab_size=len(tokenizer))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(name, dtype="auto", device_map=device)
    return model.eval()


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_run_info(out: Path, args: argparse.Namespace, model) -> None:
    """Record what produced these traces. Refuse to mix runs with different settings."""
    config = model.config
    info = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "max_new_tokens": args.max_new_tokens,
        "reasoning_effort": args.reasoning_effort,
        "fixed_date": FIXED_DATE,
        "vocab_size": config.vocab_size,
        "num_moe_layers": len(find_router_modules(model)),
        "num_experts": config.num_local_experts,
        "top_k": config.num_experts_per_tok,
        "experts_module": type(model.model.layers[0].mlp.experts).__name__,
    }
    path = out / "run.json"
    if path.exists():
        previous = json.loads(path.read_text())
        mismatched = {k for k in info if previous.get(k) != info[k]}
        if mismatched:
            raise SystemExit(f"{path} has different settings for {sorted(mismatched)}; use a new --out")
    info |= {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "git_commit": git_commit(),
    }
    path.write_text(json.dumps(info, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_model(args.model, tokenizer)
    write_run_info(args.out, args, model)
    writer = TraceWriter(args.out)

    for dataset in args.datasets:
        samples = load_samples(dataset, args.max_samples)
        todo = [s for s in samples if not writer.has(s)]
        print(f"{dataset}: {len(samples)} samples, {len(samples) - len(todo)} already done", flush=True)
        for i, sample in enumerate(todo, 1):
            ids = prompt_ids(tokenizer, sample.user_message, args.reasoning_effort)
            gen = trace_generation(model, ids, args.max_new_tokens)
            writer.write(sample, split_of(sample.dataset, sample.sample_id), gen)
            new = len(gen.tokens) - gen.prompt_len
            print(f"  [{i}/{len(todo)}] {sample.sample_id}: prompt {gen.prompt_len}, "
                  f"new {new}, {new / gen.seconds:.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
