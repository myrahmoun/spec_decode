"""Confirm the EC2 environment can run gpt-oss-20b in MXFP4, before any long job.

    python setup/check_env.py

Checks: CUDA + compute capability, triton/kernels/accelerate, that the experts
really stayed MXFP4 (transformers silently falls back to bf16 otherwise), GPU
memory, router discovery, and a short traced generation. Writes
results/env_check.json and exits non-zero if MXFP4 is not active.
"""
import importlib.metadata as md
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ecospec.collect import prompt_ids, trace_generation
from ecospec.routing_trace import find_router_modules

MODEL = "openai/gpt-oss-20b"
OUT = Path(__file__).resolve().parent.parent / "results" / "env_check.json"


def version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; this script is for the GPU instance.")
        return 1
    report = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        **{pkg: version(pkg) for pkg in ["transformers", "triton", "kernels", "accelerate"]},
    }
    print(json.dumps(report, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="cuda").eval()
    except torch.cuda.OutOfMemoryError:
        print("STOP: out of GPU memory while loading. Most likely MXFP4 fell back to bf16 "
              "(check triton/kernels/accelerate above). Ask before continuing.")
        return 1

    experts_class = type(model.model.layers[0].mlp.experts).__name__
    routers = find_router_modules(model)
    names = {id(m): n for n, m in model.named_modules()}
    report |= {
        "experts_module": experts_class,
        "mxfp4_active": "Mxfp4" in experts_class,
        "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
        "num_routers_found": len(routers),
        "num_hidden_layers": model.config.num_hidden_layers,
        "router_names": [names[id(r)] for _, r in routers[:2]] + ["..."],
    }

    ids = prompt_ids(tokenizer, "What is 17 * 23? Answer briefly.")
    gen = trace_generation(model, ids, max_new_tokens=64)
    new = len(gen.tokens) - gen.prompt_len
    report |= {
        "smoke_tokens_per_second_with_hooks": round(new / gen.seconds, 1),
        "smoke_trace_shape": list(gen.experts.shape),
        "smoke_output": tokenizer.decode(gen.tokens[gen.prompt_len:]),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }

    print(json.dumps(report, indent=2))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {OUT}")

    ok = report["mxfp4_active"] and report["num_routers_found"] == report["num_hidden_layers"]
    if not report["mxfp4_active"]:
        print("STOP: experts are not MXFP4 (fell back to bf16). Ask before continuing.")
    elif not ok:
        print("STOP: router count does not match layer count.")
    else:
        print("OK: ready for scripts/collect_traces.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
