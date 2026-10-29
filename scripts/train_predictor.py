"""Train the expert predictor on routing traces (paper App. B.1).

Real run (EC2), paper settings:
    python scripts/train_predictor.py --traces traces/gpt-oss-20b --out checkpoints/predictor
Local smoke test on tiny-model traces:
    python scripts/train_predictor.py --traces traces/tiny --backbone tiny --epochs 2 --out checkpoints/tiny

Trains on the 'train' split, reports loss and top-k overlap on 'test' every
--eval-every epochs, and saves the final checkpoint (no best-on-test selection,
which would leak the test set into model choice).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ecospec.collect import load_trace, read_manifest
from ecospec.predictor import (
    MAX_CONTEXT, ExpertPredictor, collate, loss_fn, make_windows, save_predictor,
    target_embedding_for, topk_overlap,
)
from ecospec.tiny import tiny_backbone_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--backbone", default="Qwen/Qwen3-0.6B", help="HF id, or 'tiny'")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--micro-batch", type=int, default=None,
                   help="split each batch into chunks of this size if memory is short (same result)")
    p.add_argument("--max-len", type=int, default=MAX_CONTEXT)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_windows(traces: Path, split: str, max_len: int) -> list:
    windows = []
    for record in read_manifest(traces):
        if record["split"] == split:
            trace = load_trace(traces, record)
            windows += make_windows(trace.tokens, trace.experts, max_len)
    return windows


def frequency_baseline(train: list, test: list, num_experts: int) -> float:
    """Accuracy of always predicting each layer's most frequent training experts.

    A floor for interpreting predictor accuracy: routing can be skewed enough
    that this alone scores well.
    """
    train_e = np.concatenate([e for _, e in train])                 # [n, L, k]
    layers, k = train_e.shape[1:]
    counts = np.stack([np.bincount(train_e[:, l].ravel(), minlength=num_experts)
                       for l in range(layers)])                     # [L, E]
    top = np.argsort(-counts, axis=1)[:, :k]                        # [L, k]
    test_e = np.concatenate([e for _, e in test])
    hits = (test_e[..., None] == top[None, :, None, :]).any(-1).sum(-1)
    return float(hits.mean() / k)


@torch.no_grad()
def evaluate(predictor, windows, batch_size, top_k, device, autocast) -> dict:
    predictor.eval()
    loss_sum = overlap_sum = tokens = 0.0
    for i in range(0, len(windows), batch_size):
        ids, experts, mask = (t.to(device) for t in collate(windows[i:i + batch_size]))
        with autocast():
            logits = predictor(ids)
        n = mask.sum().item()
        loss_sum += loss_fn(logits, experts, top_k, mask).item() * n
        overlap_sum += topk_overlap(logits, experts, mask).item() * n
        tokens += n
    predictor.train()
    return {"test_loss": loss_sum / tokens, "test_topk_overlap": overlap_sum / tokens}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_info = json.loads((args.traces / "run.json").read_text())
    top_k, num_layers, num_experts = run_info["top_k"], run_info["num_moe_layers"], run_info["num_experts"]

    train = load_windows(args.traces, "train", args.max_len)
    test = load_windows(args.traces, "test", args.max_len)
    if not train or not test:
        raise SystemExit(f"need both splits; got {len(train)} train / {len(test)} test windows")
    baseline = frequency_baseline(train, test, num_experts)
    print(f"{len(train)} train / {len(test)} test windows; "
          f"frequency-baseline top-{top_k} overlap on test: {baseline:.3f}", flush=True)

    backbone = tiny_backbone_config() if args.backbone == "tiny" else args.backbone
    embedding = target_embedding_for(run_info)
    predictor = ExpertPredictor(embedding, backbone, num_layers, num_experts).to(device).train()
    params = [p for p in predictor.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    # fp32 master weights + bf16 autocast: with lr 1e-5, pure-bf16 weights would
    # round most updates away.
    use_bf16 = device.type == "cuda"
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)  # noqa: E731
    micro = args.micro_batch or args.batch_size

    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / "metrics.jsonl"
    metrics_path.write_text("")
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        random.shuffle(train)
        epoch_loss = 0.0
        for i in range(0, len(train), args.batch_size):
            batch = train[i:i + args.batch_size]
            batch_tokens = sum(len(t) for t, _ in batch)
            optimizer.zero_grad(set_to_none=True)
            for j in range(0, len(batch), micro):
                ids, experts, mask = (t.to(device) for t in collate(batch[j:j + micro]))
                with autocast():
                    logits = predictor(ids)
                # Weight each chunk by its token share so the total equals the full-batch loss.
                loss = loss_fn(logits, experts, top_k, mask) * mask.sum() / batch_tokens
                loss.backward()
                epoch_loss += loss.item() * len(batch)
            optimizer.step()

        row = {"epoch": epoch, "train_loss": epoch_loss / len(train),
               "seconds": round(time.perf_counter() - start, 1)}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            row |= evaluate(predictor, test, args.batch_size, top_k, device, autocast)
            row["frequency_baseline"] = baseline
        print(json.dumps(row), flush=True)
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    save_predictor(predictor, args.out / "predictor.pt",
                   {"backbone": args.backbone, "top_k": top_k, "max_len": args.max_len,
                    "traces_run": run_info, "args": {k: str(v) for k, v in vars(args).items()}})
    print(f"saved {args.out / 'predictor.pt'}")


if __name__ == "__main__":
    main()
