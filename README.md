# EcoSpec reproduction

A reproduction of **EcoSpec** ("Less Experts, Faster Decoding: Cost-Aware Speculative Decoding
for Mixture-of-Experts", arXiv 2607.12696; PDF in [`ecospec/ecospec.pdf`](ecospec/ecospec.pdf))
on **gpt-oss-20b**, using one 24 GB GPU. It checks the paper's claims and serves as the base for
later work on expert caching for MoE models under speculative decoding.

## The idea in brief

In speculative decoding, a small *drafter* proposes a tree of candidate tokens and the large
*target* model checks several of them in one forward pass. In a Mixture-of-Experts target, that
pass must load every expert used by *any* checked token, so checking tokens that route to
different experts makes verification slower. EcoSpec trains a small **expert predictor** to
guess each candidate's experts in advance, then **selects** which candidates to verify by
weighing draft probability against how many *new* experts each one would add:

```
score(node) = path_prob / (|experts on root→node path − experts already selected| + ε)
```

One decoding round, and where each step lives:

```
drafter.py        target.py               predictor.py        selection.py        target.py
draft tree  ──►  (hidden states)  ──►  annotate_tree()  ──►  select()  ──►  verify selected nodes
(11 nodes)                             predicted experts     γ = 4 nodes      in one forward pass
```

## Status

| Step | Code | State |
|---|---|---|
| Draft-tree data structure + selection (EcoSpec, confidence baseline, Global-Cost ablation) | `tree.py`, `selection.py` | done, tested |
| Tiny random gpt-oss for CPU tests | `tiny.py` | done, tested |
| Recording which experts the target actually uses | `routing_trace.py` | done, tested |
| Collecting routing traces on benchmarks | `benchmarks.py`, `collect.py`, `scripts/collect_traces.py` | done, tested on CPU; not yet run on GPU |
| Expert predictor + training | `predictor.py`, `scripts/train_predictor.py` | done, tested on CPU; not yet run on GPU |
| EAGLE-3 drafter, tree verification, decoding loop | `drafter.py`, `target.py`, `decode.py` | not started (draft plan in `ecospec/agents.md`, task 6) |
| Evaluation (speedup, acceptance length α, active experts E) | — | not started |

"Tested" means unit tests on a tiny random model on CPU. Nothing has run on the real 20b yet.

## Repository map

```
ecospec/            library code (each module's docstring explains its role)
  tree.py           DraftTree / Node: the structure every step reads or writes
  selection.py      select() = EcoSpec; select_by_confidence() = EAGLE baseline; select_global_cost() = ablation
  routing_trace.py  RouterTraceLogger: records the experts the target picks per token and layer
  benchmarks.py     GSM8K / HumanEval / MT-Bench prompts and the fixed train/test split
  collect.py        greedy generation with tracing; trace file format
  predictor.py      ExpertPredictor, loss, accuracy metric, annotate_tree
  tiny.py           tiny random gpt-oss and Qwen3 models for CPU tests
  drafter.py, target.py, decode.py   skeletons (task 6)
  agents.md         detailed working notes: paper details, scope, every design decision
scripts/            entry points: collect_traces.py, train_predictor.py
setup/              EC2 setup: bootstrap.sh (environment + lock files), check_env.py (GPU checks)
tests/              pytest suite, runs on CPU in ~2 s
results/            small outputs that are committed (e.g. env_check.json)
```

## Running

**Laptop (CPU, tests and smoke runs).** Needs Python ≥ 3.11. Downloads only tokenizers and
datasets, never the 20b weights.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/collect_traces.py --model tiny --max-samples 10 --max-new-tokens 16 --out traces/tiny
python scripts/train_predictor.py --traces traces/tiny --backbone tiny --epochs 2 --out checkpoints/tiny
```

**EC2 (g5.xlarge, 1× A10G 24 GB, Ubuntu with an NVIDIA driver).**

```bash
bash setup/bootstrap.sh && source .venv/bin/activate   # Python 3.11, packages, ~13 GB weights, lock files
python setup/check_env.py                              # must end with "OK" (model stays in MXFP4)
python scripts/collect_traces.py --out traces/gpt-oss-20b
python scripts/train_predictor.py --traces traces/gpt-oss-20b --out checkpoints/predictor
```

`collect_traces.py` resumes where it stopped if re-run with the same `--out`.

## Output files

Traces, checkpoints and weights are not committed (see `.gitignore`); large ones belong in S3.

**`traces/<run>/`** (from `collect_traces.py`)
- `run.json`: model, settings, library versions, git commit. A run refuses to add to a folder
  made with different settings.
- `manifest.jsonl`: one line per sample: `dataset`, `sample_id`, `user_message`, `split`
  (`train`/`test`), `file`, `prompt_len`, `new_tokens`, `seconds_with_hooks`.
- `<dataset>/<sample_id>.npz`:
  - `tokens`: int32 `[prompt_len + new_tokens]`, the prompt then the greedy response.
  - `experts`: int16 `[len(tokens) − 1, num_moe_layers, top_k]`. Row *i* is the experts token
    *i* was routed to, strongest first. The last token has no row because it is never fed back
    into the model.
  - `prompt_len`.

**`checkpoints/<run>/`** (from `train_predictor.py`)
- `predictor.pt`: predictor weights (without the target embedding, which is reloaded from the
  target) plus settings; load with `ecospec.predictor.load_predictor`.
- `metrics.jsonl`: per-epoch train loss; every few epochs, test loss and test top-k overlap next
  to a "most frequent experts" baseline.

**`results/env_check.json`** (from `check_env.py`): GPU, library versions, whether MXFP4 is
active, memory use, and a short traced generation.

## Differences from the paper

| | Paper | Here |
|---|---|---|
| Target | gpt-oss-120b (36 layers, 128 experts, top-4), plus Qwen3-235B and DeepSeek-V3.1 | gpt-oss-20b (24 layers, 32 experts, top-4) |
| Hardware | 8× H200 | 1× A10G 24 GB |
| Drafter | `lmsys/EAGLE3-gpt-oss-120b-bf16` | `RedHatAI/gpt-oss-20b-speculator.eagle3` |
| Benchmarks | 7 | GSM8K, HumanEval, MT-Bench (turn 1) |
| Decoding | greedy and sampling | greedy only |

The hardware-independent metrics (α, E, predictor accuracy) should follow the paper's trends.
With 32 experts instead of 128, verified tokens share experts more often by chance, so the
reduction in E should be smaller. Wall-clock speedup on an A10G is a new measurement, not a
reproduction of the H200 numbers.

Where the paper is silent, the choice made and the reason are recorded in the "Implementation
decisions" table in [`ecospec/agents.md`](ecospec/agents.md). That file also holds the agreed
scope and paper details beyond this README.
