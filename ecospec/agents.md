# EcoSpec reproduction — notes for coding agents

## What this project is

A student reproduction of **EcoSpec** ("Less Experts, Faster Decoding: Cost-Aware Speculative
Decoding for Mixture-of-Experts", arXiv 2607.12696). It is the first step of a larger course
project about expert caching for MoE models under speculative decoding; this repo is only the
EcoSpec reproduction for now.

**The user is new to GPUs and MoE models.** Work in small, reviewable steps. Do one task at a
time, explain what you did and why in plain language, and wait for confirmation before moving
to the next task. Do not implement several components in one go.

## Background in one paragraph

A MoE layer has many experts (small FFNs); a router picks the top-k per token. Speculative
decoding has a small drafter (EAGLE-3) propose a tree of candidate tokens; the target model
verifies a subset of them in one forward pass. In a MoE, that pass must load the *union* of
experts used by all verified tokens, so verification gets slower as candidates route to
different experts ("expert scattering"). EcoSpec trains a small **expert predictor** to guess
each draft token's experts before verification, then **selects** which tree nodes to verify by
trading off draft probability against how many *new* experts a node adds.

## The paper's method (what we are reproducing)

- **Predictor**: target-token ids → the *target's* embedding layer → linear projection →
  Qwen3-0.6B decoder layers → head with `num_layers × num_experts` logits, softmax per layer.
  Loss: per-layer cross-entropy against a uniform distribution over the true top-k experts.
  Training: Adam, lr 1e-5, batch 16, 100 epochs, 80/20 split. Metric: average top-k overlap.
- **Selection** (Algorithm 1): greedy. Score = `path_prob / (|path_experts − buffer| + ε)`.
  Pick the best node, add its path's experts to the buffer, repeat until γ nodes.
- **Config**: draft depth 3, top-2 per step, γ = 4 (includes the root/bonus token), batch size 1,
  greedy (T=0) and sampling (T=1).
- **Metrics**: speedup vs autoregressive decoding, mean acceptance length α, average unique
  experts per MoE layer per verification step (E), predictor accuracy.
- The paper's setting is all experts resident in GPU memory (8× H200). No code release is known.

### Details from the paper (`ecospec/ecospec.pdf`) not stated above

- **Draft tree shape** (Fig. 2): EAGLE-2/3 style — each step expands the top-2 *nodes* of the
  previous depth (by path prob), each into its top-2 children. Depth 3 → 1 + 2 + 4 + 4 = 11
  nodes, not a full binary tree (15).
- **Trace / training data** (App. B.1): the same seven benchmarks used for evaluation — GSM8K,
  HumanEval, AIME-25, Math500, AMC22-24, MT-Bench, MMStar (text only). Random 80/20 split by
  *sample*; all reported evaluations use the held-out 20%. Whether traces cover prompt tokens,
  generated tokens, or both is not stated.
- **Prompts** (App. E): GSM8K 5-shot CoT; AIME `{question}\nPlease reason step by step, and put
  your final answer within \boxed{}.`; Math500/AMC `Problem:\n{problem}\n\nSolution:`;
  HumanEval benchmark prompts; MMStar with images removed. Max new tokens and sample counts are
  not stated (App. A.2 mentions 100 random instances for one analysis).
- **Backend** (§5.1): HF Transformers prototype following the EAGLE-3 pipeline, tree attention
  mask, one target forward per verification. The predictor runs once per step (~4 ms, Table 3).
- **Extra experiments**: Global-Cost ablation (Eq. 10: `P / (|path_experts| + ε)`, no buffer;
  Table 4); oracle expert sets instead of predictions (Table 10); γ ∈ {3,4,5,6} (Tables 11–12);
  predictor checkpoints at ~50/60% accuracy (Table 9).
- **gpt-oss-120b reference numbers** (T=0, averages): EAGLE-3 E 11.6, α 1.88, 1.14×; EcoSpec
  E 10.6, α 1.86, 1.31×; AR E = 4.0; predictor accuracy 93%.
- **Expect less headroom on 20b**: the 120b has 128 experts/layer; the 20b has far fewer (read
  from config), so verified tokens overlap more by chance and EcoSpec's E reduction should shrink.

## Our setup and why

- **Target**: `openai/gpt-oss-20b` (same family as the paper's gpt-oss-120b; MXFP4 fits on a
  24 GB GPU). Verify layer/expert/top-k counts from the model config; do not hardcode them.
- **Drafter**: `RedHatAI/gpt-oss-20b-speculator.eagle3` (vLLM `speculators` format; may need
  weight-name conversion to load in our own HF Transformers loop).
- **Predictor backbone**: `Qwen/Qwen3-0.6B`.
- **Hardware**: development on the user's laptop (CPU, no GPU). Real runs on AWS EC2 g5.xlarge
  (1× A10G 24 GB). The GPU instance may not be available yet (quota pending).
- **Reproducibility target**: hardware-independent metrics (α, E, predictor accuracy) should be
  comparable to the paper's trends. Wall-clock speedup on an A10G is a new measurement, not a
  reproduction of the paper's H200 numbers.

## Repo layout

```
ecospec/          library code
  tree.py         DraftTree / Node — shared data structure for every component
  selection.py    select() (EcoSpec) and select_by_confidence() (EAGLE baseline)
  routing_trace.py RouterTraceLogger: forward hooks on target routers
  benchmarks.py   dataset loaders, paper prompts, stable train/test split
  collect.py      traced greedy generation + per-sample .npz / manifest.jsonl storage
  predictor.py    ExpertPredictor, loss_fn, annotate_tree
  drafter.py      EAGLE-3 tree drafting
  target.py       load target, tree verification
  decode.py       generation loop: draft → predict → select → verify
  tiny.py         tiny random gpt-oss for CPU testing
scripts/          entry points: collect_traces.py, train_predictor.py, generate.py
tests/            pytest unit tests
setup/            bootstrap.sh, check_env.py (EC2 setup; see "EC2" below)
results/          small versioned outputs (JSON, tables)
```

Tasks 1–5 are implemented and tested on CPU (not yet on GPU). `drafter.py`, `target.py` and
`decode.py` are still skeletons whose bodies raise `NotImplementedError`. Keep their signatures
unless there is a good reason; if you change one, say so explicitly. The human-facing overview
(status, run commands, file formats) is the root `README.md`; keep it current when a task lands.

## Task order

Do these one at a time. Each has a "done when" check. Tasks 1–5: done (CPU-tested).

1. **`tree.py` + `selection.py`** (pure Python, no torch).
   Done when `pytest tests/test_selection.py` passes, including the paper's Figure 3 example:
   root t0 (p=1.0, experts {E1..E4}), children t1 (p=0.53, {E4,E6,E7,E8}) and t2 (p=0.4,
   {E3,E4,E5,E6}); with γ=2 the result must be `[t0, t2]` (scores after t0: t1 = 0.53/3 ≈ 0.177,
   t2 = 0.4/2 = 0.2). Add tests for ties, γ larger than the tree, and the confidence baseline.
2. **`tiny.py`**: build a tiny randomly initialised gpt-oss via its config class in
   `transformers` (2 layers, small hidden size, few experts). Done when a forward pass runs on CPU
   in a few seconds.
3. **`routing_trace.py`** against the tiny model. Find router modules generically (do not
   hardcode names). Done when a test shows the logged array has shape
   `[num_tokens, num_layers, top_k]` and matches the router's own top-k when recomputed by hand.
4. **`scripts/collect_traces.py`** (runs on tiny model locally; real model on EC2 later).
5. **`predictor.py` + `scripts/train_predictor.py`** — test on tiny-model traces first.
6. **`drafter.py`, `target.py`, `decode.py`** — the hardest part (tree attention mask,
   acceptance, EAGLE-3 hidden-state plumbing). Consult the official EAGLE repository
   (SafeAILab/EAGLE) for tree verification. Plan this step with the user before writing code.
   **Draft plan (awaiting user review)** — facts from the drafter's `config.json`/`eagle3.py`:
   one Llama-style layer (hidden 2880), `fc` fusing 3 target hidden states, `draft_vocab_size`
   64000 with a `d2t` map to target ids, `eagle_aux_hidden_state_layer_ids: null` (vLLM default
   to confirm: layers 2, L/2, L−3), weights fp32 (1.7 GB). Build in this order, each with a
   CPU test on tiny models (random drafter ⇒ low acceptance, but correctness is still testable):
   a. `target.verify` with a tree attention mask over a KV cache + cache compaction to the
      accepted path. Test: verifying a single chain must give the same logits as a plain forward.
   b. `decode.generate_speculative` with a *stub* drafter (e.g. replays the AR reference with
      random noise). Test: output == AR greedy output exactly, for EcoSpec, confidence and
      Global-Cost selection, γ ∈ {1..6}. This is the losslessness check, independent of EAGLE.
   c. `drafter.py`: load the speculators checkpoint into a small port of its module, capture the
      3 aux hidden states with hooks, expand the 11-node tree (top-2 nodes per depth, 3 steps).
      Test on EC2: mean acceptance α of plain EAGLE-3 selection in a sane range (paper's 120b:
      ~1.9 at γ=4).
   d. Per-step logging (selected nodes, predicted and true experts per verified token, E, α,
      timings) to JSONL, reusing `RouterTraceLogger` during verification.
   Predictor at decode time starts with `annotate_tree` as is (no KV cache); add caching only
   if its time matters.
7. Evaluation harness (not yet specified; ask the user).

## Scope (agreed with the user)

Goal: verify the paper's claims and build a clean base for later expert-caching work, at low
cost and low bug surface. Keep what tests a claim; drop what only adds code paths.

- **Greedy (T=0) only, batch size 1.** No tree rejection sampling. Correctness test: speculative
  output must equal AR greedy output token for token.
- **Datasets: GSM8K, HumanEval, MT-Bench (turn 1 only).** Adding others = one registry entry.
- **Traces = prompt + greedy response** from plain `model.generate` with `RouterTraceLogger` on;
  256 new tokens; ≤160 samples per dataset. Responses double as AR reference outputs.
  AR *timing* is measured separately (hooks add overhead).
- **gpt-oss chat template**, paper's prompt text as the user message, `reasoning_effort="low"`.
- **Extra experiments:** Global-Cost ablation, oracle expert sets, γ ∈ {3,4,5,6}. Not: batch-size
  scaling, predictor-checkpoint sweeps.
- **Save every verification step's true routing** (selected nodes + experts per verified token)
  so caching ideas can be replayed offline.
- gpt-oss-20b (from its config): 24 MoE layers, 32 experts/layer, top-4; routers not quantized.

## Implementation decisions (the paper is silent; record new ones here)

| Decision | Choice | Why |
|---|---|---|
| ε in the score | 1e-6 | not stated in the paper |
| Prefix closure | a node is eligible only if its parent is already selected; root always selected first | Algorithm 1 scores every unselected node, relying on "child score ≤ parent score" (§4.2, Prefix Consistency), which can fail on ties. Our rule gives the same picks except on ties; the root wins Algorithm 1's first pick anyway (P=1, smallest cost) |
| Predictor context for a tree node | context + root-to-node path, one sequence per node (initially) | paper does not specify |
| Train/test split | by sample, not by token | avoids leakage between neighbouring tokens |
| Selection tie-breaking | equal score → higher path_prob, then lower node id | makes results deterministic; same rule for EcoSpec and the confidence baseline |
| Where routing is traced | pre-hook on the router's parent MoE block, recompute `linear + top-k` from router weights | MXFP4's `Mxfp4GptOssMLP` never calls `router.forward`, so router hooks record nothing on the real model |
| Router discovery | duck-typed: module with int `top_k`, int `num_experts`, `weight` of shape `[num_experts, hidden]` | no hardcoded class/attribute names; layer index = position among MoE layers |
| Train/test assignment | sha256 of `dataset/sample_id` < 0.2 → test | stable: adding samples never moves existing ones between splits |
| GSM8K 5-shot exemplars | first 5 problems of the train split, `Question: …\nAnswer: …`, calculator annotations `<<…>>` stripped | paper says "standard 5-shot CoT" without listing exemplars |
| HumanEval / MT-Bench user message | raw benchmark prompt / turn-1 text, no extra instruction | paper: "benchmark prompts as inputs" |
| Chat-template date | pinned to 2025-08-05 (template stamps today's date otherwise) | same prompt → same tokens on any day; needed for the speculative == AR check |
| Predictor target embedding | frozen, shared with the target, not saved in checkpoints | it is the target's; training 1.2 GB of embeddings adds nothing |
| Predictor precision | fp32 weights + bf16 autocast on CUDA | with lr 1e-5, pure-bf16 weights would round most Adam updates away |
| Predictor sequence length | non-overlapping 512-token windows in training; at inference the last 512 − depth context tokens + path | bounded memory and cost; training and inference see the same length |
| Predictor loss reduction | mean over valid tokens × layers (paper sums) | only rescales the loss; makes lr independent of batch length |
| Predictor checkpoint | the final epoch, no best-on-test selection | choosing by test score would leak the test split; paper reports "converged" |
| Predictor sanity floor | report the "most frequent experts per layer" overlap next to predictor accuracy | skewed routing alone can score well; shows what the predictor adds |
| γ < 1 | `ValueError` | root is always verified, so γ ≥ 1; γ > tree size returns every node |

When you make a decision the paper doesn't specify, add a row and tell the user.

## Conventions

- Python 3.11, type hints, small functions, docstrings explaining *why* when non-obvious.
- Tests with pytest; anything that can be tested without a GPU must be.
- Code must run on CPU (tiny model) and GPU; select device from availability, never assume CUDA.
- Read model dimensions from configs, never hardcode 24/32/4.
- Dependencies: exact versions come from `requirements.lock` produced on EC2; do not add heavy
  dependencies without asking.

## Do not

- Commit model weights, traces, checkpoints, `.env`, or `*.pem` files (see `.gitignore`).
  Large artifacts go to S3.
- Download the real 20B model on the user's laptop.
- Modify the target model's routing or the speculative verification rule — the method must
  stay lossless.
- Implement several tasks at once or silently restructure the repo.

## EC2 (for later)

`setup/bootstrap.sh` installs Python 3.11 via uv, then a pinned environment, and writes
`requirements.lock` and `models.lock` at the repo root (commit both); later runs install from the
lock. `setup/check_env.py` confirms gpt-oss-20b stays in MXFP4 on the
A10G (weights < 20 GB) and prints router module names. In transformers 5.22, MXFP4 stays active on
CUDA if compute capability ≥ 7.5 (A10G is 8.6), triton ≥ 3.4, and `kernels` + `accelerate` are
installed (the `gpu` extra); otherwise it silently dequantizes to bf16 (~40 GB, OOM on 24 GB).
If `mxfp4_active` is false, stop and ask the user.
Run order on EC2 (tasks 1–5):
```bash
bash setup/bootstrap.sh && source .venv/bin/activate
python setup/check_env.py                                    # must end with "OK"
python scripts/collect_traces.py --out traces/gpt-oss-20b    # resumable
python scripts/train_predictor.py --traces traces/gpt-oss-20b --out checkpoints/predictor
```
