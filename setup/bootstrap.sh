#!/usr/bin/env bash
# One-time setup on the EC2 GPU instance (g5.xlarge, Ubuntu, NVIDIA driver installed).
#
#   git clone <repo> && cd <repo> && bash setup/bootstrap.sh
#
# First run: installs from pyproject.toml and writes requirements.lock + models.lock
# at the repo root (commit both). Later runs: install exactly what requirements.lock says.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# uv gives us Python 3.11 regardless of what the AMI ships.
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
[ -d .venv ] || uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

if [ -f requirements.lock ]; then
    echo "Installing pinned versions from requirements.lock"
    uv pip install -r requirements.lock
    uv pip install --no-deps -e .
else
    echo "No requirements.lock yet: resolving from pyproject.toml"
    uv pip install -e ".[dev,gpu]"
    uv pip freeze --exclude-editable > requirements.lock
    echo "Wrote requirements.lock"
fi

# Download the target weights (~13 GB) and record exact revisions of everything we use.
python - <<'EOF'
import json
from huggingface_hub import HfApi, snapshot_download

api = HfApi()
models = ["openai/gpt-oss-20b", "RedHatAI/gpt-oss-20b-speculator.eagle3", "Qwen/Qwen3-0.6B"]
datasets = ["openai/gsm8k", "openai/openai_humaneval", "HuggingFaceH4/mt_bench_prompts"]
lock = {
    "models": {m: api.model_info(m).sha for m in models},
    "datasets": {d: api.dataset_info(d).sha for d in datasets},
}
snapshot_download("openai/gpt-oss-20b", revision=lock["models"]["openai/gpt-oss-20b"],
                  ignore_patterns=["original/*", "metal/*"])
with open("models.lock", "w") as f:
    json.dump(lock, f, indent=2)
    f.write("\n")
print("Wrote models.lock")
EOF

echo
echo "Next: python setup/check_env.py"
