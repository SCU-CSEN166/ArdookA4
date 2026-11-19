#!/usr/bin/env bash
# One-time setup on WAVE (run on the LOGIN node, which has internet), from the repository root:
#   bash reproduce/wave/setup_env.sh
# Makes .repro-env/ (the pipeline + Jupyter, CPU PyTorch only: the model runs in vLLM's own env)
# and .vllm-env/ (vllm 0.18.0, the version the team's server ran). Both are git-ignored.
# Optional: REPRO_PREFETCH_MODEL=Qwen/Qwen3-32B also downloads the weights now (needs HF_HOME on a big disk).
set -euo pipefail
cd "$(dirname "$0")/../.."

mkdir -p outputs/reproduce  # Slurm writes the job logs here and won't create the folder itself

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# uv's own Python build: includes the headers vLLM's Triton kernels compile against.
uv venv --python 3.12 --python-preference only-managed .repro-env
uv pip install --python .repro-env/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .repro-env/bin/python -r requirements.txt jupyterlab nbconvert ipykernel

uv venv --python 3.12 --python-preference only-managed .vllm-env
uv pip install --python .vllm-env/bin/python "vllm==0.18.0"

if [ -n "${REPRO_PREFETCH_MODEL:-}" ]; then
  echo "Downloading ${REPRO_PREFETCH_MODEL} to ${HF_HOME:-$HOME/.cache/huggingface} ..."
  .vllm-env/bin/hf download "${REPRO_PREFETCH_MODEL}"
fi
echo "Done. Next: sbatch reproduce/wave/jupyter.sbatch (interactive) or sbatch reproduce/wave/run.sbatch (batch)."
