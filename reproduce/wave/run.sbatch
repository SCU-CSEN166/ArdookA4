#!/usr/bin/env bash
# Run reproduce/reproduce.ipynb start to finish on a WAVE GPU node, no browser needed.
#   sbatch reproduce/wave/run.sbatch                                   # live, quick subset (40 grants per set)
#   sbatch --export=ALL,REPRO_SUBSET=full reproduce/wave/run.sbatch    # live, all 478 grants
#   sbatch --export=ALL,REPRO_LAUNCHER=apptainer reproduce/wave/run.sbatch   # if the vLLM wheel won't run on WAVE
# The executed notebook (with every table) is written to outputs/reproduce/reproduce-<jobid>.ipynb.
#
# Check partition / GPU names first (sinfo -o "%P %G %l") and adjust the lines below.
#SBATCH --job-name=ardook-repro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/reproduce/run-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
export REPRO_MODE="${REPRO_MODE:-live}" REPRO_SUBSET="${REPRO_SUBSET:-quick}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)   mode=${REPRO_MODE} subset=${REPRO_SUBSET}"

.repro-env/bin/jupyter nbconvert --to notebook --execute reproduce/reproduce.ipynb \
  --ExecutePreprocessor.timeout=-1 --output-dir outputs/reproduce --output "reproduce-${SLURM_JOB_ID}.ipynb"
echo "Finished: outputs/reproduce/reproduce-${SLURM_JOB_ID}.ipynb"
