#!/usr/bin/env bash
# Jupyter Lab on a WAVE GPU node, for running reproduce/reproduce.ipynb interactively.
#   sbatch reproduce/wave/jupyter.sbatch        (from the repository root, after setup_env.sh)
# Then read outputs/reproduce/jupyter-<jobid>.log for the SSH tunnel command and the login URL.
#
# Check the partition and GPU names on WAVE first:  sinfo -o "%P %G %l"
# and adjust the --partition / --gres lines. Qwen3-32B in bf16 needs one 80 GB GPU; 40-48 GB runs the AWQ build.
#SBATCH --job-name=ardook-jupyter
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/reproduce/jupyter-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# Model weights (66 GB for Qwen3-32B): point HF_HOME at a disk with room, e.g. your scratch/project space.
export HF_HOME="${HF_HOME:-$PWD/.hf-cache}"
PORT=$(( 8800 + RANDOM % 100 ))
NODE=$(hostname -s)

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "On your laptop run:   ssh -N -L ${PORT}:${NODE}:${PORT} ${USER}@<WAVE login host>"
echo "then open the http://127.0.0.1:${PORT}/lab?token=... URL printed below, and open reproduce/reproduce.ipynb"
.repro-env/bin/jupyter lab --no-browser --ip=0.0.0.0 --port="${PORT}" --notebook-dir="$PWD"
