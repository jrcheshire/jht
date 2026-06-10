#!/bin/bash
# Cannon SLURM submission to CONFIRM the real on-grid transform at nside=2048 (item 2).
# The combined-gather ring assembly (commit 83993fa) FIXED the nside=2048 compile: the
# old per-group scatter ptxas-FAILed, the gather's jit_synth compiles and synth runs at
# nside=2048 (verified -- ran even on a 5 GB MIG; only map2alm OOMed there at runtime for
# lack of device memory).  This run confirms the END-TO-END public transform (jht.synthesis
# + jht.map2alm) matches CPU to ~1e-12 at nside=2048 on a slice big enough to hold both --
# the basis for documenting the (compile + ~13 GB runtime) requirement and clearing the
# release-blocker.  Parity-only (the vmap timing leg OOMs at nside=2048 batch=8).
#
# Pre-install once (see submit_gpu_diagnostic.sh): CONDA_OVERRIDE_CUDA=12.9 pixi install -e gpu
#
# Usage (from the repo root on Cannon):
#     cd ~/jht && sbatch scripts/submit_confirm_2048.sh
#
# Output -> runs/gpu-diag/confirm2048_<jobid>.{out,err} (copy the .out back to analyze).

# Slice: request a specific 20 GB A100 MIG on the gpu_test partition (guaranteed size,
# vs gpu_requeue's random 5/10/20 GB) -- see FASRC docs "Using GPUs".  20 GB holds the
# nside=2048 synth AND map2alm at runtime (a 5 GB MIG OOMed map2alm).  gpu_test caps each
# MIG at <64 GB RAM / <8 CPUs and 12 h, so --mem=60G / -c 4 stay under the limit (the
# gather'd synth compiles well under 60 G host RAM).  (Drop --account if it is rejected
# on gpu_test; it only sets fairshare.)
#SBATCH --job-name=jht-confirm-2048
#SBATCH --account=kovac_lab
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --time=02:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=runs/gpu-diag/confirm2048_%j.out
#SBATCH --error=runs/gpu-diag/confirm2048_%j.err

set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(realpath "$0")")/..}"
mkdir -p runs/gpu-diag

echo "=== jht real-transform confirmation at nside=2048 (GPU-vs-CPU parity) ==="
echo "  host = $(hostname)   date = $(date)   SLURM_JOB_ID = ${SLURM_JOB_ID:-<none>}"
nvidia-smi -L || echo "  (nvidia-smi -L unavailable)"
echo "========================================================================"

# --frozen: use the preinstalled env as-is (no re-solve / no network on the GPU node).
pixi run --frozen -e gpu python scripts/gpu_check.py --min 2048 --max 2048 --parity-only

echo "=== done $(date) ==="
