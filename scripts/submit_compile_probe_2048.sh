#!/bin/bash
# Cannon SLURM submission to CHARACTERIZE the nside=2048 on-grid compile (item 2).
# The first probe (profile_offgrid_20592188) showed the ~nside-way FFT-unroll -- not the
# map scatter -- is the compile wall (nogath=158s @ nside=1024) and nside=2048 only
# *timed out mid-compile* with OOM events, so we have no 2048 verdict.  This run gives the
# compile its room: big --mem (host RAM is what the unrolled-module compile exhausts) and
# a 2h wall, and skips the slowest `full` (current-scatter) candidate since we already know
# full ~= gath + ~60s.  Question answered: does nside=2048 compile AT ALL with the
# combined-gather, and how long?  -> decides land-gather-cap-1024 vs cap-FFT-restructure.
#
# Pre-install once (see submit_gpu_diagnostic.sh): CONDA_OVERRIDE_CUDA=12.9 pixi install -e gpu
#
# Usage (from the repo root on Cannon):
#     cd ~/jht && sbatch scripts/submit_compile_probe_2048.sh
#
# Output -> runs/gpu-diag/compile2048_<jobid>.{out,err} (copy the .out back to analyze).

#SBATCH --job-name=jht-compile-2048
#SBATCH --account=kovac_lab
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/gpu-diag/compile2048_%j.out
#SBATCH --error=runs/gpu-diag/compile2048_%j.err

set -euo pipefail

# Honest device memory + real ptxas/OOM errors (not the prealloc pool); see
# submit_gpu_diagnostic.sh for the rationale.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(realpath "$0")")/..}"
mkdir -p runs/gpu-diag

echo "=== jht nside=2048 compile characterization (nogath + gath, skip full) ==="
echo "  host = $(hostname)   date = $(date)   SLURM_JOB_ID = ${SLURM_JOB_ID:-<none>}"
nvidia-smi -L || echo "  (nvidia-smi -L unavailable)"
echo "=========================================================================="

# --frozen: use the preinstalled env as-is (no re-solve / no network on the GPU node).
# 1024 first (a ~5-min sanity with the new resources), then the 2048 main event.
pixi run --frozen -e gpu python scripts/profile_ongrid_compile.py \
    --dtype fp64 --nsides 1024,2048 --skip-full

echo "=== done $(date) ==="
