#!/bin/bash
# Cannon SLURM submission for BOTH this session's perf probes -- one slot answers both
# gating measurements before any refactor:
#   1) profile_offgrid_forward.py -- pins which N-independent stage carries the off-grid
#      forward ~35 s phantom at lmax=1000 (leading suspect: the nufft2d2 grid-build
#      SCATTER `C.at[].set`).  [Item 1a]
#   2) profile_ongrid_compile.py  -- decides the nside=2048 ptxas-FAIL: is it the map
#      SCATTER (then the combined-gather fix works) or the FFTs themselves (then it is
#      insufficient)?  This leg traces a ~2048-way unrolled kernel and can be slow to
#      compile (minutes); --open-mode=append means the off-grid result above is already
#      written even if this leg runs long or ptxas-fails.  [Item 2a]
#
# Pre-install once (see submit_gpu_diagnostic.sh): CONDA_OVERRIDE_CUDA=12.9 pixi install -e gpu
#
# Usage (from the repo root on Cannon):
#     cd ~/jht && sbatch scripts/submit_profile_offgrid.sh
#
# Output -> runs/gpu-diag/profile_offgrid_<jobid>.{out,err} (copy the .out back to analyze).

#SBATCH --job-name=jht-perf-probes
#SBATCH --account=kovac_lab
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/gpu-diag/profile_offgrid_%j.out
#SBATCH --error=runs/gpu-diag/profile_offgrid_%j.err

set -euo pipefail

# Honest device memory + real ptxas/OOM errors (not the prealloc pool); see
# submit_gpu_diagnostic.sh for the rationale.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(realpath "$0")")/..}"
mkdir -p runs/gpu-diag

echo "=== jht perf probes (off-grid forward pin + on-grid compile probe) ==="
echo "  host = $(hostname)   date = $(date)   SLURM_JOB_ID = ${SLURM_JOB_ID:-<none>}"
nvidia-smi -L || echo "  (nvidia-smi -L unavailable)"
echo "====================================================================="

# --frozen: use the preinstalled env as-is (no re-solve / no network on the GPU node).
echo "--- [1a] off-grid forward profiler, fp64 spin 0 ---"
pixi run --frozen -e gpu python scripts/profile_offgrid_forward.py --dtype fp64 --spin 0
echo "--- [1a] off-grid forward profiler, fp64 spin 2 ---"
pixi run --frozen -e gpu python scripts/profile_offgrid_forward.py --dtype fp64 --spin 2

echo "--- [2a] on-grid compile probe, fp64 nside 1024,2048 ---"
pixi run --frozen -e gpu python scripts/profile_ongrid_compile.py --dtype fp64 --nsides 1024,2048

echo "=== done $(date) ==="
