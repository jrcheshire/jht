#!/bin/bash
# SLURM submission for the adjoint-stage profiler (scripts/profile_adjoint.py).
# Profile-confirmation step before the on-grid scatter refactor: isolates the
# ring-assembly vs recursion-adjoint stages on GPU, in fp64 and fp32, to pin the
# bottleneck (and to check whether the recursion alone compiles at nside=2048 where
# the full path ptxas-fails). A quick job -- a single ~30-min gpu_requeue slot.
#
# Pre-install once (see submit_gpu_diagnostic.sh): CONDA_OVERRIDE_CUDA=12.9 pixi install -e gpu
#
# Usage (from the repo root on the cluster):
#     cd ~/jht && sbatch scripts/submit_profile_adjoint.sh
#
# Output -> runs/gpu-diag/profile_<jobid>.{out,err} (copy the .out back to analyze).

#SBATCH --job-name=jht-profile-adj
# #SBATCH --account=<your-slurm-account>   # or: export SBATCH_ACCOUNT=<acct>
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/gpu-diag/profile_%j.out
#SBATCH --error=runs/gpu-diag/profile_%j.err

set -euo pipefail

# Honest device memory + real ptxas/OOM errors (not the prealloc pool); see
# submit_gpu_diagnostic.sh for the rationale.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(realpath "$0")")/..}"
mkdir -p runs/gpu-diag

echo "=== jht adjoint profiler ==="
echo "  host = $(hostname)   date = $(date)   SLURM_JOB_ID = ${SLURM_JOB_ID:-<none>}"
nvidia-smi -L || echo "  (nvidia-smi -L unavailable)"
echo "============================"

# --frozen: use the preinstalled env as-is (no re-solve / no network on the GPU node).
echo "--- fp64 ---"
pixi run --frozen -e gpu python scripts/profile_adjoint.py --dtype fp64
echo "--- fp32 ---"
pixi run --frozen -e gpu python scripts/profile_adjoint.py --dtype fp32

echo "=== done $(date) ==="
