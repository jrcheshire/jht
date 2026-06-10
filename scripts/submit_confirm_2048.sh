#!/bin/bash
# Cannon SLURM submission to CONFIRM the real on-grid transform at nside=2048 (item 2).
# The compile-characterization probe (compile2048_20633198) showed the ring assembly
# compiles in ~7 min with 64G host RAM and runs ~0.1s -- so the earlier "ptxas-FAIL" was
# host-RAM OOM during compile, not a defect.  This run confirms the END-TO-END public
# transform (jht.synthesis + jht.map2alm, the recursion ON TOP of the assembly) compiles,
# runs, and matches CPU to ~1e-12 at nside=2048 -- the basis for documenting the resource
# requirement and clearing the release-blocker.  Parity-only (the vmap timing leg OOMs at
# nside=2048 batch=8); generous 64G / 90 min for the several ~7-8 min GPU compiles.
#
# Pre-install once (see submit_gpu_diagnostic.sh): CONDA_OVERRIDE_CUDA=12.9 pixi install -e gpu
#
# Usage (from the repo root on Cannon):
#     cd ~/jht && sbatch scripts/submit_confirm_2048.sh
#
# Output -> runs/gpu-diag/confirm2048_<jobid>.{out,err} (copy the .out back to analyze).
# NOTE: needs a >=24 GB GPU slice (nside=2048 on-grid peak ~11-13 GB); a small MIG will OOM.

#SBATCH --job-name=jht-confirm-2048
#SBATCH --account=kovac_lab
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --requeue
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
