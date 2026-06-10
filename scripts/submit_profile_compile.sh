#!/bin/bash
# Cannon SLURM submission to CHARACTERIZE the nside>=1024 on-grid compile time (v0.1.1
# item 2).  The nside=2048 ptxas-FAIL is fixed (combined-gather de-unroll, 83993fa); this
# measures the remaining multi-minute COMPILE and attributes it between the Legendre
# recursion and the per-ring-length FFT-unroll assembly (scripts/profile_compile_time.py).
# Local CPU runs already show the FFT unroll is ~85% of the compile and scales as n_groups
# (= nside); this confirms the shape on GPU at the real target nside=2048.
#
# Pre-install once (see submit_gpu_diagnostic.sh): CONDA_OVERRIDE_CUDA=12.9 pixi install -e gpu
#
# Usage (from the repo root on Cannon):
#     cd ~/jht && sbatch scripts/submit_profile_compile.sh
#
# Output -> runs/gpu-diag/profile_compile_<jobid>.{out,err} (copy the .out back to analyze).

# Slice: a guaranteed 20 GB A100 MIG on gpu_test (the nside=2048 full synth compile + run
# needs it; gpu_requeue's random 5/10 GB MIGs OOM).  gpu_test caps <64 GB RAM / <8 CPU / 12 h
# per MIG, so --mem=60G / -c 4 stay under.  --time generous: the nside=2048 synth compile
# alone is multi-minute and the probe compiles it plus the asm/rec stages at four nsides.
#SBATCH --job-name=jht-profile-compile
#SBATCH --account=kovac_lab
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:nvidia_a100_3g.20gb:1
#SBATCH --time=02:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append
#SBATCH --output=runs/gpu-diag/profile_compile_%j.out
#SBATCH --error=runs/gpu-diag/profile_compile_%j.err

set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(realpath "$0")")/..}"
mkdir -p runs/gpu-diag

echo "=== jht on-grid compile-time characterization (recursion vs FFT-unroll assembly) ==="
echo "  host = $(hostname)   date = $(date)   SLURM_JOB_ID = ${SLURM_JOB_ID:-<none>}"
nvidia-smi -L || echo "  (nvidia-smi -L unavailable)"
echo "================================================================================="

# --frozen: use the preinstalled env as-is (no re-solve / no network on the GPU node).
pixi run --frozen -e gpu python scripts/profile_compile_time.py --nsides 256,512,1024,2048

echo "=== done $(date) ==="
