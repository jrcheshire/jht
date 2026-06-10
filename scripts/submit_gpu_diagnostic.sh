#!/bin/bash
# SLURM submission for the jht single-slot GPU performance diagnostic
# (scripts/gpu_diagnostic.py). jht's GPU numbers have been deferred -- never
# measured; this is the one-shot run that captures them.
#
# Runs on ONE gpu_requeue GPU -- whatever the scheduler hands us, most often a
# 10-20 GB A100 MIG slice, occasionally a whole 80 GB card. The diagnostic
# auto-sizes its ladder to the slice it lands on (memory-driven gating), so no
# tuning per slice is needed; it writes every completed point to JSONL as it goes
# and self-bounds its wall time (--max-wall), so a short or preempted slot still
# yields usable data. Within a slot it runs ongrid -> off-grid -> vmap, i.e. the
# never-measured off-grid (ducc-replacement) capability before the more-inferable
# vmap sweep, so a cut-short run loses the least.
#
# gpu_requeue is a PREEMPTIBLE / backfill partition: a higher-priority job can
# requeue this one at any time. --requeue lets SLURM restart it (from scratch --
# the per-jobid JSONL is rewritten); --open-mode=append keeps the preemption
# history in the .out. With poor FairShare a shorter --time backfills far more
# easily; the defaults below are a middle ground -- bump --time AND MAX_WALL
# together if you want the full ladder to finish on a slow MIG.
#
# PRE-INSTALL ONCE from any networked CPU node -- a login or dev node. This
# downloads + links the CUDA packages into ./.pixi on global
# home, which the gpu_requeue GPU node then reads (so the GPU job needs no network).
# Such nodes have no GPU, so mock the CUDA driver virtual package (__cuda) for the
# solve; the real driver is detected at runtime on the GPU node:
#     cd ~/jht && CONDA_OVERRIDE_CUDA=12.9 pixi install -e gpu
#
# Usage (from the repo root on the cluster):
#     cd ~/jht
#     sbatch scripts/submit_gpu_diagnostic.sh
#     # longer slot for the full ladder on a slow MIG:
#     MAX_WALL=6600 sbatch --time=02:30:00 scripts/submit_gpu_diagnostic.sh
#     # preview / pin a specific budget, or pass any gpu_diagnostic.py flag:
#     DIAG_ARGS="--limit-gb 10" MAX_WALL=2400 sbatch --time=00:50:00 scripts/submit_gpu_diagnostic.sh
#
# Outputs -> runs/gpu-diag/slurm_<jobid>.{out,err} (human table + log) and
#            runs/gpu-diag/diag_<jobid>.jsonl (machine-readable; copy back to
#            analyze locally). runs/ is gitignored.

#SBATCH --job-name=jht-gpu-diag
# #SBATCH --account=<your-slurm-account>   # or: export SBATCH_ACCOUNT=<acct>
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=runs/gpu-diag/slurm_%j.out
#SBATCH --error=runs/gpu-diag/slurm_%j.err

set -euo pipefail

# Honest per-point device memory + real OOMs: disable JAX's default 75 %
# preallocation so memory_stats()['peak_bytes_in_use'] reflects actual growth (not
# the prealloc pool) and an over-large point raises RESOURCE_EXHAUSTED (which the
# diagnostic catches per point) instead of failing the whole job at startup.
# (For maximum memory accuracy at the cost of timing fidelity, swap in
#  XLA_PYTHON_CLIENT_ALLOCATOR=platform.)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# The diagnostic times a CPU leg (the GPU/CPU speedup reference); give BLAS cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

: "${MAX_WALL:=3600}"   # python self-bound (s); keep < --time so it summarizes before SLURM kills
: "${DIAG_ARGS:=}"      # extra args forwarded to gpu_diagnostic.py (e.g. --limit-gb 10)

# Under sbatch $0 is the spool copy at /var/slurmd/...; use the recorded cwd.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(realpath "$0")")/..}"
mkdir -p runs/gpu-diag

echo "=== jht GPU diagnostic ==="
echo "  host           = $(hostname)   date = $(date)"
echo "  SLURM_JOB_ID   = ${SLURM_JOB_ID:-<none>}   restart # = ${SLURM_RESTART_COUNT:-0}"
echo "  CUDA_VISIBLE   = ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "  MAX_WALL       = ${MAX_WALL}s   DIAG_ARGS = ${DIAG_ARGS:-<none>}"
echo "  --- the card / MIG slice we got (nvidia-smi) ---"
nvidia-smi -L || echo "  (nvidia-smi -L unavailable)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "=========================="

OUT="runs/gpu-diag/diag_${SLURM_JOB_ID:-local}.jsonl"
# --frozen: run the login-node-preinstalled env as-is -- no lock re-check, no
# network (compute nodes may have none). The compute node has a real GPU, so the
# __cuda check passes here without the override needed on the login node.
# shellcheck disable=SC2086  # DIAG_ARGS is intentionally word-split into flags
pixi run --frozen -e gpu python scripts/gpu_diagnostic.py --max-wall "${MAX_WALL}" --out "${OUT}" ${DIAG_ARGS}

echo "=== done $(date) -> ${OUT} ==="
