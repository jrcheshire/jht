#!/bin/bash
# jht GPU validation on TACC Vista `gh` (one GH200): install the `gpu` env, run the GPU
# parity + reverse-mode-memory + gradient tests, then the GitHub #5 constants probe at
# nside 512 and 2048, each with the in-trace tables and with --old-tables (pre-#5).
#
# pixi runs only inside this job, on the compute node -- never on a login node.
#
# Usage (from the repo root, after `git fetch && git checkout <branch>`):
#     sbatch scripts/vista_gh_validate.slurm
#
# Every leg is logged with its own exit code; a failing leg does not stop the rest
# (the --old-tables nside=2048 leg is expected to fail with a CUBIN out-of-memory).

#SBATCH -J jht-gh-validate
#SBATCH -A JPL-PUB
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o jht_gh_validate.%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
PIXI="${HOME}/.pixi/bin/pixi"
export PYTHONUNBUFFERED=1

leg() {
    local name=$1
    shift
    echo
    echo "########## ${name} $(date -Iseconds)"
    "$@"
    local rc=$?
    echo "########## ${name} exit ${rc} $(date -Iseconds)"
    return ${rc}
}

echo "commit $(git rev-parse --short HEAD) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

leg install "${PIXI}" install --locked -e gpu || { echo "env install failed; stopping"; exit 1; }

leg tests "${PIXI}" run --frozen -e gpu pytest -v -p no:cacheprovider \
    tests/test_gpu.py tests/test_reverse_memory.py tests/test_grad.py

for nside in 512 2048; do
    leg "probe_new_${nside}" "${PIXI}" run --frozen -e gpu \
        python scripts/gpu_constants_probe.py --nside "${nside}"
    leg "probe_old_tables_${nside}" "${PIXI}" run --frozen -e gpu \
        python scripts/gpu_constants_probe.py --nside "${nside}" --old-tables
done

cg="/sys/fs/cgroup$(grep '^0::' /proc/self/cgroup | cut -d: -f3)"
echo
echo "job cgroup memory.peak: $(cat "${cg}/memory.peak" 2>/dev/null) bytes"
exit 0
