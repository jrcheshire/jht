# GPU (CUDA) — environment, contract, harness

jht is pure JAX, so the on-grid transforms run on a CUDA GPU with **no code
change** — the static recursion plan is host-side numpy/fp64, and the `jit`-ed
contraction + ring FFTs run on whatever device JAX defaults to. This note pins
the install story, the one footgun (x64), the accuracy tier, and how to validate
on hardware.

**Status (2026-06-08):** measured on Cannon `gpu_requeue` A100 (MIG) slices and a
V100 — see "Measured GPU performance" below. fp64 GPU==CPU parity holds; forward
synthesis and (after the `dense_to_tri` gather fix) the adjoint / `map2alm` and the
off-grid path all run well at nside ≤ 1024. Two GPU items remain: off-grid *forward*
speed and the nside=2048 compile ceiling. Development is on osx-arm64, which has no
fp64 GPU (see "Mac-local tier" below).

## The `gpu` pixi environment

conda-forge `jaxlib` defaults to a CPU build; the CUDA build is selected in a
dedicated pixi environment (`pyproject.toml`):

```toml
[tool.pixi.feature.gpu]
platforms = ["linux-64"]                 # no CUDA on osx-arm64
[tool.pixi.feature.gpu.system-requirements]
cuda = "12"
[tool.pixi.feature.gpu.dependencies]
python = "3.13.*"                        # conda-forge CUDA jaxlib is py312/py313, NOT py314
jaxlib = { build = "cuda*" }
cuda-version = ">=12.9,<13"
[tool.pixi.environments]
gpu = { features = ["gpu"] }
```

Notes:
- **linux-64 only.** The `gpu` environment does not exist on osx-arm64.
- **python 3.13.** conda-forge ships the CUDA `jaxlib` for py312/py313 but not
  py314, so the `gpu` env pins 3.13 while the default CPU env stays on 3.14.
  Numerics are identical — it is pure JAX.
- The env pulls the CUDA runtime (cudart / cupti / nvrtc / nvtx, CUDA 12.9) plus
  the usual validation oracles (healpy, ducc0) so the gated suite runs there too.

The lock is solved for all platforms from anywhere (`pixi install` re-solves;
a GPU is **not** needed to produce/commit `pixi.lock`). You only need the GPU to
*run* the env:

```bash
# on the NVIDIA box (e.g. Cannon):
pixi run -e gpu python scripts/gpu_check.py
pixi run -e gpu test                       # the full suite, incl. the GPU parity gate
```

## The x64 footgun (read this)

Library code never touches the global JAX config — float64 is opt-in **per entry
point**. On GPU the failure mode is silent: without x64 every array is float32
and you drop from the ~1e-13 tier to ~1e-3 or worse. Set it **before any array is
created**:

```python
import jax
jax.config.update("jax_enable_x64", True)   # FIRST, before importing/allocating
```

## Accuracy tier on CUDA

Same contract as CPU: float64, weighted + iterated → ~1e-13 on band-limited
maps; bare floor ~1e-3. The GPU parity gate asserts **GPU == CPU to ~1e-12** in
fp64 (it is the same XLA program). Caveat: fp64 throughput is throttled on many
NVIDIA cards (consumer/older parts especially) — fp64 *correctness* is unaffected,
but the *speedup* depends on the card's fp64:fp32 ratio. This is the production
GPU tier.

**Mac-local tier (out of scope here).** Apple Silicon GPUs have no fp64, so any
Mac-GPU path (MLX) is a separate lower-accuracy fp32 tier — a future
local-dev convenience, not this CUDA contract. (Internal note:
`jht-mac-gpu-fp32`.)

## The harness — `scripts/gpu_check.py`

Across the BK regime (nside / lmax / spin) it:
1. reports the active device(s);
2. checks **fp64 parity** between the CPU and GPU backends (`jax.device_put` to
   each) to ~1e-12 — skipped if no GPU is visible;
3. times `synthesis` / `adjoint_synthesis` / `map2alm`, including a
   `vmap`-over-realizations batch;
4. reports peak memory.

On a CPU-only machine it still runs (self-consistency + timing) so it is never
dead code; on the GPU box it produces the parity + speedup numbers. The pytest
gate `tests/test_gpu.py` makes the parity assertion part of the suite and
`pytest.skip`s cleanly when no GPU is present.

## The single-slot diagnostic — `scripts/gpu_diagnostic.py`

`gpu_check.py` is the parity gate plus a quick ladder. For the **one** moderate-scale
`gpu_requeue` slot the GPU numbers have been waiting on, use the purpose-built
`scripts/gpu_diagnostic.py` — designed to extract everything from a single job so a
second is not needed:

- GPU-vs-CPU **speedup** and the card's **fp64/fp32 throttle** factor (the same op
  timed in both dtypes — fp32 is *throughput-only*, never an accuracy claim);
- fp64 **GPU==CPU parity** measured on-device;
- the on-device **memory ceiling** per `(nside, lmax, spin)`, including nside=2048;
- **vmap throughput** vs batch size (the forecast-sweep per-realization knee);
- **off-grid NUFFT at production scale** (lmax~1000, N up to 1e6), measured vs
  predicted footprint (the `(Npts, W, W)` stencil — see `docs/offgrid.md`);
- the off-grid **pointing-gradient** cost (the ducc-`NotImplementedError`
  capability); compile time, reported separately everywhere.

Each measurement point runs in a **fresh subprocess**: this gives a clean per-point
device peak (`peak_bytes_in_use` is a cumulative counter), survives per-point OOM
(one OOM never wastes the slot), and toggles x64 per point.

`gpu_requeue` hands you *whatever is free* — most often a **10–20 GB A100 MIG
slice**, occasionally a whole 80 GB card. So the ladder is **memory-driven**: each
heavy point is gated by a predicted device footprint against the detected (or
`--limit-gb`) memory, and the run fills whatever it lands on — a MIG slice still
reaches the production off-grid point (lmax≈1000), which is memory-light
(`grid + Npts·W²`), even when it cannot reach nside=2048 on-grid. The per-point OOM
guard is the true boundary; `--max-wall` makes a short slot still save every
completed point; output is a human table + machine-readable JSONL (re-plot without
a second job) + a derived summary.

```bash
# preview what a given slice would run, from anywhere (laptop is fine):
python scripts/gpu_diagnostic.py --dry-run --limit-gb 10      # a 10 GB MIG slice
python scripts/gpu_diagnostic.py --dry-run --limit-gb 80      # a full A100
# the run on the box (auto-detects the slice; cap total wall time for a short slot):
pixi run -e gpu python scripts/gpu_diagnostic.py --max-wall 1800 --out cannon_run.jsonl
```

On a CPU-only box it runs a reduced self-test exercising every path, so it can be
smoke-tested before Cannon.

### On Cannon (batch only) — `scripts/submit_gpu_diagnostic.sh`

There are no interactive GPU nodes, so run it via SLURM. The submit script targets
`--account=kovac_lab --partition=gpu_requeue --gres=gpu:1` and inherits all the
single-slot design above: it auto-sizes to whatever MIG slice / card it lands on,
`--max-wall`-bounds itself, and runs off-grid before vmap so a preempted slot loses
the least. It logs `nvidia-smi -L` so the result is tied to the exact slice, and
sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` for honest per-point device memory.

```bash
# ONCE from any networked CPU node (a login node, or a dev node like bicep-dev) --
# it has no GPU, so mock the CUDA driver virtual package (__cuda) for the solve;
# the real driver is detected at runtime on the GPU node. The env lands in ./.pixi
# on global home, which the gpu_requeue node reads; the submit script then runs it
# with `pixi run --frozen` (no re-solve / no network on the GPU node).
cd ~/jht && CONDA_OVERRIDE_CUDA=12.9 pixi install -e gpu
# submit (gpu_requeue is preemptible -- the per-jobid JSONL is the protection):
sbatch scripts/submit_gpu_diagnostic.sh
# full ladder on a slow MIG (bump --time AND MAX_WALL together):
MAX_WALL=6600 sbatch --time=02:30:00 scripts/submit_gpu_diagnostic.sh
```

Outputs land in `runs/gpu-diag/` (gitignored): `slurm_<jobid>.out` (human table) and
`diag_<jobid>.jsonl` (copy back here to analyze).

## Measured GPU performance (2026-06-08)

Measured on Cannon `gpu_requeue` A100 MIG slices (10–25 GB) and a V100-16GB, fp64
(`scripts/gpu_diagnostic.py`, `scripts/profile_adjoint.py`):

- **fp64 GPU==CPU parity:** relerr 1e-14 … 1e-13 across nside 256–1024, spin 0/2 —
  the ~1e-12 contract holds on hardware.
- **Forward synthesis:** 14–60× the (8-core) CPU; **fp64/fp32 ≈ 2.2×** (clean A100
  fp64 — no consumer-card throttle on these parts).
- **Adjoint / `map2alm`:** the dense→triangular alm packing was a **scatter**
  (`at[idx].set(mode="drop")`), which is catastrophically slow in fp64 on GPU
  (~38000× a gather; nside=512 fp64 it was 3.8 s of a 4.0 s adjoint). Fixed by
  packing via a **gather** (`dense_to_tri`, mirroring the forward `tri_to_dense`):
  adjoint **4041 → 187 ms** at nside=512 (~21×), so `map2alm` drops ~16 s → ~1.3 s.
  The on-grid analysis / MUSE-Wiener adjoint path is now usable on GPU. *(Lesson:
  never `at[idx].set` for fp64/complex alm packing on GPU — use a gather.)*
- **Off-grid:** correct and **memory-light** — lmax=1000, N=1e6 peaks ~1.1 GB, well
  under the predicted ceiling (XLA fuses the `(Npts,W,W)` stencil rather than
  materializing it). The **pointing gradient costs ~1× a forward** (the
  ducc-`NotImplementedError` capability, cheap on GPU). The adjoint shares the
  gather fix.
- **Batched (`vmap`) synthesis:** per-realization time falls with batch (the
  forecast-sweep knee), e.g. ~286 → ~168 ms/realization at nside=512 (batch 1→4).

### Two GPU items still open

- **Off-grid *forward* synthesis is slow** — ~35 s at lmax=1000, *independent of N*
  (so recursion/DFS-bound, the general `synth_contract`, not the NUFFT) and ~50× the
  equivalent on-grid synthesis. Un-tuned for GPU; separate from the adjoint fix.
- **nside=2048 does not compile on GPU** — `ptxas` exits non-zero (code 9): the
  ~nside-way *unrolled* per-ring-group cap-FFT assembly is too large a kernel for the
  PTX assembler (and already compiles for minutes at nside=1024). This is a
  **compile** ceiling, not memory (measured footprints are small, ~1–2 GB at
  nside=1024). The fix is to **de-unroll** that assembly (batch/bucket the ring FFTs
  + a single combined gather/scatter). Above the typical MUSE nside (≤1024), so
  deferred.

## Open / to settle on Cannon

- Exact `cuda-version` pin vs Cannon's CUDA driver/module (the env targets 12.9
  runtime; the driver only needs to be forward-compatible). Confirmed working on the
  A100/V100 `gpu_requeue` nodes via `pixi run --frozen -e gpu`.
- ~~fp64 throttle factor~~ → measured ≈ 2.2× (A100/V100; good fp64 parts).
- ~~Whether the memory ceiling forces a rewrite~~ → the nside=2048 blocker is the
  `ptxas` **compile** ceiling (de-unroll), not memory; see above.
