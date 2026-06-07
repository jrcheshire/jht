# GPU (CUDA) — environment, contract, harness

jht is pure JAX, so the on-grid transforms run on a CUDA GPU with **no code
change** — the static recursion plan is host-side numpy/fp64, and the `jit`-ed
contraction + ring FFTs run on whatever device JAX defaults to. This note pins
the install story, the one footgun (x64), the accuracy tier, and how to validate
on hardware.

**Status:** the GPU env and the parity/benchmark harness are in place; the
*measured* GPU speedup/parity run is **deferred to an NVIDIA box** (Cannon).
Development is on osx-arm64, which has no fp64 GPU (see "Mac-local tier" below).

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

## Known caveat — memory ceiling

At the nside=2048 ceiling the isolated footprint is ~11–13 GB, driven by the
**unrolled per-group cap-FFT scatters** into the full map (XLA does not always
fuse these in place) — *not* the recursion (see `docs/performance.md`). This is
the next memory lever (a pad-and-fold to a common ring length / single combined
scatter). It is **deferred**: bench on the GPU first, optimize only if it bites.

## Open / to settle on Cannon

- Exact `cuda-version` pin vs Cannon's CUDA driver/module (the env targets 12.9
  runtime; the driver only needs to be forward-compatible).
- The card's fp64 throttle factor → expected speedup envelope.
- Whether the memory ceiling forces the pad-and-fold lever in practice.
