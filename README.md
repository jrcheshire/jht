# jht — JAX Harmonic Transforms

JAX-native spherical harmonic transforms (map ↔ aₗₘ): **GPU-capable**, **fully
differentiable**, and **dependency-controlled** (pure JAX + numpy at runtime — no
compiled C++ extension, no heavyweight third-party SHT library). Scoped to the
BICEP/Keck regime — **spin-0 and spin-2** on the **HEALPix RING** pixelization,
ℓ_max ≲ 1000, nside ≤ ~2048 — but written cleanly so it can serve as a general
transform dependency.

It exists to serve the GPU / differentiable tier of analysis that a CPU-only C++
transform (ducc0) structurally cannot, while *owning the numerics*. See
[`docs/motivation.md`](docs/motivation.md) for the full decision record.

## Status (2026-06-07)

Phases 0–2 **complete and validated** (128 tests, CPU/float64):

- **Transforms** — spin-0 & spin-2 synthesis (`aₗₘ→map`) and the exact adjoint
  `Sᵀ`, validated to machine precision vs healpy **and** ducc0; spin-2 inverse at
  the HEALPix floor with **no s2fft-style structural defect**.
- **Accuracy** — jht's own ring quadrature weights + Jacobi iteration reach
  ~1e-13 on band-limited maps (matches `healpy.map2alm(use_weights=True)`); see
  [`docs/accuracy.md`](docs/accuracy.md).
- **Partial-sky** — masked pseudo-aₗₘ + a cut-sky CG deconvolution; see
  [`docs/masked.md`](docs/masked.md).
- **Differentiability** — native JAX autodiff (`jacfwd ≡ jacrev`, tight adjoint
  identity), plus a convention-clean real-DOF layer `jht.diff`; see
  [`docs/design.md`](docs/design.md) §Differentiability.

**Phase 3 (current): GPU enablement + standalone-dependency surface.** A locked
CUDA environment (`pixi … -e gpu`) and a GPU parity/benchmark harness are in
place; the measured GPU speedup/parity run is deferred to an NVIDIA box (Cannon).
See [`docs/gpu.md`](docs/gpu.md).

## Install

Standard env is [pixi](https://pixi.sh):

```bash
pixi install          # CPU env (osx-arm64 / linux-64)
pixi run test         # the gated suite
```

GPU (CUDA, linux-64 — see [`docs/gpu.md`](docs/gpu.md)):

```bash
pixi run -e gpu python scripts/gpu_check.py   # on an NVIDIA box
```

As a dependency in another project (runtime deps are just `jax` + `numpy`):

```bash
pip install "jht @ git+https://github.com/jrcheshire/jht.git"
```

## Quick start

float64 is **opt-in per entry point** — enable it *before creating any array*
(library code never touches the global config):

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jht

nside, lmax, spin = 256, 512, 0
m = jht.synthesis(alm, nside, lmax, spin=spin)          # aₗₘ -> map
a = jht.map2alm(m, nside, lmax, spin=spin, niter=3)     # map -> aₗₘ (weighted + iterated)
cl = jht.bandpower(a, lmax, spin=spin)                  # angular auto-power C_ℓ
```

`spin=2` takes/returns `(E, B)` aₗₘ of shape `(2, …)` and `(Q, U)` maps of shape
`(2, npix)`. `jht.adjoint_synthesis` is the **exact unweighted transpose** `Sᵀ`
(the operator seam / VJP), distinct from `map2alm` (the approximate inverse). For
gradient-based work use the real-DOF layer `jht.synthesis_real` /
`jht.analysis_real` (plain ℝⁿ→ℝᵐ, no complex-conjugate convention subtlety).

## Conventions

healpy m-major triangular aₗₘ packing, orthonormal Yₗₘ with the Condon–Shortley
phase, HEALPix-internal (COSMO) polarization — verified against healpy 1.19.0 and
ducc0 0.41.0. Pinned in [`docs/design.md`](docs/design.md).

## Accuracy tiers (the contract)

jht targets the **GPU/differentiable tier** where the HEALPix ~1e-3 sampling
floor is acceptable; weights + iteration close it to ~1e-13 on band-limited
inputs. It is **not** a drop-in for ducc's purity-critical (~1e-4 E→B-leakage)
production path. Tolerances are a-priori and gate-driven, never relaxed without
sign-off. Residual mismatches are logged in
[`DISCREPANCIES.md`](DISCREPANCIES.md).

## Using jht as a dependency

jht is standalone and consumer-agnostic. The operator/grad seam a downstream
needs (e.g. to use jht *in place of ducc0*) — and the accuracy boundary — are
documented in [`docs/consumers.md`](docs/consumers.md). Any backend-selection
wiring lives in the consumer, not here.

## Docs

- [`docs/design.md`](docs/design.md) — technical design, conventions, the crux, differentiability.
- [`docs/accuracy.md`](docs/accuracy.md) — the accuracy contract + ring-weight algorithm.
- [`docs/masked.md`](docs/masked.md) — partial-sky estimators.
- [`docs/performance.md`](docs/performance.md) — CPU perf model + memory.
- [`docs/gpu.md`](docs/gpu.md) — the GPU env, the x64 requirement, the harness.
- [`docs/consumers.md`](docs/consumers.md) — the downstream-dependency seam.
- [`docs/motivation.md`](docs/motivation.md) — why jht exists.
- [`ROADMAP.md`](ROADMAP.md) — phased plan + gates.
