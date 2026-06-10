# jht — JAX Harmonic Transforms

JAX-native spherical harmonic transforms (map ↔ aₗₘ): **GPU-capable**, **fully
differentiable**, and **minimal-dependency** — pure JAX + NumPy at runtime, no
compiled C++ extension and no third-party SHT library, so it installs with `pip`
and needs no build toolchain. Scoped to **spin-0 and spin-2** fields on the
**HEALPix RING** pixelization at moderate band-limit (ℓ_max ≲ 1000, nside ≤
~2048), with both **on-grid** (pixel) and **off-grid** (arbitrary-pointing)
transforms.

Reach for jht when you need spherical harmonic transforms that run on a GPU,
differentiate end-to-end, and whose numerics you can read and own. The full
rationale — and the accuracy boundary versus a C++ production transform — is in
[`docs/motivation.md`](docs/motivation.md).

## What it does

Validated by a gated suite (190 tests, float64) against **healpy** and **ducc0**:

- **On-grid transforms** — spin-0 & spin-2 synthesis (`aₗₘ→map`), its exact
  adjoint `Sᵀ`, and a weighted + Jacobi-iterated inverse (`analysis`, aka
  `map2alm`), to machine precision versus healpy and ducc0.
- **Off-grid (NUFFT)** — evaluate a band-limited field at **arbitrary pointings**
  (spin 0–3), differentiable in **both** the aₗₘ and the pointings.
- **Differentiable** — native JAX autodiff throughout (`jacfwd ≡ jacrev`, tight
  adjoint identity), plus a convention-clean real-DOF layer (`synthesis_real` /
  `analysis_real` on-grid, `synthesis_general_real` off-grid) with no
  complex-conjugate subtlety.
- **GPU** — pure JAX, so every transform runs on CUDA with no code change;
  measured GPU==CPU parity ~1e-13 (fp64) through nside=2048.
- **Partial-sky** — masked pseudo-aₗₘ, a cut-sky CG deconvolution, and a masked
  Wiener filter / constrained realization.
- **Accuracy** — jht's own ring quadrature weights + iteration reach ~1e-13 on
  band-limited maps (matching `healpy.map2alm(use_weights=True)`).

## Install

Released on PyPI as **`jaxht`** (the import name stays `jht`):

```bash
pip install jaxht        # then:  import jht
```

Runtime dependencies are just `jax` + `numpy`. To track the repo directly:

```bash
pip install "jaxht @ git+https://github.com/jrcheshire/jht.git"
```

Development uses [pixi](https://pixi.sh):

```bash
pixi install          # CPU env (osx-arm64 / linux-64)
pixi run test         # the gated suite (parity vs healpy + ducc0)
pixi run -e gpu python scripts/gpu_check.py   # GPU parity check on an NVIDIA box
```

## Quick start

float64 is **opt-in per entry point** — enable it *before creating any array*
(library code never touches the global config):

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jht

nside, lmax = 256, 512
m = jnp.asarray(my_map)                       # your (12*nside**2,) HEALPix RING map (spin 0)

alm = jht.analysis(m, nside, lmax, niter=3)   # map -> aₗₘ  (healpy-packed; weighted + iterated)
cl  = jht.bandpower(alm, lmax)                # angular auto-power C_ℓ
m2  = jht.synthesis(alm, nside, lmax)         # aₗₘ -> map  (round-trips m)

# off-grid: evaluate the same aₗₘ at arbitrary pointings (theta, phi)
loc = jnp.stack([theta, phi], axis=-1)        # (npts, 2)
f   = jht.synthesis_general(alm, loc, lmax=lmax)
```

`spin=2` takes/returns `(E, B)` aₗₘ of shape `(2, …)` and `(Q, U)` maps of shape
`(2, npix)`. `jht.adjoint_synthesis` is the **exact unweighted transpose** `Sᵀ`
(the operator seam / VJP), distinct from `analysis` (the approximate inverse). For
gradient-based work use the real-DOF layer `jht.synthesis_real` /
`jht.analysis_real` (plain ℝⁿ→ℝᵐ — no complex-conjugate convention to track).

## Conventions

healpy m-major triangular aₗₘ packing, orthonormal Yₗₘ with the Condon–Shortley
phase, HEALPix-internal (COSMO) polarization — verified against healpy 1.19.0 and
ducc0 0.41.0. Pinned in [`docs/design.md`](docs/design.md).

## Accuracy

HEALPix has no sampling theorem, so any HEALPix SHT is approximate. jht targets
the tier where the ~1e-3 sampling floor is acceptable; its ring quadrature
weights + Jacobi iteration close that to ~1e-13 on band-limited inputs. It is
**not** a drop-in for a purity-critical (~1e-4) production pipeline. Tolerances
are a-priori and gate-driven, never relaxed without sign-off; residual mismatches
are logged in [`DISCREPANCIES.md`](DISCREPANCIES.md). Full contract and the
weight-solve algorithm: [`docs/accuracy.md`](docs/accuracy.md).

## Performance

Pure JAX runs unchanged on CUDA. Measured on A100 (incl. a 20 GB MIG) / V100, fp64:

- **GPU==CPU parity ~1e-13** through nside=2048 (synthesis and `analysis`).
- **Forward synthesis 14–60×** the 8-core CPU; fp64/fp32 ≈ 2.2×.
- **Off-grid forward** ~0.5–0.9 s at ℓ_max=1000 — independent of the number of
  points (recursion-bound) — with the pointing gradient ~1× a forward.
- **nside=2048** compiles and runs on a ~20 GB GPU slice (synthesis + `analysis`);
  the one-time compile is multi-minute (jit-cached).

The recurring GPU lesson: fp64/complex scatters are catastrophic on GPU, so jht
packs and assembles via **gathers**. CPU perf model + memory:
[`docs/performance.md`](docs/performance.md); GPU detail:
[`docs/gpu.md`](docs/gpu.md).

## Using jht as a dependency

jht is standalone and consumer-agnostic; runtime deps are just `jax` + `numpy`.
The operator / gradient seam a downstream needs — and the accuracy boundary — are
documented in [`docs/consumers.md`](docs/consumers.md). Any backend-selection
wiring lives in the consumer, not here.

## Docs

- [`docs/design.md`](docs/design.md) — technical design, conventions, differentiability.
- [`docs/accuracy.md`](docs/accuracy.md) — the accuracy contract + ring-weight algorithm.
- [`docs/masked.md`](docs/masked.md) — partial-sky estimators.
- [`docs/performance.md`](docs/performance.md) — CPU perf model + memory.
- [`docs/gpu.md`](docs/gpu.md) — the GPU env, the x64 requirement, the harness.
- [`docs/consumers.md`](docs/consumers.md) — the downstream-dependency seam.
- [`docs/motivation.md`](docs/motivation.md) — why jht exists.
