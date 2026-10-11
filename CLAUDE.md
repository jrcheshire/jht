# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What jht is

JAX-native spherical harmonic transforms for **spin-0 and spin-2** fields on the
**HEALPix RING** grid: GPU-capable, fully differentiable, pure JAX + NumPy at runtime
(no compiled extension, no third-party SHT library). Validated to nside ≤ 4096 /
ℓ_max ≲ 6000. Published on PyPI as `jaxht`; the import name is `jht`.

Out of scope: arbitrary spin, non-HEALPix samplings, and drop-in replacement of a
purity-critical (~1e-4) production transform. Keep the core consumer-agnostic:
nothing in `src/` should assume a particular downstream pipeline.

## Layout

| module | contents |
|---|---|
| `healpix.py` | on-grid `synthesis` and its exact transpose `adjoint_synthesis`; ring geometry, aₗₘ packing |
| `_recursion.py` | normalized associated-Legendre ℓ-recursion with branch-free log-renormalization |
| `_azimuth.py` | per-ring azimuthal FFTs (unrolled, or the looped/chirp-z mode for O(1) compiled kernels) |
| `_analysis.py`, `weights.py` | `analysis` / `map2alm` (ring quadrature weights + Jacobi iteration), `bare_analysis` |
| `diff.py` | real-DOF transforms (`synthesis_real`, `analysis_real`, …), `bandpower`, reverse-only `*_vjp` |
| `offgrid.py`, `_nufft.py` | `synthesis_general` / `adjoint_synthesis_general` at arbitrary pointings (spin 0–3) |
| `masked.py` | `pseudo_alm`, `deconvolve`, `wiener`, `constrained_realization`, `wiener_preconditioner`, `SolveInfo` |
| `_cache.py` | opt-in persistent compilation cache |

Docs: `docs/design.md` (design, conventions, differentiability), `docs/accuracy.md`
(the accuracy contract and the measured HEALPix floor), `docs/masked.md`,
`docs/offgrid.md`, `docs/performance.md`, `docs/gpu.md`, `docs/consumers.md` (the
operator seam a downstream depends on).

## Conventions (pinned in `docs/design.md`)

- aₗₘ: healpy m-major triangular packing, complex, m ≥ 0; orthonormal Yₗₘ with the
  Condon–Shortley phase. Spin 2 is `(2, K)` (E, B) and maps are `(2, Npix)` (Q, U),
  HEALPix-internal (COSMO) polarization.
- `adjoint_synthesis` is the exact transpose in the `(2 − δ_m0)`-weighted aₗₘ inner
  product, not the Euclidean one. The real-DOF isometry (`alm_to_real` /
  `real_to_alm`) makes operators real-symmetric in plain Euclidean form; use it for
  linear algebra and gradients.
- **float64 is opt-in per entry point** (`jax.config.update("jax_enable_x64", True)`
  in tests and scripts). Library code never touches global JAX config.
- Static per-(nside, lmax) tables are built in-trace or held as NumPy, never cached as
  jnp arrays: a cache first populated under a trace leaks tracers.

## Validation contract

- Every transform is gated against **healpy and ducc0** at an a-priori tolerance,
  recorded in the relevant doc's gate table.
- Tolerances are not relaxed to make a test pass. Find the mechanism first; log any
  residual numeric mismatch in `DISCREPANCIES.md`.
- Gate spin-2 analysis per (ℓ, m) below the ℓ ≤ 1.5·nside band-limit ceiling, so
  recursion, quadrature and ceiling errors stay separate.
- Iterative solvers are gated against a dense oracle at small nside.

## Tooling

- Pixi: `pixi run test` (full suite), `pixi run test-fast` (excludes `slow`),
  `pixi run lint`, `pixi run typecheck`. Tests marked `heavy` exceed a hosted CI
  runner's memory and run locally only.
- Runtime dependencies are `jax` + `numpy` only; healpy / ducc0 are test oracles in
  the pixi env, never `[project.dependencies]`.
- GPU env: `pixi run -e gpu ...` on an NVIDIA machine (see `docs/gpu.md`).
- Scripts + `ipython %run` for exploration; no notebooks.
- Before committing: `lint` + `typecheck` + `test-fast`. Commit messages: one terse line.
- Releases: bump `__version__` in `src/jht/__init__.py` (the single source; hatch reads
  it) and `tests/test_api.py::test_version`, date the CHANGELOG section, tag `vX.Y.Z`;
  `.github/workflows/release.yml` builds and publishes on the tag. Author metadata lives
  in `pyproject.toml` and `CITATION.cff`; keep them consistent.

## Writing style for code and docs

Docstrings and docs describe what the code does now: purpose, call signature,
conventions, cost, and the reason for any non-obvious choice. Development history
belongs in git and `CHANGELOG.md`, not in docstrings, comments, or docs.
