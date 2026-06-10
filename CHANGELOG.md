# Changelog

All notable changes to jht are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Docs** — rewrote `docs/` for a public audience: removed consumer-specific
  (BICEP/Keck, bk-jax) and development-process references, trimmed `motivation.md`
  to a concise justification, and de-jargoned (e.g. MUSE → field-level inference).
  No code or API changes.

## [0.1.1] - 2026-06-10

### Added
- **`enable_compilation_cache(dir)`** — opt in to JAX's persistent on-disk
  compilation cache. The nside≥1024 on-grid compile is multi-minute and structural
  (~458 s at nside=2048, ~93% the per-ring-length FFT unroll); the cache makes it
  pay-once-ever rather than per run. Consumer-opted-in (like x64); numerics untouched.
  Measured table in `docs/performance.md`.
- **Off-grid real-DOF layer** — `synthesis_general_real` (`S_g ∘ T⁻¹`) and
  `adjoint_synthesis_general_real` (`T ∘ S_gᵀ`): the arbitrary-pointing duals of
  `synthesis_real`, a plain real-linear `ℝⁿ→ℝᵐ` over spin 0–3 (`jacfwd ≡ jacrev`,
  native VJP == the exact transpose, no `2·conj` bridge). The ergonomic
  gradient-based entry point to the NUFFT path.

### Changed
- **`map2alm` → `analysis`** — the map→aₗₘ inverse is now canonically `jht.analysis`,
  the field-standard mirror of `synthesis` (and consistent with `bare_analysis` /
  `analysis_real`). **`jht.map2alm` stays as a back-compat alias** (same object), so
  nothing breaks. The internal module `jht/analysis.py` moved to `jht/_analysis.py`.
- **CI** — a `slow` marker splits the heavy off-grid oracle suite out of the fast
  subset; `test.yml` runs that fast subset (`pixi run test-fast`, ~5 min) on pushes
  to `main` and on PRs, while the full suite (`full-suite.yml`, `pixi run test`)
  stays **manual-only** (`workflow_dispatch`).
- Pixi manifest table `[tool.pixi.project]` → `[tool.pixi.workspace]` (the deprecated
  form).

## [0.1.0] - 2026-06-10

Inaugural release. Published on PyPI as **`jaxht`** (`pip install jaxht` → `import jht`;
the name `jht` is unavailable on PyPI). GitHub repo and import package stay `jht`.

### Added
- **On-grid transforms** — spin-0 and spin-2 `synthesis` (a_lm → map) and the
  exact adjoint `adjoint_synthesis`, validated to machine precision against
  healpy **and** ducc0 (spin-2 inverse at the HEALPix floor, no s2fft-style
  structural defect).
- **Approximate inverse** — `bare_analysis` (Sᵀ W) and `map2alm` (jht's own ring
  quadrature weights + Jacobi iteration; ~1e-13 on band-limited maps).
- **Quadrature weights** — `ring_weights`, `pixel_weights`.
- **Partial-sky / masked** — `pseudo_alm`, `deconvolve` (cut-sky CG), `wiener`
  (masked Wiener filter / MUSE inner solve), `constrained_realization`.
- **Off-grid (NUFFT)** — `synthesis_general` / `adjoint_synthesis_general` for
  spin 0–3 at arbitrary pointings; alm- **and** pointing-differentiable under
  native autodiff. JAX-native replacement for ducc0's `synthesis_general`.
- **Differentiable real-DOF interface** — `synthesis_real`, `analysis_real`,
  `bandpower`, with the `alm_to_real` / `real_to_alm` isometry and the
  `alm_metric_weight` (2 − δ_m0) bridge.
- **GPU (CUDA)** — pure JAX, runs on GPU with no code change. Measured on Cannon
  A100/V100 (fp64): GPU==CPU parity ~1e-13 across the BK regime **including
  nside=2048**; forward synthesis 14–60× CPU. Three fp64-scatter→gather reworks (the
  `dense_to_tri` adjoint packing, the `nufft2d2` off-grid grid build, and the on-grid
  ring assembly) make the adjoint and off-grid forward fast and let nside=2048 compile.
  Parity + diagnostic harnesses `scripts/gpu_check.py`, `scripts/gpu_diagnostic.py`.

[Unreleased]: https://github.com/jrcheshire/jht/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jrcheshire/jht/releases/tag/v0.1.0
