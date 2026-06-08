# Changelog

All notable changes to jht are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-07

Inaugural release.

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
- **GPU (CUDA)** — pure JAX, runs on GPU with no code change; a parity harness
  (`scripts/gpu_check.py`) and a single-slot performance diagnostic
  (`scripts/gpu_diagnostic.py`).

[Unreleased]: https://github.com/jrcheshire/jht/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jrcheshire/jht/releases/tag/v0.1.0
