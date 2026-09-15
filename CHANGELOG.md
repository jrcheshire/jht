# Changelog

All notable changes to jht are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-09-14

### Added
- **Reverse-only transforms** `jht.synthesis_vjp` / `jht.adjoint_synthesis_vjp` (in
  `jht.diff`): the same values and cotangents as native AD via a transpose-pair
  `custom_vjp`, where each VJP is one call to the partner kernel. Native reverse-mode AD
  through the Legendre-recursion `lax.scan` keeps one `(lmax+1, lmax+1, 2·nside)` table
  per transform for the backward (~nside³ bytes); these keep none. They block forward
  mode (`jvp` / `jacfwd` raise), so the plain transforms stay the default (GitHub #4).

### Changed
- **Static tables are built in-trace.** The recursion coefficients and seeds, the
  azimuth phase and the looped-mode cap gather/mask are computed inside each kernel's
  trace behind `optimization_barrier` instead of being closed over as NumPy arrays.
  Closed-over arrays are embedded as XLA constants with a copy per use site, and the
  CUDA driver loads that constant data outside JAX's memory pool, so SHT-heavy programs
  failed to load at high nside (`Failed to load in-memory CUBIN ... OUT_OF_MEMORY`,
  GitHub #5). In a downstream design gradient the optimized graph's constant data fell
  from 53.3 to 5.4 MB at nside=64 and from 209.8 to 19.1 MB at nside=128.
  `RecursionPlan` / `CapPlan` now hold only small vectors; `_recursion.recursion_tables_np`
  is the NumPy reference.
  - Numerics: every table matches the reference exactly except `seed_log` (≤1 ulp under
    jit, where XLA contracts multiply-add) and the phase (≤1 ulp, XLA vs libm `cos`/`sin`).
    Transforms move by ~5e-15 relative at lmax=192 and ~2e-13 at lmax=4000 versus 0.2.0.
- **Save-nothing checkpoint around each on-grid kernel**, on by default: under `grad` of a
  scanned body, JAX can no longer hoist the grid-only recursion out of the loop as a
  stacked constant. Forward mode is unaffected; the backward recomputes each transform.

## [0.2.0] - 2026-06-30

### Added
- **Opt-in looped/chirp-z azimuth-FFT mode** (`set_azimuth_fft_mode("looped")` /
  `enable_looped_fft()`; default stays `"unrolled"`). The default path emits one FFT
  kernel per distinct HEALPix ring length (`~nside` kernels), so compile time /
  executable size scale as `nside x (#SHTs in the graph)` — an SHT-heavy differentiable
  graph (e.g. a masked Wiener + bandpower forecast) can exceed XLA's 2 GB executable cap
  before it runs out of memory (GitHub #2). The looped mode reroutes the polar-cap FFTs
  through **one common-length Bluestein (chirp-z) transform inside a single `lax.scan`**
  (the equatorial belt keeps its native FFT), dropping the compiled FFT-kernel count from
  **~nside to O(1)**. A chirp-z evaluates the exact pruned/aliased length-N DFT via a
  fixed length-L convolution, so the ~nside distinct cap lengths share one FFT kernel
  without the (invalid) padding to a common length.
  - The mode is a process-global flip (like `enable_compilation_cache`), read by
    `synthesis` / `adjoint_synthesis` and forwarded as part of the transform cache key,
    so the whole `analysis` / masked / Wiener chain inherits it and both variants coexist.
  - Numerically an exact FFT-algebra identity: gated at `atol=1e-12` vs the unrolled path
    (spin 0/2, synthesis + adjoint, incl. the band-ceiling `N=4` aliasing and the spin-2
    +2/-2 channel asymmetry), plus healpy/ducc0 parity and the transpose inner-product
    identity with the mode on, and jit/grad/`lax.scan` safety (`tests/test_azimuth_fft.py`,
    `tests/test_azimuth_compile.py`). Chirp phases use an exact `k^2 mod 2N` argument
    reduction (mandatory: without it the `N=4` rings at large `lmax` lose ~8 digits).
  - Tradeoff: a bounded per-run FLOP tax on the cap rings (each at the common length
    `L ≈ 5.5·nside`); the belt — the runtime bulk — is untouched. Measured (CPU, fp64):
    ~6–7× faster compile for an ~11–15% steady-runtime tax at nside 128–256 (well under
    the ≤2× budget; `scripts/profile_compile_time.py` reports both modes side by side).

## [0.1.4] - 2026-06-16

### Fixed
- **Trace-safe Wiener-path caches.** The `lru_cache`'d geometry / index helpers on
  the `wiener` + SHT path (`healpix._prepare`, `_analysis._wvec`,
  `masked._dof_layout` / `_prior_ell_index`) now return NumPy instead of device
  arrays. When first populated inside a `jit` / `grad` / `lax.scan` trace, a cached
  value produced by a jnp op (`jnp.conj` in `_prepare`) was a tracer that leaked
  (`UnexpectedTracerError`) on reuse from a later trace; static NumPy tables bake
  into the jitted transforms as constants instead. No numerics change — the tables
  are static constants (ducc0 / healpy parity unchanged) — so `wiener`,
  `synthesis`, and `analysis` now run inside `lax.scan` and `grad`-of-`scan`
  without a pre-warm workaround.

## [0.1.3] - 2026-06-14

### Added
- **High-ℓ / high-nside validation** establishing the transform holds far past the
  former ℓ_max ≲ 1000 / nside ≤ 2048 scope:
  - Recursion fp64 roundoff vs a 50-digit mpmath reference: `ε·√ℓ`, flat at a few
    ×10⁻¹⁴ to **ℓ = 32000**, spin-2 ≡ spin-0 — the libsharp-style log-renorm needs
    no two-part (X-number) scaling (`scripts/exploratory/highL_recursion_growth.py`).
  - Forward `synthesis` vs ducc0 **and** healpy to < 1e-10 through **nside = 4096 /
    ℓ_max ≈ 6000**, both spins; gated routinely to nside ≤ 2048
    (`tests/test_highL.py`, `slow`).
  - Ring-weight solve verified well-conditioned (cond ≈ 1.24·nside) and m=0-exact
    (~1e-16) through nside = 4096 (`tests/test_weights.py`, `slow`); the weighted
    inverse reaches the deep ~3e-14 floor at nside = 2048.
  - Empirical compute ceiling (`scripts/highL_ceiling.py`): nside ≤ 4096
    compiles/runs on one CPU box; nside = 8192 is the per-ring-FFT compile wall.
- `accuracy_sweep.py --ladder` for arbitrary (nside, lmax) inverse round-trip points.

### Changed
- **Scope statements** in the package docstring, README, and design/motivation docs
  updated to the validated **nside ≤ 4096 / ℓ_max ≲ 6000** envelope (was
  ℓ_max ≲ 1000, nside ≤ 2048); the band-ceiling warning notes the transform is
  validated up to the ceiling.
- New "High-ℓ / high-nside validation" section in `docs/accuracy.md`; the prior
  weight-conditioning "documented follow-up" is resolved.

## [0.1.2] - 2026-06-10

### Fixed
- **`constrained_realization` with zero-power multipoles** (e.g. `Cl[0:2] = 0`,
  the standard zeroed monopole/dipole) returned garbage for the *physical* modes:
  the old `1/Cl → 1e30` prior injected ~1e15-scale noise into the CG RHS, so the
  relative-residual stopping rule fired before the physical modes were solved
  (measured relative error ~1 vs the dense solve; ~3e-9 with all-positive Cl).
  `wiener` and `constrained_realization` now solve in **prior-whitened**
  coordinates (`P = diag(√Cl)`, operator `P·A_x·P + I`): zero-power modes are
  pinned to 0 exactly, no `1/Cl` scale ever enters the system, and the operator
  is much better conditioned (eigenvalues ≥ 1). Same math, same posterior;
  regression-gated with zero-Cl spectra in `tests/test_wiener.py`.
- **Off-grid pointing gradients at grid-aligned points** — `jax.grad` of
  `synthesis_general` w.r.t. `loc` returned `±inf` whenever θ or φ landed exactly
  on an oversampled-grid node (θ ∈ {0, π}, φ = 0, …): the ES-kernel support
  boundary has an infinite `sqrt` derivative. The kernel now excludes the
  boundary with a double-`where` guard (value change ≤ e^(−β), below every
  kernel tier); gradients are finite everywhere. Regression-gated in
  `tests/test_diff_offgrid.py`.

### Added
- **Input validation** — `synthesis` / `adjoint_synthesis` /
  `synthesis_general` / `adjoint_synthesis_general` now raise `ValueError` on
  wrong-shape `alm` / map / `field` / `loc` (previously silently clamped by the
  gather → silently wrong results), and `adjoint_synthesis_general` rejects
  complex fields. `wiener` / `constrained_realization` validate the `signal_cl`
  length.
- **Warnings** — the on-grid transforms warn once per geometry when
  `lmax > 1.5·nside` (the documented design ceiling, previously unenforced), and
  all transforms warn when `jax_enable_x64` is off (silent float32 execution).

### Changed
- **Accuracy docs** — corrected the ring-weight exactness claim: `Lw = 2·nside`
  covers the analysis quadrature products in full only for `lmax ≤ nside`, not
  the whole `1.5·nside` band. Added the measured band-ceiling table (weighted
  `niter=3`: ~1e-13 at `lmax = nside` → ~5e-7 at `lmax = 1.5·nside`; `niter=8`
  recovers ~1e-14) to `docs/accuracy.md`, a band-ceiling row (`nside=64,
  lmax=96`) to the accuracy gate, and a `DISCREPANCIES.md` entry.
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
