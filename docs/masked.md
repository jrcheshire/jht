# jht — Partial-sky / masked analysis

Cut-sky map → a_lm on HEALPix: the two estimators in `jht.masked`, their
conventions, and what is and isn't gated. Companion to `docs/accuracy.md` (the
full-sky contract). All numbers are float64, CPU, `jax 0.9.2`; reproduce the
table with `pixi run python scripts/masked_sweep.py` and gate with
`pixi run python -m pytest tests/test_masked.py`.

A **mask** is a per-pixel array `M` in RING order, `(Npix,)` (one sky mask, used
for both Q and U), binary `{0,1}` or apodized `∈[0,1]`.

## The operators — and why they differ

A masked SHT is not one operation. `jht.masked` exposes three: two cut-sky
estimators that differ only in *where the mask enters* (that difference is the
whole point), and a noise-aware **Wiener filter** that adds a signal prior.

### 1. `pseudo_alm` — the masked pseudo-a_lm (zero-fill)

Mask the **map**, keep the **full** quadrature weights, run the existing
full-residual Jacobi (`jht.analysis`):

> `pseudo_alm(m, M) = analysis(M·m, …)`.

With uniform weights (`use_weights=False, niter=0`) this is exactly the canonical
pseudo-a_lm `(4π/Npix) Sᵀ(M·m)` — the standard CMB estimator (the input to a
pseudo-Cℓ). It is **biased by mode-coupling**: the cut mixes (ℓ,m), so it does
*not* recover the true a_lm and is not meant to. It is the cheap forward
building block.

### 2. `deconvolve` — cut-sky deconvolution (CG)

Mask the **weights** (`M·W`) and solve the masked normal equations with
conjugate gradient:

> `A a = b`,  `A = Sᵀ(M W)S`,  `b = Sᵀ(M W)m`.

`A` is the masked weighted-least-squares operator (Hermitian-PSD). For noiseless
band-limited data `m = S a_true` it recovers `a = a_true` **exactly and
independently of the weights** wherever the cut leaves the modes constrained
(where `A` restricted to the active band is well-conditioned). It reduces to the
full-sky `SᵀWS ≈ I` as `M→1` (CG then converges in ~1 step). Near-null modes
(supported almost entirely under the cut — e.g. spin-2 E/B *ambiguous* modes)
are not recoverable from the data; CG from `x0=0` returns the minimum-norm
solution, and an optional Tikhonov `reg` (`A → A + reg·I` in the real-DOF
metric) stabilises ill-conditioned / apodized masks (trading bias for variance).

### 3. `wiener` — the Wiener filter / field-level-inference inner solve

Add a noise model and a signal prior. For the data model `m = S a + n` with
per-pixel inverse-noise `N⁻¹` and a diagonal Gaussian prior `a ~ N(0, C)`,
`C_{lm} = Cℓ`, the posterior mean (MAP) is

> `â = (Sᵀ N⁻¹ S + C⁻¹)⁻¹ Sᵀ N⁻¹ m`.

This is the **same operator family** as `deconvolve`: in the real-DOF `x`-space
(below), the prior `C⁻¹` is the **exact diagonal** `D = diag(1/Cℓ)` (the isometry
sends each `(ℓ,m)` DOF to real components, and `Cℓ` is constant over `m`), so the
solve is stock CG on `(A_x + D) x̂ = b_x`. The scalar `reg` of `deconvolve` is the
special case `Cℓ = 1/reg` constant. The `N⁻¹` weighting generalizes `deconvolve`'s
`M·W`: pass `inv_noise` (the per-pixel inverse-variance, mask folded in), or omit
it to fall back to `mask·W`. The Cℓ prior **bounds the near-null / E-B ambiguous
modes** that the `reg=0` deconvolution amplifies — a principled (Cℓ-informed)
bias-for-variance trade, the home of apodization-vs-exact-solve tension from §2.

`constrained_realization(…, key)` draws a **posterior sample** (constrained
realization) — the same CG solve with a stochastic source added to the RHS:

> `x_sample = (A_x + D)⁻¹ (b_x + s)`,  `s = T·Sᵀ(√N⁻¹·ω₁) + √D·ξ`,

with `ω₁ ~ N(0, I)` in pixel space and `ξ ~ N(0, I)` in `x`-space. Then
`Cov(s) = A_x + D`, so the draw has mean `â` and covariance `(A_x + D)⁻¹` — the
posterior. (The `s₁` factorization is exact because `T·Sᵀ` is the Euclidean
transpose of `S·T⁻¹` — the gated operator symmetry.) This is the stochastic piece
a full field-level-inference gradient needs.

## The inner-product subtlety and the isometry `T`

`S`/`Sᵀ` (`jht.healpix`) are adjoint in the **`(2−δ_m0)`-weighted** a_lm inner
product, not the Euclidean one (the `2·conj` gotcha). So `A` is
Hermitian-PD under `⟨·,·⟩_w`, not Euclidean, and stock conjugate gradient
(which assumes Euclidean) would solve the wrong system.

We handle it once with a real-DOF **isometry** `T: a ↔ x`
(`alm_to_real`/`real_to_alm`) with `‖x‖₂ = ‖a‖_w`:

> `a_{l0} → a_{l0}` (real);  `a_{lm>0} → [√2·Re, √2·Im]`,

dropping the structurally-zero `l<|spin|` modes. In `x`-space `A_x = T A T⁻¹` is
**real-symmetric-PD in plain Euclidean**, so `jax.scipy.sparse.linalg.cg` (no new
dependency) is exactly correct. `T` is gated as an isometry (`‖Tx‖² = ‖a‖_w²`,
`T⁻¹T = I` to ~1e-16) and makes the dense oracle a clean real `lstsq`.

## Conventions

- a_lm: healpy m-major triangular packing, spin-2 = `(2,K)` (E,B); maps RING
  order, spin-2 = `(2,Npix)` (Q,U) — same as `jht.healpix`.
- `deconvolve` data model (interpretation): `M` is a per-pixel *weight* in the
  least-squares fit `min_a ‖√(MW)(m − Sa)‖²`; pixels with `M=0` are ignored,
  apodized `M∈(0,1)` are down-weighted. (`pseudo_alm` instead *windows* the map,
  `M·m` — the standard pseudo-Cℓ convention.) Both are linear in `M`.

## What is gated (oracle-backed, tight) — `tests/test_masked.py`

| gate | quantity | a-priori | measured |
|---|---|---|---|
| M-isometry | `T⁻¹T=I`, `‖Tx‖²=‖a‖_w²` | 1e-12 | ~2e-16 |
| M-operator | `A_x` == dense `Sᵀdiag(MW)S`; symmetric; PSD | 1e-10 | ~4e-15 |
| M-pseudo | unweighted `pseudo_alm` == healpy/ducc (spin 0 & 2, binary + apod) | 1e-10 | ~3e-15 |
| M-fsky→1 | `pseudo_alm(·,𝟙)` == full-sky `analysis`; `deconvolve(·,𝟙)` == truth | 1e-12 / 1e-6 | ~1e-13 |
| M-deconv | `deconvolve` == truth == dense solve, mild cut (spin 0 & 2) | 1e-6 | ~1e-13 |
| W-prior | `_prior_diag(Cℓ)` == independent `diag(1/Cℓ)` build | 1e-12 | 0 (exact) |
| W-operator | `A_x + D` == dense `Sᵀdiag(N⁻¹)S + diag(1/Cℓ)`; symmetric; strictly PD | 1e-10 | ~5e-16 |
| W-adjoint | `T·Sᵀ` == Euclidean transpose of `S·T⁻¹` (makes the CR source exact) | 1e-10 | ~1e-15 |
| W-mean | `wiener` == explicit dense `(A_x+D)⁻¹ b_x` (spin 0 & 2) | 1e-8 | ~1e-12 |
| W-limits | `wiener(Cℓ→∞)` == `deconvolve(reg=0)`; `wiener(Cℓ=1/reg)` == `deconvolve(reg)` | 1e-7 / 1e-8 | ~1e-13 / 0 (exact) |
| W-CR | constrained-realization sample mean/cov == posterior (MC, 4σ a-priori budget) | 4σ | within budget |
| W-grad | `jax.grad(wiener)` wrt `Cℓ`/data finite + FD-consistent + `jit`-clean | 1e-2 (FD) | FD-consistent |

(`tests/test_wiener.py`. The Wiener *correctness* rests on the deterministic
W-operator / W-adjoint / W-mean / W-limits gates; the W-CR Monte-Carlo gate is the
end-to-end wiring check, budgeted a-priori at 4σ of the known sample-covariance
sampling distribution `E‖Ĉ−Σ‖_F² = ((trΣ)²+‖Σ‖_F²)/N` — not hand-tuned.)

The **unweighted** pseudo-a_lm is the weight-unambiguous quantity → matched to
healpy/ducc at machine precision. The *ring-weighted/iterated* pseudo-a_lm is
jht's own conditioned estimator (a different weight solver from healpy's), so it
is only *comparable*, not identical — characterised below, not gated tight
(exactly as the full-sky `test_vs_healpy` is). Tolerances are a-priori and not
relaxed without sign-off (the house rule).

## What is characterised, not gated (the cut physics) — `scripts/masked_sweep.py`

Recovery is a property of the **cut** (information loss), not the quadrature, so
it is reported, not gated tight (never conflate the two). Polar-cap ladder,
nside=32, lmax=24, max-abs a_lm error vs ground truth:

| spin | mask | fsky | pseudo (biased) | deconv (recovery) |
|---:|---|---:|---:|---:|
| 0 | binary t<0.2 | 0.991 | 5.1e-1 | **3.1e-11** |
| 0 | binary t<0.4 | 0.961 | 9.1e-1 | **6.0e-11** |
| 0 | binary t<0.6 | 0.910 | 1.2e+0 | 3.5e-2 |
| 0 | binary t<0.8 | 0.849 | 1.3e+0 | 6.1e-1 |
| 0 | apod  t<0.4 | 0.961 | 1.1e+0 | 2.1e+1 |
| 2 | binary t<0.2 | 0.991 | 2.9e-1 | **1.2e-12** |
| 2 | binary t<0.4 | 0.961 | 9.9e-1 | **2.2e-9** |
| 2 | binary t<0.6 | 0.910 | 1.2e+0 | 2.8e-1 |
| 2 | binary t<0.8 | 0.849 | 1.4e+0 | 7.0e-1 |
| 2 | apod  t<0.4 | 0.961 | 1.1e+0 | 1.1e+2 |

Takeaways: for **mild** cuts (large fsky) `deconvolve` recovers the truth to
machine precision while the pseudo-a_lm sits at O(1) bias; recovery **degrades**
as the cut grows and the active band develops near-null modes (the reg=0 CG then
amplifies them — `reg>0` bounds the solution). **Heavy apodization down-weights
data and so *hurts* an exact solve** — apodization's home is the *pseudo*-a_lm
(it suppresses ringing), not the deconvolution. spin-2 behaves like spin-0 modulo
the additional E↔B ambiguous-mode floor (see `DISCREPANCIES.md`).

## The Wiener win — characterised (the field-level-inference inner-solve tier)

`reg=0` `deconvolve` is exact where the cut leaves modes constrained but
**amplifies near-null modes** on aggressive / apodized cuts (the `apod t<0.4`
blow-up above; for spin-2 the E/B *ambiguous* subspace). The Cℓ prior of `wiener`
bounds exactly those modes. With white noise injected and a matched Cℓ prior,
`scripts/masked_sweep.py --wiener` reports the max-abs a_lm error of the Wiener
mean vs the unregularized deconvolution across the cap ladder. nside=32, lmax=24,
white-noise σ=0.1, matched Cℓ prior, max-iter=800:

| spin | mask | fsky | deconv (reg=0) | wiener |
|---:|---|---:|---:|---:|
| 0 | binary t<0.2 | 0.991 | 1.6e-2 | 1.6e-2 |
| 0 | binary t<0.4 | 0.961 | 1.3e+0 | **2.1e-1** |
| 0 | binary t<0.6 | 0.910 | 2.3e+1 | **6.3e-1** |
| 0 | binary t<0.8 | 0.849 | 6.4e+1 | **8.6e-1** |
| 0 | apod  t<0.4 | 0.961 | 2.1e+1 | **4.8e+0** |
| 2 | binary t<0.2 | 0.991 | 9.6e-3 | 9.6e-3 |
| 2 | binary t<0.4 | 0.961 | 2.0e+0 | **3.6e-1** |
| 2 | binary t<0.6 | 0.910 | 3.8e+1 | **6.1e-1** |
| 2 | binary t<0.8 | 0.849 | 4.8e+0 | **1.2e+0** |
| 2 | apod  t<0.4 | 0.961 | 1.1e+2 | **3.7e+0** |

The story: comparable on mild cuts (high fsky — both sit at the injected-noise
floor, the prior barely shrinks), and a large Wiener advantage once the cut grows
or apodization down-weights the data and the unregularized `deconvolve` diverges
(up to ~100×). The prior trades the blown-up variance for prior-controlled bias,
keeping the solve bounded. This is the field-level-inference tier;
`constrained_realization` supplies the posterior draws its gradient needs. It
resolves the mitigation path flagged for the spin-2 E/B ambiguous-mode limitation
in `DISCREPANCIES.md`.

## Scope

This rung delivers the masked operators (`pseudo_alm`, `deconvolve`) and the
Wiener filter + constrained realizations (`wiener`, `constrained_realization`).
What remains explicitly out: off-grid (NUFFT) synthesis at arbitrary pointings is
a separate capability (see `docs/offgrid.md`), not part of the on-grid masked
analysis.
