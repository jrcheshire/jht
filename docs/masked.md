# jht — Partial-sky / masked analysis

Cut-sky map → a_lm on HEALPix: the two estimators in `jht.masked`, their
conventions, and what is and isn't gated. Companion to `docs/accuracy.md` (the
full-sky contract). All numbers are float64, CPU, `jax 0.9.2`; reproduce the
table with `pixi run python scripts/masked_sweep.py` and gate with
`pixi run python -m pytest tests/test_masked.py`.

A **mask** is a per-pixel array `M` in RING order, `(Npix,)` (one sky mask, used
for both Q and U), binary `{0,1}` or apodized `∈[0,1]`.

## Two estimators — and why they differ

A masked SHT is not one operation. `jht.masked` exposes two, which differ only
in *where the mask enters*, and that difference is the whole point:

### 1. `pseudo_alm` — the masked pseudo-a_lm (zero-fill)

Mask the **map**, keep the **full** quadrature weights, run the existing
full-residual Jacobi (`jht.analysis.map2alm`):

> `pseudo_alm(m, M) = map2alm(M·m, …)`.

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

## The inner-product subtlety and the isometry `T`

`S`/`Sᵀ` (`jht.healpix`) are adjoint in the **`(2−δ_m0)`-weighted** a_lm inner
product, not the Euclidean one (the bk-jax `2·conj` gotcha). So `A` is
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
| M-fsky→1 | `pseudo_alm(·,𝟙)` == full-sky `map2alm`; `deconvolve(·,𝟙)` == truth | 1e-12 / 1e-6 | ~1e-13 |
| M-deconv | `deconvolve` == truth == dense solve, mild cut (spin 0 & 2) | 1e-6 | ~1e-13 |

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

## Scope

This rung delivers the masked operators + deconvolution. A full Wiener filter
(`Sᵀ N⁻¹ S` + a signal/`Cℓ` prior, i.e. the MUSE inner solve) builds *on* these
operators and is a later phase, not here.
