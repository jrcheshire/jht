# jht — Accuracy (Phase-1 contract)

The measured accuracy of the jht HEALPix inverse (`analysis`, aka `map2alm`), the ring-weight
algorithm behind it, and the committed tolerance. Companion to
`docs/performance.md`. All numbers are float64, CPU, `jax 0.9.2`; reproduce with
`pixi run python scripts/accuracy_sweep.py` and gate with
`pixi run python -m pytest tests/test_accuracy.py tests/test_weights.py`.

## The contract (a-priori, signed off)

HEALPix has **no sampling theorem**, so any HEALPix SHT is approximate. The
quantity gated is the **broadband band-limited round-trip**: a random a_lm for a
real spin-`s` field (`lmax ≈ nside`, below the `ℓ ≤ 1.5·nside` ceiling) is
synthesized to a map and recovered with `analysis`; the error is the max-abs
difference from the known input a_lm.

- **Committed gate:** weighted + `niter=3` round-trip ≤ **1e-4**, across
  nside ∈ {32,64,128,256} at `lmax = nside`, plus a band-ceiling row
  (`nside=64, lmax=96`), spin ∈ {0,2}. (`tests/test_accuracy.py`, Gate A.)
- **Measured:** ~**1e-13** (≈9 orders of headroom) — machine precision, matching
  `healpy.map2alm(use_weights=True, iter=3)`. The gate is held at the a-priori
  1e-4 (not tightened to one machine's float64 floor; the headroom is documented
  here instead, per the standing "tolerances a-priori, don't chase one machine"
  rule).
- This **resolves the pending Phase-0 floor sign-off**: the loose ~2–3e-3 bare
  carryover is now the *unweighted baseline*; the committed tier is the weighted
  + iterated 1e-4 above.

## Measured floor (broadband round-trip, max-abs a_lm error vs ground truth)

| nside | lmax | spin | unweighted bare | unweighted niter=3 | **weighted bare** | **weighted niter=3** | healpy niter=3 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32  | 32  | 0 | 1.7e-2 | 5.0e-7 | 1.2e-3 | **1.1e-12** | 1.6e-12 |
| 32  | 32  | 2 | 8.4e-3 | 2.3e-7 | 1.0e-3 | **5.0e-13** | — |
| 64  | 64  | 0 | 8.4e-3 | 2.0e-7 | 3.3e-4 | **2.1e-13** | 3.1e-13 |
| 64  | 64  | 2 | 5.3e-3 | 1.3e-7 | 5.6e-4 | **2.9e-13** | — |
| 128 | 128 | 0 | 4.7e-3 | 1.1e-7 | 3.6e-4 | **1.3e-13** | 2.0e-13 |
| 128 | 128 | 2 | 5.1e-3 | 1.2e-7 | 4.2e-4 | **2.7e-13** | — |
| 256 | 256 | 0 | 4.2e-3 | 9.8e-8 | 2.3e-4 | **1.0e-13** | 1.7e-13 |
| 256 | 256 | 2 | 1.8e-3 | 4.1e-8 | 2.5e-4 | **8.9e-14** | — |

Takeaways: ring weights drop the **bare** floor ~10× (~1e-3 → ~1e-4 tier) and
let the Jacobi iteration reach machine precision (the unweighted iteration stalls
~1e-7); **spin-2 ≡ spin-0** at every tier (no spin-2 m<ℓ structural defect); jht's
weighted result matches/beats healpy's ring-weighted `map2alm`.

## Ring weights — jht's own, pure-numpy, no files (`jht.weights`)

The bare estimator `A0 = S^T W` weights each pixel by its solid angle `4π/Npix`.
The error is dominated by the **colatitude** quadrature (the per-ring azimuthal
FFT is already exact in m), so a per-ring factor `W_i = (4π/Npix)(1 + w_i)`
chosen to make the m=0 colatitude quadrature exact removes most of it. Applied to
all m (the weight is azimuthally symmetric) this is a heuristic correction that
drops the floor ~10× and, crucially, **conditions the normal equations** so the
iteration converges to machine precision in a few steps.

**These are jht's own weights, not HEALPix's.** HEALPix ships precomputed
`weight_ring_n*.fits` from an in-house solver whose exactness-degree target has no
closed form (empirically 2, 8, 24, 48, 100, 226, 454, … for nside = 2…128) and is
not cleanly reproducible; depending on those files would also break the
jax+numpy-only / no-files contract. jht instead solves a well-posed, well-
conditioned system in numpy (once per nside, cached, off the JAX hot path):

> minimize ‖w‖ s.t. `Σ_rings c_i n_i (1 + w_i) P_ℓ(x_i) = Npix·δ_{ℓ0}`
> for even ℓ = 0, 2, …, `Lw`, with **`Lw = 2·nside`**.

- Nodes `x_i` and pixel counts `n_i` are the northern half + equator
  (`RingInfo.z[:2·nside]`), the same half-grid the recursion runs on; multiplicity
  `c_i = 2` (N/S pair) except `c = 1` for the equatorial ring → exactly `2·nside`
  weights. `P_ℓ` via `numpy.polynomial.legendre.legvander`.
- `Lw = 2·nside` makes the m=0 quadrature exact for Legendre polynomials to
  degree `2·nside`. The analysis integrals involve *products* `λ_ℓ0 λ_ℓ'0` of
  degree up to `2·lmax`, so the quadrature is fully exact only for
  `lmax ≤ nside` — which is where the deep ~1e-13 floor in the table above is
  measured. Above `nside` the residual error grows (see "Behavior toward the
  band ceiling" below). The fully-determined system (`Lw = 4·nside−2`) is
  Vandermonde-ill-conditioned and was rejected; the minimum-norm (`lstsq`)
  solution regularizes the remaining out-of-band freedom.
- **Convention:** the stored `w_i` is the deviation from 1 (matching HEALPix's
  file convention); `T == Q == U`, so one array serves both spins.

The weights are gated on their **defining math property** — m=0 quadrature exact
to `Lw` (≤1e-9, to nside=512) plus `Σ W_pix = 4π` — in `tests/test_weights.py`,
and on **end-to-end accuracy** (the table above) in `tests/test_accuracy.py`. They
are *not* gated against HEALPix's weight array (a different solver; the
end-to-end performance is what matches).

## Behavior toward the band ceiling (lmax > nside)

The deep ~1e-13 floor above is a property of the `lmax ≈ nside` regime, where
the ring weights make the colatitude quadrature exact for the full set of
analysis products. Toward the `lmax ≤ 1.5·nside` ceiling the default `niter=3`
floor rises (the contract still holds with ~2 orders of headroom), and more
iterations recover machine precision — the degradation is a convergence-rate
effect, not a floor:

| nside | lmax | spin | weighted niter=3 | weighted niter=8 |
|---:|---:|---:|---:|---:|
| 32 | 32 (= nside)    | 0/2 | 1.1e-12 / 5.0e-13 | 4.4e-16 / 5.2e-16 |
| 32 | 40 (1.25·nside) | 0/2 | 2.6e-8 / 2.2e-8   | 5.1e-16 / 6.3e-16 |
| 32 | 48 (1.5·nside)  | 0/2 | 8.4e-7 / 7.8e-7   | 2.6e-14 / 3.0e-14 |
| 64 | 64 (= nside)    | 0/2 | 2.1e-13 / 2.9e-13 | 6.3e-16 / 8.3e-16 |
| 64 | 80 (1.25·nside) | 0/2 | 2.4e-8 / 7.5e-9   | 7.8e-16 / 9.2e-16 |
| 64 | 96 (1.5·nside)  | 0/2 | 4.0e-7 / 5.0e-7   | 1.3e-14 / 1.8e-14 |

The contract gate matrix (`tests/test_accuracy.py`) includes a ceiling row
(`nside=64, lmax=96`) at the committed 1e-4. If you need machine precision at
`lmax > nside`, raise `niter` (8 suffices through the ceiling). The transforms
warn (once per geometry) if called *above* the `1.5·nside` ceiling, where
accuracy is unvalidated.

## High-ℓ / high-nside validation (ℓ > 1000)

The contract above is gated at nside ≤ 256. The transform is validated well past
that on two independent axes — the recursion's *absolute* accuracy and the *full
forward transform* vs external libraries — establishing that nothing structural
breaks at high ℓ. (Reproduce: `scripts/exploratory/highL_recursion_growth.py` and
`scripts/highL_ceiling.py`; gated in `tests/test_highL.py`, `slow`.)

**Recursion roundoff is flat to ℓ = 32000.** jht's float64 normalized
(spin-weighted) Legendre/Wigner recursion, compared against a 50-digit mpmath
reference of the *same* recursion (isolating the float64 roundoff), has a
worst-case relative error — normalized by the addition-theorem amplitude
`√((2ℓ+1)/4π)` — that follows `ε·√ℓ` and stays at a few ×10⁻¹⁴, with spin-2 ≡ spin-0:

| ℓ | spin-0 | spin-2 |
|---:|---:|---:|
| 500 (anchor) | 4.9e-15 | 6.1e-15 |
| 2000 | 8.5e-15 | 1.1e-14 |
| 8000 | 4.0e-14 | 2.3e-14 |
| 16000 | 2.7e-14 | 2.8e-14 |
| 32000 | 4.5e-14 | 4.2e-14 |

Extrapolating the `√ℓ` law, fp64 roundoff would not reach the ~1e-3 HEALPix floor
until ℓ ~ 10²⁵ — so the libsharp-style log-renorm needs no two-part (X-number)
scaling at any band limit one could pixelize. The ℓ=500 row also rules out
convention drift (jht is independently correct there to ~1e-12 vs scipy).

**Forward synthesis matches ducc0 and healpy to ~1e-11 at high nside.** A
broadband band-limited `synthesis` (the quadrature-free forward operator) vs both
libraries, spin 0 and 2 (CPU, float64):

| nside | lmax | compile | run | peak RAM | rel vs ducc / healpy |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1500 | — | — | — | 1.9e-11 / 3.1e-12 |
| 2048 | 2000 | 48 s | 25 s | 9.7 GB | 2.5e-11 |
| 2048 | 3000 | 84 s | 61 s | 11.6 GB | 6.5e-11 |
| 4096 | 4000 | 4.4 min | 3.3 min | 31 GB | 5.6e-11 |
| 4096 | 6000 | 8.6 min | 7.5 min | 39 GB | 9.1e-11 |

(spin-0 timings shown. spin-2 reaches the same frontier — nside=4096/lmax=4000 at
8.9e-12 vs ducc, 353 s compile / 298 s run / 41 GB — and runs ~1.5× the spin-0
cost. All rows are inside the 1e-10 operator tolerance, including the near-ceiling
`lmax ≈ 1.5·nside` rows.)

**The weighted inverse reaches the deep floor at scale.** The end-to-end
broadband round-trip (`analysis`, weighted, niter=3) recovers the input a_lm to
**~3e-14** at nside=1024 and nside=2048 (lmax=nside), spin 0 and 2 — the same deep
floor as the nside ≤ 256 contract table, and 30–100× below healpy's iterated
`map2alm` (1–3e-12). So the ring weights' `ℓ ≤ nside` exactness translates into
machine-precision recovery at high nside, not just an exact quadrature.
(Reproduce: `scripts/accuracy_sweep.py --ladder 1024:1024,2048:2048`.)

**The ceiling is compute, not correctness.** The recursion is exact to arbitrary
ℓ and the geometry only needs `nside ≳ ℓ/1.5`. The practical limit is the XLA
compile of the per-ring-length FFT unroll (≈ nside distinct kernels, super-linear
in nside): **nside ≤ 4096 (ℓ up to ~6000) compiles in minutes** on one CPU box,
while **nside = 8192 exceeds a 40-minute compile budget**. The lever is the
pay-once persistent compile cache (`jht.enable_compilation_cache`), not a numeric
change. Memory scales ~linearly with pixel count (nside=4096 ≈ 31–39 GB float64).

## Iteration

`analysis` runs Jacobi / stationary-Richardson on the normal equations
`S^T W S a = S^T W m`: `a_{k+1} = a_k + A0 (m − S a_k)`. It converges because the
HEALPix points are quasi-uniform; ring weights precondition it. `niter=3` is the
default and reaches the floor above; `niter=0` is the bare weighted estimator.
The exact (unweighted) transpose `adjoint_synthesis = S^T` (the operator seam / the
VJP) is **not** weighted and is unchanged by this work.

## Notes / open

- **Weight-solve conditioning at scale (resolved):** the `Lw = 2·nside` lstsq stays
  well-conditioned — `cond(A) ≈ 1.24·nside` (→ ~5e3 at nside=4096, ~3 digits of 16)
  — and the m=0 quadrature stays exact to ~1e-16 **through nside=4096**, so the deep
  inverse floor's `ℓ ≤ nside` quadrature exactness holds at scale. Gated `slow` in
  `tests/test_weights.py`; measured by `scripts/exploratory/weight_conditioning.py`.
- **Partial-sky (masked) analysis** has its own contract and document,
  `docs/masked.md` (the masked pseudo-a_lm + the cut-sky CG deconvolution); this
  contract is full-sky.
