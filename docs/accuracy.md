# jht — Accuracy (Phase-1 contract)

The measured accuracy of the jht HEALPix inverse (`map2alm`), the ring-weight
algorithm behind it, and the committed tolerance. Companion to
`docs/performance.md`. All numbers are float64, CPU, `jax 0.9.2`; reproduce with
`pixi run python scripts/accuracy_sweep.py` and gate with
`pixi run python -m pytest tests/test_accuracy.py tests/test_weights.py`.

## The contract (a-priori, signed off)

HEALPix has **no sampling theorem**, so any HEALPix SHT is approximate. The
quantity gated is the **broadband band-limited round-trip**: a random a_lm for a
real spin-`s` field (`lmax ≈ nside`, below the `ℓ ≤ 1.5·nside` ceiling) is
synthesized to a map and recovered with `map2alm`; the error is the max-abs
difference from the known input a_lm.

- **Committed gate:** weighted + `niter=3` round-trip ≤ **1e-4**, across
  nside ∈ {32,64,128,256}, spin ∈ {0,2}. (`tests/test_accuracy.py`, Gate A.)
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
~1e-7); **spin-2 ≡ spin-0** at every tier (no s2fft-style m<ℓ defect); jht's
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
- `Lw = 2·nside` makes the m=0 quadrature exact across the whole usable band
  (`lmax ≤ 1.5·nside`), which is what lets the iteration reach machine precision.
  The fully-determined system (`Lw = 4·nside−2`) is Vandermonde-ill-conditioned
  and was rejected; the minimum-norm (`lstsq`) solution regularizes the remaining
  out-of-band freedom.
- **Convention:** the stored `w_i` is the deviation from 1 (matching HEALPix's
  file convention); `T == Q == U`, so one array serves both spins.

The weights are gated on their **defining math property** — m=0 quadrature exact
to `Lw` (≤1e-9, to nside=512) plus `Σ W_pix = 4π` — in `tests/test_weights.py`,
and on **end-to-end accuracy** (the table above) in `tests/test_accuracy.py`. They
are *not* gated against HEALPix's weight array (a different solver; the
end-to-end performance is what matches).

## Iteration

`map2alm` runs Jacobi / stationary-Richardson on the normal equations
`S^T W S a = S^T W m`: `a_{k+1} = a_k + A0 (m − S a_k)`. It converges because the
HEALPix points are quasi-uniform; ring weights precondition it. `niter=3` is the
default and reaches the floor above; `niter=0` is the bare weighted estimator.
The exact (unweighted) transpose `adjoint_synthesis = S^T` (the bk-jax seam / the
VJP) is **not** weighted and is unchanged by this work.

## Notes / open

- **Conditioning at the ceiling (nside→2048):** the gated matrix tops out at 256;
  the weight solve at `Lw = 2·nside` stays well-conditioned via `lstsq`. Behavior
  at nside=2048 is a documented follow-up, not a blocker (the weights are static
  numpy, free to use higher precision if ever needed).
- **Partial-sky (masked) analysis** has its own contract and document,
  `docs/masked.md` (the masked pseudo-a_lm + the cut-sky CG deconvolution); this
  contract is full-sky.
