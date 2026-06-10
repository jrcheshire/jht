# HEALPix accuracy floor — measured characterization

HEALPix has no sampling theorem, so a HEALPix SHT has an intrinsic accuracy floor
set by the grid. This note records the measured floor for jht's spin-0 and spin-2
transforms and the recursion / quadrature behaviour behind it. The headline:
spin-2 sits at the **same** floor as spin-0, with **no m<ℓ structural defect**, and
ring weights + iteration close the floor to machine precision on band-limited maps.
(All numbers float64, CPU.)

## Gate results

| Gate | What | Result | a-priori |
|---|---|---|---|
| **R** (recursion isolation) | `lambda_{l,m}` vs scipy, all m, l→512 | **1e-12** (scipy itself NaNs at l=1000, m≪l; jht stays finite/bounded to 1000) | ≤1e-12 |
| **S-synth** spin-0 | synthesis vs healpy **and** ducc | **≤1e-13** | ≤1e-10 |
| **S-synth** spin-2 | synthesis vs healpy & ducc (E(2,0)→Q only) | **≤1e-13** | ≤1e-10 |
| **adj** spin-0/2 | `<Sa,v>=<a,Sᵀv>`; ==`(Npix/4π)map2alm` | **1e-16 / 3–4e-14** | ≤1e-10 |
| **Spin2-floor** | single-mode round-trip, all m, below ceiling | see below | ≤1e-3 bare / ≤1e-4 iter |
| **Spin0-floor** (control) | same, spin-0 | floor 2.9e-3 → 1.1e-13 (niter=3) | — |

## Spin-2 single-mode floor

A single E-mode injected at `(l0, m0)`, synthesized to (Q,U), analyzed back; max
error over all modes / channels. `nside=32, lmax=40` (below the `l≤1.5·nside=48`
ceiling, so this measures quadrature, not aliasing):

| l0 | jht m=l0 (sectoral) | jht worst m<l0 (bare) |
|---|---|---|
| 8 | 3.3e-5 | **1.1e-3** |
| 16 | 4.4e-6 | **1.5e-3** |
| 32 | 2.7e-8 | **2.2e-3** |

jht is **flat at the ~1e-3 floor across all m — no m<ℓ defect**. Jacobi iteration
on the band-limited mode then converges to machine precision (l0=16, m0=8):

```
niter=0: 7.67e-05   niter=1: 7.36e-08   niter=3: 6.96e-14   niter=5: 4.84e-17
```

The spin-0 control behaves identically (floor 2.9e-3 → 1.1e-13 at niter=3), so
**spin-2 sits at the same floor as spin-0** — no structural defect.

### Tolerance note (a-priori)

The measured *bare* HEALPix floor is **~2–3e-3 for both spins**, modestly above a
1e-3 estimate of the weights-free floor; the floor is a physical property of the
grid, and spin-0 sits at the same ~2.9e-3. With jht's own pure-numpy ring weights
+ iteration the band-limited recovery reaches ~1e-13 (machine precision), matching
healpy; the committed a-priori contract is **weighted + niter=3 ≤ 1e-4**. See
`docs/accuracy.md`.

## Characterization notes

- **Recursion is not the limiting factor.** The ℓ-recursion holds where scipy's own
  `sph_harm_y` underflows to NaN (ℓ=1000, m≪ℓ); the bare floor is set by the
  quadrature, not recursion underflow.
- **The 2·conj subtlety.** The spin-2 adjoint's naive two-channel sum is exactly
  2× the strict transpose (the m>0 conjugate-symmetry weight). The strict
  `adjoint_synthesis` (÷2) is the operator seam; the AD-convention VJP carries
  `G = 2−δ_{m0}` (native AD returns `G ⊙ conj(adjoint_synthesis)`, exact to ~1e-15
  — see `docs/design.md` §Differentiability and `tests/test_grad.py`).

Implementation: `src/jht/_recursion.py` (the ℓ-recursion), `src/jht/healpix.py`
(RING geometry, synthesis + exact adjoint), `src/jht/_analysis.py` (weighted
analysis + iteration). Gated by `tests/test_recursion.py`, `tests/test_healpix.py`,
and `tests/test_floor.py`.
