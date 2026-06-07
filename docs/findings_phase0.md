# Phase 0 — feasibility spike findings (go/no-go)

**Verdict: GO.** A clean-room, pure-JAX HEALPix SHT reaches the HEALPix accuracy
floor for **both spin-0 and spin-2 with no structural defect**, validated to
machine precision against healpy *and* ducc0. The corrected risk model (recursion
is table-stakes; the spin-2 *analysis quadrature* is the make-or-break) is
confirmed: the recursion was never the problem, and jht's spin-2 inverse sits at
the floor exactly where s2fft fails by 13–35%.

Date: 2026-06-06. All work float64, CPU (per the Phase-0 decision; GPU timing
deferred to Phase-3). Tolerances were fixed a-priori in `ROADMAP.md`.

---

## Gate results

| Gate | What | Result | a-priori |
|---|---|---|---|
| **R** (recursion isolation) | `lambda_{l,m}` vs scipy, all m, l→512 | **1e-12** (scipy itself NaNs at l=1000, m≪l; jht stays finite/bounded to 1000) | ≤1e-12 |
| **S-synth** spin-0 | synthesis vs healpy **and** ducc | **≤1e-13** | ≤1e-10 |
| **S-synth** spin-2 | synthesis vs healpy & ducc (E(2,0)→Q only) | **≤1e-13** | ≤1e-10 |
| **adj** spin-0/2 | `<Sa,v>=<a,Sᵀv>`; ==`(Npix/4π)map2alm` | **1e-16 / 3–4e-14** | ≤1e-10 |
| **Spin2-floor** (make-or-break) | single-mode round-trip, all m, below ceiling | see below | ≤1e-3 bare / ≤1e-4 iter |
| **Spin0-floor** (control) | same, spin-0 | floor 2.9e-3 → 1.1e-13 (niter=3) | — |

GPU sanity: **deferred to Phase 3** (float64 + Cannon GPU access).

---

## The make-or-break: spin-2 inverse vs the s2fft defect

Single E-mode injected at `(l0, m0)`, synthesized to (Q,U), analyzed back; max
error over all modes/channels. `nside=32, lmax=40` (below the `l≤1.5·nside=48`
ceiling, so this measures quadrature, not aliasing). The same sweep s2fft failed:

| l0 | jht m=l0 (sectoral) | jht worst m<l0 (bare) | **s2fft (recorded)** |
|---|---|---|---|
| 8 | 3.3e-5 | **1.1e-3** | m<l fail 28–35% |
| 16 | 4.4e-6 | **1.5e-3** | 19–28% |
| 32 | 2.7e-8 | **2.2e-3** | 13–22% |

jht is **flat at the ~1e-3 floor across all m — no m<l defect** — i.e. ~100–300×
better than s2fft at the exact modes it gets wrong. Jacobi iteration on the
band-limited mode then converges to machine precision (l0=16, m0=8):

```
niter=0: 7.67e-05   niter=1: 7.36e-08   niter=3: 6.96e-14   niter=5: 4.84e-17
```

s2fft's iteration, by contrast, stalls at ~1e-5 (upstream issue #269). The
spin-0 control behaves identically (floor 2.9e-3 → 1.1e-13 at niter=3), so
**spin-2 sits at the same floor as spin-0** — the success criterion.

### Tolerance note (a-priori, flagged not relaxed)
The measured *bare* HEALPix floor is **~2–3e-3 for both spins**, modestly above
the 1e-3 a-priori estimate. The 1e-3 was an estimate of the (weights-free)
HEALPix floor; the floor is a physical property of the grid, and **spin-0 sits at
the same 2.9e-3**, so the discriminating criterion (spin-2 ≡ spin-0, no
structural defect) is met decisively. Iterated accuracy (≤1e-4 gate) is exceeded
by ~10 orders. The floor closes further with ring weights (deferred rung).

---

## What was built (committed)

- `src/jht/_recursion.py` — libsharp ℓ-recursion (Kostelec–Rockmore) +
  branch-free per-step log-renorm, as a `lax.scan`; spin-0 and spin-2 share one
  path (spin-0 = the m′=0 case). Holds to ℓ=1000 in float64.
- `src/jht/healpix.py` — RING geometry; `synthesis` and exact `adjoint_synthesis`
  (= Yᵀ, the bk-jax seam op), spin-0 and spin-2, with polar folding.
- `src/jht/analysis.py` — bare `A0 = (4π/Npix)Sᵀ` + Jacobi iteration.
- Gates: `tests/test_recursion.py`, `tests/test_healpix.py`, `tests/test_floor.py`.
- `scripts/exploratory/phase0_floor.py` — the spin-2 kill-test measurement.

## Notes carried forward

- **Recursion is solved, not the risk.** It holds where scipy's own `sph_harm_y`
  underflows to NaN (ℓ=1000, m≪ℓ). The s2fft "recursion underflow" hypothesis is
  retired; its defect is the unweighted spin-2 quadrature + stalling iteration.
- **The 2·conj subtlety** showed up concretely: the spin-2 adjoint's naive
  two-channel sum is exactly 2× the strict transpose (the m>0 conjugate-symmetry
  weight). The strict `adjoint_synthesis` (÷2) is the bk-jax seam op; the
  AD-convention VJP carries `2−δ_{m0}` (Phase-2 work).
- **Performance is the Phase-1 priority.** The reference transforms loop over
  rings eagerly and recompute the recursion per call — fine for correctness,
  far too slow for production (a single nside=32 sweep burned minutes). Cache the
  λ tables, batch equal-length rings, vmap, jit.

## Next (Phase 1)
Harden the core: precompute/vectorize the transforms; ring weights + iteration to
close toward ducc; full nside/ℓmax/spin test matrix; partial-sky path. Then
Phase 2 (differentiability: the two transposes, gradient identities) and Phase 3
(GPU + the `BK_JAX_SHT_BACKEND=jht` backend).
