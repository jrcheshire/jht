# jht — Technical design notes

Working design notes for the JAX-native SHT. Written 2026-06-06 as scaffolding;
this is the place to record algorithm choices, conventions, and accuracy
decisions as they are made. Nothing here is implemented yet.

## Accuracy tiers (the contract)

HEALPix has **no sampling theorem**, so a HEALPix SHT is intrinsically
approximate. Three accuracy regimes matter:

| Tier | What it is | Use |
|---|---|---|
| Bare floor (~1e-3) | Direct quadrature, no ring weights | Not good enough alone |
| + ring weights | Full HEALPix ring quadrature weights | Closes toward ducc |
| + Jacobi iteration | `map2alm`-style iterative refinement on band-limited input | ducc-class (~1e-4+) |

- **jht's first target:** reach the *bare floor with no structural defect* —
  spin-2 must sit at ~1e-3 like spin-0, NOT s2fft's 3–28%.
- **Then** add ring weights + iteration to approach ducc on band-limited maps.
- ducc remains the production oracle in bk-jax; jht serves the GPU/diff tier
  where ~1e-3 is acceptable. Tolerances are a-priori and not relaxed without
  sign-off (standing rule).

## Algorithm skeleton (to be settled in Phase 0/1)

The standard analysis/synthesis structure:
- **Synthesis (aₗₘ → map):** for each ring (fixed colatitude θ), evaluate the
  associated-Legendre (spin-0) or Wigner-d / spin-weighted (spin-2) functions
  Pₗₘ(θ), contract with aₗₘ over ℓ to get per-m ring coefficients, then an FFT
  over φ along the ring.
- **Analysis (map → aₗₘ):** FFT over φ per ring → per-m ring values, contract
  with Pₗₘ(θ) and the ring quadrature weight over rings, sum to aₗₘ; optionally
  Jacobi-iterate to refine.

Design choices to make empirically:
- **Dense-Legendre vs FFT-per-ring + on-the-fly recursion** — memory vs speed
  at BK ℓ_max. Dense Pₗₘ tables may be fine at ℓ_max≲1000 on GPU; measure.
- **Real-aₗₘ vs complex-aₗₘ** internal representation.
- Batching over realizations (vmap) for the sim/forecast use cases.

Reference algorithms (study, don't copy a broken codebase): Price & McEwen for
the recursion structure; libsharp / ducc for the stability scaling and ring
weights.

## The crux: recursion stability

Associated-Legendre / Wigner-d recursions **underflow/overflow** at high ℓ; a
naive `float64` recursion silently loses precision well before ℓ~1000. The
solved technique is **logarithmic / "X-number" scaling** (carry a separate
integer exponent alongside the mantissa, à la libsharp). This is almost
certainly the class of bug behind s2fft's spin-2 failure pattern (clean only at
m≈ℓ). **Test this explicitly and early** — single-mode sweeps across *all* m at
several ℓ up to ~1000, not just round-trip totals which can mask localized
leakage.

## Conventions to pin (and cross-validate vs healpy AND ducc)

Leaving these implicit is how `bk-jax` accumulated sign/convention bugs. Decide,
document here, and test against both oracles:

- **Polarization U-sign:** HEALPix-internal vs IAU. (bk-jax stores
  HEALPix-internal, flips U at I/O boundaries.) Pick jht's internal convention
  and state it loudly.
- **Spin sign** (spin +2 vs −2; E/B sign).
- **aₗₘ packing:** healpy triangular m-major layout, real vs complex, ℓ/m
  ordering.
- **Normalization:** orthonormal vs 4π. Match healpy/ducc and document the map.

## Differentiability

- The on-grid SHT is **linear in aₗₘ**, so JVP/VJP are clean to register.
- **Watch the JAX-VJP-convention vs strict-math-adjoint subtlety** that bit
  bk-jax: JAX's complex-VJP convention introduces a `2·conj(·)` factor on m>0
  modes relative to the strict math adjoint. Decide which convention each
  registered transpose exposes, and provide a strict-math-adjoint helper if a
  consumer (e.g. an operator-form CG solve) needs `Aᵀ` rather than the AD
  cotangent. Test `jacfwd ≡ jacrev` and ⟨A x, y⟩ = ⟨x, Aᵀ y⟩.
- Geometry/pointing differentiability is the **off-grid NUFFT** story
  (Phase 4), not the on-grid core.

## Explicit non-goals (keep the surface small)

- Arbitrary spin-s; non-HEALPix samplings; ℓ_max ≫ 1000.
- Off-grid synthesis / NUFFT (separate, deferred).
- A compiled (C++/CUDA) kernel. Pure JAX is the dependency-control point; only
  revisit under a deliberate, documented decision.

## Validation harness (planned)

- Parity tests vs **healpy** (`alm2map`/`map2alm`, `pol=True`) and **ducc0**
  for every transform, at documented a-priori tolerances.
- Single-mode sweeps (all m, ℓ up to ~1000) to catch localized leakage that
  round-trip totals hide — the diagnostic that exposed s2fft.
- A `DISCREPANCIES.md` log (mirroring bk-jax) for any residual mismatch: where
  seen, suspected cause, what would close it.
