# jht — Technical design notes

Working design notes for the JAX-native SHT. This records algorithm choices,
conventions, and accuracy decisions. Phase 0 (feasibility) and Phase 1
(performance + accuracy) are implemented; the measured accuracy contract lives
in `docs/accuracy.md` and the performance characterization in
`docs/performance.md`.

## Accuracy tiers (the contract)

HEALPix has **no sampling theorem**, so a HEALPix SHT is intrinsically
approximate. Three accuracy regimes matter:

| Tier | What it is | Measured (broadband) | Use |
|---|---|---|---|
| Bare floor | Direct quadrature, uniform `4π/Npix` | ~2–17e-3 | Not good enough alone |
| + ring weights | jht's own min-norm ring weights (`jht.weights`) | bare ~2–12e-4 | Closes toward ducc |
| + Jacobi iteration | `map2alm` iterative refinement, band-limited | **~1e-13** | ducc-class |

- **Phase-0 target (met):** reach the *bare floor with no structural defect* —
  spin-2 sits at ~1e-3 like spin-0, NOT s2fft's 3–28%.
- **Phase-1 accuracy (met):** ring weights + iteration reach machine precision on
  band-limited maps, matching `healpy.map2alm(use_weights=True)`. The **committed
  a-priori contract is weighted + niter=3 ≤ 1e-4** (measured ~1e-13, ~9 orders of
  headroom). Full numbers + the ring-weight algorithm: `docs/accuracy.md`.
- **Ring weights are jht's own**, not HEALPix's shipped `weight_ring` array
  (different undocumented solver; reproducing it would also break the no-files
  contract). They are the minimum-norm per-ring correction making the m=0
  quadrature exact to `Lw = 2·nside`; validated end-to-end, not by array match.
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

Reference algorithms (study, don't copy a broken codebase): **libsharp /
Kostelec–Rockmore** for the recursion structure *and* the stability scaling
(the 3-term ℓ-recursion + enhanced-exponent — see "The crux" below); ducc for
ring weights + iteration; Price & McEwen only for the JAX branch-free renorm
*technique* (NOT their m-recursion, the suspected s2fft-defect home).

## The crux: two risks (recursion stability + spin-2 analysis quadrature)

The 2026-06 literature dive corrected the project's risk model. There are two
distinct risks, and **the recursion is necessary but is not the make-or-break.**

### Risk 1 — recursion stability to ℓ_max~1000 (necessary, table-stakes)

Associated-Legendre / Wigner-d recursions **underflow** at high ℓ (the seed
`P̄_mm` carries `(sinθ)^m`, vanishingly small near the poles); a naive `float64`
recursion loses precision well before ℓ~1000. **Chosen scheme:** the libsharp
3-term recursion in ℓ at fixed (m, s) (Kostelec–Rockmore 2008; libsharp
Eq. 10–11), run in the **increasing-ℓ** direction (libsharp §4.1 — stable that
way, *contra* McEwen–Wiaux), stabilized by a **branch-free per-step log-renorm**
(divide the running pair by `|d_ℓ|`, accumulate `log|d_ℓ|`, reconstruct with
`exp`), as a fixed-trip `lax.scan` → `jit`/`vmap`/`grad`-clean. **Spin-0 is
m′=0, spin-2 is m′=s=2 — one code path**, and libsharp §5.1.2 documents spin-2
reaching the *same* accuracy as spin-0 with exactly this machinery. Borrow
Price & McEwen's renorm *technique* (their §3.4, proven JAX-workable), NOT their
m-recursion. Consult Prézeau & Reinecke 2010 (arXiv:1002.1050) for the exact
enhanced-exponent algebra, and **verify the coefficients + seed elementwise vs
`scipy.special`/healpy before trusting any round-trip.** Avoid the
Risbo/Trapani–Navaza Δ(π/2)-matrix family (O(L²)/ℓ memory; double-precision
instability cliff at L~2048) and threshold-*gated* rescaling (data-dependent
branch, XLA-hostile).

### Risk 2 — spin-2 HEALPix analysis quadrature (the actual make-or-break)

This is where s2fft fails, and it is *not* the recursion: s2fft already
rescales, spin-0 (the identical recursion) is machine-precision, and its spin-2
single-mode errors appear at ℓ=8/16/32, *shrink* with ℓ (35→28→22%), and sit far
below the ℓ≤1.5·nside ceiling — none of which fit underflow. The signature
(`m≈ℓ` clean, `m<ℓ` fails O(10%)) is an **aliasing / polar-folding +
missing-ring-weights** defect in the spin-2 map2alm, compounded by an unweighted
Jacobi iteration that stalls (upstream issue #269: "iterations don't fix it").
jht must own the **weighted spin-2 analysis** and validate it **per-(ℓ,m), below
the ℓ≤1.5·nside ceiling**, so recursion / quadrature / band-ceiling errors are
never conflated — single-mode sweeps across *all* m, never round-trip totals
(which mask localized leakage). **Test both risks explicitly and early.**

## Conventions (verified vs healpy 1.19.0 / ducc0 0.41.0 — pin in code)

Leaving these implicit is how `bk-jax` accumulated sign/convention bugs. These
were checked **empirically against both oracles** during the 2026-06 scoping
(healpy ≡ ducc on the transform seam — no adapter needed; the only divergence is
the IAU U-flip, which lives at the *application* layer):

- **aₗₘ layout:** m-major triangular, only m≥0 stored (real maps → conjugate
  symmetry `a_{ℓ,−m} = (−1)^m conj(a_{ℓ,m})`); index
  `idx(ℓ,m) = m·(2·ℓmax+1−m)//2 + ℓ`; size `(ℓmax+1)(ℓmax+2)/2`; m=0 real.
  (Matches `healpy.Alm`.)
- **Normalization:** orthonormal `Y_ℓm` with the **Condon–Shortley** phase,
  `map = Σ a_ℓm Y_ℓm`; **no** extra 4π / √(4π/(2ℓ+1)) factor (a constant map of
  value c gives `a₀₀ = √(4π)·c` — verified; *not* the 4π-normalized convention
  some geodesy libraries use).
- **Spin-2 (E,B)↔(Q,U):** `(Q±iU) = Σ ₊₂/₋₂a_ℓm · ₊₂/₋₂Y_ℓm` with
  `₊₂a = −(aE + i·aB)`, `₋₂a = −(aE − i·aB)`; inverse `aE = −(₂a+₋₂a)/2`,
  `aB = i(₂a−₋₂a)/2`. healpy `alm2map(pol=True)` == ducc `synthesis(spin=2)` to
  ~5e-13, same sign. *(Re-verify these signs against the HEALPix polarization
  primer at implementation time — historically the dangerous ones.)*
- **Polarization U-sign:** jht is **HEALPix-internal (COSMO)-native** (= healpy
  default, = bk-jax's internal storage); `U_IAU = −U_HEALPix`. Flip only at an
  explicit I/O boundary if a consumer asks for IAU.

## Differentiability

- The on-grid SHT is **linear in aₗₘ**, so JVP/VJP are clean to register.
- **The transpose of `synthesis` is `adjoint_synthesis = Sᵀ = Yᵀ` — the exact,
  weight-free adjoint, NOT `map2alm`.** map2alm (`A = SᵀW`, quadrature-weighted
  and iterative) is the *approximate inverse* — a different operator that equals
  the adjoint only on exact-quadrature grids (never on HEALPix). The VJP/JVP of
  synthesis is `Sᵀ`; keep the two distinct. (`Sᵀ` is also the only operator the
  bk-jax seam needs — bk-jax keeps its weighted analysis on ducc.)
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
