# jht — Technical design notes

Design notes for the JAX-native SHT — the algorithm choices, conventions, and
accuracy decisions. The transforms are implemented and validated; the measured
accuracy contract lives in `docs/accuracy.md` and the performance characterization
in `docs/performance.md`.

## Accuracy tiers (the contract)

HEALPix has **no sampling theorem**, so a HEALPix SHT is intrinsically
approximate. Three accuracy regimes matter:

| Tier | What it is | Measured (broadband) | Use |
|---|---|---|---|
| Bare floor | Direct quadrature, uniform `4π/Npix` | ~2–17e-3 | Not good enough alone |
| + ring weights | jht's own min-norm ring weights (`jht.weights`) | bare ~2–12e-4 | Closes toward ducc |
| + Jacobi iteration | `analysis` iterative refinement, band-limited | **~1e-13** | ducc-class |

- **Bare floor, no structural defect:** spin-2 sits at ~1e-3 like spin-0 — no
  spin-2-specific accuracy defect.
- **Weighted accuracy:** ring weights + iteration reach machine precision on
  band-limited maps, matching `healpy.map2alm(use_weights=True)`. The **committed
  a-priori contract is weighted + niter=3 ≤ 1e-4** (measured ~1e-13, ~9 orders of
  headroom). Full numbers + the ring-weight algorithm: `docs/accuracy.md`.
- **Ring weights are jht's own**, not HEALPix's shipped `weight_ring` array
  (different undocumented solver; reproducing it would also break the no-files
  contract). They are the minimum-norm per-ring correction making the m=0
  quadrature exact to `Lw = 2·nside`; validated end-to-end, not by array match.
- ducc remains the high-accuracy CPU reference; jht serves the GPU/diff tier
  where ~1e-3 is acceptable. Tolerances are a-priori and not relaxed without
  sign-off (standing rule).

## Algorithm skeleton

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
  at ℓ_max ≲ 1000. Dense Pₗₘ tables may be fine at that ℓ_max on GPU; measure.
- **Real-aₗₘ vs complex-aₗₘ** internal representation.
- Batching over realizations (vmap) for the sim/forecast use cases.

Reference algorithms (study the published methods): **libsharp /
Kostelec–Rockmore** for the recursion structure *and* the stability scaling
(the 3-term ℓ-recursion + enhanced-exponent — see "The crux" below); ducc for
ring weights + iteration; Price & McEwen for the JAX branch-free renorm
*technique* (the recursion only, not their m-recursion).

## The crux: two risks (recursion stability + spin-2 analysis quadrature)

Two distinct risks drive the design, and **the recursion is necessary but is not
the make-or-break.**

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

The harder risk is *not* the recursion (spin-0, the identical recursion, is
machine-precision) but the **spin-2 analysis quadrature**. The failure mode it
guards against is a single-mode `m<ℓ` leakage signature — `m≈ℓ` clean, `m<ℓ` off
by O(10%) — the symptom of an **aliasing / polar-folding + missing-ring-weights**
defect in spin-2 map2alm, which an unweighted Jacobi iteration does not fix. jht
therefore owns the **weighted spin-2 analysis** and validates it **per-(ℓ,m),
below the ℓ≤1.5·nside ceiling**, so recursion / quadrature / band-ceiling errors
are never conflated — single-mode sweeps across *all* m, never round-trip totals
(which mask localized leakage). **Both risks are tested explicitly.**

## Conventions (verified vs healpy 1.19.0 / ducc0 0.41.0 — pin in code)

Leaving these implicit is a classic source of sign/convention bugs. They are
checked **empirically against both oracles** (healpy ≡ ducc on the transform seam
— no adapter needed; the only divergence is the IAU U-flip, which lives at the
*application* layer):

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
  default); `U_IAU = −U_HEALPix`. Flip only at an
  explicit I/O boundary if a consumer asks for IAU.

## Differentiability

The on-grid SHT is **linear in aₗₘ** and differentiates cleanly under JAX's
**native** autodiff. jht registers **no** custom VJP/JVP rule.

- **Mechanism = native AD.** `jax.grad` / `vjp` / `jacrev` / `jacfwd` all work and
  are numerically correct. A `custom_vjp` was evaluated and **rejected**: it
  *blocks forward-mode AD* (so `jacfwd` — and hence `jacfwd ≡ jacrev` — fails on
  `synthesis` and everything downstream), and the only mechanism that both keeps
  forward mode and routes reverse through the hand kernel needs JAX's internal
  `jax.core.Primitive` + manual MLIR lowering (already removed/moved in jax 0.9.2)
  — fragile, against the pure-JAX dependency-control point. Native AD avoids all
  of it. Forward scatters carry `unique_indices=True` (the indices are genuinely
  unique) to keep the kernels transpose-friendly; forward numerics are unchanged.
- **Two distinct operators, kept separate.** `adjoint_synthesis = Sᵀ = Yᵀ` is the
  exact, weight-free **strict transpose** (the operator seam / the operator a CG
  solve needs); `analysis` (`A = SᵀW`, weighted + iterative, aka `map2alm`) is the
  *approximate inverse*. Neither is the AD cotangent — keep all three distinct.
- **The convention (the `2·conj` subtlety), pinned.** The packed
  aₗₘ store only m≥0; a real field's m<0 half is implied, so the alm inner product
  carries the diagonal metric `G = 2 − δ_{m0}` (`jht.healpix.alm_metric_weight`).
  JAX's native reverse-mode returns the Euclidean cotangent **in this packing**:

      jax.vjp(synthesis)(v)  ==  G ⊙ conj(adjoint_synthesis(v))

  exact for spin 0 (all modes) and spin 2 (m>0), verified to ~1e-15
  (`tests/test_grad.py`). So native AD is *numerically identical* to the validated
  `adjoint_synthesis` kernel — cotangent and strict adjoint differ only by the
  documented diagonal `G` and a conjugation — and `jax.grad` is therefore
  finite-difference consistent (JAX's complex-grad returns `dL/dRe − i·dL/dIm`).
- **Spin-2 m=0 phantom.** `synthesis` keeps both Re→Q and Im→U at every m, so
  unlike spin 0 (where the final `Re(·)` annihilates Im(aₗ₀)) the map *does* depend
  on the unphysical Im(aₗ₀) at m=0; the native cotangent reflects it with an extra
  E/B-mixing term there (`cot_E = bE − i·bB`, `cot_B = bB + i·bE`, with `bE,bB` the
  real strict-adjoint values — see the bridge in `tests/test_grad.py`). Physical
  fields have real aₗ₀; the real-DOF layer drops this direction entirely.
- **Real-DOF layer = the recommended differentiable interface.** `jht.diff`
  exposes `synthesis_real` / `analysis_real` over the real isometry coordinates `x`
  (the `jht.masked` isometry `T`, `‖x‖₂ = ‖a‖_w`): plain `R^n → R^m` maps with **no**
  conj / `2·conj` subtlety — `jacfwd ≡ jacrev` exactly and finite differences are
  unambiguous. Plus `bandpower` (Cℓ with the `G` fold, == `healpy.alm2cl` to 1e-16).
  Use this for optimisation / field-level inference.
- **Tested (`tests/test_grad.py`, gates 1e-12 algebraic / 1e-6 FD):** `jacfwd ≡
  jacrev`; ⟨S a, v⟩ = ⟨a, Sᵀ v⟩; native-VJP == the `G`-bridge; FD agreement;
  `jit` / `vmap` cleanliness (note: pre-warm the `_prepare` cache before `vmap`,
  else a first-trace-inside-vmap leaks a tracer); end-to-end map→alm→Cℓ grad.
- Geometry/pointing differentiability is the **off-grid NUFFT** story
  (`docs/offgrid.md`), not the on-grid core.

## Explicit non-goals (keep the surface small)

- Arbitrary spin-s; non-HEALPix samplings; ℓ_max ≫ 1000.
- Off-grid synthesis / NUFFT (separate, deferred).
- A compiled (C++/CUDA) kernel. Pure JAX is the dependency-control point; only
  revisit under a deliberate, documented decision.

## Validation harness

- Parity tests vs **healpy** (`alm2map`/`map2alm`, `pol=True`) and **ducc0**
  for every transform, at documented a-priori tolerances.
- Single-mode sweeps (all m, ℓ up to ~1000) to catch localized leakage that
  round-trip totals hide.
- A `DISCREPANCIES.md` log for any residual mismatch: where seen, suspected
  cause, what would close it.
