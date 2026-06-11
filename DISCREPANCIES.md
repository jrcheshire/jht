# DISCREPANCIES

Residual numeric mismatches and known-limitation notes, with their cause — logged
rather than buried as TODOs (mirroring bk-jax discipline). Each entry says
whether it is a *defect* (to fix) or an *expected property* (to document).

## Weighted analysis floor rises toward the band ceiling (expected, not a defect)

The headline weighted + `niter=3` round-trip ~1e-13 is measured at `lmax ≈ nside`.
The ring weights (`Lw = 2·nside`) make the m=0 quadrature exact for the analysis
products `λ_ℓ0 λ_ℓ'0` only up to `lmax = nside`; between there and the
`lmax ≤ 1.5·nside` ceiling the `niter=3` floor rises to ~5e-7 (measured; the
committed 1e-4 contract still holds with ~2 orders of headroom).

- **Status:** expected property of the quadrature degree, **not** a transform
  defect — `niter=8` recovers ~1e-14 at the ceiling (convergence-rate effect,
  not a floor). Measured table + gate row (`nside=64, lmax=96`) in
  `docs/accuracy.md` / `tests/test_accuracy.py`.
- **Mitigation:** raise `niter` for `lmax > nside`; the transforms warn (once per
  geometry) above the `1.5·nside` ceiling, where accuracy is unvalidated.

## Spin-2 cut-sky recovery — E/B ambiguous modes (expected, not a defect)

`jht.masked.deconvolve` recovers the true a_lm from a cut sky only where the cut
leaves the modes constrained. For **spin-2** there is an additional fundamental
floor: on a cut sky, E and B are not cleanly separable — a subspace of modes
(the *ambiguous modes*, Bunn et al. 2003) is supported almost entirely under the
mask and is therefore unconstrained by the data. These near-null directions of
`Sᵀ(MW)S` are not recoverable; CG from `x0=0` returns the minimum-norm solution
within that subspace, and `reg>0` (Tikhonov) bounds it.

- **Status:** expected physics (information loss from the cut), **not** a
  transform defect. The *operator* is correct — gated to ~1e-13 against the
  explicit dense `Sᵀ diag(MW) S` (`tests/test_masked.py::test_operator_matches_dense`)
  and symmetric/PSD — and the pseudo-a_lm reproduces healpy/ducc to machine
  precision. Only the *recoverability* of cut-localized E/B modes is limited.
- **Manifestation:** in `scripts/masked_sweep.py`, deconvolution error grows as
  fsky shrinks and blows up under heavy apodization (which down-weights data).
  spin-2 reaches the same recovery floor as spin-0 for mild cuts (no s2fft-style
  m<ℓ defect — that was ruled out in Phase 0); the spin-2-specific limitation is
  only the ambiguous-mode subspace at aggressive cuts.
- **Mitigation / scope:** use `reg>0` to stabilise, restrict the recovered band,
  or — now implemented — bound the ambiguous subspace with a signal prior via
  `jht.masked.wiener` (the Cℓ-informed generalization of `reg`; the MUSE
  inner-solve tier), with `constrained_realization` for posterior draws. The
  operator and recovery are unchanged; the prior simply gives the unconstrained
  subspace a principled, finite answer instead of the min-norm CG one. See
  `docs/masked.md`.

## Differentiability: `jax.linear_transpose` / forward-mode of the raw kernels (limitation, not a defect)

The supported autodiff path is JAX's **native reverse/forward AD** through the
public transforms — `jax.grad` / `vjp` / `jacrev` (complex aₗₘ), and `jacfwd ≡
jacrev` / FD on the real-DOF layer (`jht.diff`). All Phase-2 gates pass on these
(`tests/test_grad.py`).

Two lower-level entry points are **not** supported, by deliberate choice:

- `jax.linear_transpose(synthesis, …)` — the forward `lax.scan` in the recursion
  engine is not directly transposable as a standalone linear function ("Error
  interpreting argument to scan as a JAX value"). `jax.vjp`/`grad` are unaffected:
  they linearize first (baking the static recursion plan as constants) and
  transpose the linearized jaxpr, which is clean. Forward scatters carry
  `unique_indices=True` so the *scatter*-transpose blocker is gone; the residual is
  the scan, which native AD sidesteps.
- `jax.jacfwd` / `jax.jvp` on the **complex** `synthesis` — `jacfwd` requires
  real-valued inputs (a JAX constraint, not jht-specific). Forward-mode lives on
  the **real-DOF layer** (`synthesis_real`/`analysis_real`), where `jacfwd ≡
  jacrev` to ~1e-16.

- **Status:** expected JAX-mechanics limitations with first-class supported
  alternatives, **not** transform defects. A `custom_vjp` would *block* forward
  mode entirely; the only kernel-routing fix needs JAX's internal `jax.core.
  Primitive` API (removed in jax 0.9.2). Native AD is the chosen, version-stable
  path. See `docs/design.md` §Differentiability.

## Off-grid (`synthesis_general`): spin>0 m=0 phantom vs ducc (expected, not a defect)

For spin > 0 the off-grid synthesis keeps both real components at every m (Re→first
plane, Im→second), so the field depends on the **unphysical `Im(a_{l,0})`** at m=0 —
the same phantom documented on-grid in `docs/design.md`. jht's *strict adjoint*
`adjoint_synthesis_general` therefore carries a nonzero m=0 imaginary part, while
ducc0's `adjoint_synthesis_general` zeros it.

- **Status:** expected; jht's adjoint is the exact transpose (the inner-product
  identity `<S a, v> == <a, Sᵀ v>_G` holds to ~1e-16, `tests/test_offgrid.py::
  test_adjoint_identity`), ducc applies a different m=0 convention. **Harmless** —
  physical skies have real `a_{l,0}`, so the phantom is never excited. The ducc
  cross-check (`test_adjoint_vs_ducc`) compares the physical m≥1 modes (matched to
  ~1e-13); the m≥1 / real-m=0 content agrees with ducc bit-for-bit.

## Off-grid: the −spin channel carries `(−1)^s` (convention pin, now matched to ducc)

The spin-weighted E/B → ±spin mapping is `+s a = −(aE + i aB)`, `−s a =
−(−1)^s (aE − i aB)`. The `(−1)^s` on the −spin channel is **invisible for even
spin** (the on-grid 0/2 path never exposed it) but flips the result for **odd spin
(1, 3)**. Pinned by matching ducc0's `synthesis_general` for spin 1/3 (a sign probe;
`tests/test_offgrid.py::test_forward_vs_ducc` covers all of spin 0–3). Documented so
the even-only on-grid convention is not assumed to carry over to odd spin.
