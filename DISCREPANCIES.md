# DISCREPANCIES

Residual numeric mismatches and known-limitation notes, with their cause — logged
rather than buried as TODOs (mirroring bk-jax discipline). Each entry says
whether it is a *defect* (to fix) or an *expected property* (to document).

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
  or treat the ambiguous subspace explicitly — the latter belongs to the Wiener
  / MUSE inner-solve phase, not this rung. See `docs/masked.md`.
