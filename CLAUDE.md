# jht — JAX Harmonic Transforms

JAX-native spherical harmonic transforms. GPU-capable, fully differentiable,
and **dependency-controlled** — pure JAX + numpy at runtime, no compiled C++
extension and no heavyweight third-party SHT library. Born from the BICEP/Keck
`bk-jax` pipeline's need for a transform it can run on GPU, differentiate
through, and *own* the numerics of.

**Status (2026-06-06): scaffolding only.** This file, `ROADMAP.md`, and the
`docs/` notes were written to carry the design into a future development
session. No transforms are implemented yet. **Read `ROADMAP.md` next; the first
action is the Phase-0 feasibility spike.**

---

## What jht is, in one paragraph

A focused, clean-room implementation of the forward and inverse spherical
harmonic transform (map ↔ aₗₘ) for **spin-0 and spin-2** fields on the
**HEALPix ring** pixelization, in the **BICEP/Keck angular regime**
(ℓ_max ≲ 1000, nside ≤ ~2048). It exists to serve the GPU / differentiable
tier of analysis that a CPU-only C++ transform structurally cannot. It is
*not* an attempt to reimplement a general all-spin / all-sampling library.

## Scope

**In scope (the target):**
- Spin-0 and spin-2 (T and Q/U / E/B) transforms, forward and adjoint.
- HEALPix ring pixelization.
- BK regime: ℓ_max ≲ 1000, nside ≤ ~2048; partial-sky (masked) analysis.
- GPU execution (the primary motivation) and JAX autodiff (alm-linear).
- Validation to the relevant accuracy tier against ducc0 / healpy.

**Out of scope (for now — keep the surface small):**
- Arbitrary spin-s, non-HEALPix samplings, extreme ℓ_max. We are not chasing
  s2fft-style generality.
- **Off-grid synthesis (the NUFFT)** — synthesis at arbitrary detector
  pointings, i.e. the `bk-jax` *sim-forward* path. That is a distinct
  capability (a JAX NUFFT) and a separate, later phase. "JAX-native SHT" ≠
  "escape ducc everywhere": jht replaces the **on-grid** SHT only.
- Being gratuitously BK-coupled. Write the core so BK assumptions live at the
  edges; the transform itself should be reusable.

## Why jht exists (decision record — see `docs/motivation.md` for the full story)

The recurring constraint is a single C++ black box (ducc0) that does the
harmonic transforms. It is excellent but:
- **CPU-only.** The forecast / field-level-inference (MUSE) ambitions are
  GPU-hungry; ducc can't go there.
- **A black box.** ducc is vendored in `bk-jax` as a submodule wrapped in a
  custom XLA FFI — that is *build* control, not *code* control. Its internals
  are unmodifiable C++, and off-grid pointing tangents raise
  `NotImplementedError`.
- **A build tax.** The C++/scikit-build/CUDA toolchain has repeatedly broken
  CI and local installs.

The leading JAX-native alternative, **s2fft**, was evaluated and rejected:
its **spin-2 HEALPix inverse has a structural precision defect** (measured
3–28% error at production ℓ, vs its own documented ~1e-3), and as of
2026-06 it is **still at v1.4.0 (Feb 2025)** with the relevant issues triaged
and parked ("no time") and a stagnant PR queue. **Do not depend on s2fft, and
do not fork it** (owning a large codebase you find broken is the opposite of
dependency control). Clean-room, *informed by* the published methods
(Price & McEwen for the algorithm; libsharp for recursion stability), is the
control-aligned path.

Crucially, the s2fft evaluation **validated the architecture**: a JAX-native
SHT integrates cleanly (JIT-clean, spin-0 machine-precision, finite/well-scaled
VJP). The concept works; one library's spin-2 corner does not.

## Relationship to bk-jax (the consumer)

- jht is a **future dependency** of `bk-jax` (`~/bicepkeck/bk-jax`), added as a
  path/pip dependency once Phase 0–1 land. It is developed and versioned here,
  independently — that's the point.
- **ducc stays in bk-jax for the production path.** The ~1e-4 purity /
  E→B-leakage-critical bandpower pipeline keeps using ducc; that accuracy is
  person-decades of hard-won tuning and is not the thing to risk first.
- **jht earns its keep in the tier ducc can't serve:** GPU forecast sweeps,
  the MUSE inner Wiener solve, and (eventually) fully-differentiable geometry —
  where the HEALPix ~1e-3 accuracy floor is acceptable.
- `bk-jax` already dispatches its transform via a `BK_JAX_SHT_BACKEND` env var;
  jht becomes a new backend option there (Phase 3).

## Accuracy philosophy (the validation contract)

This is the make-or-break of the project; treat it with the same rigor `bk-jax`
applies — **a-priori tolerances, gate-driven, and tolerances are NOT relaxed
without an explicit discussion and sign-off.**

- HEALPix has **no sampling theorem**, so any HEALPix SHT is approximate. The
  fundamental floor without ring weights is ~1e-3; ducc reaches ~1e-4 and
  better via **ring quadrature weights + Jacobi (map2alm-style) iteration**.
- jht's first target is **reach the HEALPix floor with no structural defect**
  (i.e. spin-2 must NOT show s2fft's 3–28% — it must sit at ~1e-3 like spin-0),
  then close toward ducc by adding ring weights + iteration.
- **Spin-2 is the gate.** It is exactly what s2fft got wrong; if jht's spin-2
  HEALPix transform is correct to the floor, the project is viable.
- Every transform lands with a parity test vs **both** healpy and ducc0
  (cross-checking conventions), at a documented tolerance. Log any residual
  mismatch (a `DISCREPANCIES.md`, mirroring bk-jax discipline) rather than
  burying it.

## The crux technical risk

**Associated-Legendre / Wigner-d recursion numerical stability to ℓ_max~1000.**
The classic underflow/overflow problem: naive recursions silently lose
precision at high ℓ. The solved technique is libsharp-style log / "X-number"
scaling. This is almost certainly the class of bug s2fft never fixed. It is the
single thing most likely to make or break the spike — test it explicitly and
early (Phase 0).

## Conventions — pin and document these from day one

These have repeatedly bitten `bk-jax`; do not leave them implicit.

- **Polarization sign:** HEALPix-internal vs IAU U-sign. Decide the internal
  convention, document it, and cross-validate against healpy (`pol=True`) and
  ducc. (bk-jax stores HEALPix-internal and flips U at I/O boundaries.)
- **Spin sign convention** (spin +2 vs −2 / E,B sign).
- **aₗₘ layout** (healpy m-major triangular packing; real vs complex; which ℓ,m
  ordering) and **normalization** (4π vs orthonormal). Mirror healpy/ducc and
  document the exact mapping.
- **Differentiability convention:** the on-grid SHT is alm-linear, so JVP/VJP
  are straightforward — but note the **JAX-VJP-convention vs strict-math-adjoint**
  subtlety (the `2·conj(·)` on m>0 modes) that bit bk-jax. Register transposes
  deliberately and test `jacfwd ≡ jacrev` and the inner-product adjoint
  identity.

## Tooling & conventions

- **Pixi** is the env tool (`pixi run test|lint|format|typecheck`). Standard.
- **Pure Python / JAX — no compiled extension.** This is a deliberate feature:
  zero build-toolchain pain, unlike bk-jax's vendored-ducc FFI. Keep it that
  way; if a kernel ever truly needs C++/CUDA, that's a major decision, not a
  drive-by.
- **Runtime deps = jax + numpy only.** ducc0 / healpy are dev/test oracles in
  the pixi env, never `[project.dependencies]`.
- **No Jupyter notebooks for working code** — scripts + `ipython %run` for
  exploration (user-level standard).
- **JAX float64 is opt-in per entry point** (`jax.config.update("jax_enable_x64",
  True)` in tests/scripts; library code does not touch global config) — same
  rule as bk-jax; SHT accuracy work needs float64.

## Attribution (release blocker)

Published / shippable author fields must read **"James Cheshire"**
(`cheshire@caltech.edu`) — already set in `pyproject.toml` `authors`/
`maintainers`. "jamie" is fine only in the local git author config. This repo
may become a published dependency; verify the name before any tag / build /
publish. ("Jamie" must never appear in a shipped artifact.)

## Dev workflow (per transform)

1. Implement the transform over typed JAX arrays (parametric where a systematic
   knob could later live).
2. Gate it against healpy **and** ducc0 at an a-priori documented tolerance.
3. Pin & document the conventions it assumes (`docs/design.md`).
4. `pixi run lint` + `typecheck` before committing; terse single-line commit
   messages (user preference).
5. Log any numeric mismatch in `DISCREPANCIES.md`, don't bury it as a TODO.

## Pointers

- `ROADMAP.md` — phased plan + hard gates + open questions.
- `docs/motivation.md` — why jht exists (full decision record).
- `docs/design.md` — technical design, accuracy tiers, conventions, algorithm.
- Consumer / sibling context: `~/bicepkeck/bk-jax` (the pipeline that motivates
  jht; do not make jht depend on reading bk-jax to be understood).
