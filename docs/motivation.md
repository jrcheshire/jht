# Why jht exists — the decision record

Captured 2026-06-06, from the conversation that spun this repo out of `bk-jax`.
Recorded so a future session does not relitigate the "why" or repeat the
evaluation that's already been done.

## The recurring constraint

`bk-jax` (the JAX BICEP/Keck pipeline) does its spherical harmonic transforms
through **ducc0**, Martin Reinecke's C++ library, wrapped in a hand-written XLA
FFI custom call. ducc is the gold standard for SHT accuracy and CPU speed — but
the same constraint keeps surfacing across unrelated efforts:

1. **CPU-only.** The directions bk-jax most wants to grow into are GPU-hungry:
   - differentiable forecasting (sweeping instrument knobs through the full
     R → purification → BPCM → likelihood stack),
   - field-level inference (a MUSE bridge; the inner Wiener-filter MAP solve is
     a CG over `(S⁻¹ + RᵀN⁻¹R)`, run many times),
   - large sim ensembles.
   ducc cannot run on GPU, so all of this is capped at CPU throughput.

2. **A black box we don't control.** ducc is vendored as a submodule and
   wrapped in our own FFI — that is *build* control, not *code* control. Its
   internals are unmodifiable C++. Concretely, off-grid pointing (`loc`)
   tangents raise `NotImplementedError`, which blocks differentiating through
   detector geometry — exactly the kind of thing a JAX-native transform would
   give for free.

3. **A build tax.** The C++ / scikit-build / CUDA / submodule toolchain has
   repeatedly broken CI and local installs (config-setting rejections, macOS
   wheel-tag mismatches, flaky submodule fetches). Every one is fixable, but
   it's a standing maintenance cost that a pure-JAX library simply doesn't have.

The user's framing: *"This is not the first time our ambitions have been
stymied by the C-shaped black box that quacks and does harmonic transforms,"*
plus a clear preference for **being in control of dependencies**.

## Why not just use s2fft

s2fft (Price & McEwen) is the leading JAX-native SHT and was the obvious
candidate. It was evaluated (the bk-jax "M0 SHT spike", 2026-05-03) and
**rejected for our use**:

- Its **spin-2 HEALPix inverse has a structural precision defect**: measured
  3–28% error at production ℓ (L=257–769, nside 128–256), versus its own
  documented ~1e-3. The failure pattern: at high ℓ only `m ≈ ℓ` single modes
  pass; `m < ℓ` fail by O(10%). Spin-0 is machine-precision fine — the bug is
  specific to the spin-±2 + HEALPix corner, i.e. **polarization**, which is the
  whole point for CMB B-modes.
- jax-healpy delegates to s2fft for SHTs, so it inherits the same bug — not a
  viable wrapper alternative.
- As of 2026-06 s2fft is **still at v1.4.0 (12 Feb 2025)** — no release since
  the spike. The relevant accuracy issues are triaged and parked by the
  maintainers ("we don't have time"), and the PR queue is stagnant / broken.

So: **don't depend on s2fft** (its broken corner is exactly our corner, and
it's unmaintained on that front), and **don't fork it** either — inheriting a
large codebase that's broken where you need it and that you find hard to follow
is the *opposite* of dependency control.

## What the s2fft evaluation did prove

Importantly, the spike **validated the architecture**, not just rejected a
library. s2fft's `inverse_jax` was JIT-clean (1 eqn, 0 callbacks), spin-0
matched healpy at 3.9e-13, the VJP was finite and well-scaled, and it was only
~1.7× ducc on CPU at a small grid. A JAX-native SHT *integrates cleanly* into
this kind of pipeline. The concept works; one library's spin-2 implementation
does not.

## The reframe that makes this tractable

The first JAX SHT **does not have to replace ducc**:

- **ducc stays** on the ~1e-4 purity / E→B-leakage-critical production path in
  bk-jax. That accuracy is person-decades of expert tuning; it is not the thing
  to risk first.
- **jht targets the tier ducc structurally can't serve** — GPU forecast sweeps,
  the MUSE inner solve, fully-differentiable geometry — where the HEALPix ~1e-3
  accuracy floor is perfectly acceptable.

That collapses the bar from "match ducc to machine precision" down to
"BK-regime spin-0/2 on HEALPix, good-enough + GPU + differentiable, with no
structural defect" — which is genuinely achievable, and is exactly the gap that
keeps biting.

## Honest costs (so the decision is eyes-open)

- Owning a transform means owning its **numerics forever** — ring quadrature
  weights and, especially, **recursion stability to ℓ_max~1000** (the classic
  underflow problem; libsharp solves it with log/X-number scaling). Bounded for
  the BK regime, but a real, permanent surface.
- ducc is already fast on CPU, so jht offers **no CPU win** — the payoff is
  purely on GPU, and that GPU payoff currently lands on FairShare-constrained
  Cannon GPUs (development and benchmarking themselves are cheap).
- jht is the **on-grid SHT only**; the sim-forward off-grid NUFFT is a separate
  capability (deferred to Phase 4).

These are why the plan leads with a cheap, hard-gated **feasibility spike**
(ROADMAP Phase 0) rather than a commitment: it converts "should we own an SHT?"
from a values/vibes question into data, for ~2 days of work.

## Lineage

- Motivating consumer: `~/bicepkeck/bk-jax`.
- s2fft spike findings live in bk-jax memory
  (`feedback_s2fft_spin2_healpix_defect`) and the build-tax history in
  `reference_bk_jax_build_toolchain`.
- Downstream science drivers that want the GPU/diff transform: the bk-jax
  differentiable-forecast and MUSE-bridge directions.
