# jht — Roadmap

Phased plan for the JAX-native spherical harmonic transform. Written
2026-06-06 as scaffolding; **nothing here is implemented yet.** Read `CLAUDE.md`
first for scope, conventions, and the decision record.

The guiding principle: **validation-gated, smallest-viable-first.** Each phase
has an explicit exit criterion. We do not advance on vibes, and per the standing
rule we do **not** relax accuracy tolerances without an explicit discussion.

The overall bet, stated plainly so it can be killed cheaply: *a clean-room JAX
SHT for the BK regime can hit the HEALPix accuracy floor (no s2fft-style
structural defect) at acceptable GPU performance, giving bk-jax a transform it
can run on GPU, differentiate, and own.* Phase 0 exists to confirm or refute
that before any real investment.

---

## Phase 0 — Feasibility spike (the go/no-go)

**Goal:** answer "is owning a JAX SHT viable?" cheaply, before committing. Pure
JAX, **float64**, CPU (Mac / bicep-dev), validated against ducc0 **and** healpy.
No packaging polish, no generality. Structured as a **discriminating** spike: it
*diagnoses* any failure (recursion vs quadrature vs band-ceiling) rather than
just observing one — the seed's original single-gate design would have conflated
all three.

> **Corrected risk model (2026-06 dive).** The make-or-break is the **spin-2
> analysis quadrature**, not the recursion. s2fft already rescales its recursion
> yet still fails spin-2 (errors at ℓ=8/16/32, *shrinking* with ℓ, below the
> ceiling; spin-0 fine). So Phase 0 tests the recursion *in isolation*
> (table-stakes) AND the weighted spin-2 round-trip *below the ℓ≤1.5·nside
> ceiling, per-(ℓ,m)* (the real gate). See `docs/design.md` "The crux".

**Build (three layers):**
- **L0 — recursion in isolation:** normalized `P̄_ℓm` (spin-0) and `ₛλ_ℓm`
  (spin-2) via the libsharp ℓ-recursion + branch-free log-renorm; validated
  *elementwise* vs `scipy.special`/healpy, all m, ℓ up to ~1000, both spins,
  incl. polar θ.
- **L1 — synthesis + exact adjoint:** `synthesis` (aₗₘ→map) and
  `adjoint_synthesis = Sᵀ` (map→aₗₘ, unweighted), spin-0 and spin-2, with HEALPix
  ring geometry + polar folding. (This pair is the bk-jax-tier capability.)
- **L2 — weighted spin-2 round-trip:** bare analysis `A₀=(4π/N)·Sᵀ` + Jacobi
  iteration; per-(ℓ,m) round-trip **below the ℓ≤1.5·nside ceiling**, single-mode
  sweeps across all m.

**Hard gates (a-priori — set here, before running; not relaxed without sign-off):**
1. **Gate R (recursion isolation):** elementwise rel error ≤ **1e-12** across all
   m at ℓ∈{2,8,16,32,128,256,512,768,1000}, both spins, incl. polar θ.
2. **Gate S-synth:** jht `synthesis` vs healpy `alm2map` **and** ducc
   `synthesis`, band-limited input (ℓmax≤1.5·nside), both spins: ≤ **1e-10**.
3. **Gate adj:** inner-product identity
   `|⟨S a, m⟩ − ⟨a, Sᵀ m⟩| / |⟨S a, m⟩|` ≤ **1e-10**.
4. **Gate Spin2-floor (make-or-break):** single-mode spin-2 round-trip below the
   ceiling reaches the bare floor ≤ **1e-3** flat across **all m** (explicitly
   NOT s2fft's m<ℓ O(10%) leakage); with ≤5 Jacobi iters on band-limited input
   → ≤ **1e-4**. Reproduce s2fft's failing sweep (ℓ=8,16,32) and show jht flat.
5. **Gate Spin0-floor (control):** same structure, spin-0 ≤ 1e-3 bare → much
   better with iteration.

**GPU gate — DEFERRED to Phase 3.** Phase-0 accuracy needs float64, which
JAX-metal does not reliably support; a rough jht-GPU-vs-ducc-CPU timing is a
later Cannon spike. Go/no-go is **not** blocked on GPU access.

**Exit criterion:**
- Gates R, S-synth, adj, Spin2-floor, Spin0-floor green → proceed to Phase 1.
- Weighted spin-2 round-trip can't reach the floor below the ceiling *even with*
  correct recursion + bare weights + iteration → **document and stop** (the bet
  fails). The layered structure guarantees the stop is *diagnosed*. (A narrower
  fallback — lower ℓ_max ceiling — can be considered here.)

**Deliverable:** the spike script(s) + the pytest gates + a short findings note
(`docs/findings_phase0.md`: gate results, the recursion approach that held, the
discriminating evidence on the s2fft root cause), and a clear go/no-go.

---

## Phase 1 — Core transforms, hardened

*(Only if Phase 0 says go.)*

- Spin-0 + spin-2, **forward and adjoint**, production-quality.
- **Ring quadrature weights** + **Jacobi (map2alm-style) iteration** to close
  accuracy from the bare floor toward ducc on band-limited inputs.
- Numerically stable recursion (libsharp-style log / X-number scaling) as the
  standing implementation, not just a spike hack.
- **Conventions pinned and documented** (`docs/design.md`): U-sign, spin sign,
  aₗₘ layout, normalization — with cross-validation against both healpy and ducc.
- Full vs-ducc/healpy **test matrix** across nside / ℓ_max / spin.
- Partial-sky (masked) analysis path, since that's what BK actually uses.
  *(Done — `jht.masked`: pseudo-a_lm + cut-sky CG deconvolution; see
  `docs/masked.md`, `tests/test_masked.py`.)*

**Exit criterion:** the test matrix is green at documented tolerances; a
`DISCREPANCIES.md` captures any residual with its cause.

---

## Phase 2 — Differentiability  *(Done — see `docs/design.md` §Differentiability, `jht.diff`, `tests/test_grad.py`)*

- JVP / VJP for the (alm-linear) transforms; `jit` / `vmap` clean. *(Done —
  **native AD**, no custom rule: `custom_vjp` was rejected because it blocks
  forward-mode AD / `jacfwd`, and the kernel-routing alternative needs JAX's
  internal removed-in-0.9.2 primitive API.)*
- Handle the **JAX-VJP-convention vs strict-math-adjoint** subtlety
  deliberately (the `2·conj(·)` on m>0 modes that bit bk-jax). *(Done — pinned:
  `jax.vjp(synthesis)(v) == G ⊙ conj(adjoint_synthesis(v))` with the `(2−δ_m0)`
  metric `G = jht.healpix.alm_metric_weight`, exact to ~1e-15; `adjoint_synthesis`
  stays the strict transpose; `jht.diff` adds a real-DOF layer with no convention
  subtlety.)*
- Gradient tests: `jacfwd ≡ jacrev`, and the inner-product adjoint identity
  ⟨A x, y⟩ = ⟨x, Aᵀ y⟩ at tight tolerance. *(Done — `tests/test_grad.py`, 1e-12
  algebraic / 1e-6 FD; 15 tests.)*

**Exit criterion:** gradient identities pass; a downstream `jax.grad` through a
toy "map → aₗₘ → bandpower" chain works end-to-end. *(Met — `bandpower` == 
`healpy.alm2cl`; end-to-end `jax.grad` through map→aₗₘ→Cℓ finite, FD-consistent,
`jit`-clean, both spins.)*

---

## Phase 3 — GPU enablement + standalone dependency

**Reframed (2026-06-07):** the goal is *not* to couple jht to bk-jax. jht
becomes a clean, GPU-ready **standalone dependency** that any consumer (bk-jax
first) can adopt *in place of ducc0* — and that adoption is done from the
*consumer's* side, not built into jht. jht stays bk-jax-agnostic (it already is).
Two tracks:

**GPU enablement** *(prepare now; measured run deferred to an NVIDIA box / Cannon)*:
- **GPU JAX install** settled as a locked **pixi `gpu` environment** (conda-forge
  CUDA `jaxlib`, linux-64, py313 — see `docs/gpu.md`). conda-forge jax defaults to
  CPU; the CUDA build lives in this env. Solvable/committable without a GPU.
- A GPU **parity gate + benchmark harness** (`scripts/gpu_check.py`,
  `tests/test_gpu.py`): fp64 GPU==CPU to ~1e-12, timing + `vmap`-batch, memory;
  skips cleanly with no GPU. *(Code is device-agnostic — no change needed.)*
- The **x64-per-entry-point** contract documented (the silent fp64→fp32 footgun).
- Deferred memory lever: the unrolled cap-FFT scatters (~11–13 GB at the ceiling
  — pad-and-fold); bench on GPU first, optimize only if it bites.

**Standalone-dependency surface** *(the decoupling)*:
- A curated public API (`jht.__all__`) + version bump; the stale README rewritten.
- A consumer-seam doc (`docs/consumers.md`): operator path
  (`synthesis`/`adjoint_synthesis`/`map2alm`), grad path (`jht.diff`), the
  `(2−δ_m0)` convention bridge, and the accuracy boundary (GPU/diff tier, **not**
  the ducc purity tier). Backend wiring (e.g. `BK_JAX_SHT_BACKEND`) is the
  consumer's responsibility — explicitly out of jht's scope.
- *(Deferred to publish time: PyPI build/tag — with the "James Cheshire"
  name-check release blocker — LICENSE file, CHANGELOG, CI.)*

**Exit criterion:** jht imports and round-trips through a curated public API; the
GPU env solves + locks and `pixi run -e gpu` runs the harness on an NVIDIA box at
the documented fp64 tier with a measured speedup; the consumer seam is documented.
A consumer can depend on jht as a ducc0 replacement for its GPU/diff tier without
any jht-side coupling.

---

## Phase 4 — Off-grid synthesis (stretch / deferred)

The sim-forward path needs synthesis at arbitrary **detector pointings** — a
NUFFT, not an on-grid SHT, and the piece whose `loc` tangents ducc's FFI can't
differentiate. This is a separate capability (a JAX NUFFT, e.g. building on
jax-finufft-class primitives) and is explicitly deferred until the on-grid core
is solid and a consumer needs it. Likewise any general-sampling support stays
out until something demands it.

---

## Cross-cutting constraints (bind every phase)

- **Accuracy tiers:** ducc for purity-critical production (in bk-jax); jht for
  the GPU/diff tier (~1e-3 HEALPix floor acceptable). Don't conflate.
- **Validation-gated, tolerances a-priori, not relaxed without sign-off.**
- **Conventions documented**, cross-checked vs healpy AND ducc.
- **Pure JAX, no compiled extension; runtime deps = jax + numpy.**
- **Dependency minimalism** is a feature, not an accident — it's the reason this
  repo exists.

---

## Open questions (resolve as phases begin)

- GPU JAX install path on Cannon (conda-forge CPU vs pip CUDA vs a hybrid env)?
  — and note the GPU payoff lands on FairShare-constrained Cannon GPUs, so the
  *benefit* is somewhat deferred even though dev/bench is cheap.
- Exact ℓ_max / nside ceiling to support for BK (drives recursion-stability
  effort and memory budget).
- For the first useful version: stop at the HEALPix floor, or go all the way to
  ducc-class accuracy (weights + full iteration)? Depends on the first real
  consumer's tolerance.
- Real-alm vs complex-alm internal representation (memory vs simplicity).
- Dense-Legendre vs FFT-per-ring + on-the-fly recursion — the perf/memory
  trade at BK ℓ_max (settle empirically in Phase 0/1).
- Whether the NUFFT (Phase 4) ever lives in this repo or as a sibling.
- Eventual packaging / naming for publication (jht = "JAX Harmonic Transforms").

---

## This session (2026-06-06)

Repository scaffolded only: `git init`, `CLAUDE.md`, this roadmap, `docs/`
notes, minimal `pyproject.toml` + package stub. **No transform code.** The
development session picks up at **Phase 0**.

**Scoping pass (same day, after a 4-agent literature dive + primary-source
verification):** corrected the risk model (recursion → spin-2 analysis
quadrature; see the Phase-0 box above and `docs/design.md`), chose the recursion
scheme (libsharp ℓ-recursion + branch-free log-renorm), pinned + verified the
healpy/ducc conventions, and locked three decisions: Phase-0 = a discriminating
3-layer spike; first useful accuracy = bare floor + iteration (no weight files);
GPU timing deferred to Phase 3. Full plan:
`~/.claude/plans/yeah-let-s-get-to-lovely-feather.md`.
