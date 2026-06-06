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

**Goal:** answer "is owning a JAX SHT viable?" cheaply, before committing. A
couple of days of focused work. Pure JAX, float64, validated against ducc0 and
healpy. No packaging polish, no generality.

**Build:**
- Minimal spin-0 HEALPix forward (map→aₗₘ) and inverse (aₗₘ→map) in pure JAX.
- Minimal spin-2 HEALPix forward/inverse (the part s2fft gets wrong).
- A recursion-stability probe: associated-Legendre / Wigner-d evaluated up to
  ℓ_max ~ 1000 with explicit checks for underflow/precision loss.

**Hard gates (a-priori — set the numbers before running, do not move them after):**
1. **Spin-0** round-trip and vs-healpy/ducc on a band-limited map: reaches the
   HEALPix floor (≲1e-3, ideally ~machine with weights+iteration on band-limited
   input).
2. **Spin-2** round-trip and vs-ducc on a band-limited Q/U map: ≲1e-3 with **no
   structural defect** — explicitly NOT s2fft's 3–28%. This is the make-or-break
   gate.
3. **Recursion stability:** precision does not degrade pathologically as ℓ→1000
   (single-mode sweeps stay clean across all m, not just m≈ℓ — the failure
   pattern s2fft showed).
4. **GPU sanity:** rough GPU-vs-ducc-CPU timing confirms there is actually a win
   to chase (jht on CPU is expected to be *slower* than ducc; the win is GPU).

**Exit criterion:**
- All gates pass → proceed to Phase 1 (the bet holds).
- Spin-2 floor unreachable or recursion fights us → **document it and stop**;
  bk-jax stays on ducc, and we've spent ~2 days, not 2 months, to learn it.
  (A narrower fallback — e.g. lower ℓ_max ceiling — can be considered here.)

**Deliverable:** a spike script + a short findings note (gate results, the
recursion approach that worked, GPU timing), and a clear go/no-go.

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

**Exit criterion:** the test matrix is green at documented tolerances; a
`DISCREPANCIES.md` captures any residual with its cause.

---

## Phase 2 — Differentiability

- JVP / VJP for the (alm-linear) transforms; `jit` / `vmap` clean.
- Handle the **JAX-VJP-convention vs strict-math-adjoint** subtlety
  deliberately (the `2·conj(·)` on m>0 modes that bit bk-jax). Provide both the
  AD-convention transpose and a strict-math-adjoint helper if needed.
- Gradient tests: `jacfwd ≡ jacrev`, and the inner-product adjoint identity
  ⟨A x, y⟩ = ⟨x, Aᵀ y⟩ at tight tolerance.

**Exit criterion:** gradient identities pass; a downstream `jax.grad` through a
toy "map → aₗₘ → bandpower" chain works end-to-end.

---

## Phase 3 — GPU + bk-jax integration

- Settle the **GPU JAX install** story (conda-forge jax is CPU; CUDA needs the
  right channel/pip + a Cannon-compatible setup). Bench on an actual GPU.
- Performance tuning for the BK regime (memory footprint at nside/ℓ_max,
  batching over realizations).
- Expose jht as a `BK_JAX_SHT_BACKEND` backend in bk-jax; add jht as a
  dependency there.
- End-to-end parity in bk-jax's **GPU/differentiable tier** (a forecast sweep
  or the MUSE inner solve) within the agreed accuracy tier — explicitly NOT
  displacing ducc on the purity-critical production path.

**Exit criterion:** a real bk-jax GPU/diff workload runs through jht and agrees
with the ducc path to the agreed tier, with a measured GPU speedup.

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
