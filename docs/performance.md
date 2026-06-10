# jht — Phase-1 performance

Performance characterization of the on-grid HEALPix transforms after the Phase-1
hardening (vectorized recursion + jit + batched ring FFTs + equatorial symmetry).
The Phase-0 spike was correctness-first and eager: a single nside=32 single-mode
sweep burned **30+ CPU-minutes**. Phase 1 makes the transforms production-fast
while keeping every Phase-0 accuracy gate green at its original tolerance
(numerics are unchanged — see `tests/test_vectorized.py` for the
fast-path-vs-eager-reference equivalence, and `tests/test_healpix.py` for the
healpy/ducc parity).

**Headline:** the eager-dispatch penalty is gone. nside=32 transforms now run in
**~1–2 ms** (was the dominant cost of that 30-min sweep); the full test suite went
**139 s → ~60 s** (now dominated by the slow eager *reference oracle* the
equivalence tests deliberately exercise, not the fast path). GPU timing is
deferred to Phase 3 (JAX-metal float64 is unreliable); the code is written
jit/vmap-clean and is GPU-ready.

## What changed (and why it was slow)

The Phase-0 reference (`jht._reference`, retained as a validation oracle) was slow
for four compounding reasons, all addressed:

1. **No `jit`** — every op dispatched eagerly with a host round-trip. → the public
   transforms are `jit`-compiled; geometry / recursion plans / index maps are
   built once per `(nside, lmax, spin)` and cached (`lru_cache` on `_prepare`), so
   `map2alm`'s Jacobi iteration reuses one compiled kernel.
2. **Python loop over `m`** (lmax+1 separate scans) → one fused all-m `lax.scan`
   that vectorizes the recursion over `m` and **fuses the ℓ-contraction**, so the
   full λ table is never materialized (peak λ memory `O(M·n_θ)`, not
   `O(L·M·n_θ)` ≈ 16 GB/component at nside=2048).
3. **Python loop over rings** (one small FFT each) → the φ-FFTs are batched by ring
   length: the equatorial belt (`2·nside+1` rings) in a single FFT, the polar cap
   by length-group.
4. **No equatorial symmetry** → the recursion (≈98 % of the runtime; see below)
   runs only on the north half + equator (`2·nside` colatitudes), using the
   reflection parity `λˢ_{ℓm}(π−θ) = (−1)^{ℓ+m} λ⁻ˢ_{ℓm}(θ)`. spin-0 splits the
   ℓ-sum into even/odd; spin-2 advances the +2 and −2 recursions together (the
   ±s coupling). This is a ~1.8× win at nside=512 (fixed lmax), trending to 2× as
   the recursion dominates more at the ceiling.

The recursion is the bottleneck by a wide margin — a fixed-lmax probe found
synthesis is **90–98 % recursion**, FFTs ≲10 % — which is why the on-the-fly
fused contraction and the N/S halving (both targeting the recursion) are the
levers that matter, and why a dense-λ cache (memory-blocked at the ceiling) was
not pursued.

## Benchmark (CPU, float64)

Reproduce with `pixi run python scripts/bench_transforms.py` (Apple-silicon
osx-arm64; jax 0.9.2). `compile_s` is the first-call wall time (trace + XLA
compile, cached thereafter); `synth`/`adj`/`map2alm` are steady-state best-of-N
jitted run times; `map2alm` uses `niter=3` (= 1 + 2·niter = 7 transform passes).
`lmax` follows the BK regime (~1.5·nside, capped at 1000). peak RSS is the
process high-water mark (cumulative, single process — indicative, not per-call).

| nside | lmax | npix | spin | compile (s) | synth (ms) | adj (ms) | map2alm (ms) | peak RSS (MB) |
|------:|-----:|-----:|:----:|------------:|-----------:|---------:|-------------:|--------------:|
| 32   | 48   | 12 288     | 0 | 0.3 | 1.1   | 1.1   | 6.8     | 380 |
| 32   | 48   | 12 288     | 2 | 0.5 | 1.9   | 1.8   | 12.6    | 460 |
| 128  | 192  | 196 608    | 0 | 0.9 | 28.9  | 30.0  | 206     | 730 |
| 128  | 192  | 196 608    | 2 | 1.8 | 38.2  | 41.6  | 279     | 1 015 |
| 256  | 384  | 786 432    | 0 | 1.7 | 144   | 156   | 1 052   | 1 460 |
| 256  | 384  | 786 432    | 2 | 3.1 | 170   | 203   | 1 323   | 2 123 |
| 512  | 768  | 3 145 728  | 0 | 4.6 | 1 004 | 1 168 | 7 683   | 3 667 |
| 512  | 768  | 3 145 728  | 2 | 7.6 | 1 282 | 1 542 | 10 271  | 5 468 |
| 1024 | 1000 | 12 582 912 | 0 | 11.2 | 3 190 | 3 465 | 23 397 | 9 269 |
| 1024 | 1000 | 12 582 912 | 2 | 17.8 | 4 373 | 4 934 | 33 311 | 12 230 |
| 2048 | 1000 | 50 331 648 | 0 | 24.0 | 6 631 | 7 050 | n/a | 18 788 |
| 2048 | 1000 | 50 331 648 | 2 | 39.2 | 9 138 | 10 302 | n/a | 25 909 |

(map2alm skipped above nside=1024 in the default run — it is ≈7× a single
synthesis; derive from the synth/adj columns.)

The peak-RSS column is the **cumulative** process high-water of the single
laddered run (one process climbing from nside=32, retaining every compiled kernel
and cached plan), so it is monotonic and overstates any one size's footprint. The
**isolated** ceiling footprint (fresh process, nside=2048 alone) is **10.8 GB
(spin-0)** and **13.2 GB (spin-2)**.

## Memory model

Per transform, the dominant live arrays are:

- recursion carry (the fused scan): `O(M · n_θ)` per recursion state, with
  `n_θ = 2·nside` (north half) and `M = lmax+1`. spin-2 runs two states (±2)
  with a few complex accumulators → ~hundreds of MB at the ceiling, not GB.
- the per-m ring coefficients `F` / `V`: `O(M · nrings)` complex.
- the map and the batched belt FFT: `O(npix)`.
- static plan tables (recurrence coeffs, seeds, index maps): `O(M² + M·n_θ)`,
  built in numpy once and captured as compile-time constants.

There is **no dense `O(L·M·n_θ)` λ table** — that is the deliberate on-the-fly
choice (a dense table would be ≈16 GB *per spin component* at the ceiling).

The FFT-assembly stage (not the recursion) processes the polar cap as ~`nside`
length-groups — a length-N HEALPix ring needs an exact length-N FFT, so the
per-ring FFTs cannot be padded to a common length. The per-group **output writes**
were a ~`nside`-way unrolled fp64 scatter into the `O(npix)` map; they are now
hoisted out of the ring loop into a **single combined gather** (`pix_to_buf` /
`ring_to_col` static permutations in `src/jht/healpix.py`). That de-unroll is what
makes nside=2048 compile on GPU (see below) — bit-identical, runtime unaffected
(FFTs are ≲10 %).

## Notes / future levers (deferred)

- **Compile time grows with the cap-FFT length-groups** (~nside distinct lengths
  unrolled in one jit). One-time (jit caches by shape) and modest on CPU; on GPU
  the *one-time* compile is multi-minute at nside ≥ 1024. The combined-gather
  de-unroll (above) removed the per-group scatter that pushed the full module past
  `ptxas` at nside=2048, so it now compiles; the remaining multi-minute compile is
  the per-length-FFT unroll itself (a future lever, not a blocker) — see `docs/gpu.md`.
- **`map2alm` recomputes the recursion** each of its 7 passes (the memory-safe
  on-the-fly choice). A memory-gated cached-λ fast path for small nside is an
  easy future optimization.
- Ring quadrature weights, the full nside/lmax/spin accuracy matrix, and
  partial-sky (masked) analysis were the Phase-1 accuracy rungs (done — see
  `docs/accuracy.md`, `docs/masked.md`); differentiability is Phase 2; GPU is
  Phase 3.

## GPU performance (measured 2026-06-10)

Method and full numbers in `docs/gpu.md`. Headline (Cannon A100 MIG / V100, fp64):

- **Forward synthesis** GPU-accelerates **14–60×** the (8-core) CPU; fp64/fp32 ≈
  2.2×; fp64 GPU==CPU parity 1e-14 … 1e-13 across the BK regime **incl. nside=2048**.
- **Adjoint / `map2alm`** were initially ~21× too slow on GPU: the dense→triangular
  alm packing used a **scatter** (`at[idx].set`), which is **~38000× slower than a
  gather in fp64 on GPU**. Fixed by packing via a gather (`dense_to_tri`, the mirror
  of the forward `tri_to_dense`) — adjoint **4041 → 187 ms** at nside=512, `map2alm`
  ~16 s → ~1.3 s. Numerically identical (the gather index is the scatter's inverse).
- **Off-grid forward** was ~35 s at lmax=1000 (N-independent): the `nufft2d2` grid
  build was an fp64/complex `C.at[].set` **scatter** = 97 % of it. Replaced by an
  inverse-index **gather** → ~1.4 ms, so the forward is recursion-bound (~0.5–0.9 s,
  ~40×). Off-grid is otherwise correct and memory-light (N=1e6 ~1.1 GB); the pointing
  gradient costs ~1× a forward.
- **nside=2048 now compiles + runs on GPU.** The ~`nside`-way unrolled fp64 map
  scatter (fused with the recursion) exceeded `ptxas` (exit 9); the **combined-gather**
  de-unroll (above) shrinks the module enough to compile. At nside=2048, lmax=1000 on
  a 20 GB A100 MIG, synth + `map2alm` match CPU to ~1e-13 and the runtime fits the
  slice (one-time compile is multi-minute, jit-cached).

Every fix here is the same lesson — **fp64/complex `at[idx].set` scatters are
catastrophic on GPU; use a gather** — and each is bit-identical (gated by the full
CPU suite + GPU==CPU parity).
