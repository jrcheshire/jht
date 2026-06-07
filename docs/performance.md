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

The **measured** isolated footprint at nside=2048 is nonetheless ~11–13 GB —
well above the sum of the live arrays above. The excess is the FFT-assembly
stage, not the recursion: the polar cap is processed as ~`nside` length-groups,
each unrolled in the jit graph with a scatter into the full `O(npix)` map, and
XLA does not always fuse these in place. This is the next **memory** lever (a
pad-and-fold to a common ring length, or a single combined scatter, would cut
it); runtime is unaffected (FFTs are ≲10 %).

## Notes / future levers (deferred)

- **Compile time grows with the cap-FFT length-groups** (~nside distinct lengths
  unrolled in one jit). It is a one-time cost (jit caches by shape) and modest
  through the ceiling; a pad-and-fold scheme could cut it if it ever bites.
- **`map2alm` recomputes the recursion** each of its 7 passes (the memory-safe
  on-the-fly choice). A memory-gated cached-λ fast path for small nside is an
  easy future optimization.
- Ring quadrature weights, the full nside/lmax/spin accuracy matrix, and
  partial-sky (masked) analysis were the Phase-1 accuracy rungs (done — see
  `docs/accuracy.md`, `docs/masked.md`); differentiability is Phase 2; GPU is
  Phase 3.
