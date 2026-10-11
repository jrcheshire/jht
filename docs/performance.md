# jht — performance

Performance characterization of the on-grid HEALPix transforms. The fast path is
gated against the eager reference implementation (`jht._reference`) by
`tests/test_vectorized.py`, and against healpy / ducc0 by `tests/test_healpix.py`.

**Headline:** small transforms run in ~1–2 ms (nside=32), dominated by the
ℓ-recursion. The code is jit/vmap-clean and runs unchanged on GPU (`docs/gpu.md`).

## How the transforms are made fast

1. **jit + cached plans.** The public transforms are jit-compiled; geometry, recursion
   plans and index maps are built once per `(nside, lmax, spin)` (`lru_cache` on
   `_prepare`), so `analysis`'s Jacobi iteration reuses one compiled kernel.
2. **One fused all-m recursion.** A single `lax.scan` vectorizes the recursion over m
   and fuses the ℓ-contraction, so the full λ table is never materialized: peak λ
   memory is `O(M·n_θ)`, not `O(L·M·n_θ)` (≈ 16 GB per component at nside=2048).
3. **Batched ring FFTs.** The φ-FFTs are batched by ring length: the equatorial belt
   (`2·nside+1` rings) in one FFT, the polar caps by length group.
4. **Equatorial symmetry.** The recursion runs only on the north half + equator
   (`2·nside` colatitudes), using `λˢ_{ℓm}(π−θ) = (−1)^{ℓ+m} λ⁻ˢ_{ℓm}(θ)`. Spin 0 splits
   the ℓ-sum into even/odd; spin 2 advances the ±2 recursions together. ~1.8× at
   nside=512, tending to 2× as the recursion dominates.

Synthesis is **90–98% recursion** (FFTs ≲ 10%), which is why the fused on-the-fly
contraction and the N/S halving are the levers that matter, and why there is no dense
λ cache (it would not fit at the ceiling).

## Benchmark (CPU, float64)

Reproduce with `pixi run python scripts/bench_transforms.py` (Apple-silicon
osx-arm64; jax 0.9.2). `compile_s` is the first-call wall time (trace + XLA
compile, cached thereafter); `synth`/`adj`/`map2alm` are steady-state best-of-N
jitted run times; `map2alm` uses `niter=3` (= 1 + 2·niter = 7 transform passes).
`lmax` follows the usual band-limit (~1.5·nside, capped at 1000). peak RSS is the
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

The FFT-assembly stage processes the polar caps as ~`nside` length groups: a
length-N HEALPix ring needs an exact length-N FFT, so the per-ring FFTs cannot be
padded to a common length. The per-group outputs are assembled by a **single combined
gather** (`pix_to_buf` / `ring_to_col` static permutations in `src/jht/healpix.py`)
outside the ring loop; an unrolled per-group scatter would not compile on GPU at
nside=2048.

## Known costs

- **Compile time** grows with the number of cap-FFT length groups (~nside distinct
  lengths in one jit). It is one-time per program and modest on CPU, but multi-minute
  at nside ≥ 1024 on GPU; see "Compile time" below.
- **`analysis` recomputes the recursion** in each of its `1 + 2·niter` passes (the
  memory-safe on-the-fly choice).

Ring weights and the accuracy matrix are in `docs/accuracy.md`; partial-sky analysis in
`docs/masked.md`.

## GPU performance

Full method and numbers in `docs/gpu.md`. A100 / V100, fp64: synthesis 14–60× an 8-core
CPU; fp64/fp32 ≈ 2.2×; GPU == CPU to 1e-14–1e-13 through nside=2048; the off-grid
forward is recursion-bound (~0.5–0.9 s at lmax=1000). The kernels assemble with gathers,
never fp64/complex scatters, which are ~10⁴× slower on GPU.

## Compile time (nside ≥ 1024) — and the persistent cache

The first compile at nside ≥ 1024 is **multi-minute**, and that cost is structural.
The on-grid `synth`
loops over the distinct HEALPix ring lengths (caps 4, 8, …, 4·nside; the equatorial
belt is one batched group), so it compiles **one ifft kernel per distinct length =
`n_groups` = nside** of them. Measured first-compile (GPU A100 MIG, fp64,
`scripts/profile_compile_time.py`), split into the recursion vs the FFT-unroll
assembly:

| nside | n_groups | synth compile | of which FFT-unroll | recursion | synth run |
|------:|---------:|--------------:|--------------------:|----------:|----------:|
| 256   | 256      | 46 s          | 26 s (57%)          | 0.4 s     | 29 ms     |
| 512   | 512      | 70 s          | 63 s (89%)          | 0.7 s     | 173 ms    |
| 1024  | 1024     | 168 s         | 154 s (92%)         | 1.1 s     | 442 ms    |
| 2048  | 2048     | **458 s**     | **425 s (93%)**     | 1.6 s     | 881 ms    |

The FFT-unroll assembly is ~93% of the compile at nside=2048 and scales
super-linearly (the kernel *count* ∝ nside and each kernel's length grows too); the
recursion compiles in ~1.6 s. (At *runtime* the inversion holds — the recursion
dominates, the many tiny FFTs are cheap to run.) A length-N ring needs an exact
length-N FFT, so distinct ring lengths cannot share a kernel and padding to a common
length is numerically invalid (it changes which harmonics alias, `m -> m mod N`).

There are two levers:

**1. Pay the compile once — persistent cache.** Opt in to JAX's on-disk compilation
cache, which jht exposes as a one-liner (it flips global JAX config, so, like x64, the
entry point opts in and the library never does):

```python
import jht
jht.enable_compilation_cache("~/.cache/jht-xla")   # before the first transform compiles
```

The first nside≥1024 compile is then written to disk and reused by every later process
(keyed by jaxlib version + accelerator + the program); numerics are untouched.

**2. Structurally collapse the kernel count — looped/chirp-z mode.** Opt in with
`jht.set_azimuth_fft_mode("looped")` (or `jht.enable_looped_fft()`). It reroutes the
polar-cap FFTs — every distinct cap length, ~nside of them — through **one common-length
Bluestein (chirp-z) transform inside a single `lax.scan`**, keeping the equatorial belt
on its native FFT. A chirp-z evaluates the exact pruned/aliased length-N DFT via a
fixed length-L convolution, so all cap lengths share one FFT kernel: the compiled
FFT-kernel count drops from **~nside to O(1)**. This is the lever for SHT-heavy /
high-nside *differentiable* graphs (e.g. a masked Wiener + bandpower forecast), where the
unrolled path's executable size — not memory — is the wall (it can exceed XLA's 2 GB
executable cap before it OOMs).

The tradeoff is a bounded per-run FLOP tax on the cap rings (each computed at the common
length `L ≈ 5.5·nside ≥ N`); the belt — the runtime bulk — is untouched. Measured (CPU,
fp64, `scripts/profile_compile_time.py`):

| nside | compile: unrolled → looped | steady runtime tax |
|------:|---------------------------:|-------------------:|
| 128   | 1123 ms → 153 ms (7.3×)    | 1.15×              |
| 256   | 2160 ms → 341 ms (6.3×)    | 1.11×              |

i.e. ~6–7× faster to compile with only an ~11–15% steady-runtime tax (well under the
≤2× a-priori budget), and the compile win widens with nside (unrolled compile ∝ nside,
looped ≈ flat). The default stays `"unrolled"` — faster for a single transform, and its
one-time compile is cached by lever 1. Numerically the two modes agree to FFT roundoff
(gated at `atol=1e-12` vs the unrolled path, plus healpy/ducc0 parity;
`tests/test_azimuth_fft.py`).
