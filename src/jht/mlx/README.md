# jht.mlx — Apple-GPU (MLX, fp32) backend — **SHELVED**

This subpackage is an **archived, dormant** experiment: an MLX (Apple Silicon GPU,
float32) backend for jht's on-grid spherical harmonic transforms, intended as a
lower-accuracy local-development tier alongside the production JAX (fp64) path. It lives
only on the `archive/mlx-fp32-shelved` branch — it is **not** part of any release and is
**not** imported by the package.

## Why it exists, and why it's shelved

Apple Silicon GPUs have no float64, so any Mac-GPU SHT is a separate ~fp32-precision
tier. The enabling idea — validated here — is **fp64-table / fp32-apply**: run the
numerically dangerous λ-recursion in fp64 on the CPU (`_lambda_table.py`, pure numpy,
reuses the validated `jht._recursion` machinery), materialize the spin-weighted λ table,
cast it to fp32, and run only the well-conditioned **contraction + FFT** in fp32 on the
GPU.

**The approach works.** A feasibility spike (`scripts/mlx_spike.py`) measured this
against the JAX-fp64 oracle at **~2–3×10⁻⁷ (fp32 machine precision), flat across
ℓ = 96…1000**, spin-2 identical to spin-0, with no high-m defect. The implementation in
`_apply.py` (synthesis / adjoint_synthesis / analysis / map2alm, spin 0 & 2) is
**mathematically correct**: on the MLX **CPU** device it reproduces the JAX path to fp32
precision deterministically across the whole surface.

**The blocker is upstream.** On the MLX **Metal (GPU)** backend tested (Apple M4 Max,
MLX 0.29.4 and 0.31.2 — version-independent), this *specific* multi-kernel SHT graph
returns **non-deterministic, corrupt results**: two identical calls can differ by
O(10²–10³), at roughly a 25% (synthesis) / 50% (adjoint) per-run rate, correlated with
background GPU activity. It was **not** reproducible by any isolated building block
(matmul stacks, reductions, complex matmuls, standalone complex-FFT loops, einsum/gather
over large cached buffers were all bit-exact) — only the full fused graph triggers it,
and it was not fixable from this side via `mx.eval`/`mx.synchronize` discipline,
disabling the buffer cache, an MLX version swap, or kernel warmup. The MLX-CPU path being
exact confirms it is a Metal-runtime hazard, not a numerics problem in this code.

## How to resume

If MLX-Metal becomes reliable for this workload (a newer MLX release, different Apple
silicon, or an upstream fix), this code should be ready. The first step on any new
setup is a **determinism gate**: call each transform twice with identical inputs and
assert bit-exact agreement on the GPU device; only proceed if that is clean. A
single-process, no-other-GPU-load environment and a two-phase test harness (compute all
JAX references first, then all MLX — never interleave JAX dispatch with MLX-GPU in one
process, which independently corrupts MLX-GPU output) are also required for meaningful
parity testing.

## Contents

| file | role |
|---|---|
| `../_lambda_table.py` | fp64 numpy spin-weighted λ-table builder (the fp64→fp32 hand-off; standalone, MLX-independent, reusable) |
| `_apply.py` | fp32 MLX on-grid kernels (synthesis / adjoint / analysis / map2alm, spin 0 & 2; scatter-free gather ring-fold) |
| `__init__.py` | public surface of the MLX tier |
| `../../../scripts/mlx_spike.py` | the Phase-0 feasibility spike (the ~3e-7 validation) |
