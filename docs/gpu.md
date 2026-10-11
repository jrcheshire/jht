# GPU (CUDA) — environment, accuracy, harness

jht is pure JAX, so every transform (on-grid, `analysis`, off-grid, masked) runs on a
CUDA GPU with no code change: the static recursion plan is host-side NumPy, and the
jitted contraction and ring FFTs run on JAX's default device.

## The `gpu` pixi environment

conda-forge `jaxlib` defaults to a CPU build; the CUDA build lives in a dedicated pixi
environment (`pyproject.toml`):

```toml
[tool.pixi.feature.gpu]
platforms = ["linux-64", "linux-aarch64"]  # no CUDA on osx-arm64
[tool.pixi.feature.gpu.system-requirements]
cuda = "12"
[tool.pixi.feature.gpu.dependencies]
python = "3.13.*"                        # conda-forge CUDA jaxlib is py312/py313, not py314
jaxlib = { build = "cuda*" }
cuda-version = ">=12.9,<13"
[tool.pixi.environments]
gpu = { features = ["gpu"] }
```

- linux-64 and linux-aarch64 (x86 NVIDIA nodes, Grace Hopper / Grace Blackwell).
- Python 3.13, because conda-forge ships the CUDA `jaxlib` for py312/py313 only. The
  default CPU env uses 3.14; numerics are identical.
- The env includes the CUDA 12.9 runtime and the validation oracles (healpy, ducc0), so
  the gated suite runs there too. The driver only needs to be forward-compatible with
  CUDA 12.9.

The lock solves for all platforms from any machine. To install on a cluster from a
node without a GPU, mock the CUDA driver for the solve; the real driver is detected at
run time:

```bash
CONDA_OVERRIDE_CUDA=12.9 pixi install -e gpu      # once, on any networked node
pixi run --frozen -e gpu python scripts/gpu_check.py
pixi run --frozen -e gpu test                     # full suite incl. the GPU parity gate
```

## float64 must be enabled

Library code never touches global JAX config; float64 is opt-in **per entry point**. On
GPU the failure is silent: without x64 every array is float32 and accuracy drops from
~1e-13 to ~1e-3 or worse. Enable it before any array is created:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Accuracy on CUDA

Same contract as CPU (`docs/accuracy.md`). The GPU parity gate asserts GPU == CPU to
~1e-12 in float64; it is the same XLA program. fp64 throughput depends on the card's
fp64:fp32 ratio (throttled on consumer parts); correctness does not.

## Harness

- **`scripts/gpu_check.py`:** reports devices, checks float64 CPU-vs-GPU parity to
  ~1e-12 (skipped without a GPU), and times `synthesis` / `adjoint_synthesis` /
  `map2alm`, including a `vmap` batch, plus peak memory. On a CPU-only machine it runs
  the self-consistency and timing parts. `tests/test_gpu.py` makes the parity check
  part of the suite and skips cleanly without a GPU.
- **`scripts/gpu_diagnostic.py`:** a fuller characterisation from a single job. It
  measures:
  - GPU-vs-CPU speedup, and the fp64/fp32 throttle (fp32 is timed for throughput
    only);
  - on-device float64 parity;
  - the memory ceiling per `(nside, lmax, spin)`;
  - `vmap` throughput vs batch size;
  - off-grid NUFFT cost at production scale, and the pointing-gradient cost.

  How it runs:
  - **Isolation:** each point runs in a fresh subprocess, which gives a clean
    per-point peak (`peak_bytes_in_use` is cumulative) and confines an OOM to that
    point.
  - **Sizing:** heavy points are gated by a predicted footprint against the
    detected memory or `--limit-gb`, so the ladder fills whatever card or MIG
    slice it lands on.
  - **Wall time:** `--max-wall` bounds it, and every finished point is written to
    JSONL as it goes.

  ```bash
  python scripts/gpu_diagnostic.py --dry-run --limit-gb 10        # preview the ladder
  pixi run -e gpu python scripts/gpu_diagnostic.py --max-wall 1800 --out run.jsonl
  ```

  On a preemptible batch partition, set `XLA_PYTHON_CLIENT_PREALLOCATE=false` for
  honest per-point memory and log `nvidia-smi -L` to tie results to the card.

## Measured performance

A100 (whole card and 10–25 GB MIG slices) and V100-16GB, float64
(`scripts/gpu_diagnostic.py`, `scripts/profile_adjoint.py`):

- **Parity:** GPU == CPU to 1e-14–1e-13 across nside 256–2048, spin 0/2.
- **Synthesis:** 14–60× an 8-core CPU; fp64/fp32 ≈ 2.2× on these parts.
- **Adjoint:** 187 ms at nside=512; a full `map2alm` (niter=3) ~1.3 s there.
- **Off-grid:**
  - the forward is recursion-bound, ~0.5–0.9 s at lmax=1000 and independent of the
    number of points;
  - lmax=1000 with 10⁶ points peaks at ~1.1 GB, because XLA fuses the `(Npts, W, W)`
    stencil rather than materializing it;
  - the pointing gradient costs ~1× a forward.
- **Batched synthesis:** per-realization time falls with batch size, e.g. 286 → 168 ms
  at nside=512 (batch 1 → 4).
- **nside=2048:** synthesis and `map2alm` compile and run on a 20 GB MIG slice and
  match CPU to ~1e-13 (lmax=1000).
  - The first compile is multi-minute (~458 s), 93% of it the per-ring-length FFT
    unroll (≈ nside distinct kernels).
  - `jht.enable_compilation_cache(dir)` pays it once across processes.
  - `jht.set_azimuth_fft_mode("looped")` collapses the cap FFTs to O(1) kernels: ~6–7×
    faster compile for an ~11–15% runtime cost. See `docs/performance.md`.

**Implementation rule:** the kernels pack, unpack and assemble with **gathers**, never
`at[idx].set` scatters. An fp64/complex scatter is ~10⁴× slower than the equivalent
gather on GPU, and a ~nside-way unrolled scatter overflows `ptxas` at nside=2048.
