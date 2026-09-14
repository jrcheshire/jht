"""GPU probe for GitHub #4 / #5: an SHT-heavy ``value_and_grad`` at high nside loads and runs.

    pixi run -e gpu python scripts/gpu_constants_probe.py --nside 2048 --pairs 16
    pixi run -e gpu python scripts/gpu_constants_probe.py --nside 2048 --pairs 16 --old-tables

Each of ``--sims`` sims (a checkpointed ``lax.map`` body) runs ``--pairs`` spin-2
``synthesis_vjp -> adjoint_synthesis_vjp`` pairs with a scalar weight inside the chain;
the gradient is taken w.r.t. that weight, so the backward also runs every transform. The
maps are rescaled by ``4 pi / npix`` after each pair so values stay O(1).

``--old-tables`` reinstates the pre-#5 recursion tables (NumPy arrays closed over, so XLA
embeds a copy per use) for an A/B in one environment. Run with JAX's default GPU
preallocation: the pre-#5 failure is the CUDA driver loading that constant data outside
the preallocated pool (``Failed to load in-memory CUBIN ... OUT_OF_MEMORY``).

Prints the devices, compile+first and steady wall times, the value and gradient, peak
device bytes (if reported) and peak host RSS. Exits 1 on a failed evaluation.
"""

from __future__ import annotations

import argparse
import math
import resource
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jht  # noqa: E402
from jht import _recursion  # noqa: E402


def _peak_device_gb():
    """Peak device bytes if the backend reports them, else None."""
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:  # noqa: BLE001 -- backend without memory stats
        return None
    for k in ("peak_bytes_in_use", "peak_bytes", "bytes_in_use"):
        if stats and k in stats:
            return stats[k] / 1e9
    return None


def _peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6  # darwin: bytes, linux: KB


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=2048)
    ap.add_argument("--lmax", type=int, default=None, help="default round(1.5 * nside)")
    ap.add_argument("--pairs", type=int, default=16, help="synthesis/adjoint pairs per sim")
    ap.add_argument("--sims", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=2, help="steady-state evaluations")
    ap.add_argument("--fft-mode", default="looped", choices=["looped", "unrolled"])
    ap.add_argument("--old-tables", action="store_true", help="pre-#5 embedded recursion tables")
    args = ap.parse_args()

    nside = args.nside
    lmax = args.lmax if args.lmax is not None else round(1.5 * nside)
    if args.old_tables:
        _recursion._plan_arrays = lambda plan: tuple(  # type: ignore[assignment]
            jnp.asarray(a) for a in _recursion.recursion_tables_np(plan.x, plan.spin, plan.lmax)
        )
    jht.set_azimuth_fft_mode(args.fft_mode)
    print(
        f"jht {jht.__version__} from {jht.__file__}\ndevices {jax.devices()}\n"
        f"nside={nside} lmax={lmax} pairs={args.pairs} sims={args.sims} "
        f"fft={args.fft_mode} old_tables={args.old_tables}",
        flush=True,
    )

    rng = np.random.default_rng(0)
    K = jht.alm_size(lmax)
    A = jnp.asarray(rng.standard_normal((args.sims, 2, K)) + 1j * rng.standard_normal((args.sims, 2, K)))
    scale = 4.0 * math.pi / (12 * nside**2)

    def one(s, a):
        for _ in range(args.pairs):
            m = jht.synthesis_vjp(a, nside, lmax, 2) * s
            a = jht.adjoint_synthesis_vjp(m, nside, lmax, 2) * scale
        return jnp.sum(jnp.abs(a) ** 2)

    body = jax.checkpoint(one)

    def loss(s, X):
        return jnp.sum(jax.lax.map(lambda a: body(s, a), X))

    vg = jax.jit(jax.value_and_grad(loss))
    try:
        t0 = time.perf_counter()
        v, g = jax.block_until_ready(vg(1.0, A))
        first = time.perf_counter() - t0
        steady = []
        for _ in range(args.repeat):
            t0 = time.perf_counter()
            jax.block_until_ready(vg(1.0, A))
            steady.append(time.perf_counter() - t0)
    except Exception as exc:  # noqa: BLE001 -- report the failure mode in-band
        print(f"FAILED {type(exc).__name__}: {str(exc)[:800]}", flush=True)
        print(f"peak device {_peak_device_gb()} GB  peak host RSS {_peak_rss_gb():.1f} GB", flush=True)
        return 1
    print(
        f"OK value={float(v):.12e} grad={float(g):.12e}\n"
        f"compile+first {first:.1f} s  steady {min(steady):.2f} s\n"
        f"peak device {_peak_device_gb()} GB  peak host RSS {_peak_rss_gb():.1f} GB",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
