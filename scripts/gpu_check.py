"""GPU (CUDA) parity + benchmark harness for the jht on-grid transforms.

jht is pure JAX, so the transforms run on a CUDA GPU with no code change. This
script validates and characterizes that on hardware::

    pixi run -e gpu python scripts/gpu_check.py            # on an NVIDIA box (Cannon)
    pixi run -e gpu python scripts/gpu_check.py --max 1024

It (1) reports the visible devices; (2) if a GPU is present, checks **fp64 parity
GPU-vs-CPU** to ~1e-12 across the BK regime; (3) times synthesis / adjoint /
map2alm and a ``vmap``-over-realizations batch on the default device; (4) reports
host peak RSS and (on GPU) device peak bytes.

On a CPU-only machine (e.g. osx-arm64 dev) it still runs -- parity is skipped and
timing is on the CPU -- so it is never dead code. The *measured* GPU numbers are
deferred to the NVIDIA box. See ``docs/gpu.md``.

Parity note: ``jht.healpix._prepare`` is ``lru_cache``-d and closes over
device-resident constants placed at first-call time, so to genuinely run the same
program on each device we clear the caches and rebuild under
``jax.default_device(dev)``.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

import jht  # noqa: E402
from jht import analysis as _analysis  # noqa: E402
from jht import healpix as _healpix  # noqa: E402

# nside -> lmax (BK regime caps lmax ~ 1000; below that ~1.5*nside)
LADDER = [(64, 96), (128, 192), (256, 384), (512, 768), (1024, 1000), (2048, 1000)]


def _clear_caches() -> None:
    """Drop the per-(nside,lmax,spin) prepared kernels + weight vectors so they
    rebuild on whatever device is currently the default."""
    _healpix._prepare.cache_clear()
    _analysis._wvec.cache_clear()


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # darwin: bytes, linux: KB


def gpu_device():
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        return None


def timed(fn, n: int = 3) -> tuple[float, float]:
    """(first-call wall incl. compile, best steady-state run)."""
    t0 = time.perf_counter()
    fn().block_until_ready()
    first = time.perf_counter() - t0
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn().block_until_ready()
        ts.append(time.perf_counter() - t0)
    return first, min(ts)


def _rand_alm(lmax, rng, lmin=0):
    a = (rng.standard_normal(jht.alm_size(lmax)) + 1j * rng.standard_normal(jht.alm_size(lmax))) / np.sqrt(2)
    a[: lmax + 1] = a[: lmax + 1].real  # real a_{l,0}
    a[:lmin] = 0.0  # zero l<|spin| would-be modes (m=0 column head)
    return a.astype(np.complex128)


def _inputs(nside, lmax, spin, rng):
    if spin == 0:
        alm = _rand_alm(lmax, rng)
    else:
        alm = np.stack([_rand_alm(lmax, rng, 2), _rand_alm(lmax, rng, 2)])
    _clear_caches()
    m0 = np.asarray(jht.synthesis(alm, nside, lmax, spin))
    return alm, m0


def _run_synth_on(dev, alm, nside, lmax, spin):
    _clear_caches()
    with jax.default_device(dev):
        return jht.synthesis(jax.device_put(alm, dev), nside, lmax, spin)


def _run_m2a_on(dev, m0, nside, lmax, spin):
    _clear_caches()
    with jax.default_device(dev):
        return jht.map2alm(jax.device_put(m0, dev), nside, lmax, spin, niter=3)


def _relerr(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    denom = max(float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a - b))) / denom


def parity(nside, lmax, spin, cpu, gpu, rng) -> dict:
    alm, m0 = _inputs(nside, lmax, spin, rng)
    syn = _relerr(_run_synth_on(gpu, alm, nside, lmax, spin), _run_synth_on(cpu, alm, nside, lmax, spin))
    m2a = _relerr(_run_m2a_on(gpu, m0, nside, lmax, spin), _run_m2a_on(cpu, m0, nside, lmax, spin))
    return dict(nside=nside, lmax=lmax, spin=spin, synth_relerr=syn, map2alm_relerr=m2a)


def bench(nside, lmax, spin, rng, batch=8) -> dict:
    alm, m0 = _inputs(nside, lmax, spin, rng)
    _clear_caches()
    c_syn, r_syn = timed(lambda: jht.synthesis(alm, nside, lmax, spin))
    _, r_adj = timed(lambda: jht.adjoint_synthesis(m0, nside, lmax, spin))
    _, r_m2a = timed(lambda: jht.map2alm(m0, nside, lmax, spin, niter=3), n=2)
    # vmap over realizations (pre-warm the prepared-kernel cache before vmap)
    batch_alm = jax.device_put(np.stack([alm] * batch))
    vsyn = jax.vmap(lambda a: jht.synthesis(a, nside, lmax, spin))
    _, r_vsyn = timed(lambda: vsyn(batch_alm), n=2)
    return dict(
        nside=nside, lmax=lmax, spin=spin, compile_s=c_syn, synth_ms=r_syn * 1e3,
        adj_ms=r_adj * 1e3, m2a_ms=r_m2a * 1e3, vsynth_ms=r_vsyn * 1e3,
        vsynth_per_ms=r_vsyn * 1e3 / batch,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=1024, help="largest nside to run")
    ap.add_argument("--batch", type=int, default=8, help="vmap batch size")
    args = ap.parse_args()

    devs = jax.devices()
    cpu = jax.devices("cpu")[0]
    gpu = gpu_device()
    print(f"jht GPU check (jax {jax.__version__}, x64={jax.config.jax_enable_x64})")
    print(f"devices: {devs}")
    print(f"default backend: {jax.default_backend()}  |  GPU present: {gpu is not None}\n")

    ladder = [(ns, lm) for ns, lm in LADDER if ns <= args.max]
    rng = np.random.default_rng(0)

    if gpu is not None:
        print("== fp64 parity: GPU vs CPU (gate ~1e-12) ==")
        hdr = f"{'nside':>5} {'lmax':>5} {'spin':>4} {'synth_relerr':>14} {'map2alm_relerr':>15}"
        print(hdr + "\n" + "-" * len(hdr))
        for nside, lmax in ladder:
            for spin in (0, 2):
                r = parity(nside, lmax, spin, cpu, gpu, rng)
                print(f"{r['nside']:>5} {r['lmax']:>5} {r['spin']:>4} "
                      f"{r['synth_relerr']:>14.2e} {r['map2alm_relerr']:>15.2e}", flush=True)
        print()
    else:
        print("== parity skipped (no GPU visible) -- run on the NVIDIA box ==\n")

    print(f"== timing on default device ({jax.default_backend()}) ==")
    hdr = (f"{'nside':>5} {'lmax':>5} {'spin':>4} {'compile_s':>9} {'synth_ms':>9} "
           f"{'adj_ms':>9} {'map2alm_ms':>10} {'vsynth_ms':>9} {'vsynth/ea':>9}")
    print(hdr + "\n" + "-" * len(hdr))
    for nside, lmax in ladder:
        for spin in (0, 2):
            r = bench(nside, lmax, spin, rng, batch=args.batch)
            print(f"{r['nside']:>5} {r['lmax']:>5} {r['spin']:>4} {r['compile_s']:>9.2f} "
                  f"{r['synth_ms']:>9.1f} {r['adj_ms']:>9.1f} {r['m2a_ms']:>10.1f} "
                  f"{r['vsynth_ms']:>9.1f} {r['vsynth_per_ms']:>9.1f}", flush=True)

    print(f"\nhost peak RSS: {peak_rss_mb():.0f} MB")
    if gpu is not None:
        try:
            stats = gpu.memory_stats() or {}
            peak = stats.get("peak_bytes_in_use")
            if peak is not None:
                print(f"GPU peak bytes in use: {peak / 1e9:.2f} GB")
        except Exception:
            pass


if __name__ == "__main__":
    main()
