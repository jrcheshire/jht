"""Phase-1 performance benchmark for the jht on-grid transforms (CPU).

Reports, per ``(nside, lmax, spin)``: the **first-call** wall time (trace +
XLA compile + run, cached thereafter), the **steady-state** run time (best of a
few jitted calls), and the process peak RSS.  Run::

    pixi run python scripts/bench_transforms.py            # default ladder
    pixi run python scripts/bench_transforms.py --max 512  # cap nside

GPU timing is deferred to Phase 3 (JAX-metal float64 is unreliable); this is the
CPU characterization that defines "done" for the performance phase.  See
``docs/performance.md`` for the committed table and the architecture/memory model.
"""

from __future__ import annotations

import argparse
import resource
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from jht import map2alm  # noqa: E402  (alias of jht.analysis)
from jht.healpix import adjoint_synthesis, alm_size, synthesis  # noqa: E402

# nside -> lmax: the BK regime caps lmax ~ 1000; below that use ~1.5*nside.
LADDER = [
    (32, 48),
    (128, 192),
    (256, 384),
    (512, 768),
    (1024, 1000),
    (2048, 1000),
]


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3  # darwin: bytes, linux: KB


def timed(fn, n: int = 3) -> tuple[float, float]:
    """Return (first-call wall incl. compile, best steady-state run)."""
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
    a = (rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))) / np.sqrt(
        2
    )
    a[: lmax + 1] = a[: lmax + 1].real
    a[:lmin] = 0.0
    return a.astype(np.complex128)


def bench_one(nside, lmax, spin, do_map2alm=True):
    rng = np.random.default_rng(0)
    npix = 12 * nside**2
    if spin == 0:
        alm = _rand_alm(lmax, rng)
        m0 = np.asarray(synthesis(alm, nside, lmax, 0))
    else:
        alm = np.stack([_rand_alm(lmax, rng, 2), _rand_alm(lmax, rng, 2)])
        m0 = np.asarray(synthesis(alm, nside, lmax, 2))

    c_syn, r_syn = timed(lambda: synthesis(alm, nside, lmax, spin))
    c_adj, r_adj = timed(lambda: adjoint_synthesis(m0, nside, lmax, spin))
    if do_map2alm:
        _, r_m2a = timed(lambda: map2alm(m0, nside, lmax, spin, niter=3), n=2)
    else:
        r_m2a = float("nan")
    return dict(
        nside=nside,
        lmax=lmax,
        npix=npix,
        spin=spin,
        compile_s=max(c_syn, c_adj),
        synth_ms=r_syn * 1e3,
        adj_ms=r_adj * 1e3,
        m2a_ms=r_m2a * 1e3,
        peak_mb=peak_rss_mb(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=2048, help="largest nside to run")
    ap.add_argument("--no-map2alm-above", type=int, default=1024, help="skip map2alm above this nside")
    args = ap.parse_args()

    hdr = (
        f"{'nside':>5} {'lmax':>5} {'npix':>10} {'spin':>4} {'compile_s':>9} "
        f"{'synth_ms':>9} {'adj_ms':>9} {'map2alm_ms':>10} {'peakRSS_MB':>10}"
    )
    print(f"jht Phase-1 transform benchmark (CPU, float64, jax {jax.__version__})\n")
    print(hdr)
    print("-" * len(hdr))
    for nside, lmax in LADDER:
        if nside > args.max:
            break
        for spin in (0, 2):
            r = bench_one(nside, lmax, spin, do_map2alm=nside <= args.no_map2alm_above)
            m2a = "       n/a" if np.isnan(r["m2a_ms"]) else f"{r['m2a_ms']:10.1f}"
            print(
                f"{r['nside']:>5} {r['lmax']:>5} {r['npix']:>10} {r['spin']:>4} "
                f"{r['compile_s']:>9.2f} {r['synth_ms']:>9.1f} {r['adj_ms']:>9.1f} "
                f"{m2a} {r['peak_mb']:>10.0f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
