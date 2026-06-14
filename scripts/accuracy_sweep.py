"""Phase-1 accuracy sweep for the jht HEALPix inverse (map2alm).

Reports, per ``(nside, lmax, spin)``, the max-abs error of a broadband
band-limited round-trip ``a -> synthesis -> map2alm -> a`` against the known
input a_lm, for four estimators -- unweighted/weighted x bare(niter=0)/iterated
(niter=3) -- plus, for spin-0, ``healpy.map2alm(use_weights=True, iter=3)`` as a
cross-check.  Run::

    pixi run python scripts/accuracy_sweep.py            # default ladder
    pixi run python scripts/accuracy_sweep.py --max 512  # cap nside

This is the accuracy analogue of ``bench_transforms.py``; the committed numbers
land in ``docs/accuracy.md``.  Ring weights (jht's own, pure-numpy; see
``jht.weights``) drop the bare floor ~10x and let the Jacobi iteration reach
machine precision.  The a-priori contract is weighted + niter=3 <= 1e-4.
"""

from __future__ import annotations

import argparse

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from jht import map2alm  # noqa: E402  (alias of jht.analysis)
from jht.healpix import alm_size, synthesis  # noqa: E402

LADDER = [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]


def _lvals(lmax: int) -> np.ndarray:
    return np.concatenate([np.arange(m, lmax + 1) for m in range(lmax + 1)])


def _rand_alm(lmax: int, spin: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lv = _lvals(lmax)

    def one() -> np.ndarray:
        a = (rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))).astype(complex)
        a[: lmax + 1] = a[: lmax + 1].real
        a[lv < abs(spin)] = 0.0
        return a

    return one() if spin == 0 else np.stack([one(), one()])


def sweep_one(nside: int, lmax: int, spin: int) -> dict:
    a = _rand_alm(lmax, spin)
    m = synthesis(np.asarray(a), nside, lmax, spin=spin)

    def err(niter: int, use_weights: bool) -> float:
        r = np.asarray(map2alm(m, nside, lmax, spin=spin, niter=niter, use_weights=use_weights))
        return float(np.max(np.abs(r - a)))

    hp_n3 = float("nan")
    if spin == 0:
        import healpy as hp

        hp_n3 = float(np.max(np.abs(hp.map2alm(np.asarray(m), lmax=lmax, iter=3, use_weights=True) - a)))

    return dict(
        nside=nside,
        lmax=lmax,
        spin=spin,
        unw_bare=err(0, False),
        unw_n3=err(3, False),
        w_bare=err(0, True),
        w_n3=err(3, True),
        hp_n3=hp_n3,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=256, help="largest nside to run (from the default ladder)")
    ap.add_argument(
        "--ladder",
        default=None,
        help="override the (nside,lmax) ladder, e.g. '1024:1024,2048:2048' "
        "(high-nside deep-floor / band-ceiling inverse validation)",
    )
    args = ap.parse_args()

    ladder = (
        [(int(n), int(lm)) for n, lm in (p.split(":") for p in args.ladder.split(","))]
        if args.ladder
        else [(n, lm) for n, lm in LADDER if n <= args.max]
    )

    hdr = (
        f"{'nside':>5} {'lmax':>5} {'spin':>4} {'unw_bare':>10} {'unw_n3':>10} "
        f"{'w_bare':>10} {'w_n3':>10} {'healpy_n3':>10}"
    )
    print(f"jht Phase-1 accuracy sweep (CPU, float64, jax {jax.__version__})")
    print("broadband band-limited round-trip, max-abs a_lm error vs ground truth\n")
    print(hdr)
    print("-" * len(hdr))
    for nside, lmax in ladder:
        for spin in (0, 2):
            r = sweep_one(nside, lmax, spin)
            hp_col = "       n/a" if np.isnan(r["hp_n3"]) else f"{r['hp_n3']:10.1e}"
            print(
                f"{r['nside']:>5} {r['lmax']:>5} {r['spin']:>4} {r['unw_bare']:>10.1e} "
                f"{r['unw_n3']:>10.1e} {r['w_bare']:>10.1e} {r['w_n3']:>10.1e} {hp_col}",
                flush=True,
            )


if __name__ == "__main__":
    main()
