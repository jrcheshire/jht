"""Wiener-solve preconditioning characterisation sweep for jht.

For a ladder of irregular Galactic-like masks (a two-sided band cut ``|b| < b_cut`` plus
random holes -- deliberately not azimuthally symmetric), reports the CG iterations to a
relative residual of ``--tol`` for the spin-2 :func:`jht.wiener` solve with

* no preconditioner (stock CG),
* the diagonal ``1 + Cl * nbar`` alone (``ell_split=None``),
* the diagonal + the exact low-l block (``ell_split`` = each of ``--ell-splits``),

plus each preconditioner's build time and the largest relative difference of its
solution from the unpreconditioned one (tol-limited: up to ~cond * tol, since both stop
at the same relative residual).  The prior is a red spectrum with a
beam-like damping, and the white noise puts S/N >> 1 at low l and ~1 near lmax.

Iteration counts are a property of the mask, prior and noise, so they are
characterised here, not gated (the deterministic gates live in
``tests/test_wiener.py``); the table lands in ``docs/masked.md``.  Run::

    pixi run python scripts/wiener_precond_sweep.py
    pixi run python scripts/wiener_precond_sweep.py --nside 32 --lmax 48 --ell-splits 8 16
"""

from __future__ import annotations

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import healpy as hp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jht.healpix import synthesis  # noqa: E402
from jht.masked import wiener, wiener_preconditioner  # noqa: E402


def _mask(nside: int, b_cut_deg: float, n_holes: int, seed: int) -> np.ndarray:
    """``|b| < b_cut`` band removed (b = 90 deg - theta), plus ``n_holes`` random discs of 3-8 deg."""
    npix = 12 * nside**2
    th = hp.pix2ang(nside, np.arange(npix))[0]
    m = (np.abs(th - np.pi / 2) > np.radians(b_cut_deg)).astype(float)
    rng = np.random.default_rng(seed)
    for c in rng.integers(0, npix, n_holes):
        m[hp.query_disc(nside, hp.pix2vec(nside, int(c)), np.radians(rng.uniform(3, 8)))] = 0.0
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--lmax", type=int, default=96)
    p.add_argument("--b-cuts", type=float, nargs="+", default=[10.0, 20.0, 30.0])
    p.add_argument("--holes", type=int, default=12)
    p.add_argument("--ell-splits", type=int, nargs="+", default=[16, 32])
    p.add_argument("--noise", type=float, default=1.0, help="white per-pixel noise rms")
    p.add_argument("--tol", type=float, default=1e-8)
    p.add_argument("--max-iter", type=int, default=3000)
    args = p.parse_args()

    nside, lmax, spin = args.nside, args.lmax, 2
    npix = 12 * nside**2
    ell = np.arange(lmax + 1)
    cl_ee = np.where(ell >= 2, 1e2 / (ell + 1.0) ** 2 * np.exp(-((ell / (0.6 * lmax)) ** 2)), 0.0)
    cl_bb = 0.05 * cl_ee
    cl = (jnp.asarray(cl_ee), jnp.asarray(cl_bb))
    np.random.seed(0)
    alm = np.stack([hp.synalm(cl_ee, lmax=lmax, new=True), hp.synalm(cl_bb, lmax=lmax, new=True)])
    sky = np.asarray(synthesis(jnp.asarray(alm), nside, lmax, spin))
    noise = args.noise * np.random.default_rng(1).standard_normal((2, npix))

    print(f"nside={nside} lmax={lmax} spin=2 tol={args.tol:g} noise rms={args.noise}")
    cols = ["none", "diagonal"] + [f"diag+l<={ls}" for ls in args.ell_splits]
    print(f"{'mask':>22} {'f_sky':>6} " + " ".join(f"{c:>12}" for c in cols)
          + "   build time (s)       max rel. diff vs none")
    for k, b_cut in enumerate(args.b_cuts):
        mask = _mask(nside, b_cut, args.holes, seed=10 + k)
        ninv = mask / args.noise**2
        m = sky + noise

        def solve(pc):
            a, info = wiener(m, cl, nside, lmax, spin=spin, inv_noise=ninv, precond=pc,
                             return_info=True, max_iter=args.max_iter, tol=args.tol)
            return np.asarray(a), int(info.n_iter), float(info.rel_residual)

        a0, n0, r0 = solve(None)
        iters, builds, diffs = [f"{n0}" + ("*" if r0 > args.tol else "")], [], []
        for ls in [None, *args.ell_splits]:
            t0 = time.perf_counter()
            pc = wiener_preconditioner(cl, nside, lmax, spin, inv_noise=ninv, ell_split=ls)
            jax.block_until_ready(pc)
            builds.append(time.perf_counter() - t0)
            a, n, r = solve(pc)
            iters.append(f"{n}" + ("*" if r > args.tol else ""))
            diffs.append(np.max(np.abs(a - a0)) / np.max(np.abs(a0)))
        label = f"|b|<{b_cut:g} + {args.holes} holes"
        print(f"{label:>22} {mask.mean():6.3f} " + " ".join(f"{s:>12}" for s in iters)
              + "   " + " ".join(f"{t:5.1f}" for t in builds)
              + "   " + " ".join(f"{d:.0e}" for d in diffs))
    print("(* = stopped at --max-iter unconverged)")


if __name__ == "__main__":
    main()
