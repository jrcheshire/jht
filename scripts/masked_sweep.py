"""Partial-sky / masked-analysis characterisation sweep for jht.

For a ladder of polar-cap masks (shrinking ``fsky``) reports, per
``(nside, lmax, spin, mask)``:

* ``pseudo`` -- the masked pseudo-a_lm bias ``max|pseudo_alm(M m) - a_true|``
  (weighted, niter=3): the mode-coupling bias the cut imprints; it does *not*
  shrink with iteration (the cut is real).
* ``deconv`` -- the cut-sky CG deconvolution error ``max|deconvolve(M m) - a_true|``:
  it recovers the truth (-> machine precision) wherever the cut leaves the modes
  constrained, and *degrades* once the active band develops near-null modes
  (aggressive cuts / heavy apodization).

This is the masked analogue of ``accuracy_sweep.py``; the table lands in
``docs/masked.md``.  Recovery is a property of the cut (information loss), not of
the quadrature -- so it is characterised here, not gated tight (the tight,
oracle-backed gates live in ``tests/test_masked.py``).  Run::

    pixi run python scripts/masked_sweep.py
    pixi run python scripts/masked_sweep.py --nside 64 --lmax 48
"""

from __future__ import annotations

import argparse

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from jht.healpix import alm_size, synthesis  # noqa: E402
from jht.masked import deconvolve, pseudo_alm  # noqa: E402

# (label, th_cut, apod_width); width=0 -> binary cap
MASKS = [
    ("binary t<0.2", 0.2, 0.0),
    ("binary t<0.4", 0.4, 0.0),
    ("binary t<0.6", 0.6, 0.0),
    ("binary t<0.8", 0.8, 0.0),
    ("apod  t<0.4", 0.4, 0.4),
]


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


def _mask(nside: int, th_cut: float, width: float) -> np.ndarray:
    import healpy as hp

    th = hp.pix2ang(nside, np.arange(12 * nside**2))[0]
    if width == 0.0:
        m = np.ones(12 * nside**2)
        m[th < th_cut] = 0.0
        return m
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.clip((th - th_cut) / width, 0.0, 1.0)))
    return np.where(th < th_cut, 0.0, ramp)


def sweep_one(nside: int, lmax: int, spin: int, th_cut: float, width: float, max_iter: int) -> dict:
    mask = _mask(nside, th_cut, width)
    a = _rand_alm(lmax, spin, seed=spin)
    m = np.asarray(synthesis(np.asarray(a), nside, lmax, spin=spin))
    mm = m * (mask if spin == 0 else mask[None, :])
    ps = np.asarray(pseudo_alm(m, mask, nside, lmax, spin=spin, niter=3))
    dec = np.asarray(deconvolve(mm, mask, nside, lmax, spin=spin, max_iter=max_iter, tol=1e-11))
    return dict(
        fsky=float(np.mean(mask > 0)),
        pseudo=float(np.max(np.abs(ps - a))),
        deconv=float(np.max(np.abs(dec - a))),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=32)
    ap.add_argument("--lmax", type=int, default=24)
    ap.add_argument("--max-iter", type=int, default=1500)
    args = ap.parse_args()

    print(f"jht masked-analysis sweep (CPU, float64, jax {jax.__version__})")
    print(f"nside={args.nside} lmax={args.lmax}; max-abs a_lm error vs ground truth\n")
    hdr = f"{'spin':>4} {'mask':>13} {'fsky':>6} {'pseudo':>10} {'deconv':>10}"
    print(hdr)
    print("-" * len(hdr))
    for spin in (0, 2):
        for label, th_cut, width in MASKS:
            r = sweep_one(args.nside, args.lmax, spin, th_cut, width, args.max_iter)
            print(
                f"{spin:>4} {label:>13} {r['fsky']:>6.3f} {r['pseudo']:>10.1e} {r['deconv']:>10.1e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
