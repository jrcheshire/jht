"""Conditioning + achieved exactness of the ring-weight lstsq at high nside.

``docs/accuracy.md`` flags the weight solve at nside->2048 as a documented follow-up
("the gated matrix tops out at 256 ... behavior at nside=2048 is a documented
follow-up").  The solve is ``min ||w|| s.t. sum_i c_i n_i (1+w_i) P_l(z_i) = Npix
delta_{l0}`` for even l up to ``Lw = 2*nside`` via ``np.linalg.lstsq`` -- pure numpy,
off the JAX hot path.  This probe reports, per nside: the Vandermonde condition number,
the lstsq residual, the max weight deviation, and the ACHIEVED m=0 quadrature exactness
(max |Q_l| over even l<=Lw, relative to 4pi) -- the property that must stay < ~1e-9 for
the deep inverse floor.

    pixi run python scripts/exploratory/weight_conditioning.py
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legvander

from jht.healpix import RingInfo
from jht.weights import pixel_weights, ring_weights


def probe(nside: int) -> dict:
    geo = RingInfo(nside)
    t_half = 2 * nside
    z = geo.z[:t_half]
    npr = geo.npix_ring[:t_half].astype(np.float64)
    c = np.full(t_half, 2.0)
    c[-1] = 1.0
    lw = 2 * nside
    P_even = legvander(z, lw)[:, ::2]
    A = ((c * npr)[:, None] * P_even).T  # (n_even, t_half)
    cond = float(np.linalg.cond(A))

    w = ring_weights(nside)
    # achieved exactness: Q_l = sum_pixels W_pix P_l(z), collapsed per-ring
    ring_w = np.add.reduceat(pixel_weights(nside), geo.startpix)
    Q = ring_w @ legvander(geo.z, lw)
    even = np.arange(0, lw + 1, 2)
    exact_err = float(np.max(np.abs(Q[even][1:])) / (4 * np.pi))  # drop l=0 monopole
    mono_err = float(abs(Q[0] - 4 * np.pi) / (4 * np.pi))
    return {
        "nside": nside,
        "Lw": lw,
        "cond": cond,
        "max|w|": float(np.max(np.abs(w))),
        "mono_err": mono_err,
        "exact_err": exact_err,
    }


def main() -> None:
    print(f"{'nside':>6} {'Lw':>6} {'cond(A)':>11} {'max|w|':>9} "
          f"{'mono_err':>10} {'m0_exact_err':>13}")
    print("-" * 60)
    for nside in (256, 512, 1024, 2048, 4096):
        r = probe(nside)
        print(f"{r['nside']:>6} {r['Lw']:>6} {r['cond']:>11.3e} {r['max|w|']:>9.3e} "
              f"{r['mono_err']:>10.2e} {r['exact_err']:>13.2e}", flush=True)
    print("\nm0_exact_err must stay < ~1e-9 (QUAD_TOL) for the deep weighted inverse floor.")


if __name__ == "__main__":
    main()
