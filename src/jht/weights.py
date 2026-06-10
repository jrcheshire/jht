"""HEALPix ring quadrature weights -- computed on the fly, pure numpy, no files.

HEALPix has no sampling theorem, so the bare quadrature ``a_lm ~ (4pi/Npix) S^T m``
sits at the ~1e-3 floor.  The dominant error is in the **colatitude** quadrature:
the per-ring azimuthal FFT is already exact in ``m``, so what remains is the
``theta`` (ring) integral.  *Ring weights* replace the uniform pixel solid angle
``4pi/Npix`` with a per-ring factor ``(4pi/Npix)(1 + w_i)`` chosen so the ``m=0``
colatitude quadrature is exact for the (even) Legendre polynomials up to a target
degree.  Applied to all ``m`` (the weight is azimuthally symmetric) this is a
heuristic correction that drops the floor ~10x (~1e-3 -> ~1e-4) and -- more
importantly -- makes the Jacobi iteration in :func:`jht.analysis`
converge several orders deeper (it conditions the normal equations).

Convention (matches the HEALPix ``weight_ring`` files, verified empirically):
the stored quantity ``w_i`` is the **deviation from 1**; the effective per-pixel
weight is ``W_i = (4pi/Npix)(1 + w_i)``.

**These are jht's own weights, not a copy of HEALPix's.**  HEALPix ships
precomputed ``weight_ring_n*.fits`` files from an in-house solver whose exact
degree/node target is undocumented and not cleanly reproducible (its achieved
exactness degree has no closed form).  Depending on those files would also break
the jax+numpy-only / no-files contract.  Instead we solve a well-posed,
well-conditioned system here:

    minimize ||w||  s.t.  sum_rings c_i n_i (1 + w_i) P_l(x_i) = Npix * delta_{l0}
    for even l = 0, 2, ..., Lw,   with Lw = 2*nside.

``Lw = 2*nside`` makes the m=0 quadrature exact across the whole usable band
(transforms are capped at ``lmax <= 1.5*nside``), which is what lets the iteration
reach machine precision; pushing ``Lw`` to the fully-determined ``4*nside-2`` is
Vandermonde-ill-conditioned and was rejected.  The minimum-norm (least-squares)
solution regularizes the remaining out-of-band freedom.  Validated **end to end**
(weighted round-trip accuracy matches ``healpy.map2alm(use_weights=True)``; see
``docs/accuracy.md``), not by matching HEALPix's weight array.

Multiplicity ``c_i``: rings come in N/S pairs (``c_i = 2``) except the equatorial
ring (``c_i = 1``), so there are exactly ``2*nside`` distinct weights -- the
northern half + equator, the same half-grid the recursion runs on
(:class:`jht.healpix.RingInfo` ``z[:2*nside]``).

This module is pure numpy and runs once per ``nside`` (cached); it is **not** on
the jit/JAX hot path.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import legvander

from .healpix import RingInfo


@lru_cache(maxsize=None)
def ring_weights(nside: int) -> np.ndarray:
    """Per-ring multiplicative weight deviations ``w_i`` (length ``2*nside``).

    The effective per-pixel weight for a ring is ``(4pi/Npix)(1 + w_i)``.  Indexed
    over the northern half + equator (``RingInfo.z[:2*nside]``); the southern
    rings reuse their N/S partner's weight.
    """
    geo = RingInfo(nside)
    t_half = 2 * nside
    z = geo.z[:t_half]
    npr = geo.npix_ring[:t_half].astype(np.float64)
    c = np.full(t_half, 2.0)
    c[-1] = 1.0  # the equatorial ring is its own N/S reflection

    lw = 2 * nside  # exact across the lmax <= 1.5*nside band; well-conditioned
    P_even = legvander(z, lw)[:, ::2]  # P_0, P_2, ..., P_lw at the ring nodes
    A = ((c * npr)[:, None] * P_even).T  # (n_even, t_half)
    rhs = -A.sum(axis=1)
    rhs[0] += geo.npix  # l=0: sum c_i n_i (1 + w_i) = Npix  ->  sum c_i n_i w_i = 0

    w, *_ = np.linalg.lstsq(A, rhs, rcond=None)  # minimum-norm solution
    w.setflags(write=False)
    return w


@lru_cache(maxsize=None)
def pixel_weights(nside: int, use_weights: bool = True) -> np.ndarray:
    """Per-pixel quadrature weights in RING order (length ``Npix``).

    ``use_weights=False`` returns the uniform ``4pi/Npix`` (the bare estimator);
    ``True`` returns ``(4pi/Npix)(1 + w_ring)`` with each pixel taking its ring's
    weight (south rings fold onto their northern partner via
    ``r -> 4*nside-2-r``).
    """
    geo = RingInfo(nside)
    base = 4.0 * np.pi / geo.npix
    if not use_weights:
        wv = np.full(geo.npix, base)
        wv.setflags(write=False)
        return wv
    w = ring_weights(nside)
    r = np.arange(geo.nrings)
    half = np.where(r < 2 * nside, r, 4 * nside - 2 - r)  # south ring -> north partner
    wv = base * (1.0 + np.repeat(w[half], geo.npix_ring))  # RING-order per-pixel
    wv.setflags(write=False)
    return wv
