"""Gate W (M1): HEALPix ring quadrature weights.

The weights are validated by their **defining math property** -- the m=0
colatitude quadrature is exact for the even Legendre polynomials up to
``Lw = 2*nside`` -- not by matching HEALPix's shipped ``weight_ring_n*.fits``
array (a different, undocumented solver; see ``jht.weights`` and
``docs/accuracy.md``).  The end-to-end accuracy this buys is gated separately in
``tests/test_accuracy.py``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from numpy.polynomial.legendre import legvander

from jht.healpix import RingInfo
from jht.weights import pixel_weights, ring_weights

NSIDES = (2, 4, 8, 16, 32, 64, 128, 256, 512)
QUAD_TOL = 1e-9  # a-priori: exactness of the m=0 quadrature, relative to 4*pi


def _quadrature_moments(nside: int) -> np.ndarray:
    """``Q_l = sum_pixels W_pix P_l(z_pix)`` for l = 0..2*nside (per-pixel weights).

    Collapsed to per-ring: every pixel in a ring shares the same ``z``, so the
    per-pixel sum factors exactly as ``sum_ring (sum_{pix in ring} W_pix)
    P_l(z_ring)``.  Identical moments, but the Vandermonde is ``(nrings,
    2*nside+1)`` rather than ``(npix, 2*nside+1)`` -- ~16 MB vs 24 GiB at
    nside=512, where the per-pixel form OOMs CI runners.
    """
    geo = RingInfo(nside)
    ring_w = np.add.reduceat(pixel_weights(nside), geo.startpix)  # exact per-ring weight sum
    return ring_w @ legvander(geo.z, 2 * nside)


def test_ring_weights_shape_and_bounded():
    for nside in NSIDES:
        w = ring_weights(nside)
        assert w.shape == (2 * nside,)
        assert np.all(np.isfinite(w))
        assert np.max(np.abs(w)) < 0.5  # small multiplicative correction to 1


def test_m0_quadrature_exact_to_2nside():
    """The defining property: m=0 quadrature is exact for even l up to 2*nside."""
    for nside in NSIDES:
        q = _quadrature_moments(nside)
        assert abs(q[0] - 4.0 * np.pi) < QUAD_TOL * 4.0 * np.pi  # monopole normalization
        even = np.arange(2, 2 * nside + 1, 2)
        assert np.max(np.abs(q[even])) < QUAD_TOL * 4.0 * np.pi  # higher even moments vanish


@pytest.mark.slow
@pytest.mark.parametrize("nside", [1024, 2048, 4096])
def test_m0_quadrature_exact_high_nside(nside):
    """Weight solve stays well-conditioned (cond ~ 2*nside) and m=0-exact (~1e-16)
    to nside=4096 -- resolves the docs/accuracy.md 'behavior at nside=2048 is a
    documented follow-up' note (measured: scripts/exploratory/weight_conditioning.py).
    So the lmax<=nside quadrature exactness (the deep inverse floor) holds at scale."""
    q = _quadrature_moments(nside)
    assert abs(q[0] - 4.0 * np.pi) < QUAD_TOL * 4.0 * np.pi
    even = np.arange(2, 2 * nside + 1, 2)
    assert np.max(np.abs(q[even])) < QUAD_TOL * 4.0 * np.pi


def test_pixel_weights_sum_to_4pi():
    """Total weighted solid angle is 4*pi (weighted and unweighted alike)."""
    for nside in (4, 16, 64, 256):
        assert abs(pixel_weights(nside, use_weights=True).sum() - 4.0 * np.pi) < 1e-11
        wv0 = pixel_weights(nside, use_weights=False)
        assert np.allclose(wv0, 4.0 * np.pi / (12 * nside**2))
        assert abs(wv0.sum() - 4.0 * np.pi) < 1e-12


def test_pixel_weights_fold_north_south():
    """A southern ring's pixels carry the same weight as its N/S partner."""
    nside = 32
    geo = RingInfo(nside)
    wv = pixel_weights(nside)
    for r_north in (0, 5, 2 * nside - 2):  # a polar-cap and a belt ring
        r_south = 4 * nside - 2 - r_north
        wn = wv[geo.startpix[r_north]]
        ws = wv[geo.startpix[r_south]]
        assert abs(wn - ws) < 1e-15


def test_healpy_ring_weights_are_spin_independent():
    """T == Q == U in HEALPix's files -> jht reuses one weight array for both spins."""
    import healpy as hp

    datapath = os.path.join(os.path.dirname(hp.__file__), "data")
    for nside in (8, 64):
        cols = np.asarray(hp.fitsfunc.read_cl(os.path.join(datapath, f"weight_ring_n{nside:05d}.fits")))
        assert cols.shape[0] == 3
        assert np.allclose(cols[0], cols[1]) and np.allclose(cols[0], cols[2])
