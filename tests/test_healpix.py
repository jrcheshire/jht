"""Gate S-synth and Gate adj (L1): HEALPix synthesis + exact adjoint.

* geometry vs ``healpy.pix2ang``;
* ``synthesis`` vs ``healpy`` AND ``ducc0`` (spin-0 and spin-2), <= 1e-10;
* ``adjoint_synthesis`` == ``(Npix/4pi) map2alm`` and the inner-product identity
  ``<S a, v>_map == <a, S^T v>_alm`` (with the ``2 - delta_{m0}`` weight), <= 1e-10.

All comparisons use band-limited input with ``lmax <= 1.5 nside`` (below the
HEALPix reliability ceiling) so the operator pair is tested, not the quadrature.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import healpy as hp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jht.healpix import (  # noqa: E402
    RingInfo,
    adjoint_synthesis,
    alm_column_base,
    alm_size,
    synthesis,
)

S_SYNTH_TOL = 1e-10  # a-priori (ROADMAP Phase-0)
ADJ_TOL = 1e-10


def _rand_alm(lmax, rng, lmin=0):
    """Random healpy-packed alm; m=0 real, l<lmin zeroed (lmin=2 for E/B)."""
    a = (rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))) / np.sqrt(
        2
    )
    for ell in range(lmax + 1):
        a[ell] = a[ell].real  # m=0 column is real
    for m in range(min(lmin, lmax + 1)):
        base = alm_column_base(m, lmax)
        a[base : base + (lmin - m)] = 0.0
    return a.astype(np.complex128)


def _ducc_synthesis(alm2d, nside, lmax, spin):
    import ducc0

    info = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    return ducc0.sht.experimental.synthesis(
        alm=alm2d, lmax=lmax, spin=spin, mmax=lmax, nthreads=1, **info
    )


def _alm_weight(lmax):
    w = np.ones(alm_size(lmax))
    for m in range(1, lmax + 1):
        base = alm_column_base(m, lmax)
        w[base : base + (lmax - m + 1)] = 2.0
    return w


# --------------------------------------------------------------------------- #
def test_geometry_vs_healpy():
    for nside in (2, 4, 8, 16, 32):
        geo = RingInfo(nside)
        th, ph = hp.pix2ang(nside, np.arange(geo.npix))
        assert geo.nrings == 4 * nside - 1
        for r in range(geo.nrings):
            s = geo.startpix[r]
            assert np.isclose(np.cos(th[s]), geo.z[r])
            assert np.isclose(ph[s], geo.phi0[r])  # first pixel azimuth
            assert (th[s : s + geo.npix_ring[r]] == th[s]).all()  # one colatitude


@pytest.mark.parametrize("nside,lmax", [(64, 96), (128, 150)])
def test_gate_s_synth_spin0(nside, lmax):
    rng = np.random.default_rng(0)
    alm = _rand_alm(lmax, rng)
    m_jht = np.asarray(synthesis(alm, nside, lmax, spin=0))
    m_hp = hp.alm2map(alm, nside, lmax=lmax, mmax=lmax, pol=False)
    m_ducc = _ducc_synthesis(alm[None, :], nside, lmax, 0)[0]
    scale = np.max(np.abs(m_hp))
    assert np.max(np.abs(m_jht - m_hp)) / scale <= S_SYNTH_TOL
    assert np.max(np.abs(m_jht - m_ducc)) / scale <= S_SYNTH_TOL


@pytest.mark.parametrize("nside,lmax", [(64, 96), (128, 150)])
def test_gate_s_synth_spin2(nside, lmax):
    rng = np.random.default_rng(1)
    aE, aB = _rand_alm(lmax, rng, lmin=2), _rand_alm(lmax, rng, lmin=2)
    QU = np.asarray(synthesis(np.stack([aE, aB]), nside, lmax, spin=2))
    QU_hp = np.asarray(hp.alm2map_spin([aE, aB], nside, 2, lmax, mmax=lmax))
    QU_ducc = np.asarray(_ducc_synthesis(np.stack([aE, aB]), nside, lmax, 2))
    scale = np.max(np.abs(QU_hp))
    assert np.max(np.abs(QU - QU_hp)) / scale <= S_SYNTH_TOL  # (Q,U) parity
    assert np.max(np.abs(QU - QU_ducc)) / scale <= S_SYNTH_TOL


@pytest.mark.parametrize("nside,lmax", [(64, 96)])
def test_gate_adj_spin0(nside, lmax):
    rng = np.random.default_rng(2)
    npix = 12 * nside**2
    v = rng.standard_normal(npix)
    b = np.asarray(adjoint_synthesis(v, nside, lmax, spin=0))
    b_id = hp.map2alm(v, lmax=lmax, mmax=lmax, iter=0, use_weights=False) * (npix / (4 * np.pi))
    assert np.max(np.abs(b - b_id)) / np.max(np.abs(b_id)) <= ADJ_TOL
    # inner-product identity <S a, v> = <a, S^T v>_alm
    a = _rand_alm(lmax, rng)
    lhs = np.dot(np.asarray(synthesis(a, nside, lmax)), v)
    rhs = np.sum(_alm_weight(lmax) * np.real(np.conj(a) * b))
    assert abs(lhs - rhs) / abs(lhs) <= ADJ_TOL


@pytest.mark.parametrize("nside,lmax", [(64, 96)])
def test_gate_adj_spin2(nside, lmax):
    rng = np.random.default_rng(3)
    npix = 12 * nside**2
    Q, U = rng.standard_normal(npix), rng.standard_normal(npix)
    b = np.asarray(adjoint_synthesis(np.stack([Q, U]), nside, lmax, spin=2))
    bE, bB = hp.map2alm_spin([Q, U], 2, lmax=lmax, mmax=lmax)
    fac = npix / (4 * np.pi)
    assert np.max(np.abs(b[0] - bE * fac)) / np.max(np.abs(bE * fac)) <= ADJ_TOL
    assert np.max(np.abs(b[1] - bB * fac)) / np.max(np.abs(bB * fac)) <= ADJ_TOL
    # inner-product identity (both polarization channels)
    aE, aB = _rand_alm(lmax, rng, lmin=2), _rand_alm(lmax, rng, lmin=2)
    QU = np.asarray(synthesis(np.stack([aE, aB]), nside, lmax, spin=2))
    lhs = np.dot(QU[0], Q) + np.dot(QU[1], U)
    w = _alm_weight(lmax)
    rhs = np.sum(w * np.real(np.conj(aE) * b[0])) + np.sum(w * np.real(np.conj(aB) * b[1]))
    assert abs(lhs - rhs) / abs(lhs) <= ADJ_TOL
