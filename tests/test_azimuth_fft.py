"""Looped (chirp-z) azimuth mode == the default unrolled path, to machine precision.

The looped mode reroutes the polar-cap ring FFTs through a common-length Bluestein
``lax.scan`` (see :mod:`jht._azimuth`).  It is an exact FFT-algebra identity, so its
output must match the default per-length-FFT path to floating-point roundoff.  These
gates check that identity (synth + adjoint, spin 0/2, incl. the cap-heavy N=4 fold at
the band ceiling and the spin-2 +2/-2 channel asymmetry), that the looped numerics are
independently correct vs healpy AND ducc0, that the transpose identity survives, and
that the mode stays jit / grad / scan-safe.

Uses ``allclose(rtol=0, atol=1e-12)`` rather than ``==``: the two paths reduce the same
sum in a different float op-order (and separately-compiled FFTs differ ~1e-13 ULP on GPU).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jht import get_azimuth_fft_mode, set_azimuth_fft_mode  # noqa: E402
from jht.healpix import adjoint_synthesis, alm_column_base, alm_size, synthesis  # noqa: E402

IDENT_TOL = 1e-12  # a-priori: looped vs unrolled is an exact identity to FFT roundoff
ORACLE_TOL = 1e-10  # a-priori: same bar as the default synthesis gates (test_healpix)


@pytest.fixture(autouse=True)
def _reset_mode():
    """Never let a test's mode flip leak into the next one."""
    yield
    set_azimuth_fft_mode("unrolled")


def _rand_alm(lmax, rng, lmin=0):
    a = (rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))) / np.sqrt(2)
    for ell in range(lmax + 1):
        a[ell] = a[ell].real  # m=0 column is real
    for m in range(min(lmin, lmax + 1)):
        base = alm_column_base(m, lmax)
        a[base : base + (lmin - m)] = 0.0
    return a.astype(np.complex128)


def _alm_weight(lmax):
    w = np.ones(alm_size(lmax))
    for m in range(1, lmax + 1):
        base = alm_column_base(m, lmax)
        w[base : base + (lmax - m + 1)] = 2.0
    return w


def _ducc_synthesis(alm2d, nside, lmax, spin):
    import ducc0

    info = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    return ducc0.sht.experimental.synthesis(
        alm=alm2d, lmax=lmax, spin=spin, mmax=lmax, nthreads=1, **info
    )


def _under(mode, fn):
    set_azimuth_fft_mode(mode)
    out = np.asarray(fn())
    set_azimuth_fft_mode("unrolled")
    return out


# --------------------------------------------------------------------------- #
# (a) looped == unrolled identity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nside,lmax", [(8, 12), (16, 24), (32, 48)])
@pytest.mark.parametrize("spin", [0, 2])
def test_looped_matches_unrolled_synth(nside, lmax, spin):
    rng = np.random.default_rng(spin * 100 + nside)
    if spin == 0:
        alm = jnp.asarray(_rand_alm(lmax, rng))
    else:
        alm = jnp.stack([jnp.asarray(_rand_alm(lmax, rng, 2)), jnp.asarray(_rand_alm(lmax, rng, 2))])
    u = _under("unrolled", lambda: synthesis(alm, nside, lmax, spin))
    lp = _under("looped", lambda: synthesis(alm, nside, lmax, spin))
    assert np.allclose(lp, u, rtol=0.0, atol=IDENT_TOL)


@pytest.mark.parametrize("nside,lmax", [(8, 12), (16, 24), (32, 48)])
@pytest.mark.parametrize("spin", [0, 2])
def test_looped_matches_unrolled_adj(nside, lmax, spin):
    rng = np.random.default_rng(spin * 200 + nside)
    npix = 12 * nside**2
    v = rng.standard_normal(npix) if spin == 0 else rng.standard_normal((2, npix))
    v = jnp.asarray(v)
    u = _under("unrolled", lambda: adjoint_synthesis(v, nside, lmax, spin))
    lp = _under("looped", lambda: adjoint_synthesis(v, nside, lmax, spin))
    assert np.allclose(lp, u, rtol=0.0, atol=IDENT_TOL)


@pytest.mark.parametrize("nside", [16, 32])
@pytest.mark.parametrize("spin", [0, 2])
def test_looped_alias_ceiling(nside, spin):
    """lmax = floor(1.5*nside): maximal m->m mod N aliasing on the short N=4/N=8 cap
    rings, the case that exercises the chirp k^2 mod 2N argument reduction hardest."""
    lmax = (3 * nside) // 2
    rng = np.random.default_rng(7 * nside + spin)
    lmin = 0 if spin == 0 else 2
    if spin == 0:
        alm = jnp.asarray(_rand_alm(lmax, rng, lmin))
    else:
        alm = jnp.stack(
            [jnp.asarray(_rand_alm(lmax, rng, lmin)), jnp.asarray(_rand_alm(lmax, rng, lmin))]
        )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # lmax==1.5*nside is at (not over) the ceiling
        u = _under("unrolled", lambda: synthesis(alm, nside, lmax, spin))
        lp = _under("looped", lambda: synthesis(alm, nside, lmax, spin))
    assert np.allclose(lp, u, rtol=0.0, atol=IDENT_TOL)


def test_looped_spin2_channel_independence():
    """Pure-E and pure-B inputs: the +2/-2 channels (Fp/Fm, asymmetric at m=0) must
    stay independent under the looped path, matching unrolled."""
    nside, lmax = 32, 48
    rng = np.random.default_rng(0)
    aE = _rand_alm(lmax, rng, 2)
    zero = np.zeros_like(aE)
    for pair in ([aE, zero], [zero, aE]):
        alm = jnp.stack([jnp.asarray(pair[0]), jnp.asarray(pair[1])])
        u = _under("unrolled", lambda a=alm: synthesis(a, nside, lmax, 2))
        lp = _under("looped", lambda a=alm: synthesis(a, nside, lmax, 2))
        assert np.allclose(lp, u, rtol=0.0, atol=IDENT_TOL)


# --------------------------------------------------------------------------- #
# (b) looped numerics are independently correct vs healpy AND ducc0
# --------------------------------------------------------------------------- #
def test_looped_two_oracle_spin0():
    import healpy as hp

    nside, lmax = 64, 96
    rng = np.random.default_rng(0)
    alm = _rand_alm(lmax, rng)
    m_jht = _under("looped", lambda: synthesis(alm, nside, lmax, 0))
    m_hp = hp.alm2map(alm, nside, lmax=lmax, mmax=lmax, pol=False)
    m_ducc = _ducc_synthesis(alm[None, :], nside, lmax, 0)[0]
    scale = np.max(np.abs(m_hp))
    assert np.max(np.abs(m_jht - m_hp)) / scale <= ORACLE_TOL
    assert np.max(np.abs(m_jht - m_ducc)) / scale <= ORACLE_TOL


def test_looped_two_oracle_spin2():
    import healpy as hp

    nside, lmax = 64, 96
    rng = np.random.default_rng(1)
    aE, aB = _rand_alm(lmax, rng, 2), _rand_alm(lmax, rng, 2)
    QU = _under("looped", lambda: synthesis(np.stack([aE, aB]), nside, lmax, 2))
    QU_hp = np.asarray(hp.alm2map_spin([aE, aB], nside, 2, lmax, mmax=lmax))
    QU_ducc = np.asarray(_ducc_synthesis(np.stack([aE, aB]), nside, lmax, 2))
    scale = np.max(np.abs(QU_hp))
    assert np.max(np.abs(QU - QU_hp)) / scale <= ORACLE_TOL
    assert np.max(np.abs(QU - QU_ducc)) / scale <= ORACLE_TOL


# --------------------------------------------------------------------------- #
# (c) transpose identity survives in looped mode
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
def test_looped_adjoint_inner_product(spin):
    nside, lmax = 64, 96
    rng = np.random.default_rng(5 + spin)
    npix = 12 * nside**2
    set_azimuth_fft_mode("looped")
    if spin == 0:
        a = _rand_alm(lmax, rng)
        v = rng.standard_normal(npix)
        lhs = float(np.dot(np.asarray(synthesis(a, nside, lmax, 0)), v))
        b = np.asarray(adjoint_synthesis(v, nside, lmax, 0))
        rhs = float(np.sum(_alm_weight(lmax) * np.real(np.conj(a) * b)))
    else:
        aE, aB = _rand_alm(lmax, rng, 2), _rand_alm(lmax, rng, 2)
        Q, U = rng.standard_normal(npix), rng.standard_normal(npix)
        QU = np.asarray(synthesis(np.stack([aE, aB]), nside, lmax, 2))
        b = np.asarray(adjoint_synthesis(np.stack([Q, U]), nside, lmax, 2))
        lhs = float(np.dot(QU[0], Q) + np.dot(QU[1], U))
        w = _alm_weight(lmax)
        rhs = float(np.sum(w * np.real(np.conj(aE) * b[0])) + np.sum(w * np.real(np.conj(aB) * b[1])))
    set_azimuth_fft_mode("unrolled")
    assert abs(lhs - rhs) / abs(lhs) <= ORACLE_TOL


# --------------------------------------------------------------------------- #
# (e) jit / grad / scan safety (the v0.1.4 trace-safety contract)
# --------------------------------------------------------------------------- #
def test_looped_scan_and_grad_safe():
    nside, lmax = 16, 24
    rng = np.random.default_rng(0)
    alms = jnp.stack([jnp.asarray(_rand_alm(lmax, rng)) for _ in range(3)])
    a0 = alms[0]

    def loss(a):  # real scalar
        return jnp.sum(synthesis(a, nside, lmax, 0) ** 2)

    def body(carry, a):
        return carry, synthesis(a, nside, lmax, 0)

    set_azimuth_fft_mode("looped")
    # synthesis inside lax.scan (must not raise UnexpectedTracerError)
    _, maps_l = jax.lax.scan(body, 0.0, alms)
    g_l = jax.grad(loss)(a0)  # grad-of-jit under looped
    set_azimuth_fft_mode("unrolled")
    _, maps_u = jax.lax.scan(body, 0.0, alms)
    g_u = jax.grad(loss)(a0)

    assert np.all(np.isfinite(np.asarray(g_l)))
    assert np.allclose(np.asarray(maps_l), np.asarray(maps_u), rtol=0.0, atol=IDENT_TOL)
    assert np.allclose(np.asarray(g_l), np.asarray(g_u), rtol=0.0, atol=IDENT_TOL)


def test_mode_toggle_validation():
    assert get_azimuth_fft_mode() == "unrolled"
    with pytest.raises(ValueError, match="unknown azimuth FFT mode"):
        set_azimuth_fft_mode("bogus")
    assert get_azimuth_fft_mode() == "unrolled"  # unchanged after a bad set
