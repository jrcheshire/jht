"""Gate H (slow): high-l synthesis correctness above the design ceiling (l > 1000).

The committed accuracy matrix (``test_accuracy.py``) and the operator gates
(``test_healpix.py``) top out at ``lmax = 256`` / ``150``.  The recursion's *fp64
roundoff* is separately measured flat to l=4000 (``scripts/exploratory/
highL_recursion_growth.py`` vs an mpmath reference); this file pins the
*independent-implementation correctness of the full transform* (recursion +
l-contraction + per-ring FFT assembly) at high l against **both** ducc0 and healpy,
spin 0 and 2 -- so a high-l regression can't slip through.

Scope: this is the **forward operator** (``synthesis``), which is quadrature-free,
so it is a clean recursion/assembly check independent of ring weights.  The weighted
*analysis* round-trip at high nside (the quadrature/weight-conditioning tier) is a
separate contract -- see ``docs/accuracy.md`` "Notes / open".

Heavy (nside up to 2048, ~13 GB, multi-second compiles) -> marked ``slow``: runs in
the full suite / nightly, not the fast per-push gate.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import healpy as hp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jht.healpix import alm_column_base, alm_size, synthesis  # noqa: E402

pytestmark = pytest.mark.slow

HIGHL_TOL = 1e-10  # a-priori; same operator tier as test_healpix (measured ~1e-12)


def _ducc_synthesis(alm2d, nside, lmax, spin):
    import ducc0

    info = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    return ducc0.sht.experimental.synthesis(
        alm=alm2d, lmax=lmax, spin=spin, mmax=lmax, nthreads=0, **info
    )


def _rand_alm(lmax, rng, lmin=0):
    """Random healpy-packed alm; m=0 real, l<lmin zeroed (lmin=2 for E/B)."""
    a = (rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))) / np.sqrt(2)
    for ell in range(lmax + 1):
        a[ell] = a[ell].real
    for m in range(min(lmin, lmax + 1)):
        base = alm_column_base(m, lmax)
        a[base : base + (lmin - m)] = 0.0
    return a.astype(np.complex128)


# (nside, lmax): 1500 is 1.46*nside=1024 (inside the ceiling, independently seen to
# agree with ducc to fp64); 2000 = nside (the deep-floor band) at nside=2048.
HIGHL_MATRIX = [(1024, 1500), (2048, 2000)]


@pytest.mark.parametrize("nside,lmax", HIGHL_MATRIX)
def test_highL_synthesis_spin0(nside, lmax):
    """l>1000 spin-0 synthesis matches ducc0 AND healpy to the operator tolerance."""
    rng = np.random.default_rng(0)
    alm = _rand_alm(lmax, rng)
    m_jht = np.asarray(synthesis(alm, nside, lmax, spin=0))
    m_ducc = np.asarray(_ducc_synthesis(alm[None, :], nside, lmax, 0)[0])
    m_hp = hp.alm2map(alm, nside, lmax=lmax, mmax=lmax, pol=False)
    scale = np.max(np.abs(m_ducc))
    assert np.max(np.abs(m_jht - m_ducc)) / scale <= HIGHL_TOL
    assert np.max(np.abs(m_jht - m_hp)) / scale <= HIGHL_TOL


@pytest.mark.parametrize("nside,lmax", HIGHL_MATRIX)
def test_highL_synthesis_spin2(nside, lmax):
    """l>1000 spin-2 (E/B) synthesis matches ducc0 AND healpy -- the s2fft failure
    mode, re-confirmed clean far above where its error appeared."""
    rng = np.random.default_rng(1)
    aE, aB = _rand_alm(lmax, rng, lmin=2), _rand_alm(lmax, rng, lmin=2)
    QU = np.asarray(synthesis(np.stack([aE, aB]), nside, lmax, spin=2))
    QU_ducc = np.asarray(_ducc_synthesis(np.stack([aE, aB]), nside, lmax, 2))
    QU_hp = np.asarray(hp.alm2map_spin([aE, aB], nside, 2, lmax, mmax=lmax))
    scale = np.max(np.abs(QU_ducc))
    assert np.max(np.abs(QU - QU_ducc)) / scale <= HIGHL_TOL
    assert np.max(np.abs(QU - QU_hp)) / scale <= HIGHL_TOL


def test_highL_single_mode_spin2_no_m_lt_l_defect():
    """Explicit per-(l,m) check: a single high-l spin-2 mode (l=1400, m=700, well
    below the l<=1.5*nside ceiling at nside=1024) synthesizes to ducc's map -- the
    s2fft m<l defect was a single-mode error, so pin a single mode directly."""
    nside, lmax = 1024, 1500
    ell, m = 1400, 700
    aE = np.zeros(alm_size(lmax), dtype=np.complex128)
    aB = np.zeros(alm_size(lmax), dtype=np.complex128)
    aE[alm_column_base(m, lmax) + (ell - m)] = 1.0 + 0.5j  # one (l,m) in E
    QU = np.asarray(synthesis(np.stack([aE, aB]), nside, lmax, spin=2))
    QU_ducc = np.asarray(_ducc_synthesis(np.stack([aE, aB]), nside, lmax, 2))
    scale = np.max(np.abs(QU_ducc))
    assert np.max(np.abs(QU - QU_ducc)) / scale <= HIGHL_TOL
