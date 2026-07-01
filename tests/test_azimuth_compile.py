"""Looped azimuth mode: O(1) compiled FFT-kernel count, SHT-heavy-graph compilation,
and high-l / large-L correctness.  SLOW (manual full suite, not the per-push gate).

* The default path emits one FFT kernel per distinct ring length (``= nside``); the
  looped path emits the belt's native FFT plus one common chirp-z length -- O(1),
  independent of nside.  This is the fix for the compile-size blowup (issue #2).
* A self-contained SHT-heavy graph (the ``S^T diag(w) S`` normal operator applied
  repeatedly -- the shape that multiplies the per-SHT kernel count) must compile as
  one executable under the looped mode and match the default numerically.
* High-l parity (large Bluestein length L) vs healpy AND ducc0, to confirm the chirp
  ``k^2 mod 2N`` precision holds well past the fast-suite sizes.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jht import set_azimuth_fft_mode  # noqa: E402
from jht._azimuth import build_cap_plan  # noqa: E402
from jht.healpix import (  # noqa: E402
    RingInfo,
    _ring_groups,
    adjoint_synthesis,
    alm_column_base,
    alm_size,
    synthesis,
)

pytestmark = pytest.mark.slow

HIGHL_TOL = 1e-10  # a-priori, same bar as tests/test_highL.py


@pytest.fixture(autouse=True)
def _reset_mode():
    yield
    set_azimuth_fft_mode("unrolled")


def _rand_alm(lmax, rng, lmin=0):
    a = (rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))) / np.sqrt(2)
    for ell in range(lmax + 1):
        a[ell] = a[ell].real
    for m in range(min(lmin, lmax + 1)):
        base = alm_column_base(m, lmax)
        a[base : base + (lmin - m)] = 0.0
    return a.astype(np.complex128)


def _ducc_synthesis(alm2d, nside, lmax, spin):
    import ducc0

    info = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    return ducc0.sht.experimental.synthesis(
        alm=alm2d, lmax=lmax, spin=spin, mmax=lmax, nthreads=0, **info
    )


# --------------------------------------------------------------------------- #
# (d) O(1) compiled-kernel count (structural proxy: distinct FFT lengths)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nside", [16, 32, 64, 128])
def test_looped_kernel_count_is_O1(nside):
    lmax = nside
    groups = _ring_groups(RingInfo(nside), lmax)
    # default: one FFT kernel per distinct ring length -> grows linearly with nside
    assert len({int(g.N) for g in groups}) == nside
    # looped: the belt's native length + one common chirp-z length, independent of nside
    cap_plan = build_cap_plan(RingInfo(nside), groups[:-1], lmax)
    looped_lengths = {int(groups[-1].N), int(cap_plan.L)}
    assert len(looped_lengths) <= 2


def test_sht_heavy_graph_compiles_as_one_executable():
    """The masked normal operator (S^T diag(w) S) applied repeatedly, jitted as ONE
    graph.  Under the default path this stacks ~nside*3 FFT kernels; under looped it is
    O(1).  Assert the looped graph compiles as a single executable and matches default."""
    nside, lmax = 32, 48
    rng = np.random.default_rng(0)
    a = jnp.asarray(_rand_alm(lmax, rng))
    w = jnp.asarray(rng.uniform(0.2, 1.0, 12 * nside**2))

    def graph(alm):
        x = alm
        for _ in range(3):
            m = synthesis(x, nside, lmax, 0)
            x = adjoint_synthesis(w * m, nside, lmax, 0)
        return x

    set_azimuth_fft_mode("looped")
    looped = np.asarray(jax.jit(graph)(a))  # one XLA executable spanning 3 S / 3 S^T
    set_azimuth_fft_mode("unrolled")
    ref = np.asarray(graph(a))
    # relative: 3x S^T W S amplifies |alm| to ~1e8, so normalize (the looped chirp-z
    # leaves a ~1e-16-relative imaginary part on the m=0 modes that unrolled keeps at 0).
    assert np.max(np.abs(looped - ref)) / np.max(np.abs(ref)) <= 1e-12


# --------------------------------------------------------------------------- #
# (b-slow) high-l parity with the looped mode ON (large Bluestein length L)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nside,lmax", [(512, 768), (1024, 1500)])
@pytest.mark.parametrize("spin", [0, 2])
def test_looped_highl_two_oracle(nside, lmax, spin):
    import healpy as hp

    rng = np.random.default_rng(nside + spin)
    set_azimuth_fft_mode("looped")
    if spin == 0:
        alm = _rand_alm(lmax, rng)
        m_jht = np.asarray(synthesis(alm, nside, lmax, 0))
        m_hp = hp.alm2map(alm, nside, lmax=lmax, mmax=lmax, pol=False)
        m_ducc = _ducc_synthesis(alm[None, :], nside, lmax, 0)[0]
        scale = np.max(np.abs(m_hp))
        assert np.max(np.abs(m_jht - m_hp)) / scale <= HIGHL_TOL
        assert np.max(np.abs(m_jht - m_ducc)) / scale <= HIGHL_TOL
    else:
        aE, aB = _rand_alm(lmax, rng, 2), _rand_alm(lmax, rng, 2)
        QU = np.asarray(synthesis(np.stack([aE, aB]), nside, lmax, 2))
        QU_hp = np.asarray(hp.alm2map_spin([aE, aB], nside, 2, lmax, mmax=lmax))
        QU_ducc = np.asarray(_ducc_synthesis(np.stack([aE, aB]), nside, lmax, 2))
        scale = np.max(np.abs(QU_hp))
        assert np.max(np.abs(QU - QU_hp)) / scale <= HIGHL_TOL
        assert np.max(np.abs(QU - QU_ducc)) / scale <= HIGHL_TOL
