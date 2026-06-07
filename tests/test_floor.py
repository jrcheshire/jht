"""Spin2-floor gate (L2, the make-or-break): the spin-2 HEALPix inverse.

Exercises the committed ``jht.analysis.map2alm`` (bare + Jacobi) on the single
modes where s2fft's spin-2 inverse fails (l0=8, m<l0 errors of 28-35%). jht must
instead sit at the HEALPix ~1e-3 floor (no m<l defect) and iterate toward machine
precision on band-limited input.

Small ``nside`` keeps the (Phase-0, un-optimized) reference transforms fast; the
precise floor across l is characterized in ``scripts/exploratory/phase0_floor.py``
and ``docs/findings_phase0.md`` (bare ~2-3e-3 at nside=32, both spins).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from jht.analysis import map2alm  # noqa: E402
from jht.healpix import alm_column_base, alm_size, synthesis  # noqa: E402

NSIDE, LMAX, L0 = 8, 12, 8  # l0=8 is below the l <= 1.5*nside = 18 ceiling

# s2fft's recorded m<l0 error at l0=8 was 28-35%; jht must instead sit at the
# HEALPix floor.  (At this coarse nside the floor is ~3e-3 -> ~100x better;
# finer grids widen the gap further -- see docs/findings_phase0.md.)
S2FFT_DEFECT_L0_8 = 0.28


def _roundtrip(m0, spin, niter, channel=0):
    if spin == 0:
        a = np.zeros(alm_size(LMAX), complex)
        a[alm_column_base(m0, LMAX) + (L0 - m0)] = 1.0
    else:
        a = np.zeros((2, alm_size(LMAX)), complex)
        a[channel, alm_column_base(m0, LMAX) + (L0 - m0)] = 1.0
    rec = np.asarray(
        map2alm(  # use_weights=False: this gate characterizes the bare *unweighted* floor
            synthesis(a, NSIDE, LMAX, spin=spin), NSIDE, LMAX, spin=spin, niter=niter, use_weights=False
        )
    )
    return float(np.max(np.abs(rec - a)))


def test_spin2_inverse_has_no_s2fft_defect():
    """Bare round-trip across m at l0=8 stays at the floor -- no m<l blowup."""
    errs = [_roundtrip(m0, 2, 0) for m0 in (0, 4, 8)]  # m<l0 and the sectoral m=l0
    assert max(errs) < 1e-2  # at the HEALPix floor, not s2fft's 0.28-0.35
    assert max(errs) < 0.1 * S2FFT_DEFECT_L0_8  # cleanly below the s2fft defect band


def test_spin2_iteration_converges_to_machine_precision():
    """Jacobi iteration drives the band-limited mode far below the bare floor."""
    bare = _roundtrip(4, 2, 0)
    iterated = _roundtrip(4, 2, 3)
    assert iterated < bare
    assert iterated < 1e-4  # a-priori gate; actually reaches ~1e-9 here


def test_spin2_sits_at_the_spin0_floor():
    """No spin-specific defect: the spin-2 floor matches the spin-0 floor."""
    assert _roundtrip(4, 2, 0) < 5.0 * _roundtrip(4, 0, 0)
