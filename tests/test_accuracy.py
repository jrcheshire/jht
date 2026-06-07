"""Gate A (M3): broadband weighted accuracy -- the Phase-1 accuracy contract.

A random *broadband* band-limited a_lm (a real field; no ``l < |spin|`` modes) is
synthesized to a map and recovered with :func:`jht.analysis.map2alm`.  The
committed a-priori contract is

    weighted + niter=3 round-trip error <= 1e-4

across the nside x lmax x spin matrix (measured ~1e-13, ~9 orders of headroom;
see ``docs/accuracy.md``).  Also gated: ring weights beat the unweighted bare
estimator, spin-2 reaches the same deep floor as spin-0 (no s2fft-style defect),
and jht's weighted recovery is no worse than ``healpy.map2alm(use_weights=True)``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jht.analysis import map2alm  # noqa: E402
from jht.healpix import alm_size, synthesis  # noqa: E402

GATE_A = 1e-4  # a-priori committed contract (weighted + niter=3)
DEEP_FLOOR = 1e-9  # weighted+iter actually converges far below the contract
MATRIX = [(32, 32), (64, 64), (128, 128), (256, 256)]


def _lvals(lmax: int) -> np.ndarray:
    return np.concatenate([np.arange(m, lmax + 1) for m in range(lmax + 1)])


def _rand_alm(lmax: int, spin: int, seed: int = 0) -> np.ndarray:
    """Random broadband a_lm for a real spin-``spin`` field (m=0 real, no l<|spin|)."""
    rng = np.random.default_rng(seed)
    lv = _lvals(lmax)

    def one() -> np.ndarray:
        a = (rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))).astype(complex)
        a[: lmax + 1] = a[: lmax + 1].real  # a_{l,0} real for a real field
        a[lv < abs(spin)] = 0.0  # spin-s fields have no l < |s| harmonics
        return a

    return one() if spin == 0 else np.stack([one(), one()])


def _roundtrip_err(nside, lmax, spin, niter, use_weights, seed=0) -> float:
    a = _rand_alm(lmax, spin, seed)
    m = synthesis(np.asarray(a), nside, lmax, spin=spin)
    r = np.asarray(map2alm(m, nside, lmax, spin=spin, niter=niter, use_weights=use_weights))
    return float(np.max(np.abs(r - a)))


@pytest.mark.parametrize("nside,lmax", MATRIX)
@pytest.mark.parametrize("spin", [0, 2])
def test_weighted_iterated_meets_contract(nside, lmax, spin):
    """The committed gate: weighted + niter=3 broadband round-trip <= 1e-4."""
    assert _roundtrip_err(nside, lmax, spin, niter=3, use_weights=True) <= GATE_A


@pytest.mark.parametrize("nside,lmax", [(32, 32), (64, 64)])
@pytest.mark.parametrize("spin", [0, 2])
def test_weights_beat_unweighted_bare(nside, lmax, spin):
    """Ring weights strictly improve the bare (niter=0) estimator."""
    weighted = _roundtrip_err(nside, lmax, spin, niter=0, use_weights=True)
    unweighted = _roundtrip_err(nside, lmax, spin, niter=0, use_weights=False)
    assert weighted < unweighted


@pytest.mark.parametrize("nside,lmax", [(64, 64), (128, 128)])
def test_spin2_reaches_spin0_floor(nside, lmax):
    """No spin-2 defect: both spins iterate to the same deep floor (far below 1e-4)."""
    assert _roundtrip_err(nside, lmax, 0, niter=3, use_weights=True) < DEEP_FLOOR
    assert _roundtrip_err(nside, lmax, 2, niter=3, use_weights=True) < DEEP_FLOOR


@pytest.mark.parametrize("nside,lmax", [(32, 32), (64, 64)])
def test_vs_healpy_use_weights(nside, lmax):
    """jht's weighted recovery is no worse than healpy's map2alm(use_weights=True)."""
    import healpy as hp

    a = _rand_alm(lmax, 0)
    m = np.asarray(synthesis(np.asarray(a), nside, lmax, spin=0))
    jht_err = _roundtrip_err(nside, lmax, 0, niter=3, use_weights=True)
    hp_err = float(np.max(np.abs(hp.map2alm(m, lmax=lmax, iter=3, use_weights=True) - a)))
    assert jht_err < DEEP_FLOOR
    assert jht_err <= 10.0 * hp_err + 1e-12  # comparable, machine-noise cushion
