"""Phase-1 equivalence gates: the vectorized/jitted fast path == the Phase-0 reference.

The fast path (one fused all-m ``lax.scan`` + batched ring FFTs + jit, in
:mod:`jht._recursion` / :mod:`jht.healpix`) must reproduce the validated eager
Phase-0 transforms (:mod:`jht._reference`) and the per-m recursion, to machine
precision -- the refactor is performance-only and changes no numerics.  This is
in addition to the healpy/ducc parity gates in ``test_healpix.py``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jht._recursion import (  # noqa: E402
    adjoint_contract,
    build_recursion_plan,
    normalized_legendre,
    spin_weighted_lambda,
)
from jht._recursion import synth_contract as _synth_contract  # noqa: E402
from jht._reference import (  # noqa: E402
    adjoint_synthesis_reference,
    synthesis_reference,
)
from jht.healpix import adjoint_synthesis, alm_column_base, alm_size, synthesis  # noqa: E402

TOL = 1e-12  # a-priori; the recursion match is ~1e-15, the FFT path ~1e-14


def _ref_lambda_dense(x, spin, lmax):
    """Dense (M, lmax+1, T) reference lambda table from the per-m functions."""
    M = lmax + 1
    lam = np.zeros((M, lmax + 1, x.shape[0]))
    for m in range(M):
        if spin == 0:
            lam[m, m:] = np.asarray(normalized_legendre(x, m, lmax))
        else:
            lmin = max(m, abs(spin))
            lam[m, lmin:] = np.asarray(spin_weighted_lambda(x, m, spin, lmax))
    return lam


def _dense_alm(M, lmax, spin, rng):
    a = (rng.standard_normal((M, lmax + 1)) + 1j * rng.standard_normal((M, lmax + 1))) / 2
    mask = np.zeros((M, lmax + 1))
    for m in range(M):
        mask[m, (m if spin == 0 else max(m, abs(spin))) :] = 1.0
    return a * mask


def _rand_alm(lmax, rng, lmin=0):
    a = (rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))) / np.sqrt(
        2
    )
    for ell in range(lmax + 1):
        a[ell] = a[ell].real
    for m in range(min(lmin, lmax + 1)):
        base = alm_column_base(m, lmax)
        a[base : base + (lmin - m)] = 0.0
    return a.astype(np.complex128)


# --------------------------------------------------------------------------- #
# the fused all-m recursion engine vs the per-m reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2, -2])
@pytest.mark.parametrize("lmax", [12, 64])
def test_synth_contract_matches_per_m(spin, lmax):
    rng = np.random.default_rng(spin + lmax)
    theta = np.linspace(0.01, np.pi - 0.01, 31)
    x = np.cos(theta)
    lam = _ref_lambda_dense(x, spin, lmax)
    alm = _dense_alm(lmax + 1, lmax, spin, rng)
    plan = build_recursion_plan(x, spin, lmax)
    F = np.asarray(_synth_contract(plan, x, alm))
    F_ref = np.einsum("ml,mlt->mt", alm, lam)
    assert np.max(np.abs(F - F_ref)) / max(np.max(np.abs(F_ref)), 1e-300) <= TOL


@pytest.mark.parametrize("spin", [0, 2, -2])
@pytest.mark.parametrize("lmax", [12, 64])
def test_adjoint_contract_matches_per_m(spin, lmax):
    rng = np.random.default_rng(spin + lmax + 1)
    theta = np.linspace(0.01, np.pi - 0.01, 31)
    x = np.cos(theta)
    lam = _ref_lambda_dense(x, spin, lmax)
    V = (rng.standard_normal((lmax + 1, x.shape[0])) + 1j * rng.standard_normal(
        (lmax + 1, x.shape[0])
    )) / 2
    plan = build_recursion_plan(x, spin, lmax)
    b = np.asarray(adjoint_contract(plan, x, V))
    b_ref = np.einsum("mlt,mt->ml", lam, V)
    assert np.max(np.abs(b - b_ref)) / max(np.max(np.abs(b_ref)), 1e-300) <= TOL


# --------------------------------------------------------------------------- #
# the public fast transforms vs the eager Phase-0 reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nside,lmax", [(8, 12), (16, 24)])
def test_fast_synthesis_matches_reference_spin0(nside, lmax):
    rng = np.random.default_rng(nside)
    alm = _rand_alm(lmax, rng)
    fast = np.asarray(synthesis(alm, nside, lmax, spin=0))
    ref = np.asarray(synthesis_reference(alm, nside, lmax, spin=0))
    assert np.max(np.abs(fast - ref)) / np.max(np.abs(ref)) <= TOL


@pytest.mark.parametrize("nside,lmax", [(8, 12), (16, 24)])
def test_fast_synthesis_matches_reference_spin2(nside, lmax):
    rng = np.random.default_rng(nside + 1)
    aE, aB = _rand_alm(lmax, rng, lmin=2), _rand_alm(lmax, rng, lmin=2)
    alm = np.stack([aE, aB])
    fast = np.asarray(synthesis(alm, nside, lmax, spin=2))
    ref = np.asarray(synthesis_reference(alm, nside, lmax, spin=2))
    assert np.max(np.abs(fast - ref)) / np.max(np.abs(ref)) <= TOL


@pytest.mark.parametrize("nside,lmax", [(8, 12), (16, 24)])
def test_fast_adjoint_matches_reference_spin0(nside, lmax):
    rng = np.random.default_rng(nside + 2)
    v = rng.standard_normal(12 * nside**2)
    fast = np.asarray(adjoint_synthesis(v, nside, lmax, spin=0))
    ref = np.asarray(adjoint_synthesis_reference(v, nside, lmax, spin=0))
    assert np.max(np.abs(fast - ref)) / np.max(np.abs(ref)) <= TOL


@pytest.mark.parametrize("nside,lmax", [(8, 12), (16, 24)])
def test_fast_adjoint_matches_reference_spin2(nside, lmax):
    rng = np.random.default_rng(nside + 3)
    Q, U = rng.standard_normal(12 * nside**2), rng.standard_normal(12 * nside**2)
    maps = np.stack([Q, U])
    fast = np.asarray(adjoint_synthesis(maps, nside, lmax, spin=2))
    ref = np.asarray(adjoint_synthesis_reference(maps, nside, lmax, spin=2))
    assert np.max(np.abs(fast - ref)) / np.max(np.abs(ref)) <= TOL
