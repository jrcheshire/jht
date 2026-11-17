"""l-blocked Legendre contraction: every block size K gives the K = 1 result.

The recursion rows are bitwise unchanged by blocking; only the order of the l-sum
differs, so contractions agree to rounding.  K = 3 does not divide lmax + 1 here,
so the zero/inactive padding rows are exercised; ``"auto"`` resolves to the cap.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import jht  # noqa: E402
from jht import _recursion as R  # noqa: E402
from jht import healpix  # noqa: E402

TOL = 1e-13  # a-priori: summation reordering only (lambda rows are identical)
TOL_T = 1e-12  # transform level (FFTs / packing on top)
LMAX = 40
X = np.cos(np.linspace(0.05, 0.5 * np.pi, 21))  # north-half-like colatitudes, incl. near-pole
BLOCKS = [3, "auto"]


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _cplx(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


@pytest.fixture
def restore_block():
    old = jht.get_ell_block()
    yield
    jht.set_ell_block(old)


def test_setter_validation(restore_block):
    for bad in (0, -1, 2.0, "big", True, None):
        with pytest.raises(ValueError):
            jht.set_ell_block(bad)
    jht.set_ell_block(7)
    assert jht.get_ell_block() == 7
    jht.set_ell_block("auto")
    assert jht.get_ell_block() == "auto"


def test_auto_rule():
    assert R.resolve_ell_block("auto", 41, 21) == 32  # small problem -> cap
    assert R.resolve_ell_block(5, 41, 21) == 5
    # nside 2048 / lmax 3000 spin 2: 2 stacks of 3001 x 4096 doubles -> budget-limited
    assert R.resolve_ell_block("auto", 3001, 4096, 2) == 2
    assert R.resolve_ell_block("auto", 10**5, 10**5) == 1  # never below 1


@pytest.mark.parametrize("K", BLOCKS)
@pytest.mark.parametrize("spin", [0, 2, -2])
def test_single_state_contractions(K, spin):
    rng = np.random.default_rng(1)
    plan = R.build_recursion_plan(X, spin, LMAX)
    M, T = LMAX + 1, X.size
    a = _cplx(rng, (M, LMAX + 1))
    V = _cplx(rng, (M, T))
    assert _rel(R.synth_contract(plan, X, a, K), R.synth_contract(plan, X, a, 1)) <= TOL
    assert _rel(R.adjoint_contract(plan, X, V, K), R.adjoint_contract(plan, X, V, 1)) <= TOL


@pytest.mark.parametrize("K", BLOCKS)
def test_spin0_eo_contractions(K):
    rng = np.random.default_rng(2)
    plan = R.build_recursion_plan(X, 0, LMAX)
    M, T = LMAX + 1, X.size
    a = _cplx(rng, (M, LMAX + 1))
    Vn, Vs = _cplx(rng, (M, T)), _cplx(rng, (M, T))
    for got, ref in zip(R.synth_contract_eo(plan, X, a, K), R.synth_contract_eo(plan, X, a, 1)):
        assert _rel(got, ref) <= TOL
    got = R.adjoint_contract_eo(plan, X, Vn, Vs, K)
    assert _rel(got, R.adjoint_contract_eo(plan, X, Vn, Vs, 1)) <= TOL


@pytest.mark.parametrize("K", BLOCKS)
def test_spin2_ns_contractions(K):
    rng = np.random.default_rng(3)
    pp, pm = R.build_recursion_plan(X, 2, LMAX), R.build_recursion_plan(X, -2, LMAX)
    M, T = LMAX + 1, X.size
    bp, bm = _cplx(rng, (M, LMAX + 1)), _cplx(rng, (M, LMAX + 1))
    V = [_cplx(rng, (M, T)) for _ in range(4)]
    got = R.synth_contract_spin2_ns(pp, pm, X, bp, bm, K)
    ref = R.synth_contract_spin2_ns(pp, pm, X, bp, bm, 1)
    for g, r in zip(got, ref):
        assert _rel(g, r) <= TOL
    got = R.adjoint_contract_spin2_ns(pp, pm, X, *V, K)
    ref = R.adjoint_contract_spin2_ns(pp, pm, X, *V, 1)
    for g, r in zip(got, ref):
        assert _rel(g, r) <= TOL


def test_compiled_matches_eager():
    rng = np.random.default_rng(4)
    pp, pm = R.build_recursion_plan(X, 2, LMAX), R.build_recursion_plan(X, -2, LMAX)
    M = LMAX + 1
    bp, bm = _cplx(rng, (M, LMAX + 1)), _cplx(rng, (M, LMAX + 1))
    f = jax.jit(lambda u, v: R.synth_contract_spin2_ns(pp, pm, X, u, v, 3))
    with jax.disable_jit():
        eager = R.synth_contract_spin2_ns(pp, pm, X, bp, bm, 3)
    for g, r in zip(f(bp, bm), eager):
        assert _rel(g, r) <= TOL


NSIDE, TLMAX = 16, 24


def _alm(rng, spin):
    shape = (healpix.alm_size(TLMAX),) if spin == 0 else (2, healpix.alm_size(TLMAX))
    a = _cplx(rng, shape)
    a[..., : TLMAX + 1] = a[..., : TLMAX + 1].real
    return a


@pytest.mark.parametrize("spin", [0, 2])
def test_vmap_matches_loop(spin):
    """The batched path consumers use: vmap over maps == one call per map."""
    rng = np.random.default_rng(5)
    alms = np.stack([_alm(rng, spin) for _ in range(5)])
    f = lambda a: jht.synthesis(a, NSIDE, TLMAX, spin)  # noqa: E731
    assert _rel(jax.vmap(f)(alms), np.stack([f(a) for a in alms])) <= TOL_T
    maps = np.stack([np.asarray(f(a)) for a in alms])
    g = lambda m: jht.adjoint_synthesis(m, NSIDE, TLMAX, spin)  # noqa: E731
    assert _rel(jax.vmap(g)(maps), np.stack([g(m) for m in maps])) <= TOL_T


@pytest.mark.parametrize("spin", [0, 2])
def test_transforms_and_vjp_match_unblocked(spin, restore_block):
    rng = np.random.default_rng(6)
    a = _alm(rng, spin)
    m = rng.standard_normal((12 * NSIDE**2,) if spin == 0 else (2, 12 * NSIDE**2))

    def run():
        y, vjp = jax.vjp(lambda z: jht.synthesis(z, NSIDE, TLMAX, spin), a)
        b, vjp_t = jax.vjp(lambda z: jht.adjoint_synthesis(z, NSIDE, TLMAX, spin), m)
        return y, vjp(m)[0], b, vjp_t(a)[0]

    jht.set_ell_block(1)
    ref = run()
    for K in BLOCKS:
        jht.set_ell_block(K)
        for g, r in zip(run(), ref):
            assert _rel(g, r) <= TOL_T


def test_block_is_part_of_cache_key(restore_block):
    p1 = healpix._prepare(NSIDE, TLMAX, 0, "unrolled", 1)
    p3 = healpix._prepare(NSIDE, TLMAX, 0, "unrolled", 3)
    assert p1 is not p3
