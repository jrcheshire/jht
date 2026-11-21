"""Stored Legendre tables (``jht.legendre_tables`` / ``tables=``) == the recursion path.

A stored table holds the lambda values the folded recursion produces per call, and
the contraction runs the same block GEMMs in the same order, so on one backend the two
paths agree bitwise; the gate is ``ALG_TOL`` (as ``tests/test_reverse_memory.py``) to
allow a different fusion of the same arithmetic on another backend.  Comparisons are
like for like (eager vs eager, compiled vs compiled).  Also gated: ``vmap`` with an
unbatched table, native AD and the ``*_vjp`` wrappers, the Jacobi ``analysis``, the
size formula, that a table passed as an argument is not compiled in as a constant, and
the guards.
"""

from __future__ import annotations

import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import jht  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:jht. lmax=.*exceeds the HEALPix band-limit")

ALG_TOL = 1e-12  # a-priori: same values, same contraction order
# (nside, lmax): odd lmax below the 1.5 nside ceiling; even lmax at it; lmax = 2 nside - 1 above
# it (warns) with two l-blocks at K = 32.
CASES = [(8, 11), (16, 24), (16, 31)]
BLOCKS = [5, "auto"]


@pytest.fixture(autouse=True)
def _restore_block():
    old = jht.get_ell_block()
    yield
    jht.set_ell_block(old)


def _rel(x, ref):
    x, ref = np.asarray(x), np.asarray(ref)
    return float(np.max(np.abs(x - ref)) / np.max(np.abs(ref)))


def _alm(rng, lmax, spin, n=None):
    K = jht.alm_size(lmax)
    shape = (K,) if spin == 0 else (2, K)
    if n is not None:
        shape = (n, *shape)
    a = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    a[..., : lmax + 1] = a[..., : lmax + 1].real  # m = 0 real
    return jnp.asarray(a)


def _map(rng, nside, spin, n=None):
    npix = 12 * nside**2
    shape = (npix,) if spin == 0 else (2, npix)
    if n is not None:
        shape = (n, *shape)
    return jnp.asarray(rng.standard_normal(shape))


@pytest.mark.parametrize("block", BLOCKS)
@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("nside,lmax", CASES)
def test_tables_match_recursion(nside, lmax, spin, block):
    jht.set_ell_block(block)
    rng = np.random.default_rng(7 * nside + lmax + spin)
    a, m = _alm(rng, lmax, spin), _map(rng, nside, spin)
    t = jht.legendre_tables(nside, lmax, spin)
    # eager
    assert _rel(jht.synthesis(a, nside, lmax, spin, tables=t), jht.synthesis(a, nside, lmax, spin)) <= ALG_TOL
    assert (
        _rel(jht.adjoint_synthesis(m, nside, lmax, spin, tables=t), jht.adjoint_synthesis(m, nside, lmax, spin))
        <= ALG_TOL
    )
    assert _rel(jht.analysis(m, nside, lmax, spin, tables=t), jht.analysis(m, nside, lmax, spin)) <= ALG_TOL
    # compiled, table as an argument
    s_t = jax.jit(lambda a, t: jht.synthesis(a, nside, lmax, spin, tables=t))(a, t)
    s_r = jax.jit(lambda a: jht.synthesis(a, nside, lmax, spin))(a)
    assert _rel(s_t, s_r) <= ALG_TOL
    b_t = jax.jit(lambda m, t: jht.adjoint_synthesis(m, nside, lmax, spin, tables=t))(m, t)
    b_r = jax.jit(lambda m: jht.adjoint_synthesis(m, nside, lmax, spin))(m)
    assert _rel(b_t, b_r) <= ALG_TOL


@pytest.mark.parametrize("spin", [0, 2])
def test_tables_under_vmap_unbatched(spin):
    nside, lmax = 16, 24
    rng = np.random.default_rng(30 + spin)
    A, Mp = _alm(rng, lmax, spin, n=3), _map(rng, nside, spin, n=3)
    t = jht.legendre_tables(nside, lmax, spin)
    vs = jax.jit(jax.vmap(lambda a, t: jht.synthesis(a, nside, lmax, spin, tables=t), in_axes=(0, None)))
    va = jax.jit(jax.vmap(lambda m, t: jht.adjoint_synthesis(m, nside, lmax, spin, tables=t), in_axes=(0, None)))
    rs = jax.jit(jax.vmap(lambda a: jht.synthesis(a, nside, lmax, spin)))
    ra = jax.jit(jax.vmap(lambda m: jht.adjoint_synthesis(m, nside, lmax, spin)))
    assert _rel(vs(A, t), rs(A)) <= ALG_TOL
    assert _rel(va(Mp, t), ra(Mp)) <= ALG_TOL


@pytest.mark.parametrize("spin", [0, 2])
def test_tables_gradients(spin):
    nside, lmax = 16, 24
    rng = np.random.default_rng(40 + spin)
    a, m = _alm(rng, lmax, spin), _map(rng, nside, spin)
    t = jht.legendre_tables(nside, lmax, spin)

    def compiled_grad(f):
        return jax.jit(jax.grad(f))

    # native AD w.r.t. alm (the table is a constant of the differentiation)
    g_t = compiled_grad(lambda a, t: jnp.sum(jht.synthesis(a, nside, lmax, spin, tables=t) ** 2))(a, t)
    g_r = compiled_grad(lambda a: jnp.sum(jht.synthesis(a, nside, lmax, spin) ** 2))(a)
    assert _rel(g_t, g_r) <= ALG_TOL
    # reverse-only wrappers, both directions
    w_t = compiled_grad(lambda a, t: jnp.sum(jht.synthesis_vjp(a, nside, lmax, spin, tables=t) ** 2))(a, t)
    w_r = compiled_grad(lambda a: jnp.sum(jht.synthesis_vjp(a, nside, lmax, spin) ** 2))(a)
    assert _rel(w_t, w_r) <= ALG_TOL

    def loss_adj(m, t=None):
        return jnp.sum(jnp.abs(jht.adjoint_synthesis_vjp(m, nside, lmax, spin, tables=t)) ** 2)

    assert _rel(compiled_grad(loss_adj)(m, t), compiled_grad(lambda m: loss_adj(m))(m)) <= ALG_TOL
    # eager wrapper gradient (bwd runs with concrete residual tables)
    e_t = jax.grad(lambda a: jnp.sum(jht.synthesis_vjp(a, nside, lmax, spin, tables=t) ** 2))(a)
    e_r = jax.grad(lambda a: jnp.sum(jht.synthesis_vjp(a, nside, lmax, spin) ** 2))(a)
    assert _rel(e_t, e_r) <= ALG_TOL


def test_tables_in_scan_body():
    nside, lmax, spin = 16, 24, 2
    rng = np.random.default_rng(50)
    A = _alm(rng, lmax, spin, n=3)
    t = jht.legendre_tables(nside, lmax, spin)

    @jax.jit
    def run(A, t=None):
        return jax.lax.map(lambda a: jht.synthesis(a, nside, lmax, spin, tables=t), A)

    assert _rel(run(A, t), run(A)) <= ALG_TOL


@pytest.mark.parametrize("spin", [0, 2])
def test_nbytes_matches_arrays(spin):
    for nside, lmax, block in [(8, 11, "auto"), (16, 31, 5), (16, 24, 3)]:
        t = jht.legendre_tables(nside, lmax, spin, ell_block=block)
        assert sum(a.nbytes for a in t.lam) == jht.legendre_tables_nbytes(nside, lmax, spin, block)
        assert t.ell_block == t.lam[0].shape[1]
        assert len(t.lam) == (1 if spin == 0 else 2)


def test_table_block_overrides_setting():
    # a table built at K=5 contracts in blocks of 5 even when the global setting differs
    nside, lmax, spin = 16, 31, 0
    rng = np.random.default_rng(60)
    a = _alm(rng, lmax, spin)
    t5 = jht.legendre_tables(nside, lmax, spin, ell_block=5)
    jht.set_ell_block(5)
    ref = jht.synthesis(a, nside, lmax, spin)
    jht.set_ell_block("auto")
    assert _rel(jht.synthesis(a, nside, lmax, spin, tables=t5), ref) <= ALG_TOL


_CONST = re.compile(r'dense<"0x([0-9A-Fa-f]+)">')


def _constant_bytes(lowered_text: str) -> int:
    return sum(len(h) // 2 for h in _CONST.findall(lowered_text))


def test_table_argument_is_not_a_constant():
    nside, lmax, spin = 16, 31, 2
    t = jht.legendre_tables(nside, lmax, spin)
    nbytes = jht.legendre_tables_nbytes(nside, lmax, spin)
    a = _alm(np.random.default_rng(70), lmax, spin)
    txt = jax.jit(lambda a, t: jht.synthesis(a, nside, lmax, spin, tables=t)).lower(a, t).as_text()
    assert _constant_bytes(txt) < nbytes // 10
    # the measure sees an embedded table (anti-vacuity): closing over one compiles it in
    closed = jax.jit(lambda a: jnp.sum(a) + jnp.sum(t.lam[0])).lower(a).as_text()
    assert _constant_bytes(closed) >= nbytes // 2


def test_guards():
    nside, lmax = 8, 11
    t = jht.legendre_tables(nside, lmax, 0)
    a = jnp.zeros(jht.alm_size(lmax), complex)
    with pytest.raises(ValueError, match="constants"):
        jax.jit(lambda a: jht.synthesis(a, nside, lmax, 0, tables=t))(a)  # closed over
    m = jnp.zeros(12 * nside**2)
    with pytest.raises(ValueError, match="constants"):  # checked before the custom_vjp
        jax.jit(lambda a: jht.synthesis_vjp(a, nside, lmax, 0, tables=t))(a)
    with pytest.raises(ValueError, match="constants"):
        jax.jit(lambda m: jht.adjoint_synthesis_vjp(m, nside, lmax, 0, tables=t))(m)
    with pytest.raises(ValueError, match="nside, lmax, spin"):
        jht.synthesis(jnp.zeros(jht.alm_size(12), complex), nside, 12, 0, tables=t)
    with pytest.raises(ValueError, match="nside, lmax, spin"):
        jht.synthesis(jnp.zeros((2, jht.alm_size(lmax)), complex), nside, lmax, 2, tables=t)
    with pytest.raises(TypeError):
        jht.synthesis(a, nside, lmax, 0, tables=t.lam)
    with pytest.raises(NotImplementedError):
        jht.legendre_tables(nside, lmax, 1)
