"""Gate grad (Phase 2): differentiability of the on-grid transforms.

jht uses JAX's **native** autodiff (no custom VJP/JVP rule); ``adjoint_synthesis``
stays the strict transpose for operator-form linear algebra (the bk-jax seam / CG).
This gate pins the convention and proves AD is correct, in both modes:

* ``jacfwd == jacrev`` on the real-DOF layer (``R^n -> R^m``, unambiguous);
* the inner-product adjoint identity ``<S a, v> == <a, S^T v>`` (the strict adjoint);
* native reverse-mode AD == the validated ``adjoint_synthesis`` kernel via the
  documented ``(2 - delta_m0)`` bridge (``vjp(S)(v) == G * conj(S^T v)``), spin 0 & 2;
* finite-difference agreement of ``jax.grad`` on the real-DOF layer;
* ``jit`` / ``vmap`` cleanliness;
* an end-to-end ``map -> a_lm -> C_ell`` ``jax.grad`` chain.

All inputs are band-limited (``lmax <= 1.5 nside``) so the differentiation, not the
HEALPix quadrature, is under test.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import healpy as hp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jht.analysis import map2alm  # noqa: E402
from jht.diff import (  # noqa: E402
    analysis_real,
    bandpower,
    n_dof,
    synthesis_real,
)
from jht.healpix import (  # noqa: E402
    adjoint_synthesis,
    alm_column_base,
    alm_metric_weight,
    alm_size,
    synthesis,
)

ALG_TOL = 1e-12  # algebraic identities (a-priori; achieves ~1e-15)
FD_TOL = 1e-6  # finite-difference agreement (step-limited)


def _rand_alm(lmax, rng, lmin=0):
    a = (rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))) / np.sqrt(
        2
    )
    for ell in range(lmax + 1):
        a[ell] = a[ell].real  # m=0 column real
    for m in range(min(lmin, lmax + 1)):
        base = alm_column_base(m, lmax)
        a[base : base + (lmin - m)] = 0.0
    return a.astype(np.complex128)


def _m_of_idx(lmax):
    return np.concatenate([np.full(lmax + 1 - m, m) for m in range(lmax + 1)])


# --------------------------------------------------------------------------- #
# the (2 - delta_m0) metric helper
# --------------------------------------------------------------------------- #
def test_metric_weight():
    lmax = 24
    G = alm_metric_weight(lmax)
    ms = _m_of_idx(lmax)
    assert np.all(G[ms == 0] == 1.0)
    assert np.all(G[ms > 0] == 2.0)


# --------------------------------------------------------------------------- #
# jacfwd == jacrev on the real-DOF layer (the headline gate)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
def test_jacfwd_eq_jacrev_synthesis_real(spin):
    nside, lmax = 8, 8
    rng = np.random.default_rng(10 + spin)
    x = jnp.asarray(rng.standard_normal(n_dof(lmax, spin)))
    f = lambda xx: synthesis_real(xx, nside, lmax, spin)  # noqa: E731
    Jf = jax.jacfwd(f)(x)
    Jr = jax.jacrev(f)(x)
    assert float(jnp.max(jnp.abs(Jf - Jr))) <= ALG_TOL


@pytest.mark.parametrize("spin", [0, 2])
def test_jacfwd_eq_jacrev_analysis_real(spin):
    nside, lmax = 8, 8
    rng = np.random.default_rng(20 + spin)
    npix = 12 * nside**2
    m = jnp.asarray(rng.standard_normal(npix if spin == 0 else (2, npix)))
    f = lambda mm: analysis_real(mm, nside, lmax, spin, niter=2)  # noqa: E731
    Jf = jax.jacfwd(f)(m)
    Jr = jax.jacrev(f)(m)
    assert float(jnp.max(jnp.abs(Jf - Jr))) <= ALG_TOL


# --------------------------------------------------------------------------- #
# inner-product adjoint identity <S a, v> == <a, S^T v>  (strict adjoint)
# --------------------------------------------------------------------------- #
def test_adjoint_identity_spin0():
    nside, lmax = 16, 24
    rng = np.random.default_rng(0)
    npix = 12 * nside**2
    a = _rand_alm(lmax, rng)
    v = rng.standard_normal(npix)
    lhs = float(np.dot(np.asarray(synthesis(a, nside, lmax)), v))
    b = np.asarray(adjoint_synthesis(v, nside, lmax))
    rhs = float(np.sum(alm_metric_weight(lmax) * np.real(np.conj(a) * b)))
    assert abs(lhs - rhs) / abs(lhs) <= ALG_TOL


def test_adjoint_identity_spin2():
    nside, lmax = 16, 24
    rng = np.random.default_rng(1)
    npix = 12 * nside**2
    aE, aB = _rand_alm(lmax, rng, 2), _rand_alm(lmax, rng, 2)
    QU = np.asarray(synthesis(np.stack([aE, aB]), nside, lmax, 2))
    Q, U = rng.standard_normal(npix), rng.standard_normal(npix)
    lhs = float(np.dot(QU[0], Q) + np.dot(QU[1], U))
    b = np.asarray(adjoint_synthesis(np.stack([Q, U]), nside, lmax, 2))
    w = alm_metric_weight(lmax)
    rhs = float(np.sum(w * np.real(np.conj(aE) * b[0])) + np.sum(w * np.real(np.conj(aB) * b[1])))
    assert abs(lhs - rhs) / abs(lhs) <= ALG_TOL


# --------------------------------------------------------------------------- #
# native reverse-mode AD == validated adjoint kernel via the (2-d_m0) bridge
# --------------------------------------------------------------------------- #
def _bridge_spin0(v, nside, lmax):
    G = jnp.asarray(alm_metric_weight(lmax))
    return G * jnp.conj(adjoint_synthesis(v, nside, lmax, 0))


def _bridge_spin2(v, nside, lmax):
    b = adjoint_synthesis(v, nside, lmax, 2)
    G = jnp.asarray(alm_metric_weight(lmax))[None, :]
    cot = G * jnp.conj(b)  # exact for m>0
    m0 = jnp.asarray(_m_of_idx(lmax) == 0)
    bE0, bB0 = jnp.real(b[0]), jnp.real(b[1])  # strict adjoint is real at m=0
    cotE = jnp.where(m0, bE0 - 1j * bB0, cot[0])  # spin-2 m=0 E/B phantom term
    cotB = jnp.where(m0, bB0 + 1j * bE0, cot[1])
    return jnp.stack([cotE, cotB])


def test_native_vjp_eq_bridge_spin0():
    nside, lmax = 16, 24
    rng = np.random.default_rng(2)
    a0 = jnp.zeros(alm_size(lmax), dtype=jnp.complex128)
    v = jnp.asarray(rng.standard_normal(12 * nside**2))
    _, vjp = jax.vjp(lambda a: synthesis(a, nside, lmax, 0), a0)
    cot = vjp(v)[0]
    ref = _bridge_spin0(v, nside, lmax)
    assert float(jnp.max(jnp.abs(cot - ref)) / jnp.max(jnp.abs(ref))) <= ALG_TOL


def test_native_vjp_eq_bridge_spin2():
    nside, lmax = 16, 24
    rng = np.random.default_rng(3)
    a0 = jnp.zeros((2, alm_size(lmax)), dtype=jnp.complex128)
    v = jnp.asarray(rng.standard_normal((2, 12 * nside**2)))
    _, vjp = jax.vjp(lambda a: synthesis(a, nside, lmax, 2), a0)
    cot = vjp(v)[0]
    ref = _bridge_spin2(v, nside, lmax)
    assert float(jnp.max(jnp.abs(cot - ref)) / jnp.max(jnp.abs(ref))) <= ALG_TOL


# --------------------------------------------------------------------------- #
# finite-difference agreement of jax.grad on the real-DOF layer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
def test_fd_grad_synthesis_real(spin):
    nside, lmax = 16, 24
    rng = np.random.default_rng(30 + spin)
    npix = 12 * nside**2
    nx = n_dof(lmax, spin)
    x = jnp.asarray(rng.standard_normal(nx))
    g = jnp.asarray(rng.standard_normal(npix if spin == 0 else (2, npix)))

    def loss(xx):
        return jnp.sum(g * synthesis_real(xx, nside, lmax, spin))

    grad = jax.grad(loss)(x)
    eps = 1e-6
    for j in (3, nx // 2, nx - 4):
        fd = (loss(x.at[j].add(eps)) - loss(x.at[j].add(-eps))) / (2 * eps)
        assert float(abs(fd - grad[j]) / (abs(grad[j]) + 1e-12)) <= FD_TOL


# --------------------------------------------------------------------------- #
# jit / vmap cleanliness
# --------------------------------------------------------------------------- #
def test_jit_and_vmap_grad():
    nside, lmax = 16, 24
    rng = np.random.default_rng(4)
    npix = 12 * nside**2

    def loss(m):
        a = map2alm(m, nside, lmax, spin=0, niter=2)
        return jnp.sum(jnp.abs(a) ** 2)

    m0 = jnp.asarray(rng.standard_normal(npix))
    g_eager = jax.grad(loss)(m0)  # also warms the _prepare / _wvec caches
    g_jit = jax.jit(jax.grad(loss))(m0)
    assert float(jnp.max(jnp.abs(g_eager - g_jit))) <= ALG_TOL
    assert np.all(np.isfinite(np.asarray(g_eager)))

    batch = jnp.asarray(rng.standard_normal((4, npix)))
    gb = jax.vmap(jax.grad(loss))(batch)  # caches pre-warmed above
    assert gb.shape == (4, npix)
    assert float(jnp.max(jnp.abs(gb[0] - jax.grad(loss)(batch[0])))) <= ALG_TOL


# --------------------------------------------------------------------------- #
# bandpower convention vs healpy, and the end-to-end map -> alm -> C_ell grad
# --------------------------------------------------------------------------- #
def test_bandpower_vs_healpy():
    lmax = 24
    rng = np.random.default_rng(5)
    a0 = _rand_alm(lmax, rng)
    cl = np.asarray(bandpower(a0, lmax, 0))
    cl_hp = hp.alm2cl(a0, lmax=lmax)
    assert float(np.max(np.abs(cl - cl_hp)) / np.max(np.abs(cl_hp))) <= ALG_TOL
    aE, aB = _rand_alm(lmax, rng, 2), _rand_alm(lmax, rng, 2)
    clEB = np.asarray(bandpower(np.stack([aE, aB]), lmax, 2))
    assert float(np.max(np.abs(clEB[0] - hp.alm2cl(aE, lmax=lmax))) / np.max(np.abs(clEB[0]))) <= ALG_TOL
    assert float(np.max(np.abs(clEB[1] - hp.alm2cl(aB, lmax=lmax))) / np.max(np.abs(clEB[1]))) <= ALG_TOL


@pytest.mark.parametrize("spin", [0, 2])
def test_end_to_end_map_to_cl_grad(spin):
    nside, lmax = 16, 24
    rng = np.random.default_rng(40 + spin)
    npix = 12 * nside**2
    m = jnp.asarray(rng.standard_normal(npix if spin == 0 else (2, npix)))

    def loss(mm):
        a = map2alm(mm, nside, lmax, spin=spin, niter=2)
        cl = bandpower(a, lmax, spin)
        return jnp.sum(cl**2)

    grad = jax.grad(loss)(m)
    grad_jit = jax.jit(jax.grad(loss))(m)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(jnp.max(jnp.abs(grad - grad_jit))) <= ALG_TOL

    # finite-difference one map element (real map -> unambiguous)
    flat = grad.ravel()
    eps = 1e-6
    for j in (5, npix // 2):
        mp = m.ravel().at[j].add(eps).reshape(m.shape)
        mm = m.ravel().at[j].add(-eps).reshape(m.shape)
        fd = (loss(mp) - loss(mm)) / (2 * eps)
        assert float(abs(fd - flat[j]) / (abs(flat[j]) + 1e-12)) <= FD_TOL
