"""Gate grad (off-grid real-DOF): the arbitrary-pointing differentiable layer.

``synthesis_general_real`` = ``S_g o T^-1`` and ``adjoint_synthesis_general_real``
= ``T o S_g^T`` are the off-grid (NUFFT) duals of ``synthesis_real``: plain
real-linear ``R^n -> R^m`` maps over arbitrary pointings, spin 0-3.  This gate
proves, for each spin:

* the wrapper equals the explicit ``synthesis_general(real_to_alm(x))`` composition
  (forward wiring: alm packing, spin, loc, epsilon);
* ``jacfwd == jacrev`` -- the real-DOF layer has no complex-conjugate subtlety;
* the real inner-product adjoint identity ``<S_g_real x, v> == <x, S_g_real^T v>``,
  so ``adjoint_synthesis_general_real`` is the exact Euclidean transpose (holds
  including the spin-2 m=0 phantom: T and the (2 - delta_m0) metric both drop
  Im(a_l0) consistently);
* native reverse-mode AD == ``adjoint_synthesis_general_real`` *exactly* -- a
  real-linear map has VJP == transpose, no bridge needed (the point of the layer).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jht.diff import adjoint_synthesis_general_real, synthesis_general_real  # noqa: E402
from jht.masked import n_dof, real_to_alm  # noqa: E402
from jht.offgrid import synthesis_general  # noqa: E402

ALG_TOL = 1e-12  # algebraic identities (a-priori; achieves ~1e-15)
LMAX = 16
NPTS = 400


def _rand_x(spin, seed):
    return np.random.default_rng(seed).standard_normal(n_dof(LMAX, spin))


def _rand_loc(n, seed):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(0.02, np.pi - 0.02, n), rng.uniform(0.0, 2.0 * np.pi, n)], axis=1)


def _rand_field(npts, spin, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(npts) if spin == 0 else rng.standard_normal((2, npts))


@pytest.mark.parametrize("spin", [0, 1, 2, 3])
def test_wrapper_equals_composition(spin):
    x, loc = jnp.asarray(_rand_x(spin, spin)), _rand_loc(NPTS, 100 + spin)
    got = synthesis_general_real(x, loc, spin=spin, lmax=LMAX)
    ref = synthesis_general(real_to_alm(x, LMAX, spin), loc, spin=spin, lmax=LMAX)
    assert float(jnp.max(jnp.abs(got - ref))) == 0.0  # same call path -> bit-identical


@pytest.mark.parametrize("spin", [0, 1, 2, 3])
def test_jacfwd_eq_jacrev(spin):
    x, loc = jnp.asarray(_rand_x(spin, 10 + spin)), _rand_loc(60, 200 + spin)
    f = lambda xx: synthesis_general_real(xx, loc, spin=spin, lmax=LMAX)  # noqa: E731
    Jf, Jr = jax.jacfwd(f)(x), jax.jacrev(f)(x)
    assert float(jnp.max(jnp.abs(Jf - Jr))) <= ALG_TOL


@pytest.mark.parametrize("spin", [0, 1, 2, 3])
def test_real_adjoint_identity(spin):
    x, loc = _rand_x(spin, 20 + spin), _rand_loc(NPTS, 300 + spin)
    v = _rand_field(NPTS, spin, 400 + spin)
    fwd = np.asarray(synthesis_general_real(jnp.asarray(x), loc, spin=spin, lmax=LMAX))
    adj = np.asarray(adjoint_synthesis_general_real(jnp.asarray(v), loc, spin=spin, lmax=LMAX))
    lhs = float(np.sum(fwd * v))
    rhs = float(np.dot(x, adj))
    assert abs(lhs - rhs) / abs(lhs) <= ALG_TOL


@pytest.mark.parametrize("spin", [0, 1, 2, 3])
def test_native_vjp_eq_adjoint(spin):
    x, loc = jnp.asarray(_rand_x(spin, 30 + spin)), _rand_loc(NPTS, 500 + spin)
    v = jnp.asarray(_rand_field(NPTS, spin, 600 + spin))
    _, vjp = jax.vjp(lambda xx: synthesis_general_real(xx, loc, spin=spin, lmax=LMAX), x)
    cot = vjp(v)[0]
    adj = adjoint_synthesis_general_real(v, loc, spin=spin, lmax=LMAX)
    assert float(jnp.max(jnp.abs(cot - adj)) / jnp.max(jnp.abs(adj))) <= ALG_TOL
