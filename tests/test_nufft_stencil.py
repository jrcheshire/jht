"""NUFFT stencil under compilation: points exactly on nodes of the oversampled grid.

A point on a grid node has an integer grid coordinate, so the stencil's first cell is
decided by a rounding. The cell indices and the stencil weights must come out of the same
decision: compiled (``jax.jit``) exactly as not. Gates:

* ``_stencil`` compiled == not compiled, indices exactly and weights to roundoff, on every
  node of a grid and at arbitrary points;
* compiled ``synthesis_general`` vs the **exact direct sum** on node points, spin 0 and 2,
  at the off-grid tier;
* the compiled adjoint on node points: the identity ``<S_g a, v> == <a, S_g^T v>_G``,
  and agreement with the uncompiled adjoint.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from test_offgrid import ADJ_TOL, FWD_TOL, _direct, _inner_alm, _rand_alm  # noqa: E402

from jht import _nufft  # noqa: E402
from jht.offgrid import _prepare, adjoint_synthesis_general, synthesis_general  # noqa: E402

TWO_PI = 2.0 * np.pi


def _node_loc(lmax, spin, n_pts, seed):
    """Points whose colatitude and longitude both lie exactly on nodes of the NUFFT's
    oversampled grid, away from the poles."""
    plan = _prepare(lmax, spin, 1e-10).nplan
    rng = np.random.default_rng(seed)
    k = rng.integers(plan.nk // 16, plan.nk // 2 - plan.nk // 16, n_pts)
    j = rng.integers(0, plan.nm, n_pts)
    return np.stack([TWO_PI * k / plan.nk, TWO_PI * j / plan.nm], axis=1)


@pytest.mark.parametrize("n", [400, 384, 250, 1000])
@pytest.mark.parametrize("W", [6, 14, 16])
def test_stencil_compiled_matches_uncompiled(n, W):
    beta = _nufft._beta(W, 2.0)
    rng = np.random.default_rng(n + W)
    nodes = TWO_PI * np.arange(n) / n
    coord = jnp.asarray(np.concatenate([nodes, rng.uniform(0.0, TWO_PI, 500)]))
    idx, w = _nufft._stencil(coord, n, W, beta)
    idx_c, w_c = jax.jit(lambda c: _nufft._stencil(c, n, W, beta))(coord)
    assert np.array_equal(np.asarray(idx), np.asarray(idx_c))
    assert np.max(np.abs(np.asarray(w) - np.asarray(w_c))) <= 1e-12
    # On a node the stencil is the kernel sampled at integer offsets, the same set of
    # weights wherever the node is. One offset is at the edge of the support, where the
    # weight is zero or, a rounding error inside it, just above the kernel's boundary
    # value exp(-beta).
    on_node = np.sort(np.asarray(w_c)[:n], axis=1)
    assert np.max(np.abs(on_node - on_node[:1])) <= 2.0 * np.exp(-beta) + 1e-12


@pytest.mark.parametrize("spin", [0, 2])
def test_compiled_forward_vs_direct_on_grid_nodes(spin):
    lmax = 48
    alm, loc = _rand_alm(lmax, 5, spin), _node_loc(lmax, spin, 600, 6)
    compiled = jax.jit(lambda a, x: synthesis_general(a, x, spin=spin, lmax=lmax))
    f = np.asarray(compiled(jnp.asarray(alm), jnp.asarray(loc)))
    ref = np.asarray(_direct(alm, loc, lmax, spin))
    assert np.max(np.abs(f - ref)) <= FWD_TOL
    plain = np.asarray(synthesis_general(alm, loc, spin=spin, lmax=lmax))
    assert np.max(np.abs(plain - ref)) <= FWD_TOL


@pytest.mark.parametrize("spin", [0, 2])
def test_compiled_adjoint_identity_on_grid_nodes(spin):
    lmax = 48
    alm, loc = _rand_alm(lmax, 7, spin), _node_loc(lmax, spin, 600, 8)
    rng = np.random.default_rng(9)
    v = rng.standard_normal(loc.shape[0]) if spin == 0 else rng.standard_normal((2, loc.shape[0]))
    forward = jax.jit(lambda a, x: synthesis_general(a, x, spin=spin, lmax=lmax))
    adjoint = jax.jit(lambda u, x: adjoint_synthesis_general(u, x, spin=spin, lmax=lmax))
    f = np.asarray(forward(jnp.asarray(alm), jnp.asarray(loc)))
    b = np.asarray(adjoint(jnp.asarray(v), jnp.asarray(loc)))
    lhs = float(np.sum(f * v))
    rhs = _inner_alm(alm, b, lmax)
    assert abs(lhs - rhs) / abs(lhs) <= ADJ_TOL
    # The identity alone would hold for a stencil misplaced the same way in both
    # directions, so the compiled adjoint is also held to the uncompiled one.
    plain = np.asarray(adjoint_synthesis_general(v, loc, spin=spin, lmax=lmax))
    assert np.max(np.abs(b - plain)) <= ADJ_TOL * np.max(np.abs(plain))
