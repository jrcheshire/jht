"""NUFFT spreading (the transposed stencil step) against the per-cell reference.

``_spread_points`` writes one ``(W, W)`` window per point on a padded grid and folds the
padding back. Gates, eager and compiled:

* equal to the per-cell scatter-add (the jaxht 0.4.1 formulation, kept here as the
  reference) to roundoff, for grids smaller and larger than the stencil, real and complex
  values, points at the periodic edges, outside ``[0, 2 pi)`` and exactly on grid nodes;
* the exact transpose of ``_interp_points``: ``Re<interp(g), v> == Re<g, spread(v)>``;
* ``vmap`` over values with fixed points == a loop.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jht import _nufft  # noqa: E402

TWO_PI = 2.0 * np.pi
ALG_TOL = 1e-12  # algebraically identical sums, reordered (a-priori; measured ~1e-16)

# (Nk, Nm, eps): grids larger than the stencil, and tiny ones where W > n (several wraps).
PLANS = [(128, 97, 1e-10), (40, 33, 1e-6), (8, 5, 1e-10), (4, 3, 1e-4)]


def _spread_reference(plan, vals, x, y):
    lk, wk = _nufft._stencil(x, plan.nk, plan.W, plan.beta_k)
    lm, wm = _nufft._stencil(y, plan.nm, plan.W, plan.beta_m)
    contrib = (wk[:, :, None] * wm[:, None, :]) * jnp.asarray(vals)[:, None, None]
    g = jnp.zeros((plan.nk, plan.nm), dtype=jnp.result_type(vals))
    return g.at[lk[:, :, None], lm[:, None, :]].add(contrib)


def _points(plan, n, seed):
    """Random points, grid nodes, points at and just inside the periodic edges, and
    points outside ``[0, 2 pi)``."""
    rng = np.random.default_rng(seed)
    rand = rng.uniform(0.0, TWO_PI, (n, 2))
    nodes = np.stack(
        [TWO_PI * rng.integers(0, plan.nk, n) / plan.nk, TWO_PI * rng.integers(0, plan.nm, n) / plan.nm], 1
    )
    edge = np.array([[0.0, 0.0], [TWO_PI - 1e-15, TWO_PI - 1e-15], [1e-13, TWO_PI - 1e-13],
                     [-0.05, 0.03], [TWO_PI + 0.02, -0.4], [np.pi, 0.0]])
    loc = np.concatenate([rand, nodes, edge])
    return jnp.asarray(loc[:, 0]), jnp.asarray(loc[:, 1])


def _vals(n, complex_, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n)
    return jnp.asarray(v + 1j * rng.standard_normal(n) if complex_ else v)


@pytest.mark.parametrize("Nk,Nm,eps", PLANS)
@pytest.mark.parametrize("complex_", [False, True])
def test_spread_matches_per_cell_reference(Nk, Nm, eps, complex_):
    plan = _nufft.nufft_plan(Nk, Nm, eps)
    x, y = _points(plan, 300, Nk + Nm)
    v = _vals(x.shape[0], complex_, 1)
    ref = np.asarray(_spread_reference(plan, v, x, y))
    scale = np.max(np.abs(ref))
    eager = np.asarray(_nufft._spread_points(plan, v, x, y))
    compiled = np.asarray(jax.jit(lambda v, x, y: _nufft._spread_points(plan, v, x, y))(v, x, y))
    assert eager.shape == (plan.nk, plan.nm)
    assert np.iscomplexobj(eager) == complex_
    assert np.max(np.abs(eager - ref)) <= ALG_TOL * scale
    assert np.max(np.abs(compiled - ref)) <= ALG_TOL * scale


@pytest.mark.parametrize("Nk,Nm,eps", PLANS)
def test_spread_is_transpose_of_interp(Nk, Nm, eps):
    plan = _nufft.nufft_plan(Nk, Nm, eps)
    x, y = _points(plan, 300, 7)
    rng = np.random.default_rng(8)
    g = jnp.asarray(rng.standard_normal((plan.nk, plan.nm)) + 1j * rng.standard_normal((plan.nk, plan.nm)))
    v = _vals(x.shape[0], True, 9)
    lhs = float(jnp.real(jnp.vdot(_nufft._interp_points(plan, g, x, y), v)))
    rhs = float(jnp.real(jnp.vdot(g, _nufft._spread_points(plan, v, x, y))))
    assert abs(lhs - rhs) <= ALG_TOL * abs(lhs)


def test_spread_vmap_over_values():
    plan = _nufft.nufft_plan(128, 97, 1e-10)
    x, y = _points(plan, 200, 3)
    rng = np.random.default_rng(4)
    vs = jnp.asarray(rng.standard_normal((3, x.shape[0])))
    batched = np.asarray(jax.vmap(lambda v: _nufft._spread_points(plan, v, x, y))(vs))
    looped = np.stack([np.asarray(_nufft._spread_points(plan, v, x, y)) for v in vs])
    assert np.max(np.abs(batched - looped)) <= ALG_TOL * np.max(np.abs(looped))


def _clumped(plan, n, seed):
    """Uniform points plus a clump that fills a few cells many times over."""
    x, y = _points(plan, n, seed)
    rng = np.random.default_rng(seed + 100)
    cx = 1.0 + 0.02 * rng.random(4 * n)
    cy = 2.0 + 0.02 * rng.random(4 * n)
    return jnp.concatenate([x, jnp.asarray(cx)]), jnp.concatenate([y, jnp.asarray(cy)])


@pytest.mark.parametrize("Nk,Nm,eps", PLANS)
@pytest.mark.parametrize("group_size", [1, 2, 3, 16])
@pytest.mark.parametrize("complex_", [False, True])
def test_grouped_spread_matches_per_cell_reference(Nk, Nm, eps, group_size, complex_):
    plan = _nufft.nufft_plan(Nk, Nm, eps)
    x, y = _clumped(plan, 200, Nk + group_size)
    v = _vals(x.shape[0], complex_, 2)
    pts = _nufft.prepare_points_plan((Nk, Nm, eps), np.asarray(x), np.asarray(y),
                                     group_size=group_size, store_weights=False)
    assert max(s.shape[1] for s, _ in pts.buckets) <= group_size
    slots = sum(s.size for s, _ in pts.buckets)
    assert slots < 2 * x.shape[0]  # power-of-two buckets pad by less than 2x
    ref = np.asarray(_spread_reference(plan, v, x, y))

    def grouped(v, x, y, pts):
        wk = _nufft._weights_from_base(x, pts.base[:, 0], plan.nk, plan.W, plan.beta_k)
        wm = _nufft._weights_from_base(y, pts.base[:, 1], plan.nm, plan.W, plan.beta_m)
        return _nufft._spread_grouped(plan, v, wk, wm, pts.buckets)

    for out in (grouped(v, x, y, pts), jax.jit(grouped)(v, x, y, pts)):
        assert np.max(np.abs(np.asarray(out) - ref)) <= ALG_TOL * np.max(np.abs(ref))


@pytest.mark.parametrize("Nk,Nm,eps", PLANS)
@pytest.mark.parametrize("complex_", [False, True])
def test_off_cpu_spreads_match_reference(Nk, Nm, eps, complex_):
    """The per-cell scatter-add used off CPU (``platform_dependent`` default branch),
    traced and prepared, against the reference; gated here on CPU."""
    plan = _nufft.nufft_plan(Nk, Nm, eps)
    x, y = _clumped(plan, 200, Nk + 7)
    v = _vals(x.shape[0], complex_, 3)
    ref = np.asarray(_spread_reference(plan, v, x, y))
    lk, wk = _nufft._stencil(x, plan.nk, plan.W, plan.beta_k)
    lm, wm = _nufft._stencil(y, plan.nm, plan.W, plan.beta_m)
    blocks = (wk[:, :, None] * wm[:, None, :]) * v[:, None, None]
    traced = np.asarray(_nufft._scatter_cells(plan, blocks, lk, lm))
    pts = _nufft.prepare_points_plan((Nk, Nm, eps), np.asarray(x), np.asarray(y),
                                     group_size=16, store_weights=True)
    prepared = np.asarray(jax.jit(lambda *a: _nufft._spread_prepared_cells(plan, *a))(
        v, pts.wk, pts.wm, pts.base))
    for out in (traced, prepared):
        assert np.max(np.abs(out - ref)) <= ALG_TOL * np.max(np.abs(ref))
