"""Two-stage off-grid interface on prepared point sets.

``prepare_points`` / ``sky_to_grid`` / ``grid_to_points`` / ``points_to_grid`` /
``grid_to_sky``. Gates:

* the stage compositions equal ``synthesis_general`` / ``adjoint_synthesis_general``
  (spin 0-3, several group sizes, eager and compiled);
* the adjoint identity of each stage pair, in the ``(2 - delta_m0)`` metric on a_lm and
  the plain real inner product on the grid;
* points exactly on grid nodes, compiled, against the exact direct sum;
* stored vs recomputed kernel weights;
* pointing gradient through the prepared path == through ``synthesis_general``, and
  forward == reverse mode;
* a point outside its prepared cell: raises when concrete, NaN when traced; motion
  inside the cell stays exact;
* ``vmap`` over values, empty and single-point sets, input validation.

The ducc0 cross-checks are in the ``slow`` tier.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from test_nufft_stencil import _node_loc  # noqa: E402
from test_offgrid import (  # noqa: E402
    ADJ_TOL,
    DUCC_TOL,
    FWD_TOL,
    LOC_REV_TOL,
    _direct,
    _ducc_adjoint,
    _ducc_synth,
    _inner_alm,
    _rand_alm,
)

from jht.offgrid import (  # noqa: E402
    _prepare,
    adjoint_synthesis_general,
    grid_to_points,
    grid_to_sky,
    offgrid_grid_shape,
    points_to_grid,
    prepare_points,
    sky_to_grid,
    synthesis_general,
)

ALG_TOL = 1e-12  # algebraically identical sums, reordered (a-priori; measured ~1e-15)
TWO_PI = 2.0 * np.pi
SPINS = [0, 1, 2, 3]


def _mixed_loc(lmax, n, seed):
    """Dense clumps (many points per grid cell), uniform points, grid nodes, poles,
    phi = 0 and phi just below 2 pi."""
    rng = np.random.default_rng(seed)
    clump = np.stack([1.1 + 0.004 * rng.random(n), 4.0 + 0.004 * rng.random(n)], 1)
    uni = np.stack([np.arccos(rng.uniform(-1, 1, n)), rng.uniform(0, TWO_PI, n)], 1)
    edge = np.array([[0.0, 0.0], [np.pi, 0.0], [np.pi, TWO_PI - 1e-15], [1e-9, 3.0],
                     [np.pi / 2, 0.0], [np.pi / 2, TWO_PI - 1e-12]])
    return np.concatenate([clump, uni, _node_loc(lmax, 0, n // 4, seed + 1), edge])


def _field(npts, spin, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(npts) if spin == 0 else rng.standard_normal((2, npts)))


def _rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b))))


# --------------------------------------------------------------------------- #
# compositions == the existing transforms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", SPINS)
@pytest.mark.parametrize("group_size", [1, 3, 16])
def test_stages_compose_to_existing_transforms(spin, group_size):
    lmax = 40
    loc = _mixed_loc(lmax, 400, spin)
    pts = prepare_points(loc, lmax=lmax, group_size=group_size)
    alm, v = jnp.asarray(_rand_alm(lmax, 1, spin)), _field(loc.shape[0], spin, 2)
    f_ref = synthesis_general(alm, loc, spin=spin, lmax=lmax)
    b_ref = adjoint_synthesis_general(v, loc, spin=spin, lmax=lmax)

    grid = sky_to_grid(alm, spin=spin, lmax=lmax)
    assert grid.shape == offgrid_grid_shape(lmax)
    assert grid.dtype == (jnp.float64 if spin == 0 else jnp.complex128)
    assert _rel(grid_to_points(grid, pts, loc, spin=spin), f_ref) <= ALG_TOL
    assert _rel(grid_to_sky(points_to_grid(v, pts, loc, spin=spin), spin=spin, lmax=lmax), b_ref) <= ALG_TOL

    fwd = jax.jit(lambda a, p, x: grid_to_points(sky_to_grid(a, spin=spin, lmax=lmax), p, x, spin=spin))
    adj = jax.jit(lambda u, p, x: grid_to_sky(points_to_grid(u, p, x, spin=spin), spin=spin, lmax=lmax))
    lc = jnp.asarray(loc)
    assert _rel(fwd(alm, pts, lc), f_ref) <= ALG_TOL
    assert _rel(adj(v, pts, lc), b_ref) <= ALG_TOL


@pytest.mark.parametrize("spin", SPINS)
def test_grids_from_blocks_add(spin):
    """Summing the spread grids of point blocks, then one grid_to_sky, == the adjoint
    over all points."""
    lmax = 32
    loc = _mixed_loc(lmax, 300, 5)
    v = _field(loc.shape[0], spin, 6)
    cut = [0, 250, 700, loc.shape[0]]
    g = 0.0
    for a, b in zip(cut[:-1], cut[1:]):
        pts = prepare_points(loc[a:b], lmax=lmax)
        vb = v[a:b] if spin == 0 else v[:, a:b]
        g = g + points_to_grid(vb, pts, loc[a:b], spin=spin)
    ref = adjoint_synthesis_general(v, loc, spin=spin, lmax=lmax)
    assert _rel(grid_to_sky(g, spin=spin, lmax=lmax), ref) <= ALG_TOL


# --------------------------------------------------------------------------- #
# adjoint identities
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", SPINS)
def test_stage_adjoint_identities(spin):
    lmax = 40
    loc = _mixed_loc(lmax, 300, 7)
    pts = prepare_points(loc, lmax=lmax)
    rng = np.random.default_rng(8)
    nk, nm = offgrid_grid_shape(lmax)
    g = rng.standard_normal((nk, nm))
    if spin:
        g = g + 1j * rng.standard_normal((nk, nm))
    g = jnp.asarray(g)
    alm, v = jnp.asarray(_rand_alm(lmax, 9, spin)), _field(loc.shape[0], spin, 10)

    # sky <-> grid: Re<S a, g> == <a, S^T g>_G
    lhs = float(jnp.real(jnp.vdot(sky_to_grid(alm, spin=spin, lmax=lmax), g)))
    rhs = _inner_alm(np.asarray(alm), np.asarray(grid_to_sky(g, spin=spin, lmax=lmax)), lmax)
    assert abs(lhs - rhs) <= ADJ_TOL * abs(lhs)

    # grid <-> points: <P g, v> == Re<g, P^T v>
    lhs = float(jnp.sum(grid_to_points(g, pts, loc, spin=spin) * v))
    rhs = float(jnp.real(jnp.vdot(g, points_to_grid(v, pts, loc, spin=spin))))
    assert abs(lhs - rhs) <= ADJ_TOL * abs(lhs)


# --------------------------------------------------------------------------- #
# grid nodes, compiled, vs the exact direct sum
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
def test_compiled_on_grid_nodes_vs_direct(spin):
    lmax = 48
    alm, loc = _rand_alm(lmax, 5, spin), _node_loc(lmax, spin, 600, 6)
    pts = prepare_points(loc, lmax=lmax)
    fwd = jax.jit(lambda a, p, x: grid_to_points(sky_to_grid(a, spin=spin, lmax=lmax), p, x, spin=spin))
    f = np.asarray(fwd(jnp.asarray(alm), pts, jnp.asarray(loc)))
    assert np.max(np.abs(f - np.asarray(_direct(alm, loc, lmax, spin)))) <= FWD_TOL
    v = _field(loc.shape[0], spin, 7)
    adj = jax.jit(lambda u, p, x: grid_to_sky(points_to_grid(u, p, x, spin=spin), spin=spin, lmax=lmax))
    b = np.asarray(adj(v, pts, jnp.asarray(loc)))
    lhs, rhs = float(np.sum(f * np.asarray(v))), _inner_alm(alm, b, lmax)
    assert abs(lhs - rhs) <= ADJ_TOL * abs(lhs)
    plain = np.asarray(adjoint_synthesis_general(v, loc, spin=spin, lmax=lmax))
    assert _rel(b, plain) <= ALG_TOL


# --------------------------------------------------------------------------- #
# stored weights
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
def test_stored_weights_match_recomputed(spin):
    lmax = 40
    loc = _mixed_loc(lmax, 300, 11)
    pr = prepare_points(loc, lmax=lmax)
    ps = prepare_points(loc, lmax=lmax, store_weights=True)
    assert ps.stores_weights and not pr.stores_weights
    grid = sky_to_grid(jnp.asarray(_rand_alm(lmax, 12, spin)), spin=spin, lmax=lmax)
    v = _field(loc.shape[0], spin, 13)
    assert _rel(grid_to_points(grid, ps, spin=spin), grid_to_points(grid, pr, loc, spin=spin)) <= ALG_TOL
    assert _rel(points_to_grid(v, ps, spin=spin), points_to_grid(v, pr, loc, spin=spin)) <= ALG_TOL
    jitted = jax.jit(lambda u, p: points_to_grid(u, p, spin=spin))
    assert _rel(jitted(v, ps), points_to_grid(v, pr, loc, spin=spin)) <= ALG_TOL


# --------------------------------------------------------------------------- #
# pointing gradient
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", SPINS)
def test_loc_grad_matches_synthesis_general(spin):
    """d/dloc through the prepared forward and transposed paths == through the existing
    transforms (themselves gated against the analytic derivative)."""
    lmax = 32
    loc = jnp.asarray(_mixed_loc(lmax, 200, 14))
    pts = prepare_points(loc, lmax=lmax)
    alm, w = jnp.asarray(_rand_alm(lmax, 15, spin)), _field(loc.shape[0], spin, 16)
    grid = sky_to_grid(alm, spin=spin, lmax=lmax)

    def loss_new(x):
        return jnp.sum(w * grid_to_points(grid, pts, x, spin=spin))

    def loss_ref(x):
        return jnp.sum(w * synthesis_general(alm, x, spin=spin, lmax=lmax))

    assert _rel(jax.grad(loss_new)(loc), jax.grad(loss_ref)(loc)) <= ALG_TOL

    G = jnp.asarray(_rand_alm(lmax, 17, spin))

    def adj_new(x):
        return jnp.sum(jnp.real(jnp.conj(G) * grid_to_sky(points_to_grid(w, pts, x, spin=spin),
                                                         spin=spin, lmax=lmax)))

    def adj_ref(x):
        return jnp.sum(jnp.real(jnp.conj(G) * adjoint_synthesis_general(w, x, spin=spin, lmax=lmax)))

    # Like for like, eager with eager and compiled with compiled: on an exact grid node
    # the edge tap sits at the kernel's support boundary, where its derivative is
    # singular, and a compiled program's rounding of xi moves the gradient there by
    # ~1e-8 relative in either path (see DISCREPANCIES.md).
    assert _rel(jax.grad(adj_new)(loc), jax.grad(adj_ref)(loc)) <= ALG_TOL
    assert _rel(jax.jit(jax.grad(adj_new))(loc), jax.jit(jax.grad(adj_ref))(loc)) <= ALG_TOL

    t = jnp.asarray(np.random.default_rng(18).standard_normal(loc.shape))
    d_fwd = jax.jvp(loss_new, (loc,), (t,))[1]
    d_rev = jnp.sum(jax.grad(loss_new)(loc) * t)
    assert abs(float(d_fwd - d_rev)) <= LOC_REV_TOL * abs(float(d_fwd))


# --------------------------------------------------------------------------- #
# points outside their prepared cell
# --------------------------------------------------------------------------- #
def test_stale_points_raise_eager_and_nan_compiled():
    lmax = 32
    loc = _mixed_loc(lmax, 100, 19)
    pts = prepare_points(loc, lmax=lmax)
    nk, nm = offgrid_grid_shape(lmax)
    moved = loc.copy()
    moved[3, 1] += 1.5 * TWO_PI / nm  # one and a half cells along phi
    grid = sky_to_grid(jnp.asarray(_rand_alm(lmax, 20, 2)), spin=2, lmax=lmax)
    v = _field(loc.shape[0], 2, 21)
    with pytest.raises(ValueError, match="1 of"):
        grid_to_points(grid, pts, moved, spin=2)
    with pytest.raises(ValueError, match="outside the grid cells"):
        points_to_grid(v, pts, moved, spin=2)
    f = jax.jit(lambda g, p, x: grid_to_points(g, p, x, spin=2))(grid, pts, jnp.asarray(moved))
    b = jax.jit(lambda u, p, x: points_to_grid(u, p, x, spin=2))(v, pts, jnp.asarray(moved))
    assert np.all(np.isnan(np.asarray(f))) and np.all(np.isnan(np.asarray(b)))
    # the fresh positions stay finite under jit
    f_ok = jax.jit(lambda g, p, x: grid_to_points(g, p, x, spin=2))(grid, pts, jnp.asarray(loc))
    assert np.all(np.isfinite(np.asarray(f_ok)))


def test_motion_inside_cell_is_exact():
    """Points moved within their prepared cells give the result of a fresh preparation."""
    lmax = 32
    plan = _prepare(lmax, 0, 1e-10).nplan
    rng = np.random.default_rng(22)
    cells = np.stack([rng.integers(plan.nk // 16, plan.nk // 2 - plan.nk // 16, 400),
                      rng.integers(0, plan.nm, 400)], 1)
    a = (cells + rng.uniform(0.05, 0.45, (400, 2))) * [TWO_PI / plan.nk, TWO_PI / plan.nm]
    b = (cells + rng.uniform(0.55, 0.95, (400, 2))) * [TWO_PI / plan.nk, TWO_PI / plan.nm]
    pts_a = prepare_points(a, lmax=lmax)
    grid = sky_to_grid(jnp.asarray(_rand_alm(lmax, 23, 2)), spin=2, lmax=lmax)
    ref = synthesis_general(jnp.asarray(_rand_alm(lmax, 23, 2)), b, spin=2, lmax=lmax)
    assert _rel(grid_to_points(grid, pts_a, b, spin=2), ref) <= ALG_TOL


# --------------------------------------------------------------------------- #
# batching, degenerate sets, validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
def test_vmap_over_values(spin):
    lmax = 32
    loc = _mixed_loc(lmax, 150, 24)
    pts = prepare_points(loc, lmax=lmax)
    vs = jnp.stack([_field(loc.shape[0], spin, s) for s in range(3)])
    batched = jax.vmap(lambda u: points_to_grid(u, pts, loc, spin=spin))(vs)
    looped = jnp.stack([points_to_grid(u, pts, loc, spin=spin) for u in vs])
    assert _rel(batched, looped) <= ALG_TOL


@pytest.mark.parametrize("spin", [0, 2])
def test_empty_and_single_point(spin):
    lmax = 24
    nk, nm = offgrid_grid_shape(lmax)
    grid = sky_to_grid(jnp.asarray(_rand_alm(lmax, 25, spin)), spin=spin, lmax=lmax)
    empty = prepare_points(np.zeros((0, 2)), lmax=lmax)
    f = grid_to_points(grid, empty, np.zeros((0, 2)), spin=spin)
    assert f.shape == ((0,) if spin == 0 else (2, 0))
    g = points_to_grid(_field(0, spin, 0), empty, np.zeros((0, 2)), spin=spin)
    assert g.shape == (nk, nm) and not np.any(np.asarray(g))
    one = np.array([[0.7, 5.9]])
    p1 = prepare_points(one, lmax=lmax)
    v1 = _field(1, spin, 26)
    ref = adjoint_synthesis_general(v1, one, spin=spin, lmax=lmax)
    assert _rel(grid_to_sky(points_to_grid(v1, p1, one, spin=spin), spin=spin, lmax=lmax), ref) <= ALG_TOL


def test_input_validation():
    lmax = 24
    loc = _mixed_loc(lmax, 50, 27)
    pts = prepare_points(loc, lmax=lmax)
    nk, nm = offgrid_grid_shape(lmax)
    with pytest.raises(ValueError, match="loc is required"):
        grid_to_points(jnp.zeros((nk, nm)), pts, spin=0)
    with pytest.raises(ValueError, match="grid must have shape"):
        grid_to_points(jnp.zeros((nk + 1, nm)), pts, loc, spin=0)
    with pytest.raises(TypeError, match="spin-0 grid is real"):
        grid_to_points(jnp.zeros((nk, nm), jnp.complex128), pts, loc, spin=0)
    with pytest.raises(ValueError, match="loc must have shape"):
        grid_to_points(jnp.zeros((nk, nm)), pts, loc[:-1], spin=0)
    with pytest.raises(ValueError, match="field shape"):
        points_to_grid(jnp.zeros(loc.shape[0]), pts, loc, spin=2)
    with pytest.raises(TypeError, match="must be real"):
        points_to_grid(jnp.zeros(loc.shape[0], jnp.complex128), pts, loc, spin=0)
    with pytest.raises(TypeError, match="prepare_points"):
        grid_to_points(jnp.zeros((nk, nm)), loc, loc, spin=0)
    with pytest.raises(ValueError, match="group_size"):
        prepare_points(loc, lmax=lmax, group_size=0)
    with pytest.raises(TypeError, match="outside jax.jit"):
        jax.jit(lambda x: prepare_points(x, lmax=lmax))(jnp.asarray(loc))


# --------------------------------------------------------------------------- #
# ducc0 cross-checks (slow)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize("spin", SPINS)
def test_stages_vs_ducc(spin):
    pytest.importorskip("ducc0")
    lmax = 128
    loc = _mixed_loc(lmax, 2000, 28)
    loc[:, 0] = np.clip(loc[:, 0], 0.0, np.pi)
    pts = prepare_points(loc, lmax=lmax)
    alm = _rand_alm(lmax, 29, spin)
    f = np.asarray(grid_to_points(sky_to_grid(jnp.asarray(alm), spin=spin, lmax=lmax), pts, loc, spin=spin))
    d = _ducc_synth(alm, loc, spin, lmax)
    assert np.max(np.abs(f - (d[0] if spin == 0 else d))) <= DUCC_TOL
    v = np.asarray(_field(loc.shape[0], spin, 30))
    b = np.atleast_2d(np.asarray(grid_to_sky(points_to_grid(v, pts, loc, spin=spin), spin=spin, lmax=lmax)))
    db = np.atleast_2d(_ducc_adjoint(v, loc, spin, lmax))
    keep = np.arange(b.shape[1]) >= (lmax + 1) if spin != 0 else np.ones(b.shape[1], bool)
    rel = max(np.max(np.abs(b[c][keep] - db[c][keep])) / np.max(np.abs(db[c][keep]))
              for c in range(b.shape[0]))
    assert rel <= DUCC_TOL


# --------------------------------------------------------------------------- #
# real-DOF stage wrappers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", SPINS)
def test_real_dof_stage_wrappers(spin):
    from jht.diff import grid_to_sky_real, sky_to_grid_real, synthesis_general_real
    from jht.masked import n_dof

    lmax = 24
    rng = np.random.default_rng(31)
    x = jnp.asarray(rng.standard_normal(n_dof(lmax, spin)))
    nk, nm = offgrid_grid_shape(lmax)
    g = rng.standard_normal((nk, nm))
    if spin:
        g = g + 1j * rng.standard_normal((nk, nm))
    g = jnp.asarray(g)
    # Euclidean transpose, Re<.,.> on the grid
    lhs = float(jnp.real(jnp.vdot(sky_to_grid_real(x, spin=spin, lmax=lmax), g)))
    rhs = float(jnp.dot(x, grid_to_sky_real(g, spin=spin, lmax=lmax)))
    assert abs(lhs - rhs) <= ADJ_TOL * abs(lhs)
    # JAX's cotangent: the transpose itself for the real spin-0 grid, of conj(g) otherwise
    _, vjp = jax.vjp(lambda x: sky_to_grid_real(x, spin=spin, lmax=lmax), x)
    assert _rel(vjp(jnp.conj(g))[0], grid_to_sky_real(g, spin=spin, lmax=lmax)) <= ALG_TOL
    # composed with grid_to_points == synthesis_general_real; jacfwd == jacrev end to end
    loc = _mixed_loc(lmax, 40, 32)
    pts = prepare_points(loc, lmax=lmax)

    def f(x):
        return grid_to_points(sky_to_grid_real(x, spin=spin, lmax=lmax), pts, loc, spin=spin)

    assert _rel(f(x), synthesis_general_real(x, loc, spin=spin, lmax=lmax)) <= ALG_TOL
    Jf, Jr = jax.jacfwd(f)(x), jax.jacrev(f)(x)
    assert float(jnp.max(jnp.abs(Jf - Jr))) <= ALG_TOL * float(jnp.max(jnp.abs(Jf)))
