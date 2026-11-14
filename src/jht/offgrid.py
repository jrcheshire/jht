"""Off-grid spherical harmonic synthesis -- ``synthesis_general`` / its adjoint.

Evaluate a band-limited field at *arbitrary* points (detector pointings) on the
sphere: a JAX-native, differentiable counterpart of ducc0's
``synthesis_general`` / ``adjoint_synthesis_general``.

Algorithm (matches ducc's ``sphere_interpol``): **Double-Fourier-Sphere + 2D
type-2 NUFFT**.  ``alm`` --(Legendre step at a Clenshaw-Curtis theta grid; reuses
``synth_contract``)--> per-m field coefficients --(DFS theta-extension + theta-FFT)-->
2D Fourier coefficients ``F(k,m)`` --(:mod:`jht._nufft`)--> field at the points.
The adjoint is the exact transpose (type-1 NUFFT + transposed DFS + ``adjoint_contract``).

Conventions are inherited from the on-grid core (healpy m-major triangular alm,
orthonormal Y, the ``(2 - delta_m0)`` metric and the E,B<->Q,U sign -- see
:mod:`jht.healpix`).  Accuracy is a deliberately new tier set by the NUFFT
``epsilon`` (default 1e-10, matching ducc) -- *not* the on-grid 1e-13.  Both the
alm gradient (native AD == ``G*conj(adjoint)``) and the **pointing gradient**
(``d field / d loc``, the capability ducc's FFI cannot provide) are exact under
JAX's native autodiff.  spin 0, 1, 2, 3 (the -spin channel carries (-1)^s).
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import _nufft
from ._nufft import NufftPlan, PreparedPoints, _next_size, nufft2d1, nufft2d2, nufft_plan
from ._recursion import RecursionPlan, adjoint_contract, build_recursion_plan, synth_contract
from .healpix import _tri_dense_maps, alm_metric_weight, alm_size

TWO_PI = 2.0 * np.pi


class _OffgridPrep(NamedTuple):
    lmax: int
    spin: int
    ncomp: int  # 1 (spin-0) or 2 (spin>0)
    ntheta_s: int
    ntheta_d: int
    x_cc: np.ndarray  # cos(theta) at the Clenshaw-Curtis rings (n_theta_s,)
    plan_pos: RecursionPlan  # +spin recursion plan at the CC grid
    plan_neg: RecursionPlan  # -spin recursion plan (== plan_pos for spin 0)
    nplan: NufftPlan
    sign_m: np.ndarray  # (2lmax+1,) DFS parity (-1)^m * (-1)^s over m=-lmax..lmax
    south_i: np.ndarray  # (ntheta_d - ntheta_s,) doubled-grid south indices
    refl: np.ndarray  # their reflected north indices
    gather: np.ndarray  # tri<->dense index maps
    valid: np.ndarray
    pack: np.ndarray  # dense.ravel() -> tri gather index (dense_to_tri)
    gmetric: np.ndarray  # (alm_size,) the (2 - delta_m0) weight


@lru_cache(maxsize=None)
def _prepare(lmax: int, spin: int, epsilon: float) -> _OffgridPrep:
    if spin not in (0, 1, 2, 3):
        raise NotImplementedError(f"spin={spin} unsupported (0, 1, 2, 3)")
    # lru_cache => warns once per (lmax, spin, epsilon); getattr: the attr is dynamic (mypy)
    if not getattr(jax.config, "jax_enable_x64", True):
        warnings.warn(
            "jht: jax_enable_x64 is OFF -- the off-grid transforms will silently run in "
            "float32, far below the requested NUFFT epsilon. Enable it before creating "
            'any array: jax.config.update("jax_enable_x64", True).',
            stacklevel=2,
        )
    ntheta_s = _next_size(lmax + 1) + 1  # CC rings on [0, pi], incl. both poles
    theta_cc = np.arange(ntheta_s) * np.pi / (ntheta_s - 1)
    x_cc = np.cos(theta_cc)
    plan_pos = build_recursion_plan(x_cc, spin, lmax)
    plan_neg = plan_pos if spin == 0 else build_recursion_plan(x_cc, -spin, lmax)
    ntheta_d = 2 * ntheta_s - 2  # DFS period
    nplan = nufft_plan(ntheta_d, 2 * lmax + 1, epsilon)
    mvals = np.arange(-lmax, lmax + 1)
    sign_m = ((-1.0) ** np.abs(mvals)) * ((-1.0) ** spin)
    south_i = np.arange(ntheta_s, ntheta_d)
    refl = ntheta_d - south_i
    gather, valid, pack = _tri_dense_maps(lmax)
    return _OffgridPrep(
        lmax, spin, 1 if spin == 0 else 2, ntheta_s, ntheta_d, x_cc,
        plan_pos, plan_neg, nplan, sign_m, south_i, refl,
        gather, valid, pack, alm_metric_weight(lmax),
    )


def _tri_to_dense(p: _OffgridPrep, alm):
    return jnp.where(jnp.asarray(p.valid), jnp.asarray(alm)[jnp.asarray(p.gather)], 0.0 + 0.0j)


def _dense_to_tri(p: _OffgridPrep, dense):
    return dense.ravel()[jnp.asarray(p.pack)]  # gather, not scatter (fp64 GPU; see healpix)


def _channels(p: _OffgridPrep, alm):
    """alm -> (alpha, beta) dense inputs for the +spin / -spin Legendre channels.

    spin-0: a single real field, both channels are the same a_lm.
    spin-2: the +-2 fields, ``+2a = -(aE + i aB)``, ``-2a = -(aE - i aB)``.
    """
    if p.spin == 0:
        d = _tri_to_dense(p, alm)
        return d, d
    aE, aB = _tri_to_dense(p, alm[0]), _tri_to_dense(p, alm[1])
    s_sign = (-1.0) ** p.spin  # the -spin channel carries (-1)^s (hidden for even spin)
    return -(aE + 1j * aB), -s_sign * (aE - 1j * aB)


def _channels_adjoint(p: _OffgridPrep, p_cot, n_cot):
    """Transpose of :func:`_channels`: (alpha_cot, beta_cot) dense -> alm cotangent."""
    if p.spin == 0:
        return _dense_to_tri(p, p_cot + n_cot)
    # (aE,aB) -> (-(aE+i aB), -(-1)^s (aE-i aB)) has Hermitian adjoint
    #   aE_cot = -(p_cot + s n_cot),  aB_cot = i (p_cot - s n_cot),  s = (-1)^spin.
    s_sign = (-1.0) ** p.spin
    aE = _dense_to_tri(p, -(p_cot + s_sign * n_cot))
    aB = _dense_to_tri(p, 1j * (p_cot - s_sign * n_cot))
    return jnp.stack([aE, aB])


def _coeffs_to_Fkm(p: _OffgridPrep, Cpos, Cneg):
    """+spin / -spin per-m field coeffs (m>=0) -> DFS-extended 2D Fourier coeffs F(k,m)."""
    lmax = p.lmax
    Cfull = jnp.zeros((2 * lmax + 1, p.ntheta_s), dtype=jnp.complex128)
    Cfull = Cfull.at[lmax:].set(Cpos)  # m = 0..lmax (the +spin field)
    Cfull = Cfull.at[:lmax].set(jnp.conj(Cneg[1:][::-1]))  # m = -lmax..-1 = conj(-spin field)
    Dd = jnp.zeros((p.ntheta_d, 2 * lmax + 1), dtype=jnp.complex128)
    Dd = Dd.at[: p.ntheta_s].set(Cfull.T)  # north hemisphere [0, pi]
    Dd = Dd.at[jnp.asarray(p.south_i)].set(
        jnp.asarray(p.sign_m)[None, :] * Cfull.T[jnp.asarray(p.refl)]
    )
    return jnp.fft.fftshift(jnp.fft.fft(Dd, axis=0), axes=0) / p.ntheta_d


def _Fkm_to_coeffs(p: _OffgridPrep, G):
    """Hermitian adjoint of :func:`_coeffs_to_Fkm`: F(k,m) cotangent -> (Cpos_cot, Cneg_cot)."""
    lmax = p.lmax
    dDd = jnp.fft.ifft(jnp.fft.ifftshift(G, axes=0), axis=0)  # adj of fftshift(fft)/Nd
    dCfullT = jnp.zeros((p.ntheta_s, 2 * lmax + 1), dtype=jnp.complex128)
    dCfullT = dCfullT.at[:].add(dDd[: p.ntheta_s])
    dCfullT = dCfullT.at[jnp.asarray(p.refl)].add(
        jnp.asarray(p.sign_m)[None, :] * dDd[jnp.asarray(p.south_i)]
    )
    dCfull = dCfullT.T  # (2lmax+1, ntheta_s)
    Cpos_cot = dCfull[lmax:]  # m = 0..lmax
    Cneg_cot = jnp.zeros((lmax + 1, p.ntheta_s), dtype=jnp.complex128)
    Cneg_cot = Cneg_cot.at[1:].set(jnp.conj(dCfull[:lmax][::-1]))  # m = 1..lmax
    return Cpos_cot, Cneg_cot


def _alm_to_Fkm(p: _OffgridPrep, alm):
    a, b = _channels(p, alm)
    Cpos = synth_contract(p.plan_pos, jnp.asarray(p.x_cc), a)  # (lmax+1, ntheta_s)
    # spin-0: plan_neg is plan_pos and a == b, so the -spin recursion is identical -- skip it.
    Cneg = Cpos if p.spin == 0 else synth_contract(p.plan_neg, jnp.asarray(p.x_cc), b)
    return _coeffs_to_Fkm(p, Cpos, Cneg)


def _Fkm_to_alm(p: _OffgridPrep, G):
    Cpos_cot, Cneg_cot = _Fkm_to_coeffs(p, G)
    p_cot = adjoint_contract(p.plan_pos, jnp.asarray(p.x_cc), Cpos_cot)
    n_cot = adjoint_contract(p.plan_neg, jnp.asarray(p.x_cc), Cneg_cot)
    return _channels_adjoint(p, p_cot, n_cot)


def _check_alm(alm, lmax: int, spin: int):
    alm = jnp.asarray(alm)
    expect = (alm_size(lmax),) if spin == 0 else (2, alm_size(lmax))
    if alm.shape != expect:
        raise ValueError(
            f"alm shape {alm.shape} != expected {expect} for lmax={lmax}, spin={spin} "
            "(a wrong-size alm would otherwise be silently clamped by the gather)"
        )
    return alm


# --------------------------------------------------------------------------- #
# public transforms
# --------------------------------------------------------------------------- #
def synthesis_general(alm, loc, *, spin: int = 0, lmax: int, epsilon: float = 1e-10):
    """``a_lm -> field`` at arbitrary points ``loc`` (the off-grid synthesis).

    Parameters
    ----------
    alm : complex array
        healpy-packed coefficients: ``(alm_size(lmax),)`` for ``spin=0``;
        ``(2, alm_size(lmax))`` (E, B) for ``spin=2``.
    loc : real array ``(npts, 2)``
        ``loc[:,0] = theta`` (colatitude in ``[0, pi]``), ``loc[:,1] = phi`` (in
        ``[0, 2pi)``).  Out-of-range angles are not validated: both coordinates
        are evaluated on the smooth period-2pi torus extension (theta through the
        double-Fourier-sphere identity ``F(-theta, phi) = (-1)^spin F(theta,
        phi + pi)``), so e.g. ``theta = -0.1`` returns the field at the reflected
        point, not an error.
    spin : int
        ``0`` (temperature) or ``2`` (polarization Q/U).
    lmax : int
    epsilon : float
        NUFFT accuracy (default 1e-10, matching ducc's production setting).

    Returns
    -------
    field : real array ``(npts,)`` for ``spin=0``, or ``(2, npts)`` (Q, U) for ``spin=2``.
    """
    p = _prepare(lmax, spin, epsilon)
    loc = jnp.asarray(loc)
    if loc.ndim != 2 or loc.shape[-1] != 2:
        raise ValueError(f"loc must have shape (npts, 2) [theta, phi]; got {loc.shape}")
    alm = _check_alm(alm, lmax, spin)
    theta, phi = loc[:, 0], loc[:, 1]
    field = nufft2d2(p.nplan, _alm_to_Fkm(p, alm), theta, phi)
    if p.spin == 0:
        return jnp.real(field)
    return jnp.stack([jnp.real(field), jnp.imag(field)])  # (Q, U)


def adjoint_synthesis_general(field, loc, *, spin: int = 0, lmax: int, epsilon: float = 1e-10):
    """Exact transpose of :func:`synthesis_general` (``field -> a_lm``).

    Satisfies ``<S_g a, v>_pts == <a, S_g^T v>_G`` with the ``(2 - delta_m0)``
    metric ``G`` -- the same strict-adjoint / VJP-bridge convention as the on-grid
    :func:`jht.healpix.adjoint_synthesis`.  ``field`` is ``(npts,)`` for ``spin=0``
    or ``(2, npts)`` (Q, U) for ``spin=2``.
    """
    p = _prepare(lmax, spin, epsilon)
    loc = jnp.asarray(loc)
    if loc.ndim != 2 or loc.shape[-1] != 2:
        raise ValueError(f"loc must have shape (npts, 2) [theta, phi]; got {loc.shape}")
    field = jnp.asarray(field)
    if jnp.iscomplexobj(field):
        raise TypeError(
            "field must be real ((npts,) for spin=0; (2, npts) real Q/U planes for spin>0); "
            "a complex field would be silently mis-folded"
        )
    npts = loc.shape[0]
    expect = (npts,) if spin == 0 else (2, npts)
    if field.shape != expect:
        raise ValueError(f"field shape {field.shape} != expected {expect} for {npts} points")
    theta, phi = loc[:, 0], loc[:, 1]
    # transpose of the (Re) / (Re, Im) field reduction; spin-0 spreads a real grid
    fc = field if p.spin == 0 else (field[0] + 1j * field[1])
    fkm_cot = nufft2d1(p.nplan, fc, theta, phi)
    # _Fkm_to_alm is the raw C^K Hermitian transpose (== conj(vjp)); the strict
    # adjoint in jht's (2 - delta_m0) metric divides by G (m<0 modes are implicit).
    return _Fkm_to_alm(p, fkm_cot) / jnp.asarray(p.gmetric)


# --------------------------------------------------------------------------- #
# two-stage interface: sky <-> NUFFT grid <-> points
# --------------------------------------------------------------------------- #
DEFAULT_GROUP_SIZE = 16


def offgrid_grid_shape(lmax: int, epsilon: float = 1e-10) -> tuple[int, int]:
    """Shape ``(nk, nm)`` of the oversampled NUFFT grid used at ``(lmax, epsilon)``."""
    plan = _prepare(lmax, 0, epsilon).nplan
    return plan.nk, plan.nm


def prepare_points(
    loc, *, lmax: int, epsilon: float = 1e-10, store_weights: bool = False,
    group_size: int | None = None,
) -> PreparedPoints:
    """Prepare a point set for :func:`grid_to_points` / :func:`points_to_grid`.

    Host-side (NumPy), once per point set; call it outside ``jax.jit``.  Fixes each
    point's NUFFT stencil cell and groups the points by cell, so the transposed step
    writes one ``(W, W)`` block per group instead of one per point.  Valid for any
    pointings, and for every spin at this ``(lmax, epsilon)``.

    Parameters
    ----------
    loc : ``(npts, 2)`` real, ``[theta, phi]`` as in :func:`synthesis_general`.
    lmax, epsilon : the transform the points will be used with.
    store_weights : keep the kernel weights (``2 W`` float64 per point, 224 B at the
        default epsilon) so calls need no ``loc``.  Calls without ``loc`` carry no
        gradient with respect to the pointing.
    group_size : the largest group (default 16).  Cells with fewer points use groups
        of the next power of two, so sparse and dense sets both pad less than 2x.

    The returned :class:`PreparedPoints` is a pytree; pass it into compiled functions
    as an argument.
    """
    if isinstance(loc, jax.core.Tracer):
        raise TypeError("prepare_points runs on the host: call it outside jax.jit / grad")
    loc = np.asarray(loc, dtype=np.float64)
    if loc.ndim != 2 or loc.shape[-1] != 2:
        raise ValueError(f"loc must have shape (npts, 2) [theta, phi]; got {loc.shape}")
    p = _prepare(lmax, 0, epsilon)
    return _nufft.prepare_points_plan(
        (p.ntheta_d, 2 * lmax + 1, epsilon), loc[:, 0], loc[:, 1],
        group_size=DEFAULT_GROUP_SIZE if group_size is None else int(group_size),
        store_weights=store_weights,
    )


def _check_points(points) -> NufftPlan:
    if not isinstance(points, PreparedPoints):
        raise TypeError(f"points must come from prepare_points; got {type(points).__name__}")
    return points.plan


def _point_weights(points: PreparedPoints, plan: NufftPlan, loc):
    """Kernel weights ``(wk, wm)`` and a scalar ``ok`` (False if a point left its cell).

    With ``loc`` the weights are recomputed at the prepared bases (differentiable in
    ``loc``); without, the stored weights are used.  A concrete stale ``loc`` raises.
    """
    if loc is None:
        if not points.stores_weights:
            raise ValueError(
                "loc is required: these points were prepared without store_weights=True"
            )
        return points.wk, points.wm, None
    loc = jnp.asarray(loc)
    if loc.shape != (points.npts, 2):
        raise ValueError(f"loc must have shape ({points.npts}, 2) [theta, phi]; got {loc.shape}")
    x, y = loc[:, 0], loc[:, 1]
    bk, bm = points.base[:, 0], points.base[:, 1]
    inside = (_nufft._in_prepared_cell(x, bk, plan.nk, plan.W)
              & _nufft._in_prepared_cell(y, bm, plan.nm, plan.W))
    ok = jnp.all(inside)
    try:
        concrete = bool(ok)
    except jax.errors.ConcretizationTypeError:  # traced: poison the output instead
        concrete = None
    if concrete is False:
        raise ValueError(
            f"{int(np.sum(~np.asarray(inside)))} of {points.npts} points are outside the "
            "grid cells they were prepared in: call prepare_points with the new loc"
        )
    wk = _nufft._weights_from_base(x, bk, plan.nk, plan.W, plan.beta_k)
    wm = _nufft._weights_from_base(y, bm, plan.nm, plan.W, plan.beta_m)
    return wk, wm, (None if concrete else ok)


def _poison(out, ok):
    return out if ok is None else jnp.where(ok, out, jnp.nan)


def _check_grid(grid, plan: NufftPlan, spin: int):
    grid = jnp.asarray(grid)
    if grid.shape != (plan.nk, plan.nm):
        raise ValueError(f"grid must have shape ({plan.nk}, {plan.nm}); got {grid.shape}")
    if spin == 0 and jnp.iscomplexobj(grid):
        raise TypeError("the spin-0 grid is real; got a complex array")
    return grid


def sky_to_grid(alm, *, spin: int = 0, lmax: int, epsilon: float = 1e-10):
    """``a_lm`` -> the oversampled NUFFT grid (the point-independent half of
    :func:`synthesis_general`).

    Legendre step, double-Fourier-sphere extension, kernel deconvolution and inverse FFT.
    Returns ``offgrid_grid_shape(lmax, epsilon)``, real float64 for ``spin=0`` and
    complex128 for ``spin>0``; :func:`grid_to_points` evaluates it at a point set.  The
    grid is the NUFFT's internal torus grid, not a HEALPix map.
    """
    p = _prepare(lmax, spin, epsilon)
    alm = _check_alm(alm, lmax, spin)
    g = _nufft._modes_to_grid(p.nplan, _alm_to_Fkm(p, alm))
    return jnp.real(g) if spin == 0 else g


def grid_to_sky(grid, *, spin: int = 0, lmax: int, epsilon: float = 1e-10):
    """Exact transpose of :func:`sky_to_grid`: grid -> ``a_lm``.

    Transpose in the ``(2 - delta_m0)`` metric on ``a_lm`` and the plain inner product
    ``Re sum conj(g1) g2`` on the grid, so
    ``adjoint_synthesis_general(v, loc) == grid_to_sky(points_to_grid(v, points, loc))``.
    Grids from several point sets may be summed first.
    """
    p = _prepare(lmax, spin, epsilon)
    grid = _check_grid(grid, p.nplan, spin)
    return _Fkm_to_alm(p, _nufft._grid_to_modes(p.nplan, grid)) / jnp.asarray(p.gmetric)


def grid_to_points(grid, points: PreparedPoints, loc=None, *, spin: int = 0):
    """Evaluate a :func:`sky_to_grid` grid at a prepared point set.

    The points must be prepared with the grid's ``lmax`` and ``epsilon``: the grid shape
    is checked, but it depends on ``lmax`` only, so an ``epsilon`` mismatch is not
    detected.  ``loc`` must be the positions the points were prepared from (or points still in
    the same NUFFT grid cells); the result is differentiable in it.  ``loc=None`` uses
    stored weights (``prepare_points(..., store_weights=True)``).  A point outside its
    prepared cell raises when ``loc`` is concrete and turns the whole output into NaN
    under a trace.  Returns ``(npts,)`` for ``spin=0`` or ``(2, npts)`` otherwise, as
    :func:`synthesis_general`.
    """
    plan = _check_points(points)
    grid = _check_grid(grid, plan, spin)
    wk, wm, ok = _point_weights(points, plan, loc)
    f = _nufft._interp_prepared(plan, grid, wk, wm, points.base)
    out = f if spin == 0 else jnp.stack([jnp.real(f), jnp.imag(f)])
    return _poison(out, ok)


def points_to_grid(field, points: PreparedPoints, loc=None, *, spin: int = 0):
    """Exact transpose of :func:`grid_to_points`: spread point values onto the grid.

    ``field`` is ``(npts,)`` real for ``spin=0`` or ``(2, npts)`` otherwise.  Returns
    the grid (real for ``spin=0``); grids from several point sets add before one
    :func:`grid_to_sky`.  ``loc`` as in :func:`grid_to_points`.
    """
    plan = _check_points(points)
    field = jnp.asarray(field)
    if jnp.iscomplexobj(field):
        raise TypeError("field must be real ((npts,) for spin=0; (2, npts) for spin>0)")
    expect = (points.npts,) if spin == 0 else (2, points.npts)
    if field.shape != expect:
        raise ValueError(f"field shape {field.shape} != expected {expect}")
    wk, wm, ok = _point_weights(points, plan, loc)
    v = field if spin == 0 else field[0] + 1j * field[1]
    return _poison(_nufft._spread_grouped(plan, v, wk, wm, points.buckets), ok)
