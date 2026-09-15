"""Differentiable interface for the on-grid transforms.

jht's transforms are alm-linear and differentiate cleanly under JAX's **native**
autodiff, which is the default (a ``custom_vjp`` on the plain transforms would block
forward-mode AD; see ``docs/design.md``).  Reverse-mode (``grad``/``vjp``/``jacrev``)
returns the JAX-native cotangent, which equals ``G * conj(S^T .)`` with the
``(2 - delta_m0)`` metric ``G`` (:func:`jht.healpix.alm_metric_weight`) -- i.e.
numerically identical to the validated :func:`jht.healpix.adjoint_synthesis` kernel,
and finite-difference consistent.

This module adds, on top of the complex transforms:

* a **real-DOF** layer (:func:`synthesis_real` / :func:`analysis_real`) that maps
  the complex healpy-packed a_lm onto the real isometry coordinates ``x`` of
  :mod:`jht.masked` (the isometry ``T``, ``||x||_2 = ||a||_w``), giving plain
  ``R^n -> R^m`` transforms with **no** complex-conjugate / ``2*conj`` convention
  subtlety: ``jacfwd == jacrev`` exactly and finite differences are unambiguous.
  This is the recommended differentiable interface for downstream optimisation /
  field-level inference.
* :func:`bandpower` -- the angular auto-power ``C_ell`` with the ``(2 - delta_m0)``
  fold, the natural scalar-valued head for a ``map -> a_lm -> C_ell`` pipeline.
* :func:`synthesis_vjp` / :func:`adjoint_synthesis_vjp` -- opt-in **reverse-only**
  transforms with a transpose-pair ``custom_vjp``: native AD's values and cotangents
  without its per-transform recursion tape, at the cost of forward mode. The
  memory-safe choice for ``grad`` at high nside.

Library code does not enable x64; callers opt in per entry point.
"""

from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from ._analysis import analysis
from .healpix import adjoint_synthesis, alm_metric_weight, alm_size, synthesis
from .masked import alm_to_real, n_dof, real_to_alm
from .offgrid import adjoint_synthesis_general, synthesis_general

__all__ = [
    "synthesis_real",
    "analysis_real",
    "synthesis_general_real",
    "adjoint_synthesis_general_real",
    "bandpower",
    "synthesis_vjp",
    "adjoint_synthesis_vjp",
    "alm_to_real",
    "real_to_alm",
    "n_dof",
]


# --------------------------------------------------------------------------- #
# real-DOF transforms (compose the complex transforms with the isometry T)
# --------------------------------------------------------------------------- #
def synthesis_real(x, nside: int, lmax: int, spin: int = 0) -> jax.Array:
    """``S o T^-1``: real-DOF vector ``x`` -> map.

    Plain real-linear (``R^n -> R^m``); ``jacfwd == jacrev`` and finite differences
    are unambiguous.  ``x`` has length :func:`n_dof(lmax, spin) <jht.masked.n_dof>`.
    """
    return synthesis(real_to_alm(x, lmax, spin), nside, lmax, spin)


def analysis_real(
    maps, nside: int, lmax: int, spin: int = 0, niter: int = 3, use_weights: bool = True
) -> jax.Array:
    """``T o analysis``: map -> real-DOF vector ``x`` (iterated approximate inverse).

    The real-DOF dual of :func:`jht.analysis`; AD-clean in both modes.
    """
    a = analysis(maps, nside, lmax, spin=spin, niter=niter, use_weights=use_weights)
    return alm_to_real(a, lmax, spin)


# --------------------------------------------------------------------------- #
# off-grid (NUFFT) real-DOF transforms -- the dual of synthesis_real for the
# arbitrary-pointing path (spin 0-3, no on-grid inverse so the adjoint is exposed)
# --------------------------------------------------------------------------- #
def synthesis_general_real(
    x, loc, *, spin: int = 0, lmax: int, epsilon: float = 1e-10
) -> jax.Array:
    """``S_g o T^-1``: real-DOF vector ``x`` -> field at arbitrary points ``loc``.

    The off-grid (NUFFT) dual of :func:`synthesis_real`: composes the real-DOF
    isometry ``T^-1`` (:func:`jht.masked.real_to_alm`) with
    :func:`jht.offgrid.synthesis_general`, giving a plain real-linear map (no
    complex-conjugate convention; ``jacfwd == jacrev``).  ``x`` has length
    :func:`n_dof(lmax, spin) <jht.masked.n_dof>`; ``loc`` is ``(npts, 2)`` of
    ``[theta, phi]``.  Returns ``(npts,)`` for ``spin=0`` or ``(2, npts)`` (Q, U)
    for ``spin=1..3``, exactly as :func:`jht.offgrid.synthesis_general`.
    """
    return synthesis_general(real_to_alm(x, lmax, spin), loc, spin=spin, lmax=lmax, epsilon=epsilon)


def adjoint_synthesis_general_real(
    field, loc, *, spin: int = 0, lmax: int, epsilon: float = 1e-10
) -> jax.Array:
    """``T o S_g^T``: field at points ``loc`` -> real-DOF vector ``x``.

    The exact transpose of :func:`synthesis_general_real` in the plain real inner
    products (``<S_g_real x, v>_2 == <x, S_g_real^T v>_2``): ``T`` is an isometry in
    the ``(2 - delta_m0)`` metric ``G``, so ``T o S_g^T`` is the Euclidean transpose
    of ``S_g o T^-1``.  Equivalently the native reverse-mode cotangent of
    :func:`synthesis_general_real` (real-linear, so VJP == transpose).
    """
    a = adjoint_synthesis_general(field, loc, spin=spin, lmax=lmax, epsilon=epsilon)
    return alm_to_real(a, lmax, spin)


# --------------------------------------------------------------------------- #
# angular auto-power C_ell  (the (2 - delta_m0) fold)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _ell_of_idx(lmax: int) -> np.ndarray:
    """ell for each healpy-packed a_lm index (m-major triangular order)."""
    return np.concatenate([np.arange(m, lmax + 1) for m in range(lmax + 1)])


def bandpower(alm, lmax: int, spin: int = 0) -> jax.Array:
    """Angular auto-power ``C_ell = (1/(2l+1)) sum_m (2 - delta_m0) |a_lm|^2``.

    The ``(2 - delta_m0)`` fold accounts for the implied ``m<0`` half of the
    healpy ``m>=0`` packing, matching ``healpy.anafast``.  ``spin=0`` returns shape
    ``(lmax+1,)``; ``spin=2`` returns ``(2, lmax+1)`` (``C_ell^EE``, ``C_ell^BB``).
    """
    ells = jnp.asarray(_ell_of_idx(lmax))
    weight = jnp.asarray(alm_metric_weight(lmax))
    norm = 1.0 / (2.0 * jnp.arange(lmax + 1) + 1.0)

    def _one(a: jax.Array) -> jax.Array:
        power = weight * jnp.abs(a) ** 2
        cl = jax.ops.segment_sum(power, ells, num_segments=lmax + 1)
        return cl * norm

    a = jnp.asarray(alm)
    if spin == 0:
        return _one(a)
    return jnp.stack([_one(a[0]), _one(a[1])])


# --------------------------------------------------------------------------- #
# reverse-only transforms: native AD's gradients without the recursion tape
# --------------------------------------------------------------------------- #
def synthesis_vjp(alm, nside: int, lmax: int, spin: int = 0) -> jax.Array:
    """:func:`jht.synthesis` with a transpose-pair ``custom_vjp`` (reverse mode only).

    Same values as :func:`jht.synthesis`. The VJP is one :func:`jht.adjoint_synthesis`
    call and equals native AD's cotangent, ``G * conj(S^T v)`` including the spin-2 m=0
    E/B term (``docs/design.md``). Native AD through the Legendre recursion keeps one
    ``(lmax+1, lmax+1, 2 nside)`` table per transform for the backward; this keeps none.
    ``jax.jvp`` / ``jacfwd`` raise. A real ``alm`` is cast to complex.
    """
    a = jnp.asarray(alm)
    if not jnp.iscomplexobj(a):
        a = a.astype(jnp.result_type(a.dtype, jnp.complex64))
    return _synthesis_vjp(a, int(nside), int(lmax), int(spin))


def adjoint_synthesis_vjp(maps, nside: int, lmax: int, spin: int = 0) -> jax.Array:
    """:func:`jht.adjoint_synthesis` with a transpose-pair ``custom_vjp`` (reverse mode only).

    Same values as :func:`jht.adjoint_synthesis`. The VJP is one :func:`jht.synthesis`
    call and equals native AD's cotangent; no recursion table is kept for the backward.
    ``jax.jvp`` / ``jacfwd`` raise. An integer ``maps`` is cast to float.
    """
    m = jnp.asarray(maps)
    if not jnp.issubdtype(m.dtype, jnp.floating):
        m = m.astype(jnp.result_type(float))
    return _adjoint_synthesis_vjp(m, int(nside), int(lmax), int(spin))


def _m0_mask(lmax: int) -> jax.Array:
    """``True`` at the m=0 a_lm entries (the first ``lmax+1``), built in-trace (jht#5)."""
    return jax.lax.optimization_barrier(jnp.arange(alm_size(lmax)) <= lmax)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _synthesis_vjp(alm, nside, lmax, spin):
    return synthesis(alm, nside, lmax, spin)


def _synthesis_vjp_fwd(alm, nside, lmax, spin):
    return synthesis(alm, nside, lmax, spin), None


def _synthesis_vjp_bwd(nside, lmax, spin, _res, v):
    b = adjoint_synthesis(v, nside, lmax, spin)
    m0 = _m0_mask(lmax)
    cot = jnp.where(m0, 1.0, 2.0) * jnp.conj(b)  # G * conj(S^T v)
    if spin == 0:
        return (cot,)
    # spin-2 m=0: the map depends on Im(a_l0) too, and native AD carries that direction
    bE, bB = jnp.real(b[0]), jnp.real(b[1])
    cot_E = jnp.where(m0, bE - 1j * bB, cot[0])
    cot_B = jnp.where(m0, bB + 1j * bE, cot[1])
    return (jnp.stack([cot_E, cot_B]),)


_synthesis_vjp.defvjp(_synthesis_vjp_fwd, _synthesis_vjp_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def _adjoint_synthesis_vjp(maps, nside, lmax, spin):
    return adjoint_synthesis(maps, nside, lmax, spin)


def _adjoint_synthesis_vjp_fwd(maps, nside, lmax, spin):
    return adjoint_synthesis(maps, nside, lmax, spin), None


def _adjoint_synthesis_vjp_bwd(nside, lmax, spin, _res, c):
    # the synthesis input whose map is the cotangent: Re(c) at m=0, conj(c)/2 at m>0
    a = jnp.where(_m0_mask(lmax), jnp.real(c).astype(c.dtype), jnp.conj(c) / 2.0)
    return (synthesis(a, nside, lmax, spin),)


_adjoint_synthesis_vjp.defvjp(_adjoint_synthesis_vjp_fwd, _adjoint_synthesis_vjp_bwd)
