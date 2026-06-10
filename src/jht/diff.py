"""Differentiable interface for the on-grid transforms.

jht's transforms are alm-linear and differentiate cleanly under JAX's **native**
autodiff -- no custom VJP/JVP rule is registered (the originally-considered
``custom_vjp`` blocks forward-mode AD; see ``docs/design.md`` for the convention
and why native AD is the supported path).  Reverse-mode (``grad``/``vjp``/
``jacrev``) returns the JAX-native cotangent, which equals ``G * conj(S^T .)`` with
the ``(2 - delta_m0)`` metric ``G`` (:func:`jht.healpix.alm_metric_weight`) -- i.e.
numerically identical to the validated :func:`jht.healpix.adjoint_synthesis`
kernel, and finite-difference consistent.

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

Library code does not enable x64; callers opt in per entry point.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from .analysis import map2alm
from .healpix import alm_metric_weight, synthesis
from .masked import alm_to_real, n_dof, real_to_alm
from .offgrid import adjoint_synthesis_general, synthesis_general

__all__ = [
    "synthesis_real",
    "analysis_real",
    "synthesis_general_real",
    "adjoint_synthesis_general_real",
    "bandpower",
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
    """``T o map2alm``: map -> real-DOF vector ``x`` (iterated approximate inverse).

    The real-DOF dual of :func:`jht.analysis.map2alm`; AD-clean in both modes.
    """
    a = map2alm(maps, nside, lmax, spin=spin, niter=niter, use_weights=use_weights)
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
