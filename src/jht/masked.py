"""Partial-sky / masked HEALPix analysis: the cut-sky map -> a_lm operators.

Two distinct estimators for a per-pixel sky ``mask`` (RING order, binary 0/1 or
apodized in ``[0,1]``):

1. :func:`pseudo_alm` -- the **masked pseudo-a_lm** (zero-fill): mask the *map*,
   keep the *full* quadrature weights, run the existing full-residual Jacobi
   (:func:`jht.analysis.map2alm`).  This is the standard CMB pseudo-a_lm; with
   uniform weights (``use_weights=False``) it is exactly
   ``(4pi/Npix) S^T (M m)`` -- the weight-unambiguous estimator that healpy/ducc
   compute.  It is *biased* by mode-coupling (the cut mixes ell, m): it does not
   recover the true a_lm, and is not meant to.

2. :func:`deconvolve` -- the **cut-sky deconvolution**: solve the masked normal
   equations ``A a = b`` with conjugate gradient, where

       A = S^T (M W) S,    b = S^T (M W) m .

   ``A`` is the masked least-squares operator; it reduces to ``S^T W S ~ I`` as
   ``M -> 1`` (so CG -> full-sky :func:`map2alm` in ~1 step).  For noiseless
   band-limited data ``m = S a_true`` it recovers ``a = a_true`` *exactly* and
   *independently of W* wherever the cut leaves the modes constrained (where ``A``
   restricted to the active band is well-conditioned).  Near-null modes
   (supported almost entirely under the cut -- e.g. spin-2 E/B ambiguous modes)
   are not recoverable from the data; CG from ``x0=0`` returns the minimum-norm
   solution, and an optional Tikhonov ``reg`` stabilises hard masks.

The inner-product subtlety (the bk-jax ``2*conj`` gotcha): ``S``/``S^T`` are
adjoint in the ``(2 - delta_{m0})``-weighted a_lm inner product, not the
Euclidean one (see :func:`jht.healpix.adjoint_synthesis`).  So ``A`` is
Hermitian-PD under ``<.,.>_w``, not Euclidean.  We handle it once with a
real-DOF **isometry** ``T: a <-> x`` (:func:`alm_to_real` / :func:`real_to_alm`)
with ``||x||_2 = ||a||_w``: ``a_{l0} -> a_{l0}`` (real), ``a_{lm>0} -> [sqrt2 Re,
sqrt2 Im]``, dropping the structurally-zero ``l < |spin|`` modes.  In ``x``-space
``A_x = T A T^-1`` is real-symmetric-PD in plain Euclidean, so stock
:func:`jax.scipy.sparse.linalg.cg` is exactly correct.

Library code does not enable x64; callers opt in per entry point (x64 is needed
for the deep accuracy here, as elsewhere in jht).
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

from .analysis import map2alm
from .healpix import adjoint_synthesis, alm_size, synthesis
from .weights import pixel_weights

_SQRT2 = float(np.sqrt(2.0))


# --------------------------------------------------------------------------- #
# combined mask * quadrature weight
# --------------------------------------------------------------------------- #
def _mw(nside: int, spin: int, mask, use_weights: bool) -> jax.Array:
    """Per-pixel ``M * W`` (mask times quadrature weight), broadcastable to the map.

    ``mask`` is a sky mask ``(Npix,)`` (same for Q and U); returns ``(Npix,)`` for
    spin 0, ``(1, Npix)`` for spin 2.
    """
    mw = jnp.asarray(mask) * jnp.asarray(pixel_weights(nside, use_weights))
    return mw if spin == 0 else mw[None, :]


# --------------------------------------------------------------------------- #
# masked pseudo-a_lm (zero-fill)
# --------------------------------------------------------------------------- #
def pseudo_alm(maps, mask, nside: int, lmax: int, spin: int = 0, niter: int = 3, use_weights: bool = True) -> jax.Array:
    """Masked pseudo-a_lm: ``map2alm(M * map)`` (zero-fill, healpy/ducc-style).

    With ``use_weights=False, niter=0`` this is the canonical uniform pseudo-a_lm
    ``(4pi/Npix) S^T (M m)``; with ring weights / iteration it is jht's own
    (weight-conditioned) pseudo-a_lm.  Biased by mode-coupling -- a building
    block (e.g. for pseudo-C_ell), not the true-a_lm recovery (use
    :func:`deconvolve` for that).
    """
    maps = jnp.asarray(maps)
    msk = jnp.asarray(mask)
    masked = (msk if spin == 0 else msk[None, :]) * maps
    return map2alm(masked, nside, lmax, spin=spin, niter=niter, use_weights=use_weights)


# --------------------------------------------------------------------------- #
# the real-DOF isometry  T : a (complex, healpy-packed) <-> x (real)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _dof_layout(lmax: int, spin: int) -> tuple[jax.Array, jax.Array, int, int]:
    """Static index arrays for ``T``: ``(k_m0, k_mpos, K, nx_per_channel)``.

    ``k_m0`` are the flat a_lm indices with ``m=0`` and ``l>=|spin|``; ``k_mpos``
    those with ``m>0`` and ``l>=|spin|``.  The structurally-zero ``l<|spin|``
    modes are excluded (they are null directions of ``A``).
    """
    ms = np.concatenate([np.full(lmax + 1 - m, m) for m in range(lmax + 1)])
    ls = np.concatenate([np.arange(m, lmax + 1) for m in range(lmax + 1)])
    active = ls >= abs(spin)
    idx = np.arange(alm_size(lmax))
    k_m0 = idx[active & (ms == 0)]
    k_mpos = idx[active & (ms > 0)]
    nx = int(k_m0.size + 2 * k_mpos.size)
    return jnp.asarray(k_m0), jnp.asarray(k_mpos), alm_size(lmax), nx


def n_dof(lmax: int, spin: int = 0) -> int:
    """Total length of the real DOF vector ``x`` (both E,B channels for spin 2)."""
    nx = _dof_layout(lmax, spin)[3]
    return nx if spin == 0 else 2 * nx


def _to_real_1(a: jax.Array, k_m0: jax.Array, k_mpos: jax.Array) -> jax.Array:
    return jnp.concatenate([a[k_m0].real, _SQRT2 * a[k_mpos].real, _SQRT2 * a[k_mpos].imag])


def _to_alm_1(x: jax.Array, k_m0: jax.Array, k_mpos: jax.Array, k: int) -> jax.Array:
    n0 = k_m0.shape[0]
    npp = k_mpos.shape[0]
    a = jnp.zeros(k, dtype=jnp.complex128)
    a = a.at[k_m0].set(x[:n0].astype(jnp.complex128))
    re = x[n0 : n0 + npp]
    im = x[n0 + npp : n0 + 2 * npp]
    return a.at[k_mpos].set((re + 1j * im) / _SQRT2)


def alm_to_real(alm, lmax: int, spin: int = 0) -> jax.Array:
    """``T``: healpy-packed a_lm -> real DOF vector ``x`` with ``||x||_2 = ||a||_w``."""
    k_m0, k_mpos, _, _ = _dof_layout(lmax, spin)
    a = jnp.asarray(alm)
    if spin == 0:
        return _to_real_1(a, k_m0, k_mpos)
    return jnp.concatenate([_to_real_1(a[0], k_m0, k_mpos), _to_real_1(a[1], k_m0, k_mpos)])


def real_to_alm(x, lmax: int, spin: int = 0) -> jax.Array:
    """``T^-1``: real DOF vector ``x`` -> healpy-packed a_lm (``l<|spin|`` modes zero)."""
    k_m0, k_mpos, k, nx = _dof_layout(lmax, spin)
    xv = jnp.asarray(x)
    if spin == 0:
        return _to_alm_1(xv, k_m0, k_mpos, k)
    return jnp.stack([_to_alm_1(xv[:nx], k_m0, k_mpos, k), _to_alm_1(xv[nx:], k_m0, k_mpos, k)])


# --------------------------------------------------------------------------- #
# cut-sky deconvolution (CG on the masked normal equations)
# --------------------------------------------------------------------------- #
def _masked_rhs(maps, mask, nside: int, lmax: int, spin: int, use_weights: bool) -> jax.Array:
    """``b = S^T (M W) m`` -- the deconvolution RHS (masked-weighted adjoint)."""
    return adjoint_synthesis(_mw(nside, spin, mask, use_weights) * jnp.asarray(maps), nside, lmax, spin)


def deconvolve(
    maps,
    mask,
    nside: int,
    lmax: int,
    spin: int = 0,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
    reg: float = 0.0,
    use_weights: bool = True,
    x0=None,
) -> jax.Array:
    """Recover the true a_lm from a cut sky by CG on ``S^T(MW)S a = S^T(MW)m``.

    Parameters
    ----------
    maps, mask : the observed map and the per-pixel sky mask (RING order). Pixels
        with ``mask=0`` are ignored; apodized ``mask in (0,1)`` are down-weighted.
    max_iter, tol : CG budget and relative residual tolerance.
    reg : optional Tikhonov ``A -> A + reg*I`` (in the real-DOF metric) for
        ill-conditioned masks; ``reg=0`` gives the minimum-norm CG solution.
    x0 : optional warm-start a_lm.

    Returns the recovered a_lm (``(K,)`` spin-0, ``(2,K)`` spin-2).
    """
    mask_j = jnp.asarray(mask)
    b_x = alm_to_real(_masked_rhs(maps, mask_j, nside, lmax, spin, use_weights), lmax, spin)

    def a_op(x: jax.Array) -> jax.Array:
        a = real_to_alm(x, lmax, spin)
        wmp = _mw(nside, spin, mask_j, use_weights) * synthesis(a, nside, lmax, spin)
        ax = alm_to_real(adjoint_synthesis(wmp, nside, lmax, spin), lmax, spin)
        return ax + reg * x

    x_init = None if x0 is None else alm_to_real(x0, lmax, spin)
    x_sol, _ = cg(a_op, b_x, x0=x_init, tol=tol, maxiter=max_iter)
    return real_to_alm(x_sol, lmax, spin)
