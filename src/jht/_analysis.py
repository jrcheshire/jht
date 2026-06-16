"""HEALPix analysis: map -> a_lm (the approximate inverse).

The bare estimator weights each pixel by its quadrature weight and applies the
adjoint, ``A0 = S^T W``.  With ring weights (the default, :mod:`jht.weights`)
``W`` makes the m=0 colatitude quadrature exact across the band, dropping the
floor from ~1e-3 (uniform ``4pi/Npix``) toward ~1e-4; ``use_weights=False``
recovers the bare uniform estimator.  Neither is exact on HEALPix (no sampling
theorem), so :func:`analysis` adds a Jacobi / stationary-Richardson iteration on
the residual,

    a_{k+1} = a_k + A0 (map - S a_k),

which refines a band-limited map toward the true a_lm (Richardson iteration on
the normal equations ``S^T W S a = S^T W map``; it converges because the HEALPix
points are quasi-uniform, and ring weights condition it so it reaches machine
precision in a few steps -- see ``docs/accuracy.md``).

Distinct from :func:`jht.healpix.adjoint_synthesis`, which is the *exact*
(unweighted) transpose ``S^T`` (the bk-jax seam / the VJP), not an inverse.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from .healpix import adjoint_synthesis, synthesis
from .weights import pixel_weights


@lru_cache(maxsize=None)
def _wvec(nside: int, spin: int, use_weights: bool) -> np.ndarray:
    """Per-pixel quadrature weights (broadcastable), cached as NumPy.

    NumPy (not jnp): an ``lru_cache`` that returns a *device* array caches a tracer
    if first called inside a ``jit`` / ``grad`` / ``lax.scan`` trace, which then
    leaks (``UnexpectedTracerError``) on a later access from a different trace.
    NumPy weights multiply jnp maps unchanged (``np * jnp -> jnp``), so
    :func:`bare_analysis` is unaffected.  Same cold-cache-under-trace hazard guarded
    in :func:`_recursion._wigner_seed_np` and :func:`masked._dof_layout`.
    """
    wv = pixel_weights(nside, use_weights)
    return wv if spin == 0 else wv[None, :]  # (Npix,) for spin 0, (1, Npix) for spin 2


def bare_analysis(maps, nside: int, lmax: int, spin: int = 0, use_weights: bool = True) -> jax.Array:
    """``A0 = S^T W`` -- the bare (non-iterated) estimator with quadrature weights."""
    wmaps = _wvec(nside, spin, use_weights) * jnp.asarray(maps)
    return adjoint_synthesis(wmaps, nside, lmax, spin)


def analysis(
    maps, nside: int, lmax: int, spin: int = 0, niter: int = 3, use_weights: bool = True
) -> jax.Array:
    """Approximate inverse with ``niter`` Jacobi refinement steps (``niter=0`` = bare)."""
    a = bare_analysis(maps, nside, lmax, spin, use_weights)
    for _ in range(niter):
        residual = maps - synthesis(a, nside, lmax, spin)
        a = a + bare_analysis(residual, nside, lmax, spin, use_weights)
    return a


# healpy-idiom alias (back-compat; ``analysis`` is the canonical name).
map2alm = analysis
