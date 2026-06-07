"""HEALPix analysis: map -> a_lm (the approximate inverse).

The bare estimator ``A0 = (4pi/Npix) S^T`` is the unweighted adjoint scaled to
the pixel solid angle; it is *not* exact on HEALPix (no sampling theorem), so it
sits at the ~1e-3 floor.  :func:`map2alm` adds a Jacobi / stationary-Richardson
iteration on the residual,

    a_{k+1} = a_k + A0 (map - S a_k),

which refines a band-limited map toward the true a_lm (Drake & Wright 2020:
this is Richardson iteration on the normal equations ``S^T S a = S^T map``;
it converges because the HEALPix points are quasi-uniform).  No ring weights yet
-- that is the next accuracy rung (deferred per the Phase-0 decision).

Distinct from :func:`jht.healpix.adjoint_synthesis`, which is the *exact*
transpose ``S^T`` (the bk-jax seam / the VJP), not an inverse.
"""

from __future__ import annotations

import jax

from .healpix import adjoint_synthesis, synthesis


def bare_analysis(maps, nside: int, lmax: int, spin: int = 0) -> jax.Array:
    """``A0 = (4pi/Npix) S^T`` -- the bare (unweighted, non-iterated) estimator."""
    npix = 12 * nside**2
    return (4.0 * 3.141592653589793 / npix) * adjoint_synthesis(maps, nside, lmax, spin)


def map2alm(maps, nside: int, lmax: int, spin: int = 0, niter: int = 3) -> jax.Array:
    """Approximate inverse with ``niter`` Jacobi refinement steps (``niter=0`` = bare)."""
    a = bare_analysis(maps, nside, lmax, spin)
    for _ in range(niter):
        residual = maps - synthesis(a, nside, lmax, spin)
        a = a + bare_analysis(residual, nside, lmax, spin)
    return a
