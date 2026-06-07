"""jht -- JAX-native spherical harmonic transforms.

A clean-room, pure-JAX implementation of the forward and inverse spherical
harmonic transform (map <-> a_lm) for **spin-0 and spin-2** fields on the
**HEALPix RING** pixelization, in the BICEP/Keck angular regime
(l_max <~ 1000, nside <= ~2048).  GPU-capable, fully differentiable under JAX's
native autodiff, and dependency-controlled (runtime deps = jax + numpy only).

Conventions (verified vs healpy 1.19.0 / ducc0 0.41.0; see ``docs/design.md``):
healpy m-major triangular a_lm packing, orthonormal Y_lm with the
Condon-Shortley phase, HEALPix-internal (COSMO) polarization.

float64 is **opt-in per entry point** -- library code never touches the global
config.  Enable it (before creating any array, on CPU or GPU) with::

    import jax
    jax.config.update("jax_enable_x64", True)

Quick start::

    import jax; jax.config.update("jax_enable_x64", True)
    import jht
    m = jht.synthesis(alm, nside, lmax, spin=0)        # a_lm -> map
    a = jht.map2alm(m, nside, lmax, spin=0, niter=3)   # map -> a_lm (weighted)

See ``README.md`` for the tour, ``docs/consumers.md`` for the downstream seam
(e.g. using jht as a ducc0 replacement), and ``docs/gpu.md`` for the GPU story.
"""

from __future__ import annotations

from .analysis import bare_analysis, map2alm
from .diff import analysis_real, bandpower, synthesis_real
from .healpix import (
    adjoint_synthesis,
    alm_metric_weight,
    alm_size,
    synthesis,
)
from .masked import (
    alm_to_real,
    constrained_realization,
    deconvolve,
    n_dof,
    pseudo_alm,
    real_to_alm,
    wiener,
)
from .weights import pixel_weights, ring_weights

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # core on-grid transforms
    "synthesis",  # S:  a_lm -> map
    "adjoint_synthesis",  # S^T: map -> a_lm (exact unweighted transpose / VJP)
    # analysis (approximate inverse)
    "bare_analysis",  # A0 = S^T W
    "map2alm",  # A0 + Jacobi iteration
    # quadrature weights
    "ring_weights",
    "pixel_weights",
    # partial-sky / masked
    "pseudo_alm",  # masked pseudo-a_lm (zero-fill)
    "deconvolve",  # cut-sky CG deconvolution
    "wiener",  # Wiener filter / MUSE inner solve (Cl prior + N^-1)
    "constrained_realization",  # posterior draw (constrained realization)
    # differentiable (real-DOF) interface
    "synthesis_real",  # S o T^-1 : R^n -> map
    "analysis_real",  # T o map2alm : map -> R^n
    "bandpower",  # angular auto-power C_ell
    # real-DOF isometry T
    "alm_to_real",
    "real_to_alm",
    "n_dof",
    # a_lm layout / inner-product helpers
    "alm_size",
    "alm_metric_weight",  # the (2 - delta_m0) metric G (the bk-jax 2*conj bridge)
]
