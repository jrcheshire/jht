"""jht -- JAX-native spherical harmonic transforms.

A clean-room, pure-JAX implementation of the forward and inverse spherical
harmonic transform (map <-> a_lm) for **spin-0 and spin-2** fields on the
**HEALPix RING** pixelization, validated to nside <= 4096 / l_max <~ 6000 (the
recursion is numerically exact far beyond -- ~eps*sqrt(l) to l=32000; nside <=
2048 is the routine-gated regime; nside=8192 is the compile ceiling).
GPU-capable, fully differentiable under JAX's native autodiff, and
dependency-controlled (runtime deps = jax + numpy only).

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
    a = jht.analysis(m, nside, lmax, spin=0, niter=3)  # map -> a_lm (weighted)

See ``README.md`` for an overview, ``docs/consumers.md`` for the API a downstream
project depends on, and ``docs/gpu.md`` for GPU use.
"""

from __future__ import annotations

from ._analysis import analysis, bare_analysis, map2alm
from ._azimuth import enable_looped_fft, get_azimuth_fft_mode, set_azimuth_fft_mode
from ._cache import enable_compilation_cache
from .diff import (
    adjoint_synthesis_general_real,
    adjoint_synthesis_vjp,
    analysis_real,
    bandpower,
    grid_to_sky_real,
    sky_to_grid_real,
    synthesis_general_real,
    synthesis_real,
    synthesis_vjp,
)
from .healpix import (
    adjoint_synthesis,
    alm_metric_weight,
    alm_size,
    synthesis,
)
from .masked import (
    SolveInfo,
    WienerPreconditioner,
    alm_to_real,
    constrained_realization,
    deconvolve,
    n_dof,
    pseudo_alm,
    real_to_alm,
    wiener,
    wiener_preconditioner,
)
from .offgrid import (
    PreparedPoints,
    adjoint_synthesis_general,
    grid_to_points,
    grid_to_sky,
    offgrid_grid_shape,
    points_to_grid,
    prepare_points,
    sky_to_grid,
    synthesis_general,
)
from .weights import pixel_weights, ring_weights

__version__ = "0.4.1"  # single source of truth; pyproject reads this via hatch
__author__ = "James Cheshire"
__email__ = "cheshire@caltech.edu"

__all__ = [
    "__version__",
    # core on-grid transforms
    "synthesis",  # S:  a_lm -> map
    "adjoint_synthesis",  # S^T: map -> a_lm (exact unweighted transpose / VJP)
    # off-grid (arbitrary points / NUFFT) -- spin 0-3, alm- AND pointing-differentiable
    "synthesis_general",  # a_lm -> field at arbitrary (theta, phi)
    "adjoint_synthesis_general",  # its exact transpose
    # off-grid in two stages, split at the NUFFT's oversampled grid (point sets prepared once)
    "prepare_points",  # host-side: fix stencil cells, group points by cell
    "PreparedPoints",
    "sky_to_grid",  # a_lm -> NUFFT grid (the point-independent half)
    "grid_to_points",  # NUFFT grid -> field at a prepared point set
    "points_to_grid",  # its exact transpose (spread); grids from point blocks add
    "grid_to_sky",  # exact transpose of sky_to_grid
    "offgrid_grid_shape",
    # analysis (approximate inverse): map -> a_lm
    "analysis",  # A0 + Jacobi iteration (canonical)
    "map2alm",  # healpy-idiom alias of `analysis`
    "bare_analysis",  # A0 = S^T W (the single-shot primitive)
    # quadrature weights
    "ring_weights",
    "pixel_weights",
    # partial-sky / masked
    "pseudo_alm",  # masked pseudo-a_lm (zero-fill)
    "deconvolve",  # cut-sky CG deconvolution
    "wiener",  # Wiener filter / MUSE inner solve (Cl prior + N^-1)
    "constrained_realization",  # posterior draw (constrained realization)
    "wiener_preconditioner",  # l-diagonal + exact low-l block M^-1 for wiener / CR
    "WienerPreconditioner",
    "SolveInfo",  # CG diagnostics (iterations, final relative residual)
    # differentiable (real-DOF) interface
    "synthesis_real",  # S o T^-1 : R^n -> map
    "analysis_real",  # T o analysis : map -> R^n
    "synthesis_general_real",  # S_g o T^-1 : R^n -> field at arbitrary points
    "adjoint_synthesis_general_real",  # T o S_g^T : field -> R^n (exact transpose)
    "sky_to_grid_real",  # sky_to_grid o T^-1 : R^n -> NUFFT grid
    "grid_to_sky_real",  # T o grid_to_sky : its exact transpose
    "bandpower",  # angular auto-power C_ell
    # reverse-only transforms: native AD's gradients without the recursion tape (no jvp)
    "synthesis_vjp",
    "adjoint_synthesis_vjp",
    # real-DOF isometry T
    "alm_to_real",
    "real_to_alm",
    "n_dof",
    # a_lm layout / inner-product helpers
    "alm_size",
    "alm_metric_weight",  # the (2 - delta_m0) metric G (the 2*conj VJP bridge)
    # compile-latency: opt in to JAX's persistent on-disk compilation cache
    "enable_compilation_cache",
    # azimuth-FFT strategy: opt in to the looped/chirp-z cap path (O(1) compiled
    # kernels for SHT-heavy / high-nside differentiable graphs)
    "set_azimuth_fft_mode",
    "get_azimuth_fft_mode",
    "enable_looped_fft",
]
