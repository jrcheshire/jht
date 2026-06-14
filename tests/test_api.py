"""Public-API contract: ``jht.__all__`` is the supported dependency surface.

This guards the surface a downstream (e.g. bk-jax adopting jht in place of ducc0,
see ``docs/consumers.md``) pins against: every advertised name is importable and
callable, the re-exports point at the real implementations, and the top-level
round-trip works -- so accidental shadowing or a broken re-export is caught.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jht  # noqa: E402
from jht import _analysis, _cache, diff, healpix, masked, offgrid, weights  # noqa: E402

EXPECTED = {
    "synthesis", "adjoint_synthesis", "analysis", "map2alm", "bare_analysis",
    "synthesis_general", "adjoint_synthesis_general",
    "ring_weights", "pixel_weights", "pseudo_alm", "deconvolve",
    "wiener", "constrained_realization",
    "synthesis_real", "analysis_real",
    "synthesis_general_real", "adjoint_synthesis_general_real",
    "bandpower",
    "alm_to_real", "real_to_alm", "n_dof", "alm_size", "alm_metric_weight",
    "enable_compilation_cache",
}


def test_version():
    assert jht.__version__ == "0.1.3"


def test_all_names_present_and_callable():
    advertised = set(jht.__all__) - {"__version__"}
    assert advertised == EXPECTED  # __all__ matches the documented surface exactly
    for name in advertised:
        assert hasattr(jht, name), f"jht.{name} missing"
        assert callable(getattr(jht, name)), f"jht.{name} not callable"


def test_reexports_are_the_real_objects():
    # re-export integrity (no accidental shadowing / stale copy)
    assert jht.synthesis is healpix.synthesis
    assert jht.adjoint_synthesis is healpix.adjoint_synthesis
    assert jht.synthesis_general is offgrid.synthesis_general
    assert jht.adjoint_synthesis_general is offgrid.adjoint_synthesis_general
    assert jht.alm_size is healpix.alm_size
    assert jht.alm_metric_weight is healpix.alm_metric_weight
    assert jht.analysis is _analysis.analysis
    assert jht.map2alm is _analysis.map2alm
    assert jht.map2alm is jht.analysis  # map2alm is the back-compat alias of analysis
    assert jht.bare_analysis is _analysis.bare_analysis
    assert jht.ring_weights is weights.ring_weights
    assert jht.pixel_weights is weights.pixel_weights
    assert jht.pseudo_alm is masked.pseudo_alm
    assert jht.deconvolve is masked.deconvolve
    assert jht.wiener is masked.wiener
    assert jht.constrained_realization is masked.constrained_realization
    assert jht.synthesis_real is diff.synthesis_real
    assert jht.analysis_real is diff.analysis_real
    assert jht.synthesis_general_real is diff.synthesis_general_real
    assert jht.adjoint_synthesis_general_real is diff.adjoint_synthesis_general_real
    assert jht.bandpower is diff.bandpower
    assert jht.enable_compilation_cache is _cache.enable_compilation_cache


def test_enable_compilation_cache_sets_config(tmp_path):
    """The opt-in helper flips the JAX persistent-cache config (keys can move across
    jax versions, so pin them); save/restore so it does not leak to other tests."""
    saved_dir = jax.config.jax_compilation_cache_dir
    saved_secs = jax.config.jax_persistent_cache_min_compile_time_secs
    try:
        jht.enable_compilation_cache(str(tmp_path), min_compile_time_secs=2.5)
        assert jax.config.jax_compilation_cache_dir == str(tmp_path)
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 2.5
    finally:
        jax.config.update("jax_compilation_cache_dir", saved_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", saved_secs)


def test_top_level_roundtrip():
    nside, lmax = 16, 24
    rng = np.random.default_rng(0)
    a = (rng.standard_normal(jht.alm_size(lmax)) + 1j * rng.standard_normal(jht.alm_size(lmax))) / np.sqrt(2)
    a[: lmax + 1] = a[: lmax + 1].real  # real a_{l,0}
    a = jnp.asarray(a.astype(np.complex128))
    m = jht.synthesis(a, nside, lmax, spin=0)
    b = jht.map2alm(m, nside, lmax, spin=0, niter=3)
    assert float(jnp.max(jnp.abs(b - a))) < 1e-4  # weighted+iter tier (gate, not the ~1e-13 floor)
