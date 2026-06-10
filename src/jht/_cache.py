"""Opt-in to JAX's persistent on-disk compilation cache.

The on-grid synthesis compile at nside>=1024 is multi-minute (the per-ring-length
FFT unroll -- ~nside distinct static FFT kernels; ~7.6 min at nside=2048, of which
the FFT-unroll assembly is ~93% and the recursion ~0.3%; see ``docs/performance.md``).
The compile is *structural* (distinct static FFT lengths cannot be batched, and
padding to a common length is numerically invalid) and *one-time per program*, so the
lever is not to shrink it but to make it pay-once-*ever*: XLA's persistent cache
writes each compiled executable to disk and reuses it across later processes (keyed by
jaxlib version + accelerator + the program).  Numerics are untouched.

Like x64, this flips global JAX config, so the library never enables it itself -- the
entry point opts in.
"""

from __future__ import annotations

import os

import jax

__all__ = ["enable_compilation_cache"]


def enable_compilation_cache(directory: str, *, min_compile_time_secs: float = 1.0) -> None:
    """Enable JAX's persistent on-disk compilation cache at ``directory``.

    Call once at program start (before the first transform compiles)::

        import jht
        jht.enable_compilation_cache("~/.cache/jht-xla")

    The first nside>=1024 synthesis compile (multi-minute -- see
    ``docs/performance.md``) is then written to ``directory`` and reused by every later
    process, so it is paid once rather than per run.  ``min_compile_time_secs`` caches
    only compilations slower than this (default 1.0 s; the synthesis compile is far
    above it).  Numerics are unaffected -- this is purely a compile-latency cache.
    """
    jax.config.update("jax_compilation_cache_dir", os.path.expanduser(directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", float(min_compile_time_secs))
