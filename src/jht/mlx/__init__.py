"""MLX (Apple-GPU, fp32) backend for jht -- a separate local-development tier.

Apple Silicon has no fp64, so this is a deliberately lower-accuracy tier (~fp32 machine
precision, ~1e-5..1e-6 vs the JAX-fp64 path -- *not* the 1e-13 contract), for
accelerating and comparing workloads locally on the Mac GPU.  The production JAX path is
untouched; these functions are a parallel implementation reachable as ``jht.mlx.*`` and
(via :func:`jht.set_backend`) as the dispatched top-level operators.

Requires the optional ``mlx`` dependency (osx-arm64).  See ``docs/mlx.md``.
"""

from __future__ import annotations

from ._apply import (
    adjoint_synthesis,
    analysis,
    bare_analysis,
    map2alm,
    synthesis,
)

__all__ = [
    "synthesis",
    "adjoint_synthesis",
    "analysis",
    "map2alm",
    "bare_analysis",
]
