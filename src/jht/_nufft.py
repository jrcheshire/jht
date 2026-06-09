"""Pure-JAX 2D NUFFT (ES kernel) -- the off-grid engine for ``jht.offgrid``.

Type-2 (uniform Fourier modes -> nonuniform points) and its exact transpose,
type-1.  Both are differentiable in the nonuniform point coordinates under JAX's
native autodiff: the integer window index is frozen with ``stop_gradient`` (it is
genuinely locally constant), so the gradient flows analytically through the
smooth ES-kernel weights -- no custom rule, ``jacfwd == jacrev`` preserved.

The kernel is the FINUFFT "exponential of semicircle"
``psi(t) = exp(beta (sqrt(1 - (2t/W)^2) - 1))`` (Barnett, Magland & af Klinteberg
2019, arXiv:1808.06736).  The accuracy-vs-(sigma, W) trade was pinned empirically
in the M0 spike (abs err vs the exact direct sum, lmax<=512): W=14/sigma=2 reaches
<=6e-10; W=12 ~5e-9; W=16 sits at the fp64 roundoff floor.

Convention: modes are *centered*, ``k = -N//2 .. -N//2 + N - 1``; points
``x, y`` live on the ``[0, 2pi)`` torus.  ``nufft2d2`` evaluates
``f_j = sum_{k,m} c[k,m] exp(i (k x_j + m y_j))``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

TWO_PI = 2.0 * np.pi

# requested epsilon -> (W, sigma).  Keyed by *achievable* accuracy; the cheapest
# (smallest W) entry meeting the request is selected.  Tuned in scripts/offgrid_spike.py.
_KERNEL_DB: dict[float, tuple[int, float]] = {
    1e-4: (6, 2.0),
    1e-6: (8, 2.0),
    1e-8: (10, 2.0),
    1e-10: (14, 2.0),
    1e-12: (16, 2.0),
}


def _eps_to_w_sigma(eps: float) -> tuple[int, float]:
    # Keys are achievable accuracies; pick the loosest (cheapest -> largest key)
    # that still meets the request.  If eps is tighter than the DB's best, fall
    # back to min(key) -- the smallest key is the most-accurate (largest-W) kernel.
    ok = [e for e in _KERNEL_DB if e <= eps * 1.0000001]
    return _KERNEL_DB[max(ok)] if ok else _KERNEL_DB[min(_KERNEL_DB)]


def _next_size(n: float) -> int:
    """Next even FFT-friendly size (a 2,3,5-smooth integer) >= n."""
    n = int(np.ceil(n))
    if n % 2:
        n += 1
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 2


def _beta(W: int, sigma: float) -> float:
    return float(np.pi * W * (1.0 - 0.5 / sigma))


def _es_kernel(t, beta: float, W: int):
    """ES window ``exp(beta(sqrt(1-(2t/W)^2)-1))``; 0 outside ``|t| <= W/2``."""
    z2 = (2.0 * t / W) ** 2
    val = jnp.exp(beta * (jnp.sqrt(jnp.clip(1.0 - z2, 0.0, 1.0)) - 1.0))
    return jnp.where(z2 <= 1.0, val, 0.0)


def _correction(N: int, n: int, W: int, beta: float) -> np.ndarray:
    """Continuous-FT deconvolution factor ``K^(k)`` for centered modes (real, even)."""
    ks = np.arange(-(N // 2), -(N // 2) + N)
    t, wq = np.polynomial.legendre.leggauss(2 * W + 24)
    t, wq = t * (W / 2.0), wq * (W / 2.0)
    psi = np.exp(beta * (np.sqrt(np.clip(1.0 - (2.0 * t / W) ** 2, 0.0, 1.0)) - 1.0))
    u = TWO_PI / n
    return (psi[None, :] * np.cos(ks[:, None] * u * t[None, :]) * wq[None, :]).sum(1)


class NufftPlan(NamedTuple):
    """Static (host-side) tables for a 2D NUFFT at a given (Nk, Nm, eps)."""

    Nk: int
    Nm: int
    nk: int  # oversampled grid size (theta/k axis)
    nm: int  # oversampled grid size (phi/m axis)
    W: int
    beta_k: float
    beta_m: float
    corr_k: np.ndarray  # (Nk,)
    corr_m: np.ndarray  # (Nm,)
    kbins: np.ndarray  # (Nk,)  centered modes mod nk
    mbins: np.ndarray  # (Nm,)  centered modes mod nm
    fill_src: np.ndarray  # (nk*nm,) gather index: oversampled cell -> source mode in c.ravel()
    fill_mask: np.ndarray  # (nk*nm,) True where a centered mode lands on this cell


@lru_cache(maxsize=None)
def nufft_plan(Nk: int, Nm: int, eps: float) -> NufftPlan:
    W, sigma = _eps_to_w_sigma(eps)
    nk, nm = _next_size(sigma * Nk), _next_size(sigma * Nm)
    bk, bm = _beta(W, sigma), _beta(W, sigma)
    kbins = np.arange(-(Nk // 2), -(Nk // 2) + Nk) % nk
    mbins = np.arange(-(Nm // 2), -(Nm // 2) + Nm) % nm
    # Inverse fill index so nufft2d2 builds the oversampled grid by a GATHER, not an
    # fp64/complex SCATTER (`C.at[].set`) -- the scatter was 97% of the off-grid forward
    # on GPU (~25000x the gather; see scripts/profile_offgrid_forward.py).  Centered modes
    # at sigma>=2 are collision-free in the oversampled grid, so the gather reproduces the
    # scatter exactly (asserted below).
    dest = (kbins[:, None] * nm + mbins[None, :]).ravel()
    if np.unique(dest).size != Nk * Nm:
        raise ValueError("NUFFT grid-build indices collide; gather build invalid")
    flat = np.full(nk * nm, -1, dtype=np.int64)
    flat[dest] = np.arange(Nk * Nm)
    return NufftPlan(
        Nk, Nm, nk, nm, W, bk, bm,
        _correction(Nk, nk, W, bk),
        _correction(Nm, nm, W, bm),
        kbins, mbins,
        np.where(flat >= 0, flat, 0), flat >= 0,
    )


def _stencil(coord, n: int, W: int, beta: float):
    """``(idx (Npts,W) mod n, weights (Npts,W))`` for the W-wide ES stencil.

    The window base is ``stop_gradient``-frozen (locally constant); the weights
    are smooth in ``coord`` -> native AD gives the analytic point-derivative.
    """
    xi = coord * n / TWO_PI
    base = jax.lax.stop_gradient(jnp.ceil(xi - W / 2.0)).astype(jnp.int64)
    off = jnp.arange(W)
    ell = base[:, None] + off[None, :]  # (Npts, W) true grid coords (un-wrapped)
    w = _es_kernel(xi[:, None] - ell, beta, W)
    return ell % n, w


def nufft2d2(plan: NufftPlan, coeffs, x, y):
    """Type-2: ``f_j = sum_{k,m} coeffs[k,m] exp(i(k x_j + m y_j))``.

    ``coeffs`` (Nk, Nm) complex (centered modes); ``x, y`` (Npts,) on ``[0,2pi)``.
    """
    c = coeffs / jnp.asarray(plan.corr_k)[:, None] / jnp.asarray(plan.corr_m)[None, :]
    # Build the oversampled grid by a GATHER (the inverse fill index lives in the plan);
    # the equivalent fp64/complex `.at[].set` scatter was 97% of the GPU forward.
    C = jnp.where(
        jnp.asarray(plan.fill_mask), c.ravel()[jnp.asarray(plan.fill_src)], 0.0 + 0.0j
    ).reshape(plan.nk, plan.nm)
    g = jnp.fft.ifft2(C) * (plan.nk * plan.nm)
    lk, wk = _stencil(x, plan.nk, plan.W, plan.beta_k)
    lm, wm = _stencil(y, plan.nm, plan.W, plan.beta_m)
    gg = g[lk[:, :, None], lm[:, None, :]]  # (Npts, W, W)
    return jnp.sum(wk[:, :, None] * wm[:, None, :] * gg, axis=(1, 2))


def nufft2d1(plan: NufftPlan, vals, x, y):
    """Type-1: ``coeffs[k,m] = sum_j vals_j exp(-i(k x_j + m y_j))`` -- transpose of type-2."""
    lk, wk = _stencil(x, plan.nk, plan.W, plan.beta_k)
    lm, wm = _stencil(y, plan.nm, plan.W, plan.beta_m)
    contrib = (wk[:, :, None] * wm[:, None, :]) * jnp.asarray(vals)[:, None, None]
    g = jnp.zeros((plan.nk, plan.nm), dtype=jnp.complex128)
    g = g.at[lk[:, :, None], lm[:, None, :]].add(contrib)
    C = jnp.fft.fft2(g)
    cc = C[jnp.asarray(plan.kbins)[:, None], jnp.asarray(plan.mbins)[None, :]]
    return cc / jnp.asarray(plan.corr_k)[:, None] / jnp.asarray(plan.corr_m)[None, :]
