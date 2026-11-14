"""Pure-JAX 2D NUFFT (ES kernel) -- the off-grid engine for ``jht.offgrid``.

Type-2 (uniform Fourier modes -> nonuniform points) and its exact transpose,
type-1.  Both are differentiable in the nonuniform point coordinates under JAX's
native autodiff: the integer window index is frozen with ``stop_gradient`` (it is
genuinely locally constant), so the gradient flows analytically through the
smooth ES-kernel weights -- no custom rule, ``jacfwd == jacrev`` preserved.

The kernel is the FINUFFT "exponential of semicircle"
``psi(t) = exp(beta (sqrt(1 - (2t/W)^2) - 1))`` (Barnett, Magland & af Klinteberg
2019, arXiv:1808.06736).  Measured accuracy vs the exact direct sum (lmax <= 512):
W=14/sigma=2 reaches <= 6e-10; W=12 ~5e-9; W=16 sits at the fp64 roundoff floor.

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
# (smallest W) entry meeting the request is selected.  Measured with scripts/offgrid_sweep.py.
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
    """ES window ``exp(beta(sqrt(1-(2t/W)^2)-1))``; 0 outside ``|t| < W/2``.

    The support boundary ``|t| = W/2`` is excluded and the sqrt argument is
    where-guarded: a target exactly on a fine-grid node (theta in {0, pi},
    phi = 0, ...) puts a stencil point exactly at the boundary, where
    ``d/dz sqrt(1-z^2)`` is infinite -- with the boundary included, the
    loc-gradient came back ``inf`` there.  Excluding it changes the value by at
    most ``psi(W/2) = e^-beta`` (4.6e-15 at W=14; 7.2e-7 at W=6), below every
    kernel tier in ``_KERNEL_DB``, and keeps native AD finite and exact.
    """
    z2 = (2.0 * t / W) ** 2
    inside = z2 < 1.0
    val = jnp.exp(beta * (jnp.sqrt(jnp.where(inside, 1.0 - z2, 1.0)) - 1.0))
    return jnp.where(inside, val, 0.0)


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
    # Inverse fill index so nufft2d2 builds the oversampled grid by a GATHER: an
    # fp64/complex scatter (`C.at[].set`) is ~10^4x slower on GPU and would dominate
    # the off-grid forward (scripts/profile_offgrid_forward.py).  Centered modes
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
    """``(idx (Npts,W) mod n, weights (Npts,W))`` for the W-wide ES stencil (``W`` even).

    The window base is ``stop_gradient``-frozen (locally constant); the weights
    are smooth in ``coord`` -> native AD gives the analytic point-derivative.

    The base is ``ceil(xi) - W/2`` with the grid coordinate ``xi`` a single product,
    not ``ceil(xi - W/2)`` of a product-and-divide.  Both are the same integer in
    exact arithmetic, but the indices and the weights must use the *same* base, and
    a compiler is free to evaluate a shared floating-point expression more than
    once, fusing or reordering its operations differently each time.  For a point
    exactly on a grid node (``xi`` an integer) two such evaluations of
    ``coord * n / 2pi - W/2`` fall either side of the integer: under ``jax.jit``
    the indices moved by one cell while the weights did not.  The ceiling of one
    correctly-rounded multiply has no second way to be evaluated.
    """
    if W % 2:
        raise ValueError(f"stencil width must be even; got {W}")
    xi = coord * (n / TWO_PI)
    base = jax.lax.stop_gradient(jnp.ceil(xi)).astype(jnp.int64) - W // 2
    off = jnp.arange(W)
    ell = base[:, None] + off[None, :]  # (Npts, W) true grid coords (un-wrapped)
    w = _es_kernel(xi[:, None] - ell, beta, W)
    return ell % n, w


def _modes_to_grid(plan: NufftPlan, coeffs):
    """Centered modes ``(Nk, Nm)`` -> the oversampled spatial grid ``(nk, nm)``.

    Deconvolves by the kernel's Fourier transform, places the modes on the oversampled
    grid and inverse-FFTs (unnormalized): ``g[a, b] = sum_{k,m} c[k,m] e^{2 pi i (ka/nk + mb/nm)}``
    with ``c = coeffs / corr``.  Hermitian adjoint: :func:`_grid_to_modes`.
    """
    c = coeffs / jnp.asarray(plan.corr_k)[:, None] / jnp.asarray(plan.corr_m)[None, :]
    # Build the oversampled grid by a GATHER (the inverse fill index lives in the plan),
    # not an fp64/complex `.at[].set` scatter, which is ~10^4x slower on GPU.
    C = jnp.where(
        jnp.asarray(plan.fill_mask), c.ravel()[jnp.asarray(plan.fill_src)], 0.0 + 0.0j
    ).reshape(plan.nk, plan.nm)
    return jnp.fft.ifft2(C) * (plan.nk * plan.nm)


def _grid_to_modes(plan: NufftPlan, g):
    """Hermitian adjoint of :func:`_modes_to_grid`: oversampled grid -> centered modes."""
    C = jnp.fft.fft2(g)
    cc = C[jnp.asarray(plan.kbins)[:, None], jnp.asarray(plan.mbins)[None, :]]
    return cc / jnp.asarray(plan.corr_k)[:, None] / jnp.asarray(plan.corr_m)[None, :]


def _interp_points(plan: NufftPlan, g, x, y):
    """Stencil interpolation of the grid ``g`` at the points: ``sum_ab wk_a wm_b g[a, b]``."""
    lk, wk = _stencil(x, plan.nk, plan.W, plan.beta_k)
    lm, wm = _stencil(y, plan.nm, plan.W, plan.beta_m)
    gg = g[lk[:, :, None], lm[:, None, :]]  # (Npts, W, W)
    return jnp.sum(wk[:, :, None] * wm[:, None, :] * gg, axis=(1, 2))


_WINDOW_DIMS = jax.lax.ScatterDimensionNumbers(
    update_window_dims=(1, 2), inserted_window_dims=(), scatter_dims_to_operand_dims=(0, 1)
)


def _fold_axis(gp, n: int, axis: int):
    """Fold a padded periodic axis of length ``n + W - 1`` back onto ``0..n-1``."""
    length = gp.shape[axis]
    if length <= 2 * n:  # one wrap: the tail adds onto the head
        head = jax.lax.slice_in_dim(gp, 0, length - n, axis=axis)
        tail = jax.lax.slice_in_dim(gp, n, length, axis=axis)
        rest = jax.lax.slice_in_dim(gp, length - n, n, axis=axis)
        return jnp.concatenate([head + tail, rest], axis=axis)
    reps = -(-length // n)  # W > n (tiny grids): pad to a multiple of n and sum the wraps
    pad = [(0, 0)] * gp.ndim
    pad[axis] = (0, reps * n - length)
    gp = jnp.pad(gp, pad)
    shape = gp.shape[:axis] + (reps, n) + gp.shape[axis + 1 :]
    return gp.reshape(shape).sum(axis=axis)


def _spread_blocks(plan: NufftPlan, blocks, starts):
    """Add ``(G, W, W)`` blocks onto the periodic ``(nk, nm)`` grid at ``starts`` (G, 2).

    One window write per block on a grid padded by ``W - 1`` (so no window wraps), then
    the padding is folded back.  ``starts`` must lie in ``[0, nk) x [0, nm)``.
    """
    W = plan.W
    gp = jnp.zeros((plan.nk + W - 1, plan.nm + W - 1), dtype=blocks.dtype)
    gp = jax.lax.scatter_add(
        gp, starts, blocks, _WINDOW_DIMS, mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS
    )
    return _fold_axis(_fold_axis(gp, plan.nk, 0), plan.nm, 1)


def _spread_points(plan: NufftPlan, vals, x, y):
    """Transpose of :func:`_interp_points`: spread point values onto the grid.

    One ``(W, W)`` window write per point (:func:`_spread_blocks`).  Real ``vals`` give
    a real grid.
    """
    lk, wk = _stencil(x, plan.nk, plan.W, plan.beta_k)
    lm, wm = _stencil(y, plan.nm, plan.W, plan.beta_m)
    vals = jnp.asarray(vals)
    blocks = (wk[:, :, None] * wm[:, None, :]) * vals[:, None, None]
    return _spread_blocks(plan, blocks, jnp.stack([lk[:, 0], lm[:, 0]], axis=1))


def nufft2d2(plan: NufftPlan, coeffs, x, y):
    """Type-2: ``f_j = sum_{k,m} coeffs[k,m] exp(i(k x_j + m y_j))``.

    ``coeffs`` (Nk, Nm) complex (centered modes); ``x, y`` (Npts,) on ``[0,2pi)``.
    """
    return _interp_points(plan, _modes_to_grid(plan, coeffs), x, y)


def nufft2d1(plan: NufftPlan, vals, x, y):
    """Type-1: ``coeffs[k,m] = sum_j vals_j exp(-i(k x_j + m y_j))`` -- transpose of type-2."""
    return _grid_to_modes(plan, _spread_points(plan, vals, x, y))


# --------------------------------------------------------------------------- #
# prepared point sets: stencil bases fixed on the host, points grouped by base cell
# --------------------------------------------------------------------------- #
@jax.tree_util.register_pytree_node_class
class PreparedPoints:
    """A point set prepared for a given NUFFT grid (see :func:`prepare_points`).

    Array leaves (passed to compiled code as arguments, never baked in as constants):

    * ``base`` ``(N, 2)`` int32: each point's un-wrapped stencil base on the (k, m)
      axes, ``ceil(coord * n / 2pi) - W/2``;
    * ``buckets``: a tuple of ``(slots (G, K) int32, starts (G, 2) int32)``, one per
      group size ``K``.  A group is up to ``K`` points sharing one base cell; ``slots``
      index the points (``N`` pads a short group: it addresses an appended zero row),
      ``starts`` is the cell modulo the grid;
    * ``wk``, ``wm`` ``(N, W)`` float64 kernel weights, or ``None`` unless stored.

    Static: ``plan_key = (Nk, Nm, eps)`` of the :func:`nufft_plan` it was prepared for,
    and ``npts``.
    """

    def __init__(self, base, buckets, wk, wm, *, plan_key: tuple[int, int, float], npts: int):
        self.base, self.buckets, self.wk, self.wm = base, buckets, wk, wm
        self.plan_key, self.npts = plan_key, npts

    @property
    def plan(self) -> NufftPlan:
        return nufft_plan(*self.plan_key)

    @property
    def stores_weights(self) -> bool:
        return self.wk is not None

    def tree_flatten(self):
        return (self.base, self.buckets, self.wk, self.wm), (self.plan_key, self.npts)

    @classmethod
    def tree_unflatten(cls, aux, children):
        plan_key, npts = aux
        return cls(*children, plan_key=plan_key, npts=npts)

    def __repr__(self) -> str:
        sizes = ", ".join(f"{s.shape[0]}x{s.shape[1]}" for s, _ in self.buckets)
        pl = self.plan
        return (f"PreparedPoints(npts={self.npts}, grid={pl.nk}x{pl.nm}, W={pl.W}, "
                f"groups=[{sizes}], stored_weights={self.stores_weights})")


def _group_cells(cell: np.ndarray, k_max: int):
    """Host-side grouping of points by cell id: ``[(K, slots (G, K), first_point (G,))]``.

    Each cell's run (stable order) is cut into chunks of at most ``k_max`` points; a chunk
    of ``r`` points goes to the bucket ``K = min(next_pow2(r), k_max)``, so padding is
    below 2x at any density.  Slots are padded with ``N``.
    """
    n = cell.size
    order = np.argsort(cell, kind="stable")
    sc = cell[order]
    first = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
    run = np.diff(np.r_[first, n])
    rank = np.arange(n) - np.repeat(first, run)
    chunk_of_cell = rank // k_max
    pos = rank % k_max
    new_chunk = pos == 0
    chunk = np.cumsum(new_chunk) - 1  # global chunk id per sorted point
    chunk_size = np.minimum(k_max, np.repeat(run, run) - chunk_of_cell * k_max)[new_chunk]
    chunk_k = np.minimum(k_max, 1 << np.ceil(np.log2(chunk_size)).astype(np.int64))
    out = []
    for K in np.unique(chunk_k):
        sel = np.flatnonzero(chunk_k == K)
        local = np.full(chunk_k.size, -1, dtype=np.int64)
        local[sel] = np.arange(sel.size)
        in_b = local[chunk] >= 0
        slots = np.full((sel.size, int(K)), n, dtype=np.int32)
        slots[local[chunk[in_b]], pos[in_b]] = order[in_b]
        out.append((int(K), slots, slots[:, 0]))
    return out


def prepare_points_plan(
    plan_key: tuple[int, int, float], x, y, *, group_size: int, store_weights: bool
) -> PreparedPoints:
    """Prepare points ``x, y`` (NumPy, torus coordinates) for ``nufft_plan(*plan_key)``; host side."""
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1; got {group_size}")
    plan = nufft_plan(*plan_key)
    W = plan.W
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    # Same correctly-rounded product as _stencil; the in-call weights use this base.
    bk = np.ceil(x * (plan.nk / TWO_PI)).astype(np.int64) - W // 2
    bm = np.ceil(y * (plan.nm / TWO_PI)).astype(np.int64) - W // 2
    sk, sm = bk % plan.nk, bm % plan.nm
    buckets = tuple(
        (jnp.asarray(slots), jnp.asarray(np.stack([sk[p], sm[p]], axis=1).astype(np.int32)))
        for _, slots, p in _group_cells(sk * plan.nm + sm, group_size)
    )
    base = jnp.asarray(np.stack([bk, bm], axis=1).astype(np.int32))
    wk = wm = None
    if store_weights:
        weights = jax.jit(_weights_from_base, static_argnums=(2, 3, 4))
        wk = weights(jnp.asarray(x), base[:, 0], plan.nk, W, plan.beta_k)
        wm = weights(jnp.asarray(y), base[:, 1], plan.nm, W, plan.beta_m)
    return PreparedPoints(base, buckets, wk, wm, plan_key=plan_key, npts=x.size)


def _weights_from_base(coord, base, n: int, W: int, beta: float):
    """``(N, W)`` kernel weights at the given (prepared) un-wrapped bases."""
    xi = coord * (n / TWO_PI)
    ell = base[:, None].astype(jnp.int64) + jnp.arange(W)[None, :]
    return _es_kernel(xi[:, None] - ell, beta, W)


def _in_prepared_cell(coord, base, n: int, W: int):
    """True where ``coord`` still lies in the cell its base was prepared for.

    The prepared stencil covers ``base .. base + W - 1``; it is the exact stencil iff
    ``xi - base - W/2`` is in ``[-1, 0]`` (at either end the dropped tap has zero weight).
    The bounds are widened by a few ulps of ``xi``: a compiled program may round ``xi``
    differently from the host (fused multiply-add), so a point on a grid node can land
    just outside.  A point that close to the boundary drops at most a tap of weight
    ``~psi(W/2) = e^-beta``, the same as :func:`_es_kernel` already discards.
    """
    xi = coord * (n / TWO_PI)
    d = xi - (base.astype(coord.dtype) + W // 2)
    tol = 8.0 * jnp.finfo(coord.dtype).eps * (jnp.abs(xi) + W)
    return (d >= -1.0 - tol) & (d <= tol)


def _interp_prepared(plan: NufftPlan, g, wk, wm, base):
    """:func:`_interp_points` with the stencil bases and weights supplied."""
    off = jnp.arange(plan.W)
    lk = (base[:, 0, None] + off[None, :]) % plan.nk
    lm = (base[:, 1, None] + off[None, :]) % plan.nm
    gg = g[lk[:, :, None], lm[:, None, :]]
    return jnp.sum(wk[:, :, None] * wm[:, None, :] * gg, axis=(1, 2))


def _spread_grouped(plan: NufftPlan, vals, wk, wm, buckets):
    """Transpose of :func:`_interp_prepared`, one block write per group of points.

    Per group, ``block[a, b] = sum_k v_k wk_k[a] wm_k[b]`` (an elementwise product then a
    reduction, which XLA spreads over the cores), then all blocks go through one
    :func:`_spread_blocks`.
    """
    W = plan.W
    vals = jnp.asarray(vals)
    dtype = jnp.result_type(vals, wk)
    if not buckets:
        return jnp.zeros((plan.nk, plan.nm), dtype=dtype)
    A = jnp.concatenate([wk * vals[:, None], jnp.zeros((1, W), dtype)])
    M = jnp.concatenate([wm, jnp.zeros((1, W), wm.dtype)])
    blocks = [jnp.sum(A[s][:, :, :, None] * M[s][:, :, None, :], axis=1) for s, _ in buckets]
    starts = [st for _, st in buckets]
    return _spread_blocks(plan, jnp.concatenate(blocks), jnp.concatenate(starts))
