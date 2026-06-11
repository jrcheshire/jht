"""MLX (Apple-GPU, fp32) on-grid HEALPix transforms -- the apply layer.

A separate, lower-accuracy (~fp32 machine precision) backend for **local development**
on Apple Silicon (which has no fp64).  It reuses jht's validated, static numpy plan
layer wholesale -- geometry (:class:`jht.healpix.RingInfo`), the a_lm index maps
(:func:`jht.healpix._tri_dense_maps`), ring weights (:mod:`jht.weights`) -- and the
fp64 lambda recursion (:func:`jht._lambda_table.lambda_table`), and runs only the
**contraction + phi-FFT in fp32 on the GPU**.  The dangerous recursion never leaves
fp64; the fp32 part is well-conditioned reductions, so this reaches fp32 machine
precision rather than garbage (Phase-0 spike: ~3e-7 vs JAX-fp64, flat to l~1000).

Conventions are identical to :mod:`jht.healpix` / :mod:`jht._reference` (healpy m-major
triangular a_lm, orthonormal Y, COSMO Q/U); the JAX path is the oracle and
``tests/test_mlx.py`` gates this against it at the fp32 tier.

MLX has no GPU complex scatter, so the synthesis ring-spectrum fold is done as a
**precomputed gather + masked sum** (the inverse of the per-ring aliasing), never a
scatter -- mirroring the gather discipline the JAX/CUDA path also uses.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, NamedTuple

import mlx.core as mx
import numpy as np

from .._lambda_table import lambda_table
from ..healpix import RingInfo, _tri_dense_maps, alm_size
from ..weights import pixel_weights


def _as_cmx(x) -> mx.array:
    """Array-like -> mlx complex64, **materialized**.

    The eval is load-bearing: a lazy ``.astype(complex64)`` (or host->device copy) that
    feeds the per-ring-group gather loop un-evaluated races on the Metal GPU and returns
    non-deterministic, corrupt values (see _STAGE_NOTE).  Every kernel input is forced.
    """
    a = x.astype(mx.complex64) if isinstance(x, mx.array) else mx.array(
        np.ascontiguousarray(np.asarray(x, dtype=np.complex64))
    )
    mx.eval(a)
    return a


def _as_rmx(x) -> mx.array:
    """Array-like -> mlx float32, **materialized** (see :func:`_as_cmx`)."""
    a = x.astype(mx.float32) if isinstance(x, mx.array) else mx.array(
        np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    )
    mx.eval(a)
    return a


# --------------------------------------------------------------------------- #
# prepared (cached) per-(nside, lmax, spin) plan
# --------------------------------------------------------------------------- #
class _Group(NamedTuple):
    N: int
    ring_idx: mx.array  # (n_g,) ring indices
    pix_idx: mx.array  # (n_g, N) flat pixel indices
    g_plus: mx.array  # (M,)  m mod N        (adjoint +m gather)
    g_minus: mx.array  # (M,)  (-m) mod N     (adjoint -m gather)
    fold_idx: mx.array  # (N, Kmax) source index into the extended (2M-1) ring spectrum
    fold_mask: mx.array  # (N, Kmax) float32 validity


class _Prepared(NamedTuple):
    nside: int
    lmax: int
    spin: int
    tables: tuple[mx.array, ...]  # (table0,) spin0 or (table_p, table_m) spin2; real fp32
    groups: list[_Group]
    phase: mx.array  # (M, nrings) exp(i m phi0)
    conj_phase: mx.array  # (M, nrings)
    gather: mx.array  # (M, M) tri->dense gather index
    valid: mx.array  # (M, M) float32 mask
    pack: mx.array  # (K,) dense->tri gather index
    pix_to_buf: mx.array  # (npix,) buf(group order) -> map order
    ring_to_col: mx.array  # (nrings,) group order -> ring order
    npix: int


def _fold_indices(N: int, M: int) -> tuple[np.ndarray, np.ndarray]:
    """Gather form of the per-ring aliasing fold for a length-``N`` ring, ``M=lmax+1``.

    The ring spectrum ``D`` (length ``N``) is built from the extended length-``2M-1``
    source ``[G_0..G_{M-1}, conj(G_1)..conj(G_{M-1})]`` placed at frequencies
    ``[m mod N (m=0..M-1), (-m) mod N (m=1..M-1)]``.  Returns ``(idx, mask)`` of shape
    ``(N, Kmax)`` with ``D[k] = sum_j source[idx[k,j]] * mask[k,j]`` -- a gather, so it
    is complex64-safe on the MLX GPU (which has no complex scatter).
    """
    kt = np.concatenate([np.arange(M) % N, (-np.arange(1, M)) % N]).astype(np.int64)  # (2M-1,)
    counts = np.bincount(kt, minlength=N)
    kmax = int(counts.max())
    idx = np.zeros((N, kmax), dtype=np.int64)
    mask = np.zeros((N, kmax), dtype=np.float32)
    pos = np.zeros(N, dtype=np.int64)
    for src in range(kt.shape[0]):
        k = int(kt[src])
        idx[k, pos[k]] = src
        mask[k, pos[k]] = 1.0
        pos[k] += 1
    return idx, mask


@lru_cache(maxsize=None)
def _prepare(nside: int, lmax: int, spin: int) -> _Prepared:
    if spin not in (0, 2):
        raise NotImplementedError(f"spin={spin} unsupported (only 0 and 2)")
    geo = RingInfo(nside)
    M = lmax + 1
    z = geo.z  # full grid (north + south); the fp32 tier does not exploit N/S symmetry

    if spin == 0:
        tables: tuple[mx.array, ...] = (mx.array(lambda_table(z, 0, lmax, dtype=np.float32)),)
    else:
        tables = (
            mx.array(lambda_table(z, 2, lmax, dtype=np.float32)),
            mx.array(lambda_table(z, -2, lmax, dtype=np.float32)),
        )

    m_pos = np.arange(M)
    phase_np = np.exp(1j * m_pos[:, None] * geo.phi0[None, :])  # (M, nrings)

    groups: list[_Group] = []
    buf_pix_parts = []
    ring_order_parts = []
    for N in np.unique(geo.npix_ring):
        N = int(N)
        ring_idx = np.flatnonzero(geo.npix_ring == N)
        pix_idx = geo.startpix[ring_idx][:, None] + np.arange(N)[None, :]
        fold_idx, fold_mask = _fold_indices(N, M)
        groups.append(
            _Group(
                N=N,
                ring_idx=mx.array(ring_idx),
                pix_idx=mx.array(pix_idx),
                g_plus=mx.array(m_pos % N),
                g_minus=mx.array((-m_pos) % N),
                fold_idx=mx.array(fold_idx),
                fold_mask=mx.array(fold_mask),
            )
        )
        buf_pix_parts.append(pix_idx.ravel())
        ring_order_parts.append(ring_idx)

    buf_pix = np.concatenate(buf_pix_parts)  # (npix,)
    pix_to_buf = np.empty(geo.npix, dtype=np.int64)
    pix_to_buf[buf_pix] = np.arange(geo.npix)
    ring_order = np.concatenate(ring_order_parts)  # (nrings,)
    ring_to_col = np.empty(geo.nrings, dtype=np.int64)
    ring_to_col[ring_order] = np.arange(geo.nrings)

    gather, valid, pack = _tri_dense_maps(lmax)
    pre = _Prepared(
        nside=nside,
        lmax=lmax,
        spin=spin,
        tables=tables,
        groups=groups,
        phase=mx.array(phase_np.astype(np.complex64)),
        conj_phase=mx.array(np.conj(phase_np).astype(np.complex64)),
        gather=mx.array(gather),
        valid=mx.array(valid.astype(np.float32)),
        pack=mx.array(pack),
        pix_to_buf=mx.array(pix_to_buf),
        ring_to_col=mx.array(ring_to_col),
        npix=geo.npix,
    )
    # MLX is lazy: force every cached device array to materialize now, while its numpy
    # source is still alive.  Without this the *first* transform that reads a freshly
    # prepared bundle races the host->device copy and returns corrupt values (later
    # calls are fine) -- an order-dependent ~O(1) error.
    leaves: list[mx.array] = [
        pre.phase, pre.conj_phase, pre.gather, pre.valid, pre.pack,
        pre.pix_to_buf, pre.ring_to_col, *pre.tables,
    ]
    for g in pre.groups:
        leaves += [g.ring_idx, g.pix_idx, g.g_plus, g.g_minus, g.fold_idx, g.fold_mask]
    mx.eval(*leaves)
    return pre


# --------------------------------------------------------------------------- #
# shared kernels
# --------------------------------------------------------------------------- #
def _tri_to_dense(a: mx.array, pre: _Prepared) -> mx.array:
    """(K,) triangular a_lm -> (M, lmax+1) dense grid (zero where l<m)."""
    return mx.take(a, pre.gather) * pre.valid  # complex * real mask


def _dense_to_tri(b: mx.array, pre: _Prepared) -> mx.array:
    """(M, lmax+1) dense -> (K,) triangular (a gather, not a scatter)."""
    mx.eval(b)  # stage the contraction (einsum) before the pack-gather (see _STAGE_NOTE)
    return mx.take(b.reshape(-1), pre.pack)


def _assemble_ring(plus: mx.array, minus: mx.array, g: _Group) -> mx.array:
    """Build the length-N ring spectra (n_g, N) from per-mode (M, n_g) ``plus``/``minus``.

    ``plus`` supplies the +m terms (m=0..M-1), ``minus`` the conj'd -m terms (m=1..M-1);
    the placement/aliasing is the precomputed gather ``g.fold_idx`` + ``g.fold_mask``.
    """
    src = mx.concatenate([plus.T, mx.conj(minus[1:].T)], axis=1)  # (n_g, 2M-1)
    gathered = mx.take(src, g.fold_idx, axis=1)  # (n_g, N, Kmax)
    return mx.sum(gathered * g.fold_mask, axis=-1)  # (n_g, N)


def _scatter_to_map(cols: list[mx.array], pre: _Prepared) -> mx.array:
    """Concatenate per-group ring values (group order) and gather into RING map order."""
    mx.eval(*cols)  # stage the per-ring-group FFT loop (see _STAGE_NOTE)
    buf = mx.concatenate(cols)  # (npix,) group order
    return buf[pre.pix_to_buf]


def _gather_to_rings(cols: list[mx.array], pre: _Prepared) -> mx.array:
    """Concatenate per-group (M, n_g) columns (group order) -> (M, nrings) ring order."""
    mx.eval(*cols)  # stage the per-ring-group FFT loop (see _STAGE_NOTE)
    V = mx.concatenate(cols, axis=1)  # (M, nrings) group order
    Vr = V[:, pre.ring_to_col]
    mx.eval(Vr)
    return Vr


# _STAGE_NOTE: MLX evaluates lazily, and on the Metal GPU a *single* lazy graph in which
# a heavy op (einsum contraction, batched FFT) feeds an advanced-index **gather** without
# an intervening eval races -- it returns NON-DETERMINISTIC, corrupt values (two identical
# calls can differ by O(1); the synthesis was deterministic, the adjoint was not).  The
# MLX CPU device is always correct, so this is a Metal-runtime hazard, not a numerics bug.
# Inserting mx.eval at each heavy-op -> gather boundary (contraction <-> FFT loop, and the
# einsum -> pack/scatter gathers) makes every call deterministic and fp32-accurate.  The
# evals are cheap (a handful of syncs per transform).  (Separately: do NOT interleave JAX
# dispatch with MLX-GPU calls in one process -- that also corrupts MLX-GPU; tests compute
# all JAX references first, then all MLX.  See docs/mlx.md / DISCREPANCIES.md.)


# --------------------------------------------------------------------------- #
# spin-0
# --------------------------------------------------------------------------- #
def _synthesis0(alm, nside: int, lmax: int) -> mx.array:
    pre = _prepare(nside, lmax, 0)
    ad = _tri_to_dense(_as_cmx(alm), pre)  # (M, L)
    F = mx.einsum("mlt,ml->mt", pre.tables[0], ad)  # (M, nrings) complex
    G = F * pre.phase
    mx.eval(G)  # stage the contraction before the FFT loop (see _STAGE_NOTE)
    cols = []
    for g in pre.groups:
        Gg = G[:, g.ring_idx]  # (M, n_g)
        D = _assemble_ring(Gg, Gg, g)  # (n_g, N)
        ring = mx.real(mx.fft.ifft(D, axis=1) * g.N)  # (n_g, N)
        cols.append(ring.reshape(-1))
    return _scatter_to_map(cols, pre)


def _adjoint_synthesis0(maps, nside: int, lmax: int) -> mx.array:
    pre = _prepare(nside, lmax, 0)
    m = _as_cmx(maps)  # materialized complex input (the un-eval'd cast races; see _STAGE_NOTE)
    cols = []
    for g in pre.groups:
        fr = mx.fft.fft(mx.take(m, g.pix_idx, axis=0), axis=1)  # (n_g, N)
        Vg = mx.take(fr, g.g_plus, axis=1).T * pre.conj_phase[:, g.ring_idx]  # (M, n_g)
        cols.append(Vg)
    V = _gather_to_rings(cols, pre)  # (M, nrings)
    b = mx.einsum("mlt,mt->ml", pre.tables[0], V)  # (M, L)
    return _dense_to_tri(b, pre)


# --------------------------------------------------------------------------- #
# spin-2
# --------------------------------------------------------------------------- #
def _synthesis2(alm, nside: int, lmax: int) -> mx.array:
    pre = _prepare(nside, lmax, 2)
    a = _as_cmx(alm)  # (2, K)
    aE = _tri_to_dense(a[0], pre)
    aB = _tri_to_dense(a[1], pre)
    cp = -(aE + 1j * aB)
    cm = -(aE - 1j * aB)
    Fp = mx.einsum("mlt,ml->mt", pre.tables[0], cp)  # (M, nrings)
    Fm = mx.einsum("mlt,ml->mt", pre.tables[1], cm)
    mx.eval(Fp, Fm)  # stage the contraction before the FFT loop (see _STAGE_NOTE)
    Qcols, Ucols = [], []
    for g in pre.groups:
        ph = pre.phase[:, g.ring_idx]
        D = _assemble_ring(Fp[:, g.ring_idx] * ph, Fm[:, g.ring_idx] * ph, g)
        qu = mx.fft.ifft(D, axis=1) * g.N
        Qcols.append(mx.real(qu).reshape(-1))
        Ucols.append(mx.imag(qu).reshape(-1))
    Q = _scatter_to_map(Qcols, pre)
    U = _scatter_to_map(Ucols, pre)
    return mx.stack([Q, U])


def _adjoint_synthesis2(maps, nside: int, lmax: int) -> mx.array:
    pre = _prepare(nside, lmax, 2)
    mm = _as_rmx(maps)  # (2, npix)
    Q = _as_cmx(mm[0])  # materialized (the un-eval'd cast races; see _STAGE_NOTE)
    U = _as_cmx(mm[1])
    pcols, mcols = [], []
    for g in pre.groups:
        QU = mx.take(Q, g.pix_idx, axis=0) + 1j * mx.take(U, g.pix_idx, axis=0)  # (n_g, N)
        W = mx.fft.fft(QU, axis=1)
        cph = pre.conj_phase[:, g.ring_idx]
        pcols.append(mx.take(W, g.g_plus, axis=1).T * cph)  # (M, n_g)
        mcols.append(mx.conj(mx.take(W, g.g_minus, axis=1).T) * cph)
    Fpb = _gather_to_rings(pcols, pre)
    Fmb = _gather_to_rings(mcols, pre)
    p2 = mx.einsum("mlt,mt->ml", pre.tables[0], Fpb)
    m2 = mx.einsum("mlt,mt->ml", pre.tables[1], Fmb)
    aE = 0.5 * (-p2 - m2)
    aB = 0.5 * (1j * p2 - 1j * m2)
    return mx.stack([_dense_to_tri(aE, pre), _dense_to_tri(aB, pre)])


# --------------------------------------------------------------------------- #
# public dispatch (spin 0 / 2)
# --------------------------------------------------------------------------- #
def synthesis(alm: Any, nside: int, lmax: int, spin: int = 0) -> mx.array:
    """``S: a_lm -> map`` on MLX (fp32).  spin 0 or 2."""
    if spin == 0:
        return _synthesis0(alm, nside, lmax)
    if spin == 2:
        return _synthesis2(alm, nside, lmax)
    raise NotImplementedError(f"spin={spin} unsupported (only 0 and 2)")


def adjoint_synthesis(maps: Any, nside: int, lmax: int, spin: int = 0) -> mx.array:
    """Exact transpose ``S^T = Y^H: map -> a_lm`` on MLX (fp32).  spin 0 or 2."""
    if spin == 0:
        return _adjoint_synthesis0(maps, nside, lmax)
    if spin == 2:
        return _adjoint_synthesis2(maps, nside, lmax)
    raise NotImplementedError(f"spin={spin} unsupported (only 0 and 2)")


@lru_cache(maxsize=None)
def _wvec(nside: int, spin: int, use_weights: bool) -> mx.array:
    w = _as_rmx(pixel_weights(nside, use_weights))
    w = w[None, :] if spin != 0 else w
    mx.eval(w)
    return w


def bare_analysis(
    maps: Any, nside: int, lmax: int, spin: int = 0, use_weights: bool = True
) -> mx.array:
    """``A0 = S^T W``: weight pixels, then adjoint-synthesize (single-shot)."""
    wmaps = _wvec(nside, spin, use_weights) * _as_rmx(maps)
    return adjoint_synthesis(wmaps, nside, lmax, spin)


def analysis(
    maps: Any, nside: int, lmax: int, spin: int = 0, niter: int = 3, use_weights: bool = True
) -> mx.array:
    """Approximate inverse ``map -> a_lm`` (A0 + Jacobi iteration) on MLX (fp32)."""
    m = _as_rmx(maps)
    a = bare_analysis(m, nside, lmax, spin, use_weights)
    for _ in range(niter):
        residual = m - synthesis(a, nside, lmax, spin)  # synthesis returns a real map
        a = a + bare_analysis(residual, nside, lmax, spin, use_weights)
    return a


def map2alm(
    maps: Any, nside: int, lmax: int, spin: int = 0, niter: int = 3, use_weights: bool = True
) -> mx.array:
    """healpy-idiom alias of :func:`analysis`."""
    return analysis(maps, nside, lmax, spin, niter, use_weights)


def alm_size_mlx(lmax: int) -> int:  # convenience re-export for harnesses
    return alm_size(lmax)
