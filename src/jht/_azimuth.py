"""Opt-in *looped* azimuth transform: a common-length chirp-z (Bluestein) path
for the polar-cap ring FFTs.

The default on-grid transform (:mod:`jht.healpix`) computes the azimuth FFTs by
grouping rings by length and emitting one FFT per distinct length.  A HEALPix map
at resolution ``nside`` has exactly ``nside`` distinct ring lengths
(``{4, 8, ..., 4*(nside-1)}`` for the polar caps, plus ``4*nside`` for the
equatorial belt), so that loop unrolls ``~nside`` distinct FFT kernels at trace
time.  Compile time / executable size therefore grow linearly with ``nside`` and
with the number of transforms in a graph -- an SHT-heavy differentiable graph
(e.g. a masked Wiener + bandpower forecast) can exceed XLA's executable-size cap
well before it exhausts memory.

The **looped** mode collapses the cap FFTs to *O(1)* compiled kernels.  Every
polar-cap length group holds exactly two rings (a north ring and its south
mirror), so the caps are shape-uniform and can be swept with a single
``lax.scan``.  The catch is that a length-``N`` ring needs an exact length-``N``
DFT and rings of different length cannot share an FFT kernel -- padding to a
common length changes which harmonics alias (``m -> m mod N``) and is invalid.
A **chirp-z (Bluestein) transform at one common FFT length** ``L`` sidesteps this:
it evaluates the pruned/aliased DFT ``s[j] = sum_m G[m] exp(2*pi*i*j*m/N)``
exactly (reproducing the ``m mod N`` fold automatically, since
``exp(2*pi*i*j*(m mod N)/N) == exp(2*pi*i*j*m/N)``) via a fixed length-``L``
convolution, so all cap groups run through one FFT kernel inside the scan.  The
equatorial belt keeps its native single FFT (already one kernel).

This trades compile size (``O(nside)`` -> ``O(1)`` kernels) for a bounded per-run
FLOP tax on the cap rings (each computed at the common length ``L >= N``); the
belt -- the bulk of the runtime -- is untouched.  The default stays the unrolled
path, which is faster for single transforms.  Select the mode with
:func:`set_azimuth_fft_mode` / :func:`enable_looped_fft`; it is read by
:func:`jht.synthesis` / :func:`jht.adjoint_synthesis` and forwarded as part of the
``_prepare`` cache key, so the whole analysis / masked / Wiener chain inherits it.

All static geometry / chirp-index tables are built as NumPy (never device arrays):
an ``lru_cache`` that first materializes a device array under a ``jit`` / ``grad`` /
``lax.scan`` trace would leak it as an ``UnexpectedTracerError`` on later reuse.
The chirps themselves are rebuilt inside the scan body from a small ``cap_N``
table (not baked), so the static constants stay ``O(nside)`` rather than the
``O(nside**2)`` a full stack of length-``L`` chirp tables would cost.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._nufft import _next_size

# --------------------------------------------------------------------------- #
# mode toggle
# --------------------------------------------------------------------------- #
_LEGAL_MODES = ("unrolled", "looped")
_AZIMUTH_FFT_MODE = "unrolled"


def set_azimuth_fft_mode(mode: str) -> None:
    """Select the azimuth-FFT strategy for subsequent transforms.

    * ``"unrolled"`` (default) -- one FFT kernel per distinct ring length
      (``~nside`` kernels); fastest for a single transform, but compile size grows
      with ``nside * (#SHTs in the graph)``.
    * ``"looped"`` -- route the polar-cap FFTs through a single common-length
      chirp-z ``lax.scan`` (*O(1)* compiled kernels), at a bounded per-run FLOP
      tax on the cap rings.  Use for SHT-heavy / high-``nside`` differentiable
      graphs where the unrolled path's compile time / executable size is the
      bottleneck.

    The mode is a process-global flip (like :func:`jht.enable_compilation_cache`):
    set it once before the first transform compiles.  It is part of the transform
    cache key, so flipping it compiles a fresh kernel and both variants coexist.
    """
    if mode not in _LEGAL_MODES:
        raise ValueError(f"unknown azimuth FFT mode {mode!r}; expected one of {_LEGAL_MODES}")
    global _AZIMUTH_FFT_MODE
    _AZIMUTH_FFT_MODE = mode


def get_azimuth_fft_mode() -> str:
    """Return the current azimuth-FFT mode (see :func:`set_azimuth_fft_mode`)."""
    return _AZIMUTH_FFT_MODE


def enable_looped_fft() -> None:
    """Shortcut for ``set_azimuth_fft_mode("looped")``."""
    set_azimuth_fft_mode("looped")


_CAP_CHUNK: int | str = "auto"
_CAP_CHUNK_AUTO = 32


def set_cap_chunk(g: int | str) -> None:
    """Set how many polar-cap rings the looped mode transforms per scan step.

    The looped mode's chirp-z transform runs one ring length per ring pair; ``g`` ring
    pairs share one scan step as a batched FFT, so a transform takes
    ``ceil(n_cap / g)`` steps (``n_cap = nside - 1``) instead of ``n_cap``.  ``"auto"``
    (default) is 32; ``1`` is one ring pair per step.  Per-ring arithmetic is the same
    for every ``g``.  Each step holds ``O(g * B * L)`` complex values (``B`` the
    channels per ring pair, ``L`` the common FFT length), times any ``vmap`` batch.

    Process-global, like :func:`set_azimuth_fft_mode`; it is part of the transform
    cache key, so changing it compiles fresh kernels.
    """
    if isinstance(g, bool) or not (g == "auto" or (isinstance(g, int) and g >= 1)):
        raise ValueError(f"cap chunk must be 'auto' or an int >= 1, got {g!r}")
    global _CAP_CHUNK
    _CAP_CHUNK = g


def get_cap_chunk() -> int | str:
    """Return the current cap-chunk setting (see :func:`set_cap_chunk`)."""
    return _CAP_CHUNK


def resolve_cap_chunk(setting: int | str, n_cap: int) -> int:
    """Ring pairs per scan step for ``n_cap`` cap groups."""
    g = _CAP_CHUNK_AUTO if setting == "auto" else int(setting)
    return max(1, min(g, n_cap))


# --------------------------------------------------------------------------- #
# cap plan (static NumPy tables)
# --------------------------------------------------------------------------- #
class CapPlan(NamedTuple):
    """Static inputs for the looped cap azimuth transform (small NumPy vectors).

    The ``(n_cap, 2, n_out)`` gather/mask and the ``(n_cap, 2, M)`` phase are built
    in-trace by :func:`cap_gather_mask` and :func:`cap_conj_phase`, so they are runtime
    buffers rather than embedded XLA constants.
    """

    M: int  # lmax + 1
    L: int  # common Bluestein FFT length
    G: int  # cap groups per scan step (see set_cap_chunk)
    n_out: int  # padded cap ring length = max cap N = 4*(nside-1)
    n_cap: int  # number of cap length-groups
    cap_N: np.ndarray  # (n_cap,)     per-group ring length
    ring_idx: np.ndarray  # (n_cap, 2)   ring columns (north, south) into (M, nrings)
    take: np.ndarray  # (npix_cap,)  gather (n_cap, 2, n_out) -> cap buffer order
    start: np.ndarray  # (n_cap, 2)   first map pixel of each cap ring
    phi0: np.ndarray  # (n_cap, 2)   azimuth of each cap ring's first pixel


def build_cap_plan(geo, cap_groups: list, lmax: int, cap_chunk: int | str = "auto") -> CapPlan:
    """Build the static cap plan from the RingInfo geometry and cap ring groups.

    ``cap_groups`` is ``_ring_groups(...)[:-1]`` -- every group except the belt --
    each with ``ring_idx`` of size 2 (north ring + south mirror).  ``cap_chunk``: see
    :func:`set_cap_chunk`.
    """
    M = lmax + 1
    n_cap = len(cap_groups)
    cap_N = np.array([g.N for g in cap_groups], dtype=np.int64)  # ascending
    n_out = int(cap_N.max())  # = 4*(nside-1), the largest cap length
    L = _next_size(M + n_out - 1)  # >= P + Q - 1 for both synth (M->n_out) and adj (n_out->M)

    ring_idx = np.stack([g.ring_idx for g in cap_groups]).astype(np.int64)  # (n_cap, 2)

    # take: gather a (n_cap, 2, n_out) buffer -> the cap portion of the assembly
    # buffer, which is concat over groups of g.pix_idx.ravel() = [north 0..N-1, south 0..N-1].
    take_list: list[int] = []
    for gi, g in enumerate(cap_groups):
        N = int(g.N)
        for r in range(2):
            for j in range(N):
                take_list.append(gi * (2 * n_out) + r * n_out + j)
    take = np.array(take_list, dtype=np.int64)  # (npix_cap,)

    # ring pixels are contiguous: g.pix_idx[r] == startpix[ring] + arange(N)
    start = geo.startpix[ring_idx].astype(np.int64)  # (n_cap, 2)
    phi0 = geo.phi0[ring_idx]  # (n_cap, 2)

    G = resolve_cap_chunk(cap_chunk, n_cap)
    return CapPlan(M=M, L=L, G=G, n_out=n_out, n_cap=n_cap, cap_N=cap_N, ring_idx=ring_idx,
                   take=take, start=start, phi0=phi0)


def cap_gather_mask(plan: CapPlan) -> tuple[jax.Array, jax.Array]:
    """In-trace ``(map_gather, mask)``, both ``(n_cap, 2, n_out)``.

    ``map_gather`` is the map pixel feeding each padded ring sample (0 in the pad);
    ``mask`` is 1.0 where the sample is a real pixel (``j < N``), else 0.0.
    """
    j = jnp.arange(plan.n_out)[None, None, :]
    inside = jnp.broadcast_to(j < jnp.asarray(plan.cap_N)[:, None, None], (plan.n_cap, 2, plan.n_out))
    gather = jnp.where(inside, jnp.asarray(plan.start)[:, :, None] + j, 0)
    return jax.lax.optimization_barrier((gather, inside.astype(jnp.float64)))


def cap_conj_phase(plan: CapPlan) -> jax.Array:
    """In-trace ``(n_cap, 2, M)`` adjoint post-phase ``e^{-i m phi0}`` per cap ring."""
    m = jnp.arange(plan.M, dtype=jnp.float64)
    z = -1j * m[None, None, :] * jnp.asarray(plan.phi0)[:, :, None]
    return jax.lax.optimization_barrier(jnp.exp(z))


# --------------------------------------------------------------------------- #
# Bluestein pruned DFT, swept over chunks of cap groups by lax.scan
# --------------------------------------------------------------------------- #
def _bluestein_scan(
    cols: jax.Array, cap_N: np.ndarray, P: int, Q: int, L: int, sign: int, G: int = 1
) -> jax.Array:
    """Chirp-z pruned DFT ``X[q] = sum_{p<P} x[p] exp(sign*2j*pi*p*q/N)``, q<Q.

    ``cols`` is ``(n_cap, B, P)``; the modulus ``N = cap_N[g]`` varies per group but
    the FFT length ``L`` is common, so the scan body compiles one length-``L`` FFT,
    batched over ``G`` groups per step (``n_cap`` is zero-padded to a multiple of
    ``G`` with a dummy ``N = 4``; the pad rows are dropped).  Chirps are built in-body
    from the per-group ``N`` with an exact ``k^2 mod 2N`` argument reduction
    (mandatory: at ``N=4`` / large ``lmax`` the naive ``exp(i*pi*k^2/N)`` loses ~8
    digits).  The chirp kernel is placed scatter-free via a static signed-index
    array.  Returns ``(n_cap, B, Q)``.
    """
    # static index tables (baked into the jit trace as constants)
    p2 = (np.arange(P) ** 2).astype(np.int64)  # (P,)
    q2 = (np.arange(Q) ** 2).astype(np.int64)  # (Q,)
    # signed convolution-kernel indices k = q - p in [-(P-1), Q-1], length-L buffer:
    #   pos 0..Q-1 -> k = pos ; pos L-(P-1)..L-1 -> k = pos - L (negative half).
    sidx = np.zeros(L, dtype=np.int64)
    kused = np.zeros(L, dtype=np.float64)
    sidx[:Q] = np.arange(Q)
    kused[:Q] = 1.0
    neg = np.arange(1, P)
    sidx[L - neg] = neg  # k^2 is even, store |k|
    kused[L - neg] = 1.0
    k2 = (sidx * sidx).astype(np.int64)  # (L,)
    pad = L - P

    def _chirp(sq_int, N, s):
        # exp(s * i*pi * (sq mod 2N) / N); sq_int static int, N dynamic scalar
        r = jnp.mod(sq_int, 2 * N)
        return jnp.exp(s * 1j * jnp.pi * r / N)

    def body(carry, xs):
        N, x = xs  # N: (G, 1), x: (G, B, P)
        apre = _chirp(p2, N, sign)  # (G, P)
        a = x * apre[:, None, :]  # (G, B, P)
        abuf = jnp.pad(a, ((0, 0), (0, 0), (0, pad)))  # (G, B, L)
        kbuf = kused * _chirp(k2, N, -sign)  # (G, L)
        conv = jnp.fft.ifft(jnp.fft.fft(abuf, axis=-1) * jnp.fft.fft(kbuf, axis=-1)[:, None, :], axis=-1)
        out = conv[:, :, :Q] * _chirp(q2, N, sign)[:, None, :]  # (G, B, Q)
        return carry, out

    n_cap, B = cols.shape[0], cols.shape[1]
    n_chunk = -(-n_cap // G)
    n_pad = n_chunk * G - n_cap
    N_all = np.concatenate([np.asarray(cap_N, dtype=np.int64), np.full(n_pad, 4, dtype=np.int64)])
    cols_p = jnp.pad(cols, ((0, n_pad), (0, 0), (0, 0))) if n_pad else cols
    xs = (N_all.reshape(n_chunk, G, 1), cols_p.reshape(n_chunk, G, B, P))
    _, ys = jax.lax.scan(body, 0.0, xs)
    return ys.reshape(n_chunk * G, B, Q)[:n_cap]  # (n_cap, B, Q)


# --------------------------------------------------------------------------- #
# cap synthesis / adjoint (spin 0 and 2 unified via +2 / -2 channels)
# --------------------------------------------------------------------------- #
def cap_synth(plan: CapPlan, Fp: jax.Array, Fm: jax.Array) -> jax.Array:
    """Cap contribution to ``synthesis``.  ``Fp, Fm`` are ``(M, nrings)`` spectra
    (already multiplied by the azimuth phase); spin-0 passes ``Fp == Fm == G``.

    Returns the complex ring samples ``qu`` for the cap pixels, in assembly-buffer
    order (``(npix_cap,)``): spin-0 takes the real part, spin-2 splits ``Re``/``Im``.
    """
    M, n_out = plan.M, plan.n_out
    m_nonzero = np.arange(M) > 0  # channel B (the -m half) excludes m=0
    A = jnp.transpose(Fp[:, plan.ring_idx], (1, 2, 0))  # (n_cap, 2, M)
    B = jnp.transpose(Fm[:, plan.ring_idx] * m_nonzero[:, None, None], (1, 2, 0))
    cols = jnp.concatenate([A, B], axis=1)  # (n_cap, 4, M): [A_n, A_s, B_n, B_s]
    out = _bluestein_scan(cols, plan.cap_N, P=M, Q=n_out, L=plan.L, sign=+1, G=plan.G)  # (n_cap, 4, n_out)
    qu = out[:, 0:2] + jnp.conj(out[:, 2:4])  # (n_cap, 2, n_out) = A_out + conj(B_out)
    return qu.reshape(-1)[plan.take]  # (npix_cap,)


def cap_adjoint(plan: CapPlan, cols: jax.Array) -> jax.Array:
    """Cap contribution to ``adjoint_synthesis`` for one or two channels.

    ``cols`` is ``(n_cap, C*2, n_out)`` padded ring samples (``C`` channels x 2 rings,
    zero beyond each ring's length).  Returns ``(C*2, M, n_cap*2)`` V-columns already
    multiplied by ``e^{-i m phi0}`` -- one ``(M, n_cap*2)`` block per (channel, kept
    for the caller to concatenate with the belt columns).  Column order per channel is
    ``[g0_north, g0_south, g1_north, ...]`` to match the unrolled ``concat(cols)`` order.
    """
    M = plan.M
    out = _bluestein_scan(cols, plan.cap_N, P=plan.n_out, Q=M, L=plan.L, sign=-1, G=plan.G)  # (n_cap, C*2, M)
    C2 = cols.shape[1]
    n_ch = C2 // 2
    # reshape to (n_ch, 2, ...) so conj_phase (n_cap, 2, M) broadcasts over rings
    out = out.reshape(plan.n_cap, n_ch, 2, M)  # (n_cap, C, 2, M)
    V = out * cap_conj_phase(plan)[:, None, :, :]  # (n_cap, C, 2, M)
    # -> per channel: (M, n_cap*2) with column order [g0_n, g0_s, g1_n, ...]
    V = jnp.transpose(V, (1, 3, 0, 2))  # (C, M, n_cap, 2)
    return V.reshape(n_ch, M, plan.n_cap * 2)  # (C, M, n_cap*2)
