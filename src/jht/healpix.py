"""HEALPix (RING) geometry and the on-grid spherical-harmonic transforms.

Conventions (verified vs healpy / ducc0; see ``docs/design.md``):

* a_lm: healpy m-major triangular packing, ``idx(l,m) = m*(2*lmax+1-m)//2 + l``,
  only ``m >= 0`` stored, size ``(lmax+1)(lmax+2)/2``.
* orthonormal Y_lm with the Condon-Shortley phase (``map = sum a_lm Y_lm``).
* polarization is HEALPix-internal (COSMO) Q/U; no pixel window is applied
  (matches healpy ``pixwin=False``).

``synthesis`` is ``S: a_lm -> map``.  ``adjoint_synthesis`` is the *exact*
transpose ``S^T = Y^T`` (unweighted; **not** the weighted analysis), which is the
operator the bk-jax seam needs and the VJP of synthesis.

**Phase-1 fast path.**  The transforms are vectorized over m (one fused
``lax.scan`` per spin via :mod:`jht._recursion`), batch the per-ring phi-FFTs by
ring length (the equatorial belt in one FFT; the polar cap by length-group), and
are ``jit``-compiled.  Geometry, recursion plans, and index maps are built once
per ``(nside, lmax, spin)`` and cached (``lru_cache`` on :func:`_prepare`), so the
Jacobi iteration in :func:`jht.analysis` reuses one compiled kernel.
Numerics are identical to the eager Phase-0 reference (:mod:`jht._reference`),
which is retained as a validation oracle.  Library code does not enable x64;
callers opt in per entry point.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._recursion import (
    adjoint_contract_eo,
    adjoint_contract_spin2_ns,
    build_recursion_plan,
    synth_contract_eo,
    synth_contract_spin2_ns,
)


# --------------------------------------------------------------------------- #
# a_lm layout (healpy)
# --------------------------------------------------------------------------- #
def alm_size(lmax: int) -> int:
    return (lmax + 1) * (lmax + 2) // 2


def alm_column_base(m: int, lmax: int) -> int:
    """Start index of the contiguous column ``a_{l,m}, l = m..lmax``."""
    return m * (2 * lmax + 1 - m) // 2 + m


@lru_cache(maxsize=None)
def alm_metric_weight(lmax: int) -> np.ndarray:
    """Per-mode inner-product weight ``2 - delta_{m0}`` for the healpy ``m >= 0`` packing.

    Only ``m >= 0`` is stored; a real field's ``m < 0`` half is implied by conjugate
    symmetry, so the alm inner product / power carries weight 1 at ``m=0`` and 2 at
    ``m>0``.  This is the metric ``G`` in which :func:`adjoint_synthesis` is the strict
    transpose of :func:`synthesis` (``<S a, v>_map == <a, S^T v>_alm`` with this weight),
    and it is the **differentiability convention bridge**: JAX's native VJP returns

        ``jax.vjp(synthesis)(v) == G * conj(adjoint_synthesis(v))``

    (exact for spin 0; for spin 2 also exact at ``m>0`` -- the ``m=0`` modes carry an
    extra E/B-mixing phantom term, see ``docs/design.md``).  Shape ``(alm_size(lmax),)``.
    """
    ms = np.concatenate([np.full(lmax + 1 - m, m) for m in range(lmax + 1)])
    return np.where(ms == 0, 1.0, 2.0)


def _tri_dense_maps(lmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index maps between healpy-triangular a_lm and the dense ``(M, lmax+1)`` grid.

    Returns ``(gather_idx, valid_mask, pack_idx)``:
    * ``gather_idx[m,l]`` -> flat triangular index (0 where invalid), for
      ``tri_to_dense`` (alm -> dense, a gather);
    * ``valid_mask[m,l]`` -> True where ``l >= m``;
    * ``pack_idx[k]`` -> flat dense index ``m*M + l`` of triangular entry ``k``, for
      ``dense_to_tri`` (dense -> alm) as a **gather** ``dense.ravel()[pack_idx]``.
      A scatter (``at[...].set(mode='drop')``) here is catastrophically slow in fp64
      on GPU -- ~38000x the gather, profiled; see ``docs/gpu.md`` -- so the adjoint
      packs via this gather, mirroring the forward.
    """
    M = lmax + 1
    gather = np.zeros((M, M), dtype=np.int64)
    valid = np.zeros((M, M), dtype=bool)
    pack = np.zeros(alm_size(lmax), dtype=np.int64)
    for m in range(M):
        base = alm_column_base(m, lmax)
        ells = np.arange(m, lmax + 1)
        idx = base + (ells - m)
        gather[m, m:] = idx
        valid[m, m:] = True
        pack[idx] = m * M + ells
    return gather, valid, pack


# --------------------------------------------------------------------------- #
# RING geometry
# --------------------------------------------------------------------------- #
class RingInfo:
    """Per-ring HEALPix RING geometry (numpy; static for a given nside)."""

    def __init__(self, nside: int):
        i = np.arange(1, 4 * nside, dtype=np.int64)  # 1-based ring index
        npr = np.empty(i.shape, dtype=np.int64)
        z = np.empty(i.shape, dtype=np.float64)
        phi0 = np.empty(i.shape, dtype=np.float64)

        north = i < nside
        south = i > 3 * nside
        equ = ~north & ~south

        npr[north] = 4 * i[north]
        z[north] = 1.0 - i[north] ** 2 / (3.0 * nside**2)
        phi0[north] = np.pi / (4.0 * i[north])

        ii = 4 * nside - i[south]
        npr[south] = 4 * ii
        z[south] = -(1.0 - ii**2 / (3.0 * nside**2))
        phi0[south] = np.pi / (4.0 * ii)

        ieq = i[equ]
        npr[equ] = 4 * nside
        z[equ] = 4.0 / 3.0 - 2.0 * ieq / (3.0 * nside)
        phi0[equ] = np.where((ieq - nside) % 2 == 0, np.pi / (4.0 * nside), 0.0)

        self.nside = nside
        self.npix = 12 * nside**2
        self.z = z
        self.npix_ring = npr
        self.phi0 = phi0
        self.startpix = np.concatenate([[0], np.cumsum(npr)[:-1]])
        self.nrings = len(i)


class _RingGroup(NamedTuple):
    """Rings sharing a length ``N`` (statically batched into one FFT)."""

    N: int
    ring_idx: np.ndarray  # (n_g,) ring indices into the (nrings,) arrays
    pix_idx: np.ndarray  # (n_g, N) flat pixel indices in the map
    k_plus: np.ndarray  # (M,)   +m fold-in columns  (m mod N)
    k_minus: np.ndarray  # (M-1,) -m fold-in columns  ((-m) mod N), m>=1
    g_plus: np.ndarray  # (M,)   +m gather columns    (m mod N)
    g_minus: np.ndarray  # (M,)   -m gather columns    ((-m) mod N)


def _ring_groups(geo: RingInfo, lmax: int) -> list[_RingGroup]:
    m_pos = np.arange(lmax + 1)
    groups = []
    for N in np.unique(geo.npix_ring):
        N = int(N)
        ring_idx = np.flatnonzero(geo.npix_ring == N)
        pix_idx = geo.startpix[ring_idx][:, None] + np.arange(N)[None, :]
        groups.append(
            _RingGroup(
                N=N,
                ring_idx=ring_idx,
                pix_idx=pix_idx,
                k_plus=m_pos % N,
                k_minus=(-m_pos[1:]) % N,
                g_plus=m_pos % N,
                g_minus=(-m_pos) % N,
            )
        )
    return groups


# --------------------------------------------------------------------------- #
# prepared (cached) transform kernels
# --------------------------------------------------------------------------- #
class _Prepared(NamedTuple):
    synth: Callable[[jax.Array], jax.Array]
    adj: Callable[[jax.Array], jax.Array]


@lru_cache(maxsize=None)
def _prepare(nside: int, lmax: int, spin: int) -> _Prepared:
    if spin not in (0, 2):
        raise NotImplementedError(f"spin={spin} unsupported (only 0 and 2)")
    # lru_cache => each warning fires once per (nside, lmax, spin), not per call.
    if not getattr(jax.config, "jax_enable_x64", True):  # getattr: the attr is dynamic (mypy)
        warnings.warn(
            "jht: jax_enable_x64 is OFF -- transforms will silently run in float32 "
            "(~1e-5 accuracy tier, far below the documented float64 contract). "
            'Enable it before creating any array: jax.config.update("jax_enable_x64", True).',
            stacklevel=2,
        )
    if 2 * lmax > 3 * nside:  # lmax > 1.5*nside, in exact integer arithmetic
        warnings.warn(
            f"jht: lmax={lmax} exceeds the design band-limit ceiling 1.5*nside "
            f"(= {1.5 * nside:g} for nside={nside}); accuracy above the ceiling is "
            "unvalidated (see docs/accuracy.md).",
            stacklevel=2,
        )
    geo = RingInfo(nside)
    npix = geo.npix
    groups = _ring_groups(geo, lmax)
    gather, valid, pack = _tri_dense_maps(lmax)
    gather_j = jnp.asarray(gather)
    valid_j = jnp.asarray(valid)
    pack_j = jnp.asarray(pack)

    # per-(m, ring) azimuth phase e^{i m phi0_r}
    m_pos = np.arange(lmax + 1)
    phase = jnp.asarray(np.exp(1j * m_pos[:, None] * geo.phi0[None, :]))  # (M, nrings)
    conj_phase = jnp.conj(phase)

    # Combined-gather assembly indices.  The per-group output writes are hoisted out of the
    # ring loop into one static-permutation GATHER: the ~nside-way fp64/complex *scatter*
    # unroll is what tips the full synth over the ptxas / compile-RAM limit at nside=2048
    # (the per-length FFTs stay -- a HEALPix ring of length N needs an exact length-N FFT,
    # so they cannot be batched to a common length).  ``buf`` is each group's ring values
    # concatenated in group order; ``pix_to_buf`` / ``ring_to_col`` are the inverse perms
    # back to map-pixel / ring-column order.
    buf_pix = np.concatenate([g.pix_idx.ravel() for g in groups])  # (npix,) permutation
    pix_to_buf_np = np.empty(npix, dtype=np.int64)
    pix_to_buf_np[buf_pix] = np.arange(npix)
    pix_to_buf = jnp.asarray(pix_to_buf_np)
    ring_order = np.concatenate([g.ring_idx for g in groups])  # (nrings,) permutation
    ring_to_col_np = np.empty(geo.nrings, dtype=np.int64)
    ring_to_col_np[ring_order] = np.arange(geo.nrings)
    ring_to_col = jnp.asarray(ring_to_col_np)

    # --- North/South symmetry geometry (recursion runs on the north half + equator) ---
    # rings 0..2*nside-1 are north cap + equator; ring (4*nside-2-r) is the south
    # reflection of north ring r (r = 0..2*nside-2); the equator (r=2*nside-1) is self.
    t_half = 2 * nside
    x_half = jnp.asarray(geo.z[:t_half])
    south_src = np.arange(t_half - 1)  # north non-equator half indices
    south_tgt = 4 * nside - 2 - south_src  # their south full-ring indices
    sign_m = jnp.asarray(((-1.0) ** np.arange(lmax + 1))[:, None])  # (M, 1)

    def build_full(Ftot, Fsig):  # (M, t_half) x2 -> (M, nrings) full F
        F = jnp.zeros((lmax + 1, geo.nrings), dtype=jnp.complex128)
        F = F.at[:, :t_half].set(Ftot, unique_indices=True)
        return F.at[:, south_tgt].set(sign_m * Fsig[:, south_src], unique_indices=True)

    def fold_south(V):  # (M, nrings) -> Vn (north half), Vs = (-1)^m V_south
        Vs = jnp.zeros((lmax + 1, t_half), dtype=jnp.complex128)
        Vs = Vs.at[:, south_src].set(V[:, south_tgt], unique_indices=True)
        return V[:, :t_half], sign_m * Vs

    def tri_to_dense(alm):  # (K,) -> (M, lmax+1)
        return jnp.where(valid_j, alm[gather_j], 0.0 + 0.0j)

    def dense_to_tri(b_dense):  # (M, lmax+1) -> (K,): gather, not scatter (fp64 GPU)
        return b_dense.ravel()[pack_j]

    if spin == 0:
        plan = build_recursion_plan(geo.z[:t_half], 0, lmax)

        @jax.jit
        def synth(alm):
            Ftot, Fsig = synth_contract_eo(plan, x_half, tri_to_dense(alm))
            G = build_full(Ftot, Fsig) * phase  # (M, nrings)
            cols = []
            for g in groups:
                Cp = G[:, g.ring_idx]  # (M, n_g)
                D = jnp.zeros((g.ring_idx.size, g.N), dtype=jnp.complex128)
                D = D.at[:, g.k_plus].add(Cp.T)
                D = D.at[:, g.k_minus].add(jnp.conj(Cp[1:].T))
                cols.append(jnp.real(jnp.fft.ifft(D, axis=1) * g.N).ravel())  # (n_g*N,)
            return jnp.concatenate(cols)[pix_to_buf]  # one gather, not ~nside scatters

        @jax.jit
        def adj(m):
            cols = []
            for g in groups:
                fr = jnp.fft.fft(m[g.pix_idx], axis=1)  # (n_g, N)
                ph = conj_phase[:, g.ring_idx].T  # (n_g, M)
                cols.append((fr[:, g.g_plus] * ph).T)  # (M, n_g)
            V = jnp.concatenate(cols, axis=1)[:, ring_to_col]  # (M, nrings), one gather
            Vn, Vs = fold_south(V)
            return dense_to_tri(adjoint_contract_eo(plan, x_half, Vn, Vs))

        return _Prepared(synth, adj)

    plan_p = build_recursion_plan(geo.z[:t_half], 2, lmax)
    plan_m = build_recursion_plan(geo.z[:t_half], -2, lmax)

    @jax.jit
    def synth2(alm2d):  # (2, K) -> (2, npix)
        aE = tri_to_dense(alm2d[0])
        aB = tri_to_dense(alm2d[1])
        FpN, FpS, FmN, FmS = synth_contract_spin2_ns(
            plan_p, plan_m, x_half, -(aE + 1j * aB), -(aE - 1j * aB)
        )
        Fp = build_full(FpN, FpS) * phase
        Fm = build_full(FmN, FmS) * phase
        cols = []
        for g in groups:
            Cp = Fp[:, g.ring_idx]
            Cm = Fm[:, g.ring_idx]
            D = jnp.zeros((g.ring_idx.size, g.N), dtype=jnp.complex128)
            D = D.at[:, g.k_plus].add(Cp.T)
            D = D.at[:, g.k_minus].add(jnp.conj(Cm[1:].T))
            cols.append((jnp.fft.ifft(D, axis=1) * g.N).ravel())  # complex (n_g*N,)
        qu = jnp.concatenate(cols)[pix_to_buf]  # (npix,) complex, one gather
        return jnp.stack([jnp.real(qu), jnp.imag(qu)])

    @jax.jit
    def adj2(maps2d):  # (2, npix) -> (2, K)
        colsp = []
        colsm = []
        for g in groups:
            W = jnp.fft.fft(maps2d[0][g.pix_idx] + 1j * maps2d[1][g.pix_idx], axis=1)  # (n_g, N)
            ph = conj_phase[:, g.ring_idx].T  # (n_g, M)
            colsp.append((W[:, g.g_plus] * ph).T)  # (M, n_g)
            colsm.append((jnp.conj(W[:, g.g_minus]) * ph).T)  # (M, n_g)
        Vp = jnp.concatenate(colsp, axis=1)[:, ring_to_col]  # (M, nrings), one gather
        Vm = jnp.concatenate(colsm, axis=1)[:, ring_to_col]
        Vpn, Sp = fold_south(Vp)
        Vmn, Sm = fold_south(Vm)
        p2, m2 = adjoint_contract_spin2_ns(plan_p, plan_m, x_half, Vpn, Sp, Vmn, Sm)
        aE = 0.5 * (-p2 - m2)
        aB = 0.5 * (1j * p2 - 1j * m2)
        return jnp.stack([dense_to_tri(aE), dense_to_tri(aB)])

    return _Prepared(synth2, adj2)


# --------------------------------------------------------------------------- #
# public transforms
# --------------------------------------------------------------------------- #
def synthesis(alm, nside: int, lmax: int, spin: int = 0) -> jax.Array:
    """``S: a_lm -> map`` on the HEALPix RING grid.

    Parameters
    ----------
    alm : complex array
        healpy-packed coefficients: shape ``(alm_size(lmax),)`` for ``spin=0``,
        ``(2, alm_size(lmax))`` (E, B) for ``spin=2``.
    nside, lmax : int
    spin : int
        ``0`` (temperature) or ``2`` (polarization Q/U).

    Returns
    -------
    map : real array, ``(12 nside**2,)`` for ``spin=0`` or ``(2, 12 nside**2)``
        (Q, U) for ``spin=2``, in RING order.
    """
    alm = jnp.asarray(alm)
    expect = (alm_size(lmax),) if spin == 0 else (2, alm_size(lmax))
    if alm.shape != expect:
        raise ValueError(
            f"alm shape {alm.shape} != expected {expect} for lmax={lmax}, spin={spin} "
            "(a wrong-size alm would otherwise be silently clamped by the gather)"
        )
    return _prepare(nside, lmax, spin).synth(alm)


def adjoint_synthesis(m, nside: int, lmax: int, spin: int = 0) -> jax.Array:
    """``S^T = Y^H : map -> a_lm`` -- the *exact* transpose of :func:`synthesis`.

    ``b_{l,m} = sum_p conj(Y_{l,m}(p)) m_p``.  This is **not** the weighted
    analysis (the approximate inverse); it is the operator the bk-jax seam needs
    and the cotangent of synthesis, satisfying

        adjoint_synthesis(v) == (Npix / 4pi) * healpy.map2alm(v, iter=0)   (no weights)

    and the inner-product identity ``<S a, v>_map == <a, S^T v>_alm`` in the
    ``(2 - delta_{m0})``-weighted a_lm inner product.  For ``spin=2`` the spin-2
    adjoint carries the factor-1/2 (m>0 conjugate-symmetry weight) so it is the
    strict transpose; the input is ``(2, npix)`` (Q, U) and the output ``(2, K)``.
    """
    m = jnp.asarray(m)
    npix = 12 * nside * nside
    expect = (npix,) if spin == 0 else (2, npix)
    if m.shape != expect:
        raise ValueError(
            f"map shape {m.shape} != expected {expect} for nside={nside}, spin={spin} "
            "(a wrong-size map would otherwise be silently clamped by the gather)"
        )
    return _prepare(nside, lmax, spin).adj(m)
