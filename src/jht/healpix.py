"""HEALPix (RING) geometry and the on-grid spherical-harmonic transforms.

Conventions (verified vs healpy / ducc0; see ``docs/design.md``):

* a_lm: healpy m-major triangular packing, ``idx(l,m) = m*(2*lmax+1-m)//2 + l``,
  only ``m >= 0`` stored, size ``(lmax+1)(lmax+2)/2``.
* orthonormal Y_lm with the Condon-Shortley phase (``map = sum a_lm Y_lm``).
* polarization is HEALPix-internal (COSMO) Q/U; no pixel window is applied
  (matches healpy ``pixwin=False``).

``synthesis`` is ``S: a_lm -> map``.  ``adjoint_synthesis`` is the *exact*
transpose ``S^T = Y^T`` (unweighted; **not** the weighted analysis), which is the
operator a linear solve needs and, up to the ``(2 - delta_m0)`` metric, the
cotangent of synthesis.

The transforms are vectorized over m (one fused
``lax.scan`` per spin via :mod:`jht._recursion`), batch the per-ring phi-FFTs by
ring length (the equatorial belt in one FFT; the polar cap by length-group), and
are ``jit``-compiled.  Geometry, recursion plans, and index maps are built once
per ``(nside, lmax, spin)`` and cached (``lru_cache`` on :func:`_prepare`), so the
Jacobi iteration in :func:`jht.analysis` reuses one compiled kernel.
Numerics are identical to the eager reference (:mod:`jht._reference`), kept as a
validation oracle.  Library code does not enable x64;
callers opt in per entry point.
"""

from __future__ import annotations

import dataclasses
import warnings
from functools import lru_cache
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._azimuth import (
    build_cap_plan,
    cap_adjoint,
    cap_gather_mask,
    cap_synth,
    get_azimuth_fft_mode,
    get_cap_chunk,
)
from ._recursion import (
    adjoint_contract_eo,
    adjoint_contract_spin2_ns,
    _fold_np,
    build_recursion_plan,
    get_ell_block,
    lambda_blocks_one,
    lambda_blocks_two,
    resolve_ell_block,
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
    synth: Callable[..., jax.Array]  # (alm, lam) -> map; lam None = run the recursion
    adj: Callable[..., jax.Array]  # (map, lam) -> alm


def _opaque(fn):
    """Wrap a transform kernel in a save-nothing ``jax.checkpoint``.

    Under ``grad`` of a scanned body (e.g. ``lax.map`` over sims), JAX partial-evaluates
    the body and hoists work that depends only on loop constants out of the loop. The
    Legendre recursion depends only on the grid, so without a boundary its stacked per-l
    output ``(lmax+1, lmax+1, 2 nside)`` becomes a loop constant. The checkpoint keeps
    each kernel one unit, so the backward recomputes it per transform instead. Forward
    mode is unaffected.
    """
    return jax.checkpoint(fn, policy=jax.checkpoint_policies.nothing_saveable)


@lru_cache(maxsize=None)
def _prepare(
    nside: int,
    lmax: int,
    spin: int,
    fft_mode: str = "unrolled",
    ell_block: int | str = "auto",
    cap_chunk: int | str = "auto",
) -> _Prepared:
    # ``fft_mode``, ``ell_block`` and ``cap_chunk`` are part of the cache key: :func:`synthesis` /
    # :func:`adjoint_synthesis` read the process-global azimuth mode / l-block / cap chunk and forward
    # them here, so flipping one compiles a fresh kernel (both variants coexist) rather than silently returning a
    # stale one.  Do NOT read the global inside this cached body -- that is the
    # ``jax_enable_x64`` hazard below (a global read here is invisible to the key).
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
            f"jht: lmax={lmax} exceeds the HEALPix band-limit ceiling 1.5*nside "
            f"(= {1.5 * nside:g} for nside={nside}); the transform is validated up to "
            "the ceiling (to nside=4096), accuracy above it is not (see docs/accuracy.md).",
            stacklevel=2,
        )
    geo = RingInfo(nside)
    npix = geo.npix
    groups = _ring_groups(geo, lmax)
    # Every static geometry / index table built below is kept as NumPy, NOT jnp.
    # _prepare is lru_cached; if it is first called inside a jit / grad / lax.scan
    # trace, a cached *device* array would be a tracer that then leaks
    # (UnexpectedTracerError) when the returned closures run under a later trace.
    # NumPy constants instead bake into the jitted synth/adj at compile time, and
    # NumPy indexes / arithmetic-promotes against jnp arrays unchanged (the x args
    # to the contract fns are jnp.asarray'd there). Same cold-cache-under-trace
    # hazard guarded in _recursion._wigner_seed_np and the masked._dof_layout /
    # _analysis._wvec caches.
    gather, valid, pack = _tri_dense_maps(lmax)

    # per-(m, ring) azimuth phase e^{i m phi0_r}, built in-trace so the (M, nrings) table is
    # a runtime buffer, not an embedded constant per use
    m_pos = np.arange(lmax + 1)

    def phase_table():  # (M, nrings)
        z = 1j * jnp.asarray(m_pos, dtype=jnp.float64)[:, None] * jnp.asarray(geo.phi0)[None, :]
        return jax.lax.optimization_barrier(jnp.exp(z))

    # Combined-gather assembly indices.  The per-group output writes are hoisted out of the
    # ring loop into one static-permutation GATHER: the ~nside-way fp64/complex *scatter*
    # unroll is what tips the full synth over the ptxas / compile-RAM limit at nside=2048
    # (the per-length FFTs stay -- a HEALPix ring of length N needs an exact length-N FFT,
    # so they cannot be batched to a common length).  ``buf`` is each group's ring values
    # concatenated in group order; ``pix_to_buf`` / ``ring_to_col`` are the inverse perms
    # back to map-pixel / ring-column order.
    buf_pix = np.concatenate([g.pix_idx.ravel() for g in groups])  # (npix,) permutation
    pix_to_buf = np.empty(npix, dtype=np.int64)
    pix_to_buf[buf_pix] = np.arange(npix)
    ring_order = np.concatenate([g.ring_idx for g in groups])  # (nrings,) permutation
    ring_to_col = np.empty(geo.nrings, dtype=np.int64)
    ring_to_col[ring_order] = np.arange(geo.nrings)

    # --- North/South symmetry geometry (recursion runs on the north half + equator) ---
    # rings 0..2*nside-1 are north cap + equator; ring (4*nside-2-r) is the south
    # reflection of north ring r (r = 0..2*nside-2); the equator (r=2*nside-1) is self.
    t_half = 2 * nside
    x_half = geo.z[:t_half]
    south_src = np.arange(t_half - 1)  # north non-equator half indices
    south_tgt = 4 * nside - 2 - south_src  # their south full-ring indices
    sign_m = ((-1.0) ** np.arange(lmax + 1))[:, None]  # (M, 1)

    def build_full(Ftot, Fsig):  # (M, t_half) x2 -> (M, nrings) full F
        F = jnp.zeros((lmax + 1, geo.nrings), dtype=jnp.complex128)
        F = F.at[:, :t_half].set(Ftot, unique_indices=True)
        return F.at[:, south_tgt].set(sign_m * Fsig[:, south_src], unique_indices=True)

    def fold_south(V):  # (M, nrings) -> Vn (north half), Vs = (-1)^m V_south
        Vs = jnp.zeros((lmax + 1, t_half), dtype=jnp.complex128)
        Vs = Vs.at[:, south_src].set(V[:, south_tgt], unique_indices=True)
        return V[:, :t_half], sign_m * Vs

    def tri_to_dense(alm):  # (K,) -> (M, lmax+1)
        return jnp.where(valid, alm[gather], 0.0 + 0.0j)

    def dense_to_tri(b_dense):  # (M, lmax+1) -> (K,): gather, not scatter (fp64 GPU)
        return b_dense.ravel()[pack]

    # Looped (chirp-z) mode reroutes the polar-cap azimuth FFTs through one common-length
    # ``lax.scan`` (O(1) compiled kernels) and keeps the belt on its native single FFT; the
    # ``azimuth_groups`` the per-length loop still handles are then just the belt.  Requires
    # >1 group (nside=1 is belt-only and already O(1), so it stays unrolled).
    looped = fft_mode == "looped" and len(groups) > 1
    if looped:
        cap_plan = build_cap_plan(geo, groups[:-1], lmax, cap_chunk)
        azimuth_groups = groups[-1:]  # belt only; caps handled by the chirp-z scan
    else:
        azimuth_groups = groups

    if spin == 0:
        plan = build_recursion_plan(geo.z[:t_half], 0, lmax)

        @jax.jit
        def synth(alm, lam):
            Ftot, Fsig = synth_contract_eo(plan, x_half, tri_to_dense(alm), ell_block, lam)
            G = build_full(Ftot, Fsig) * phase_table()  # (M, nrings)
            cols = [jnp.real(cap_synth(cap_plan, G, G))] if looped else []
            for g in azimuth_groups:
                Cp = G[:, g.ring_idx]  # (M, n_g)
                D = jnp.zeros((g.ring_idx.size, g.N), dtype=jnp.complex128)
                D = D.at[:, g.k_plus].add(Cp.T)
                D = D.at[:, g.k_minus].add(jnp.conj(Cp[1:].T))
                cols.append(jnp.real(jnp.fft.ifft(D, axis=1) * g.N).ravel())  # (n_g*N,)
            return jnp.concatenate(cols)[pix_to_buf]  # one gather, not ~nside scatters

        @jax.jit
        def adj(m, lam):
            cols = []
            if looped:
                g_in, cmask = cap_gather_mask(cap_plan)
                vg = (m[g_in] * cmask).astype(jnp.complex128)  # (n_cap,2,n_out)
                cols.append(cap_adjoint(cap_plan, vg)[0])  # (M, n_cap*2)
            conj_phase = jnp.conj(phase_table())
            for g in azimuth_groups:
                fr = jnp.fft.fft(m[g.pix_idx], axis=1)  # (n_g, N)
                ph = conj_phase[:, g.ring_idx].T  # (n_g, M)
                cols.append((fr[:, g.g_plus] * ph).T)  # (M, n_g)
            V = jnp.concatenate(cols, axis=1)[:, ring_to_col]  # (M, nrings), one gather
            Vn, Vs = fold_south(V)
            return dense_to_tri(adjoint_contract_eo(plan, x_half, Vn, Vs, ell_block, lam))

        return _Prepared(_opaque(synth), _opaque(adj))

    plan_p = build_recursion_plan(geo.z[:t_half], 2, lmax)
    plan_m = build_recursion_plan(geo.z[:t_half], -2, lmax)

    @jax.jit
    def synth2(alm2d, lam):  # (2, K) -> (2, npix)
        aE = tri_to_dense(alm2d[0])
        aB = tri_to_dense(alm2d[1])
        FpN, FpS, FmN, FmS = synth_contract_spin2_ns(
            plan_p, plan_m, x_half, -(aE + 1j * aB), -(aE - 1j * aB), ell_block, lam
        )
        phase = phase_table()
        Fp = build_full(FpN, FpS) * phase
        Fm = build_full(FmN, FmS) * phase
        cols = [cap_synth(cap_plan, Fp, Fm)] if looped else []
        for g in azimuth_groups:
            Cp = Fp[:, g.ring_idx]
            Cm = Fm[:, g.ring_idx]
            D = jnp.zeros((g.ring_idx.size, g.N), dtype=jnp.complex128)
            D = D.at[:, g.k_plus].add(Cp.T)
            D = D.at[:, g.k_minus].add(jnp.conj(Cm[1:].T))
            cols.append((jnp.fft.ifft(D, axis=1) * g.N).ravel())  # complex (n_g*N,)
        qu = jnp.concatenate(cols)[pix_to_buf]  # (npix,) complex, one gather
        return jnp.stack([jnp.real(qu), jnp.imag(qu)])

    @jax.jit
    def adj2(maps2d, lam):  # (2, npix) -> (2, K)
        colsp = []
        colsm = []
        if looped:
            g_in, cmask = cap_gather_mask(cap_plan)
            QpiU = (maps2d[0] + 1j * maps2d[1])[g_in] * cmask  # (n_cap,2,n_out)
            QmiU = (maps2d[0] - 1j * maps2d[1])[g_in] * cmask
            Vc = cap_adjoint(cap_plan, jnp.concatenate([QpiU, QmiU], axis=1))  # (2, M, n_cap*2)
            colsp.append(Vc[0])
            colsm.append(Vc[1])
        conj_phase = jnp.conj(phase_table())
        for g in azimuth_groups:
            W = jnp.fft.fft(maps2d[0][g.pix_idx] + 1j * maps2d[1][g.pix_idx], axis=1)  # (n_g, N)
            ph = conj_phase[:, g.ring_idx].T  # (n_g, M)
            colsp.append((W[:, g.g_plus] * ph).T)  # (M, n_g)
            colsm.append((jnp.conj(W[:, g.g_minus]) * ph).T)  # (M, n_g)
        Vp = jnp.concatenate(colsp, axis=1)[:, ring_to_col]  # (M, nrings), one gather
        Vm = jnp.concatenate(colsm, axis=1)[:, ring_to_col]
        Vpn, Sp = fold_south(Vp)
        Vmn, Sm = fold_south(Vm)
        p2, m2 = adjoint_contract_spin2_ns(
            plan_p, plan_m, x_half, Vpn, Sp, Vmn, Sm, ell_block, lam
        )
        aE = 0.5 * (-p2 - m2)
        aB = 0.5 * (1j * p2 - 1j * m2)
        return jnp.stack([dense_to_tri(aE), dense_to_tri(aB)])

    return _Prepared(_opaque(synth2), _opaque(adj2))


# --------------------------------------------------------------------------- #
# stored Legendre tables
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class LegendreTables:
    """Stored lambda tables for one ``(nside, lmax, spin)``, from :func:`legendre_tables`.

    A pytree: ``lam`` (``(lam,)`` for spin 0, ``(lam_p, lam_m)`` for spin 2, each
    ``(nb, K, R, T)`` float64) are its leaves; ``nside``, ``lmax``, ``spin`` and the
    l-block ``ell_block`` (= ``K``) are static metadata.  Pass it to the transforms as
    ``tables=``, and into any ``jit`` / ``lax.scan`` as an argument, never as a
    closed-over value: a closed-over array is compiled into the program as a constant
    (one copy per use, loaded outside JAX's GPU memory pool).
    """

    lam: tuple
    nside: int
    lmax: int
    spin: int
    ell_block: int


jax.tree_util.register_dataclass(
    LegendreTables, data_fields=["lam"], meta_fields=["nside", "lmax", "spin", "ell_block"]
)


def _table_block(nside: int, lmax: int, spin: int, ell_block: int | str | None) -> int:
    fold = _fold_np(lmax, abs(spin))
    setting = get_ell_block() if ell_block is None else ell_block
    return resolve_ell_block(setting, fold.R, 2 * nside, 1 if spin == 0 else 2)


def legendre_tables_nbytes(
    nside: int, lmax: int, spin: int = 0, ell_block: int | str | None = None
) -> int:
    """Device bytes of :func:`legendre_tables` for these arguments.

    ``8 * n_tab * nb * K * R * T`` with ``n_tab`` = 1 (spin 0) or 2 (spin 2),
    ``R = ceil((lmax+1)/2)`` folded rows, ``T = 2*nside`` half-grid colatitudes,
    ``K`` the l-block (``ell_block``, default the current :func:`jht.set_ell_block`
    setting) and ``nb = ceil(S/K)`` blocks over the ``S ~ lmax+2`` folded steps.
    About 2.4 GB per table at ``nside=512, lmax=768``.
    """
    if spin not in (0, 2):
        raise NotImplementedError(f"spin={spin} unsupported (only 0 and 2)")
    fold = _fold_np(lmax, spin)
    K = _table_block(nside, lmax, spin, ell_block)
    nb = -(-fold.S // K)
    return 8 * (1 if spin == 0 else 2) * nb * K * fold.R * 2 * nside


@lru_cache(maxsize=None)
def _table_builder(nside: int, lmax: int, spin: int, K: int) -> Callable[[], tuple]:
    x_half = RingInfo(nside).z[: 2 * nside]
    if spin == 0:
        plan = build_recursion_plan(x_half, 0, lmax)
        return jax.jit(lambda: (lambda_blocks_one(plan, x_half, K),))
    plan_p = build_recursion_plan(x_half, 2, lmax)
    plan_m = build_recursion_plan(x_half, -2, lmax)
    return jax.jit(lambda: tuple(lambda_blocks_two(plan_p, plan_m, x_half, K)))


def legendre_tables(
    nside: int, lmax: int, spin: int = 0, ell_block: int | str | None = None
) -> LegendreTables:
    """Precompute the Legendre (spin 0) or Wigner-d (spin 2) tables the transforms use.

    With ``tables=`` the on-grid transforms (:func:`synthesis`,
    :func:`adjoint_synthesis`, :func:`jht.analysis`) read lambda from these tables
    instead of running the l-recursion on every call: the table holds the values the
    recursion produces, and the contraction is unchanged.  Worth it when one
    ``(nside, lmax, spin)`` is transformed many times and the memory is available;
    see :func:`legendre_tables_nbytes` for the size (about 2.4 GB per table at
    ``nside=512, lmax=768``; spin 2 holds two).

    ``ell_block`` fixes the table's l-block ``K`` (default: the current
    :func:`jht.set_ell_block` setting); the transforms then contract in blocks of that
    ``K``.  Called eagerly the result is a concrete device array; called inside a
    ``jit`` it is built in-trace.  Pass it into jitted code as an argument.
    """
    if spin not in (0, 2):
        raise NotImplementedError(f"spin={spin} unsupported (only 0 and 2)")
    K = _table_block(nside, lmax, spin, ell_block)
    lam = _table_builder(int(nside), int(lmax), int(spin), K)()
    return LegendreTables(tuple(lam), int(nside), int(lmax), int(spin), K)


def _tracer_type() -> type:
    from jax._src.core import Tracer

    return Tracer


def _under_jit() -> bool:
    """True while staging a jaxpr (``jit``, ``lax.scan`` / ``map`` bodies)."""
    from jax._src import core as _core

    f = getattr(_core, "unsafe_am_i_under_a_jit", None) or getattr(
        _core, "unsafe_am_i_under_a_jit_DO_NOT_USE", None
    )
    return bool(f()) if f is not None else False


def _lam_arg(tables: Any, nside: int, lmax: int, spin: int):
    """Validate ``tables`` for one transform; return the contraction's ``lam`` argument."""
    if tables is None:
        return None
    if not isinstance(tables, LegendreTables):
        raise TypeError(f"tables must be a jht.LegendreTables, got {type(tables).__name__}")
    if (tables.nside, tables.lmax, tables.spin) != (nside, lmax, spin):
        raise ValueError(
            f"tables are for (nside, lmax, spin) = {(tables.nside, tables.lmax, tables.spin)}, "
            f"transform is {(nside, lmax, spin)}"
        )
    if _under_jit() and not all(isinstance(a, _tracer_type()) for a in tables.lam):
        raise ValueError(
            "concrete Legendre tables inside a jit / lax.scan trace would be compiled in as "
            "constants; pass the LegendreTables into the jitted function as an argument"
        )
    return tables.lam[0] if spin == 0 else tuple(tables.lam)


# --------------------------------------------------------------------------- #
# public transforms
# --------------------------------------------------------------------------- #
def synthesis(alm, nside: int, lmax: int, spin: int = 0, *, tables=None) -> jax.Array:
    """``S: a_lm -> map`` on the HEALPix RING grid.

    Parameters
    ----------
    alm : complex array
        healpy-packed coefficients: shape ``(alm_size(lmax),)`` for ``spin=0``,
        ``(2, alm_size(lmax))`` (E, B) for ``spin=2``.
    nside, lmax : int
    spin : int
        ``0`` (temperature) or ``2`` (polarization Q/U).
    tables : LegendreTables, optional
        Stored tables from :func:`legendre_tables` for this ``(nside, lmax, spin)``;
        read in place of the l-recursion.

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
    lam = _lam_arg(tables, nside, lmax, spin)
    prep = _prepare(nside, lmax, spin, get_azimuth_fft_mode(), get_ell_block(), get_cap_chunk())
    return prep.synth(alm, lam)


def adjoint_synthesis(m, nside: int, lmax: int, spin: int = 0, *, tables=None) -> jax.Array:
    """``S^T = Y^H : map -> a_lm`` -- the *exact* transpose of :func:`synthesis`.

    ``b_{l,m} = sum_p conj(Y_{l,m}(p)) m_p``.  This is **not** the weighted
    analysis (the approximate inverse); it is the operator a linear solve needs,
    satisfying

        adjoint_synthesis(v) == (Npix / 4pi) * healpy.map2alm(v, iter=0)   (no weights)

    and the inner-product identity ``<S a, v>_map == <a, S^T v>_alm`` in the
    ``(2 - delta_{m0})``-weighted a_lm inner product.  For ``spin=2`` the spin-2
    adjoint carries the factor-1/2 (m>0 conjugate-symmetry weight) so it is the
    strict transpose; the input is ``(2, npix)`` (Q, U) and the output ``(2, K)``.
    ``tables``: as in :func:`synthesis`.
    """
    m = jnp.asarray(m)
    npix = 12 * nside * nside
    expect = (npix,) if spin == 0 else (2, npix)
    if m.shape != expect:
        raise ValueError(
            f"map shape {m.shape} != expected {expect} for nside={nside}, spin={spin} "
            "(a wrong-size map would otherwise be silently clamped by the gather)"
        )
    lam = _lam_arg(tables, nside, lmax, spin)
    prep = _prepare(nside, lmax, spin, get_azimuth_fft_mode(), get_ell_block(), get_cap_chunk())
    return prep.adj(m, lam)
