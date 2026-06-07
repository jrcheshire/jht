"""HEALPix (RING) geometry and the on-grid spherical-harmonic transforms.

Conventions (verified vs healpy / ducc0; see ``docs/design.md``):

* a_lm: healpy m-major triangular packing, ``idx(l,m) = m*(2*lmax+1-m)//2 + l``,
  only ``m >= 0`` stored, size ``(lmax+1)(lmax+2)/2``.
* orthonormal Y_lm with the Condon-Shortley phase (``map = sum a_lm Y_lm``).
* polarization is HEALPix-internal (COSMO) Q/U; no pixel window is applied
  (matches healpy ``pixwin=False``).

``synthesis`` is ``S: a_lm -> map``.  ``adjoint_synthesis`` is the *exact*
transpose ``S^T = Y^T`` (unweighted; **not** the weighted map2alm), which is the
operator the bk-jax seam needs and the VJP of synthesis.

This is the Phase-0 reference implementation: correctness first.  It uses a
Python loop over rings (RING order is north->south, so the map is the
concatenation of the per-ring arrays); batching equal-length rings / vmap is a
Phase-1 performance task.  Library code does not enable x64; callers opt in.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._recursion import normalized_legendre, spin_weighted_lambda


# --------------------------------------------------------------------------- #
# a_lm layout (healpy)
# --------------------------------------------------------------------------- #
def alm_size(lmax: int) -> int:
    return (lmax + 1) * (lmax + 2) // 2


def alm_column_base(m: int, lmax: int) -> int:
    """Start index of the contiguous column ``a_{l,m}, l = m..lmax``."""
    return m * (2 * lmax + 1 - m) // 2 + m


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


# --------------------------------------------------------------------------- #
# synthesis  S: a_lm -> map
# --------------------------------------------------------------------------- #
def _ring_coeffs_spin0(alm, x, lmax: int) -> jax.Array:
    """F_m(theta_r) = sum_l a_{l,m} lambda_{l,m}(theta_r), shape (lmax+1, nrings)."""
    rows = []
    for m in range(lmax + 1):
        lam = normalized_legendre(x, m, lmax)  # (lmax-m+1, nrings), real
        base = alm_column_base(m, lmax)
        a_m = alm[base : base + (lmax - m + 1)]  # (lmax-m+1,), complex
        rows.append(jnp.tensordot(a_m, lam, axes=([0], [0])))  # (nrings,)
    return jnp.stack(rows, axis=0)


def _ring_to_pixels(Gm, npix_ring: int, lmax: int) -> jax.Array:
    """Real ring values from per-m coeffs G_m (phase already applied), with the
    polar-cap spectral folding (m wrapped mod npix_ring)."""
    m_pos = jnp.arange(lmax + 1)
    C = jnp.zeros(npix_ring, dtype=Gm.dtype)
    C = C.at[m_pos % npix_ring].add(Gm)  # +m
    C = C.at[(-m_pos[1:]) % npix_ring].add(jnp.conj(Gm[1:]))  # -m (real-map Hermitian)
    return jnp.real(jnp.fft.ifft(C) * npix_ring)


# --- spin-2 (polarization): (aE, aB) <-> (Q, U), COSMO convention ----------- #
def _spin2_ring_coeffs(aE, aB, x, lmax: int):
    """Spin +/-2 ring coefficients Fp_m, Fm_m, each shape (lmax+1, nrings)."""
    fp, fm = [], []
    for m in range(lmax + 1):
        lmin = max(m, 2)
        lp = spin_weighted_lambda(x, m, 2, lmax)  # (lmax-lmin+1, nrings)
        lm = spin_weighted_lambda(x, m, -2, lmax)
        base = alm_column_base(m, lmax)
        off = lmin - m  # E/B have no l < 2
        aEc = aE[base + off : base + (lmax - m + 1)]
        aBc = aB[base + off : base + (lmax - m + 1)]
        fp.append(jnp.tensordot(-(aEc + 1j * aBc), lp, axes=([0], [0])))
        fm.append(jnp.tensordot(-(aEc - 1j * aBc), lm, axes=([0], [0])))
    return jnp.stack(fp), jnp.stack(fm)


def _synthesis_spin2(alm, nside: int, lmax: int) -> jax.Array:
    """(aE, aB) -> (Q, U); alm shape ``(2, K)``, returns ``(2, npix)``."""
    geo = RingInfo(nside)
    x = jnp.asarray(geo.z)
    Fp, Fm = _spin2_ring_coeffs(alm[0], alm[1], x, lmax)
    m_pos = jnp.arange(lmax + 1)
    q_rings, u_rings = [], []
    for r in range(geo.nrings):
        N = int(geo.npix_ring[r])
        ph = jnp.exp(1j * m_pos * geo.phi0[r])
        D = jnp.zeros(N, dtype=jnp.complex128)
        D = D.at[m_pos % N].add(Fp[:, r] * ph)  # +m  (spin +2)
        D = D.at[(-m_pos[1:]) % N].add(jnp.conj(Fm[1:, r] * ph[1:]))  # -m  (spin -2)
        qu = jnp.fft.ifft(D) * N
        q_rings.append(jnp.real(qu))
        u_rings.append(jnp.imag(qu))
    return jnp.stack([jnp.concatenate(q_rings), jnp.concatenate(u_rings)])


def _adjoint_synthesis_spin2(maps, nside: int, lmax: int) -> jax.Array:
    """Strict transpose of spin-2 synthesis; ``(Q, U) -> (aE, aB)`` shape ``(2, K)``.

    Equals ``(Npix/4pi) * healpy.map2alm_spin([Q, U], 2, lmax)``.  The 1/2 vs the
    naive two-channel sum is the m>0 conjugate-symmetry weight (the documented
    ``2.conj`` subtlety; the AD-convention VJP carries ``2 - delta_{m0}`` instead).
    """
    geo = RingInfo(nside)
    x = jnp.asarray(geo.z)
    m_pos = jnp.arange(lmax + 1)
    fpb, fmb = [], []
    for r in range(geo.nrings):
        N = int(geo.npix_ring[r])
        s = geo.startpix[r]
        QU = maps[0][s : s + N] + 1j * maps[1][s : s + N]
        W = jnp.fft.fft(QU)
        ph = jnp.exp(-1j * m_pos * geo.phi0[r])
        fpb.append(W[m_pos % N] * ph)
        fmb.append(jnp.conj(W[(-m_pos) % N]) * ph)
    Fpb = jnp.stack(fpb, axis=1)
    Fmb = jnp.stack(fmb, axis=1)
    aE_blocks, aB_blocks = [], []
    for m in range(lmax + 1):
        lmin = max(m, 2)
        lp = spin_weighted_lambda(x, m, 2, lmax)
        lm = spin_weighted_lambda(x, m, -2, lmax)
        p2 = jnp.tensordot(lp, Fpb[m], axes=([1], [0]))
        m2 = jnp.tensordot(lm, Fmb[m], axes=([1], [0]))
        ae = 0.5 * (-p2 - m2)
        ab = 0.5 * (1j * p2 - 1j * m2)
        pad = lmin - m  # leading zeros for the absent l < 2
        if pad:
            z = jnp.zeros(pad, dtype=ae.dtype)
            ae, ab = jnp.concatenate([z, ae]), jnp.concatenate([z, ab])
        aE_blocks.append(ae)
        aB_blocks.append(ab)
    return jnp.stack([jnp.concatenate(aE_blocks), jnp.concatenate(aB_blocks)])


def synthesis(alm, nside: int, lmax: int, spin: int = 0) -> jax.Array:
    """``S: a_lm -> map`` on the HEALPix RING grid.

    Parameters
    ----------
    alm : complex array, shape ``(alm_size(lmax),)``
        healpy-packed coefficients (spin-0 / temperature).
    nside, lmax : int
    spin : int
        Only ``spin == 0`` is implemented here; spin-2 lands next.

    Returns
    -------
    m : real array, shape ``(12 nside**2,)`` in RING order.
    """
    if spin == 2:
        return _synthesis_spin2(alm, nside, lmax)
    if spin != 0:
        raise NotImplementedError(f"spin={spin} unsupported (only 0 and 2)")
    geo = RingInfo(nside)
    x = jnp.asarray(geo.z)
    Fm = _ring_coeffs_spin0(alm, x, lmax)  # (lmax+1, nrings)
    m_pos = jnp.arange(lmax + 1)
    rings = []
    for r in range(geo.nrings):
        Gm = Fm[:, r] * jnp.exp(1j * m_pos * geo.phi0[r])
        rings.append(_ring_to_pixels(Gm, int(geo.npix_ring[r]), lmax))
    return jnp.concatenate(rings)


# --------------------------------------------------------------------------- #
# adjoint synthesis  S^T = Y^T : map -> a_lm  (the exact, unweighted transpose)
# --------------------------------------------------------------------------- #
def adjoint_synthesis(m, nside: int, lmax: int, spin: int = 0) -> jax.Array:
    """``S^T = Y^H : map -> a_lm`` -- the *exact* transpose of :func:`synthesis`.

    ``b_{l,m} = sum_p conj(Y_{l,m}(p)) m_p``.  This is **not** the weighted
    map2alm (the approximate inverse); it is the operator the bk-jax seam needs
    and the cotangent of synthesis, satisfying

        adjoint_synthesis(v) == (Npix / 4pi) * healpy.map2alm(v, iter=0)   (no weights)

    and the inner-product identity ``<S a, v>_map == <a, S^T v>_alm`` in the
    ``(2 - delta_{m0})``-weighted a_lm inner product.
    """
    if spin == 2:
        return _adjoint_synthesis_spin2(m, nside, lmax)
    if spin != 0:
        raise NotImplementedError(f"spin={spin} unsupported (only 0 and 2)")
    geo = RingInfo(nside)
    x = jnp.asarray(geo.z)
    m_pos = jnp.arange(lmax + 1)

    # per-ring m-coefficients V_m(r) = e^{-i m phi0_r} * FFT(ring)[m mod N]
    Vm_cols = []
    for r in range(geo.nrings):
        N = int(geo.npix_ring[r])
        v_ring = m[geo.startpix[r] : geo.startpix[r] + N]
        fr = jnp.fft.fft(v_ring)  # sum_j v_j e^{-2pi i k j / N}
        Vm_cols.append(fr[m_pos % N] * jnp.exp(-1j * m_pos * geo.phi0[r]))
    Vm = jnp.stack(Vm_cols, axis=1)  # (lmax+1, nrings)

    # contract over rings: b_{l,m} = sum_r lambda_{l,m}(theta_r) V_m(r)
    blocks = []
    for mm in range(lmax + 1):
        lam = normalized_legendre(x, mm, lmax)  # (lmax-mm+1, nrings)
        blocks.append(jnp.tensordot(lam, Vm[mm], axes=([1], [0])))
    return jnp.concatenate(blocks)  # m-major packing == healpy layout
