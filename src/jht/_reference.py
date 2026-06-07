"""Phase-0 reference transforms (eager, per-ring) -- kept as a validation oracle.

These are the original correctness-first implementations from the Phase-0 spike:
an eager Python loop over m and over rings, recursion recomputed every call.
They are **slow** (small nside only) and are retained solely as a second oracle
for the Phase-1 vectorized fast path in :mod:`jht.healpix` (the ``*_reference``
vs fast parity test), alongside the healpy / ducc0 gates.

Do not use these in production; import the public ``synthesis`` /
``adjoint_synthesis`` from :mod:`jht.healpix` instead.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._recursion import normalized_legendre, spin_weighted_lambda
from .healpix import RingInfo, alm_column_base


def _ring_coeffs_spin0(alm, x, lmax: int) -> jax.Array:
    rows = []
    for m in range(lmax + 1):
        lam = normalized_legendre(x, m, lmax)
        base = alm_column_base(m, lmax)
        a_m = alm[base : base + (lmax - m + 1)]
        rows.append(jnp.tensordot(a_m, lam, axes=([0], [0])))
    return jnp.stack(rows, axis=0)


def _ring_to_pixels(Gm, npix_ring: int, lmax: int) -> jax.Array:
    m_pos = jnp.arange(lmax + 1)
    C = jnp.zeros(npix_ring, dtype=Gm.dtype)
    C = C.at[m_pos % npix_ring].add(Gm)
    C = C.at[(-m_pos[1:]) % npix_ring].add(jnp.conj(Gm[1:]))
    return jnp.real(jnp.fft.ifft(C) * npix_ring)


def _spin2_ring_coeffs(aE, aB, x, lmax: int):
    fp, fm = [], []
    for m in range(lmax + 1):
        lmin = max(m, 2)
        lp = spin_weighted_lambda(x, m, 2, lmax)
        lm = spin_weighted_lambda(x, m, -2, lmax)
        base = alm_column_base(m, lmax)
        off = lmin - m
        aEc = aE[base + off : base + (lmax - m + 1)]
        aBc = aB[base + off : base + (lmax - m + 1)]
        fp.append(jnp.tensordot(-(aEc + 1j * aBc), lp, axes=([0], [0])))
        fm.append(jnp.tensordot(-(aEc - 1j * aBc), lm, axes=([0], [0])))
    return jnp.stack(fp), jnp.stack(fm)


def _synthesis_spin2(alm, nside: int, lmax: int) -> jax.Array:
    geo = RingInfo(nside)
    x = jnp.asarray(geo.z)
    Fp, Fm = _spin2_ring_coeffs(alm[0], alm[1], x, lmax)
    m_pos = jnp.arange(lmax + 1)
    q_rings, u_rings = [], []
    for r in range(geo.nrings):
        N = int(geo.npix_ring[r])
        ph = jnp.exp(1j * m_pos * geo.phi0[r])
        D = jnp.zeros(N, dtype=jnp.complex128)
        D = D.at[m_pos % N].add(Fp[:, r] * ph)
        D = D.at[(-m_pos[1:]) % N].add(jnp.conj(Fm[1:, r] * ph[1:]))
        qu = jnp.fft.ifft(D) * N
        q_rings.append(jnp.real(qu))
        u_rings.append(jnp.imag(qu))
    return jnp.stack([jnp.concatenate(q_rings), jnp.concatenate(u_rings)])


def _adjoint_synthesis_spin2(maps, nside: int, lmax: int) -> jax.Array:
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
        pad = lmin - m
        if pad:
            z = jnp.zeros(pad, dtype=ae.dtype)
            ae, ab = jnp.concatenate([z, ae]), jnp.concatenate([z, ab])
        aE_blocks.append(ae)
        aB_blocks.append(ab)
    return jnp.stack([jnp.concatenate(aE_blocks), jnp.concatenate(aB_blocks)])


def synthesis_reference(alm, nside: int, lmax: int, spin: int = 0) -> jax.Array:
    """Eager reference for ``S: a_lm -> map`` (spin-0 and spin-2)."""
    if spin == 2:
        return _synthesis_spin2(alm, nside, lmax)
    if spin != 0:
        raise NotImplementedError(f"spin={spin} unsupported (only 0 and 2)")
    geo = RingInfo(nside)
    x = jnp.asarray(geo.z)
    Fm = _ring_coeffs_spin0(alm, x, lmax)
    m_pos = jnp.arange(lmax + 1)
    rings = []
    for r in range(geo.nrings):
        Gm = Fm[:, r] * jnp.exp(1j * m_pos * geo.phi0[r])
        rings.append(_ring_to_pixels(Gm, int(geo.npix_ring[r]), lmax))
    return jnp.concatenate(rings)


def adjoint_synthesis_reference(m, nside: int, lmax: int, spin: int = 0) -> jax.Array:
    """Eager reference for the exact transpose ``S^T = Y^H : map -> a_lm``."""
    if spin == 2:
        return _adjoint_synthesis_spin2(m, nside, lmax)
    if spin != 0:
        raise NotImplementedError(f"spin={spin} unsupported (only 0 and 2)")
    geo = RingInfo(nside)
    x = jnp.asarray(geo.z)
    m_pos = jnp.arange(lmax + 1)
    Vm_cols = []
    for r in range(geo.nrings):
        N = int(geo.npix_ring[r])
        v_ring = m[geo.startpix[r] : geo.startpix[r] + N]
        fr = jnp.fft.fft(v_ring)
        Vm_cols.append(fr[m_pos % N] * jnp.exp(-1j * m_pos * geo.phi0[r]))
    Vm = jnp.stack(Vm_cols, axis=1)
    blocks = []
    for mm in range(lmax + 1):
        lam = normalized_legendre(x, mm, lmax)
        blocks.append(jnp.tensordot(lam, Vm[mm], axes=([1], [0])))
    return jnp.concatenate(blocks)
