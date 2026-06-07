"""Phase-0 L2 measurement: the spin-2 HEALPix inverse floor, per (l, m).

THE make-or-break: s2fft's spin-2 inverse failed at l=8/16/32 with m<l errors of
13-35% (m=l passed). This sweeps the same single modes through jht's
synthesize -> analyze round trip and reports the error vs m -- it must be FLAT
across m (no s2fft-style m<l defect), at the HEALPix ~1e-3 floor, improving with
Jacobi iteration.

For speed this caches the validated lambda tables (from jht._recursion) once and
does the FFT assembly in numpy; the formulas are identical to the committed JAX
transforms (which match healpy/ducc to ~1e-13). All below the l <= 1.5*nside
band-limit ceiling, so this measures the quadrature, not aliasing.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from jht._recursion import normalized_legendre, spin_weighted_lambda  # noqa: E402
from jht.healpix import RingInfo, alm_column_base, alm_size  # noqa: E402


def _cache(nside, lmax):
    geo = RingInfo(nside)
    x = geo.z
    lam0 = [np.asarray(normalized_legendre(x, m, lmax)) for m in range(lmax + 1)]
    lamP = [np.asarray(spin_weighted_lambda(x, m, 2, lmax)) for m in range(lmax + 1)]
    lamM = [np.asarray(spin_weighted_lambda(x, m, -2, lmax)) for m in range(lmax + 1)]
    return geo, lam0, lamP, lamM


def _synth(a, geo, lam, lmax, spin):
    mpos = np.arange(lmax + 1)
    nr = geo.nrings
    if spin == 0:
        Fm = np.zeros((lmax + 1, nr), complex)
        for m in range(lmax + 1):
            base = alm_column_base(m, lmax)
            Fm[m] = a[base : base + (lmax - m + 1)] @ lam[0][m]
        out = np.zeros(geo.npix)
        for r in range(nr):
            N = int(geo.npix_ring[r])
            D = np.zeros(N, complex)
            g = Fm[:, r] * np.exp(1j * mpos * geo.phi0[r])
            np.add.at(D, mpos % N, g)
            np.add.at(D, (-mpos[1:]) % N, np.conj(g[1:]))
            out[geo.startpix[r] : geo.startpix[r] + N] = np.real(np.fft.ifft(D) * N)
        return out
    aE, aB = a
    Fp = np.zeros((lmax + 1, nr), complex)
    Fmn = np.zeros((lmax + 1, nr), complex)
    for m in range(lmax + 1):
        lmin = max(m, 2)
        base = alm_column_base(m, lmax)
        off = lmin - m
        aEc = aE[base + off : base + (lmax - m + 1)]
        aBc = aB[base + off : base + (lmax - m + 1)]
        Fp[m] = (-(aEc + 1j * aBc)) @ lam[1][m]
        Fmn[m] = (-(aEc - 1j * aBc)) @ lam[2][m]
    Q = np.zeros(geo.npix)
    U = np.zeros(geo.npix)
    for r in range(nr):
        N = int(geo.npix_ring[r])
        D = np.zeros(N, complex)
        ph = np.exp(1j * mpos * geo.phi0[r])
        np.add.at(D, mpos % N, Fp[:, r] * ph)
        np.add.at(D, (-mpos[1:]) % N, np.conj(Fmn[1:, r] * ph[1:]))
        qu = np.fft.ifft(D) * N
        Q[geo.startpix[r] : geo.startpix[r] + N] = qu.real
        U[geo.startpix[r] : geo.startpix[r] + N] = qu.imag
    return np.stack([Q, U])


def _adj(maps, geo, lam, lmax, spin):
    mpos = np.arange(lmax + 1)
    nr = geo.nrings
    if spin == 0:
        Vm = np.zeros((lmax + 1, nr), complex)
        for r in range(nr):
            N = int(geo.npix_ring[r])
            s = geo.startpix[r]
            Vm[:, r] = np.fft.fft(maps[s : s + N])[mpos % N] * np.exp(-1j * mpos * geo.phi0[r])
        b = np.zeros(alm_size(lmax), complex)
        for m in range(lmax + 1):
            base = alm_column_base(m, lmax)
            b[base : base + (lmax - m + 1)] = lam[0][m] @ Vm[m]
        return b
    Q, U = maps
    Fpb = np.zeros((lmax + 1, nr), complex)
    Fmb = np.zeros((lmax + 1, nr), complex)
    for r in range(nr):
        N = int(geo.npix_ring[r])
        s = geo.startpix[r]
        W = np.fft.fft(Q[s : s + N] + 1j * U[s : s + N])
        ph = np.exp(-1j * mpos * geo.phi0[r])
        Fpb[:, r] = W[mpos % N] * ph
        Fmb[:, r] = np.conj(W[(-mpos) % N]) * ph
    aE = np.zeros(alm_size(lmax), complex)
    aB = np.zeros(alm_size(lmax), complex)
    for m in range(lmax + 1):
        lmin = max(m, 2)
        off = lmin - m
        base = alm_column_base(m, lmax)
        p2 = lam[1][m] @ Fpb[m]
        m2 = lam[2][m] @ Fmb[m]
        aE[base + off : base + (lmax - m + 1)] = 0.5 * (-p2 - m2)
        aB[base + off : base + (lmax - m + 1)] = 0.5 * (1j * p2 - 1j * m2)
    return np.stack([aE, aB])


def _map2alm(maps, geo, lam, lmax, spin, niter):
    fac = 4 * np.pi / geo.npix
    a = fac * _adj(maps, geo, lam, lmax, spin)
    for _ in range(niter):
        a = a + fac * _adj(maps - _synth(a, geo, lam, lmax, spin), geo, lam, lmax, spin)
    return a


def _roundtrip_err(l0, m0, geo, lam, lmax, spin, niter):
    """Inject a unit single mode (E channel for spin-2), round-trip, return max
    error over all modes/channels."""
    if spin == 0:
        a = np.zeros(alm_size(lmax), complex)
        a[alm_column_base(m0, lmax) + (l0 - m0)] = 1.0
    else:
        a = np.zeros((2, alm_size(lmax)), complex)
        a[0, alm_column_base(m0, lmax) + (l0 - m0)] = 1.0
    rec = _map2alm(_synth(a, geo, lam, lmax, spin), geo, lam, lmax, spin, niter)
    return float(np.max(np.abs(rec - a)))


if __name__ == "__main__":
    nside, lmax = 32, 40  # ceiling 1.5*nside = 48; l0 <= 32 is below it
    geo, lam0, lamP, lamM = _cache(nside, lmax)
    lam = (lam0, lamP, lamM)
    print(
        f"nside={nside} lmax={lmax} (ceiling 1.5*nside={int(1.5 * nside)}); single-mode round-trip\n"
    )

    print("=== spin-2 (the s2fft kill-test): bare round-trip error vs m ===")
    print(f"{'l0':>4} {'m=l0':>10} {'min m<l0':>10} {'max m<l0':>10}   verdict")
    for l0 in (8, 16, 32):
        e = [_roundtrip_err(l0, m0, geo, lam, lmax, 2, 0) for m0 in range(l0 + 1)]
        sect, sub = e[-1], e[:-1]
        ok = max(e) < 5e-3  # every m at the HEALPix floor (s2fft was 0.13-0.35)
        print(
            f"{l0:>4} {sect:>10.2e} {min(sub):>10.2e} {max(sub):>10.2e}   "
            f"{'PASS (floor, no s2fft defect)' if ok else 'CHECK'}"
        )

    print("\n=== iteration convergence (l0=16, m0=8, spin-2) ===")
    for niter in (0, 1, 3, 5):
        print(
            f"  niter={niter}: max round-trip err = {_roundtrip_err(16, 8, geo, lam, lmax, 2, niter):.2e}"
        )

    print("\n=== spin-0 control (l0=16) ===")
    e0 = [_roundtrip_err(16, m0, geo, lam, lmax, 0, 0) for m0 in range(0, 17, 4)]
    print(
        f"  bare worst over m = {max(e0):.2e};  iter=3 (m0=8) = {_roundtrip_err(16, 8, geo, lam, lmax, 0, 3):.2e}"
    )
