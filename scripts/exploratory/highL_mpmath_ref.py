"""Arbitrary-precision reference for the normalized (spin-weighted) Legendre/Wigner
recursion, used to measure jht's fp64 roundoff growth at high l.

The reference runs jht's *same* three-term recursion but in mpmath at ``DPS=50``
digits, with the recurrence coefficients recomputed at full precision.  mpmath's
``mpf`` has an effectively unbounded exponent, so the increasing-l recursion needs
**no** log-renorm here -- the sectoral seed ``sin(theta)**m`` underflows gracefully
to a tiny-but-exact ``mpf`` instead of 0.  So this isolates exactly the quantity in
question: the roundoff the *fp64* implementation accumulates relative to the
roundoff-free value of the recursion.  (Independent-*implementation* correctness of
the full transform at high l is the separate ducc gate, ``tests/test_highL.py``;
the mpmath recursion's correctness is anchored here at low l against the exact
explicit Wigner sum.)

Run in an ephemeral env that has mpmath (does NOT touch the project pixi env):

    pixi exec --spec mpmath --spec numpy -- \
        python scripts/exploratory/highL_mpmath_ref.py

It writes ``scripts/exploratory/_mpmath_ref.npz`` for ``highL_recursion_growth.py``
(run in the project env) to compare against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from mpmath import mp, mpf

sys.path.insert(0, str(Path(__file__).parent))
from _highL_grid import DPS, L_LIST, LMAX, M_LIST, SPINS, THETA  # noqa: E402

mp.dps = DPS
HERE = Path(__file__).parent


def _lg(n: int):
    """log(n!) = loggamma(n+1) at working precision."""
    return mp.loggamma(n + 1)


def mp_lambda_scalar_column(theta, m: int, lmax: int) -> list:
    """Normalized lambda_{l,m}(theta) = Re Y_{l,m}(theta,0), l = m..lmax (plain mpf
    three-term recursion in l; no renorm needed at arbitrary exponent)."""
    th = mpf(theta)
    x = mp.cos(th)
    sin_t = mp.sin(th)
    # sectoral seed lambda_{m,m}
    const = -mpf(0.5) * mp.log(4 * mp.pi)
    for k in range(1, m + 1):
        const += mpf(0.5) * mp.log(mpf(2 * k + 1) / mpf(2 * k))
    sign = mpf(1) if (m % 2 == 0) else mpf(-1)
    lam_mm = sign * mp.e ** (const) * (sin_t ** m if m > 0 else mpf(1))
    out = [lam_mm]
    lam_lm2, lam_lm1 = mpf(0), lam_mm  # lambda_{m-1}=0, lambda_m
    for ell in range(m + 1, lmax + 1):
        L = mpf(ell)
        a = mp.sqrt((2 * L - 1) * (2 * L + 1) / ((L - m) * (L + m)))
        b = mp.sqrt(
            (2 * L + 1) * ((L - 1) ** 2 - m * m) / ((2 * L - 3) * (L - m) * (L + m))
        )
        lam = a * x * lam_lm1 - b * lam_lm2
        out.append(lam)
        lam_lm2, lam_lm1 = lam_lm1, lam
    return out  # index i -> l = m + i


def mp_wigner_d_column(theta, M: int, Mp: int, lmax: int) -> list:
    """Wigner d^l_{M,M'}(theta), l = lmin..lmax (plain mpf three-term recursion)."""
    th = mpf(theta)
    x = mp.cos(th)
    cos_h = mp.cos(th / 2)
    sin_h = mp.sin(th / 2)
    lmin = max(abs(M), abs(Mp))
    # exact single-term seed at l = lmin (k0 = max(0, M-M') is the sole index)
    k0 = max(0, M - Mp)
    log_fac = mpf(0.5) * (
        _lg(lmin + M) + _lg(lmin - M) + _lg(lmin + Mp) + _lg(lmin - Mp)
    ) - (_lg(k0) + _lg(lmin + M - k0) + _lg(lmin - Mp - k0) + _lg(k0 - M + Mp))
    sign = mpf(-1) if (k0 % 2) else mpf(1)
    pow_cos = 2 * lmin - 2 * k0 + M - Mp
    pow_sin = 2 * k0 - M + Mp
    d_min = sign * mp.e ** (log_fac) * cos_h ** pow_cos * sin_h ** pow_sin
    out = [d_min]
    d_lm2, d_lm1 = mpf(0), d_min  # d^{lmin-1}=0, d^{lmin}
    for ell in range(lmin, lmax):  # each step produces d^{ell+1}
        L = mpf(ell)
        D = L * mp.sqrt(((L + 1) ** 2 - M * M) * ((L + 1) ** 2 - Mp * Mp))
        A = (2 * L + 1) * L * (L + 1) / D
        C = -(2 * L + 1) * M * Mp / D
        B = (L + 1) * mp.sqrt((L * L - M * M) * (L * L - Mp * Mp)) / D
        d = (A * x + C) * d_lm1 - B * d_lm2
        out.append(d)
        d_lm2, d_lm1 = d_lm1, d
    return out  # index i -> l = lmin + i


def mp_swsh_column(theta, m: int, spin: int, lmax: int) -> list:
    """{}_s lambda_{l,m}(theta) = (-1)^m sqrt((2l+1)/4pi) d^l_{-m,s}, l = max(m,|s|)..lmax."""
    lmin = max(m, abs(spin))
    d = mp_wigner_d_column(theta, -m, spin, lmax)
    out = []
    for i, ell in enumerate(range(lmin, lmax + 1)):
        pref = ((-1) ** m) * mp.sqrt(mpf(2 * ell + 1) / (4 * mp.pi))
        out.append(pref * d[i])
    return out


def mp_wigner_d_explicit(ell: int, M: int, Mp: int, theta) -> "mpf":
    """Exact explicit alternating Wigner sum (the independent low-l anchor for the
    recursion above; brutal cancellation, but fine at low l even in mpf)."""
    th = mpf(theta)
    c, s = mp.cos(th / 2), mp.sin(th / 2)
    kmin, kmax = max(0, M - Mp), min(ell + M, ell - Mp)
    pref = mpf(0.5) * (_lg(ell + M) + _lg(ell - M) + _lg(ell + Mp) + _lg(ell - Mp))
    tot = mpf(0)
    for k in range(kmin, kmax + 1):
        logc = pref - (_lg(k) + _lg(ell + M - k) + _lg(ell - Mp - k) + _lg(k - M + Mp))
        tot += ((-1) ** k) * mp.e ** (logc) * c ** (2 * ell - 2 * k + M - Mp) * s ** (
            2 * k - M + Mp
        )
    return tot


def _anchor() -> None:
    """Pin the mpmath recursion before trusting it at high l."""
    # scalar lambda_{l,m} at l<=2 vs closed form Re Y_lm(theta,0)
    th = 0.7
    c, s = np.cos(th), np.sin(th)
    ana = {
        (0, 0): np.sqrt(1.0 / (4.0 * np.pi)),
        (1, 0): np.sqrt(3.0 / (4.0 * np.pi)) * c,
        (1, 1): -np.sqrt(3.0 / (8.0 * np.pi)) * s,
        (2, 0): np.sqrt(5.0 / (16.0 * np.pi)) * (3.0 * c * c - 1.0),
        (2, 1): -np.sqrt(15.0 / (8.0 * np.pi)) * s * c,
        (2, 2): np.sqrt(15.0 / (32.0 * np.pi)) * s * s,
    }
    for m in (0, 1, 2):
        col = mp_lambda_scalar_column(th, m, 2)
        for ell in range(m, 3):
            assert abs(float(col[ell - m]) - ana[(ell, m)]) < 1e-14, (ell, m)
    tol = mpf(10) ** (-(DPS - 8))
    # scalar recursion vs the exact explicit Wigner sum (Mp=0), l<=12, m>=1
    # (m=0 spin=0 would seed the Wigner recursion at lmin=0, where D_l = l*... = 0;
    # jht's production spin-0 path is the scalar Legendre recursion, not the Wigner one).
    for m in (1, 2, 5):
        col = mp_lambda_scalar_column(th, m, 12)
        for ell in range(m, 13):
            pref = ((-1) ** m) * mp.sqrt(mpf(2 * ell + 1) / (4 * mp.pi))
            ref = pref * mp_wigner_d_explicit(ell, -m, 0, th)
            assert abs(col[ell - m] - ref) < tol, (0, m, ell)
    # spin-2 recursion vs explicit sum, l<=12 (lmin = max(m,2) >= 2, never singular)
    for m in (0, 1, 2, 5):
        d_rec = mp_wigner_d_column(th, -m, 2, 12)
        for ell in range(max(m, 2), 13):
            d_exp = mp_wigner_d_explicit(ell, -m, 2, th)
            assert abs(d_rec[ell - max(m, 2)] - d_exp) < tol, (2, m, ell)
    print("anchor OK: mpmath recursion == closed-form / explicit Wigner sum (low l)")


def main() -> None:
    _anchor()
    nT, nM, nL = len(THETA), len(M_LIST), len(L_LIST)
    # ref[spin][m_idx, l_idx, theta_idx]; NaN where m > l (mode does not exist).
    ref = {0: np.full((nM, nL, nT), np.nan), 2: np.full((nM, nL, nT), np.nan)}
    for spin in SPINS:
        lmin0 = abs(spin)
        for im, m in enumerate(M_LIST):
            lmin = max(m, lmin0)
            for it, th in enumerate(THETA):
                if spin == 0:
                    col = mp_lambda_scalar_column(th, m, LMAX)  # l = m..LMAX
                    base = m
                else:
                    col = mp_swsh_column(th, m, spin, LMAX)  # l = lmin..LMAX
                    base = lmin
                for il, L in enumerate(L_LIST):
                    if L < lmin:
                        continue
                    ref[spin][im, il, it] = float(col[L - base])
            print(f"  spin={spin} m={m} done")
    out = HERE / "_mpmath_ref.npz"
    np.savez(
        out,
        theta=np.asarray(THETA, float),
        m_list=np.asarray(M_LIST, int),
        l_list=np.asarray(L_LIST, int),
        ref_spin0=ref[0],
        ref_spin2=ref[2],
        dps=DPS,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
