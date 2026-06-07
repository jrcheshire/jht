"""Gate R (spin-0): the normalized associated-Legendre recursion.

Validation strategy (the recursion is the project's crux, so it is checked
against an *independent* oracle, not just self-consistency):

* analytic closed-form ``Y_{l,m}`` for ``l <= 2`` pins the convention
  (orthonormal, Condon-Shortley, no 4 pi factor) and the seed/sign;
* ``scipy.special.sph_harm_y`` is anchored to those analytics, then used as the
  elementwise oracle across ``m`` and ``l`` (Gate R: rel err <= 1e-12);
* a plain-numpy mirror of the same algorithm extends the check to l = 1000
  (catches JAX/scan/dtype bugs the oracle range can't reach);
* explicit stability + near-pole checks guard the underflow corner.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.special import gammaln, sph_harm_y  # noqa: E402

from jht._recursion import (  # noqa: E402
    legendre_recurrence_coeffs,
    normalized_legendre,
    spin_weighted_lambda,
    wigner_d_column,
)

GATE_R_RTOL = 1e-12  # a-priori, per ROADMAP Phase-0; not relaxed without sign-off


# --------------------------------------------------------------------------- #
# oracles
# --------------------------------------------------------------------------- #
def lambda_scipy(ell: int, m: int, theta: np.ndarray) -> np.ndarray:
    """lambda_{l,m}(theta) = Re Y_{l,m}(theta, phi=0) via scipy."""
    return np.real(sph_harm_y(ell, m, theta, 0.0))


def lambda_numpy(x: np.ndarray, m: int, lmax: int) -> np.ndarray:
    """Plain-numpy mirror of ``normalized_legendre`` (same algorithm)."""
    x = np.asarray(x, float)
    sin_t = np.sqrt(np.clip(1.0 - x * x, 0.0, 1.0))
    k = np.arange(1, m + 1, dtype=float)
    const = -0.5 * np.log(4.0 * np.pi) + 0.5 * np.sum(np.log((2.0 * k + 1.0) / (2.0 * k)))
    if m == 0:
        s = np.full_like(x, const)
    else:
        with np.errstate(divide="ignore"):
            s = const + m * np.log(sin_t)
    sign = 1.0 if m % 2 == 0 else -1.0
    p = np.full_like(x, sign)
    q = np.zeros_like(x)
    out = [p * np.exp(s)]
    a, b = legendre_recurrence_coeffs(m, lmax)
    for i in range(len(a)):
        raw = a[i] * x * p - b[i] * q
        r = np.sqrt(raw * raw + p * p)
        r = np.where(r == 0.0, 1.0, r)
        new_p, new_q, new_s = raw / r, p / r, s + np.log(r)
        out.append(new_p * np.exp(new_s))
        p, q, s = new_p, new_q, new_s
    return np.array(out)


# analytic lambda_{l,m}(theta) = Re Y_{l,m}(theta, 0), l <= 2
def _analytic(theta: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    c, s = np.cos(theta), np.sin(theta)
    return {
        (0, 0): np.full_like(theta, np.sqrt(1.0 / (4.0 * np.pi))),
        (1, 0): np.sqrt(3.0 / (4.0 * np.pi)) * c,
        (1, 1): -np.sqrt(3.0 / (8.0 * np.pi)) * s,
        (2, 0): np.sqrt(5.0 / (16.0 * np.pi)) * (3.0 * c * c - 1.0),
        (2, 1): -np.sqrt(15.0 / (8.0 * np.pi)) * s * c,
        (2, 2): np.sqrt(15.0 / (32.0 * np.pi)) * s * s,
    }


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_scipy_oracle_matches_analytic():
    """Pin scipy's convention (incl. Condon-Shortley) before trusting it."""
    theta = np.linspace(0.1, np.pi - 0.1, 11)
    for (ell, m), val in _analytic(theta).items():
        assert np.allclose(lambda_scipy(ell, m, theta), val, rtol=0, atol=1e-13), (ell, m)


def test_jht_matches_analytic_low_ell():
    theta = np.linspace(0.05, np.pi - 0.05, 13)
    x = np.cos(theta)
    ana = _analytic(theta)
    for m in (0, 1, 2):
        lam = np.asarray(normalized_legendre(x, m, 2))
        for ell in range(m, 3):
            assert np.allclose(lam[ell - m], ana[(ell, m)], rtol=0, atol=1e-13), (ell, m)


@pytest.mark.parametrize("m", [0, 1, 2, 5, 10, 50, 100])
def test_gate_r_spin0_vs_scipy(m):
    """Gate R: elementwise rel error vs scipy across l, up to l = 512.

    (scipy.special.sph_harm_y is the independent oracle here; it stays accurate
    to ~1e-13 through l=512 but NaNs at l=1000 for m<l -- the very underflow
    regime jht's log-renorm survives.  High-l correctness is carried by the
    numpy mirror below and the L1 synthesis-vs-ducc gate.)
    """
    lmax = 512
    theta = np.linspace(0.03, np.pi - 0.03, 23)  # off-pole; scipy is reliable here
    x = np.cos(theta)
    lam = np.asarray(normalized_legendre(x, m, lmax))
    worst = 0.0
    for ell in range(m, lmax + 1):
        ref = lambda_scipy(ell, m, theta)
        scale = max(np.max(np.abs(ref)), 1.0)  # natural amplitude of this lambda column
        worst = max(worst, np.max(np.abs(lam[ell - m] - ref)) / scale)
    assert worst <= GATE_R_RTOL, f"m={m}: worst rel err {worst:.2e} > {GATE_R_RTOL:.0e}"


@pytest.mark.parametrize("m", [0, 1, 2, 16, 100, 500, 999, 1000])
def test_jax_matches_numpy_mirror_to_lmax_1000(m):
    """High-l JAX-execution correctness vs a numpy mirror of the same algorithm."""
    lmax = 1000
    theta = np.linspace(0.03, np.pi - 0.03, 17)
    x = np.cos(theta)
    lam_jax = np.asarray(normalized_legendre(x, m, lmax))
    lam_np = lambda_numpy(x, m, lmax)
    assert np.max(np.abs(lam_jax - lam_np)) <= 1e-11, m


@pytest.mark.parametrize("m", [0, 1, 17, 250, 1000])
def test_stability_and_bound_at_lmax_1000(m):
    """Finite everywhere and within the addition-theorem bound |lambda| <= sqrt((2l+1)/4pi)."""
    lmax = 1000
    theta = np.linspace(0.01, np.pi - 0.01, 29)
    x = np.cos(theta)
    lam = np.asarray(normalized_legendre(x, m, lmax))
    assert np.all(np.isfinite(lam)), m
    ell = np.arange(m, lmax + 1)[:, None]
    bound = np.sqrt((2.0 * ell + 1.0) / (4.0 * np.pi)) * (1.0 + 1e-9)
    assert np.all(np.abs(lam) <= bound), m


def test_near_pole_no_nan():
    """Underflow corner: sectoral seed ~ sin(theta)**m; m>=1 -> 0, m=0 finite, never NaN."""
    x = np.array([1.0, 1.0 - 1e-12, np.cos(1e-6), -1.0])  # incl. exact poles
    for m in (0, 1, 2, 64, 1000):
        lam = np.asarray(normalized_legendre(x, m, 1000))
        assert np.all(np.isfinite(lam)), m
    # m>=1 sectoral value vanishes at the exact pole; m=0 does not
    assert np.asarray(normalized_legendre(np.array([1.0]), 5, 5))[0, 0] == 0.0
    assert np.asarray(normalized_legendre(np.array([1.0]), 0, 0))[0, 0] > 0.0


# --------------------------------------------------------------------------- #
# Gate R (spin-2): the shared Wigner-d recursion
# --------------------------------------------------------------------------- #
def wigner_d_explicit(ell: int, M: int, Mp: int, beta: np.ndarray) -> np.ndarray:
    """Exact Wigner small-d via the explicit alternating sum (reliable to l ~ 15;
    catastrophic cancellation kills it above that -- used only as a low-l oracle)."""
    c, s = np.cos(beta / 2.0), np.sin(beta / 2.0)
    kmin, kmax = max(0, M - Mp), min(ell + M, ell - Mp)
    pref = 0.5 * (
        gammaln(ell + M + 1) + gammaln(ell - M + 1) + gammaln(ell + Mp + 1) + gammaln(ell - Mp + 1)
    )
    tot = np.zeros_like(np.asarray(beta, float))
    for k in range(kmin, kmax + 1):
        logc = pref - (
            gammaln(k + 1)
            + gammaln(ell + M - k + 1)
            + gammaln(ell - Mp - k + 1)
            + gammaln(k - M + Mp + 1)
        )
        tot = tot + ((-1) ** k) * np.exp(logc) * c ** (2 * ell - 2 * k + M - Mp) * s ** (
            2 * k - M + Mp
        )
    return tot


@pytest.mark.parametrize("m", [0, 1, 2, 5, 10])
def test_gate_r_spin2_vs_explicit_low_ell(m):
    """spin-2-specific correctness: d^l_{-m,2} vs the exact explicit Wigner sum."""
    theta = np.linspace(0.1, np.pi - 0.1, 15)
    lmin = max(m, 2)
    d = np.asarray(wigner_d_column(np.cos(theta), -m, 2, 12))
    for ell in range(lmin, 13):
        ref = wigner_d_explicit(ell, -m, 2, theta)
        assert np.max(np.abs(d[ell - lmin] - ref)) <= 1e-11, (m, ell)


@pytest.mark.parametrize("m", [1, 2, 10, 100])
def test_shared_machinery_spin0_via_wigner_to_512(m):
    """The Wigner path reproduces the scipy-anchored spin-0 to l=512 (validates the
    shared coeffs/seed/scan up to high l; spin-2 differs only by the m'=2 terms)."""
    theta = np.linspace(0.03, np.pi - 0.03, 17)
    x = np.cos(theta)
    sw = np.asarray(spin_weighted_lambda(x, m, 0, 512))
    nl = np.asarray(normalized_legendre(x, m, 512))
    assert np.max(np.abs(sw - nl)) <= 1e-11, m


@pytest.mark.parametrize("m", [0, 2, 20, 500, 1000])
def test_gate_r_spin2_bounded_to_1000(m):
    """High-l stability: |d^l_{-m,2}| <= 1 (Wigner-d bound) -- a blow-up would exceed
    it -- and the spin-weighted |{}_2 lambda| <= sqrt((2l+1)/4pi), all finite."""
    lmax = 1000
    theta = np.linspace(0.01, np.pi - 0.01, 29)
    x = np.cos(theta)
    d = np.asarray(wigner_d_column(x, -m, 2, lmax))
    assert np.all(np.isfinite(d)), m
    assert np.all(np.abs(d) <= 1.0 + 1e-9), (m, float(np.max(np.abs(d))))
    lam = np.asarray(spin_weighted_lambda(x, m, 2, lmax))
    ell = np.arange(max(m, 2), lmax + 1)[:, None]
    assert np.all(np.abs(lam) <= np.sqrt((2.0 * ell + 1.0) / (4.0 * np.pi)) * (1.0 + 1e-9)), m


def test_spin2_near_pole_no_nan():
    x = np.array([1.0, 1.0 - 1e-12, np.cos(1e-6), -1.0])
    for m in (0, 2, 3, 64, 1000):
        assert np.all(np.isfinite(np.asarray(spin_weighted_lambda(x, m, 2, 1000)))), m
