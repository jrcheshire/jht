"""Numerically stable normalized associated-Legendre recursion (spin-0).

Computes the fully-normalized associated Legendre functions

    lambda_{l,m}(theta) = sqrt((2l+1)/(4 pi) * (l-m)!/(l+m)!) * P_l^m(cos theta)

with the Condon-Shortley phase, so that ``lambda_{l,m}(theta) == Re Y_{l,m}(theta, phi=0)``
in the orthonormal spherical-harmonic convention used by healpy and ducc0
(``map = sum_{l,m} a_{l,m} Y_{l,m}``, no extra 4 pi factor).

It uses the three-term recursion in ``l`` at fixed ``m`` (libsharp /
Kostelec-Rockmore), run in the **increasing-l** direction, stabilized by a
**branch-free per-step log-renormalization**: at every step the running pair is
rescaled to unit norm and the logarithm of the scale is accumulated, so the
mantissas stay O(1) while the true magnitude is carried in the (real-valued)
log-scale.  This holds to l ~ 1000 in float64 without the under/overflow that
kills a naive recursion near the poles (where the sectoral seed carries
``sin(theta)**m``).

The spin-weighted (spin-2) functions reuse the same machinery with the Wigner-d
recursion; they are added in a separate step.  This module is spin-0 for now.

Library code does **not** enable x64; callers opt in per entry point via
``jax.config.update("jax_enable_x64", True)``.  Accuracy work needs float64.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np


def legendre_recurrence_coeffs(m: int, lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Static three-term recurrence coefficients for ``l = m+1 .. lmax``.

    ``lambda_{l,m} = a_l * x * lambda_{l-1,m} - b_l * lambda_{l-2,m}`` with
    ``x = cos(theta)``.  Returned as numpy float64 arrays of shape ``(lmax-m,)``
    (empty when ``lmax == m``).  ``b_{m+1} == 0`` so the (undefined)
    ``lambda_{m-1,m}`` term drops out of the first step.
    """
    ell = np.arange(m + 1, lmax + 1, dtype=np.float64)
    a = np.sqrt((2.0 * ell - 1.0) * (2.0 * ell + 1.0) / ((ell - m) * (ell + m)))
    b = np.sqrt(
        (2.0 * ell + 1.0) * ((ell - 1.0) ** 2 - m * m) / ((2.0 * ell - 3.0) * (ell - m) * (ell + m))
    )
    return a, b


def _sectoral_seed_logabs(m: int) -> float:
    """theta-independent part of ``log|lambda_{m,m}|`` (the sectoral seed)."""
    k = np.arange(1, m + 1, dtype=np.float64)
    return float(-0.5 * np.log(4.0 * np.pi) + 0.5 * np.sum(np.log((2.0 * k + 1.0) / (2.0 * k))))


def normalized_legendre(x, m: int, lmax: int) -> jax.Array:
    """Normalized associated Legendre column ``lambda_{l,m}(theta)``.

    Parameters
    ----------
    x : array_like, shape ``(T,)``
        ``cos(theta)`` at the ring colatitudes.
    m, lmax : int
        Order and band limit, ``0 <= m <= lmax``.

    Returns
    -------
    lam : jax.Array, shape ``(lmax - m + 1, T)``
        ``lam[i]`` is ``lambda_{m+i, m}(theta)``.
    """
    if not (0 <= m <= lmax):
        raise ValueError(f"require 0 <= m <= lmax, got m={m}, lmax={lmax}")

    x = jnp.asarray(x)
    # sin(theta) >= 0 on [0, pi]; clip guards tiny negative round-off at the poles.
    sin_theta = jnp.sqrt(jnp.clip(1.0 - x * x, 0.0, 1.0))

    const = _sectoral_seed_logabs(m)
    if m == 0:
        # avoid 0 * log(0) = NaN at the poles (the sin term is absent for m=0)
        seed_logabs = jnp.broadcast_to(jnp.asarray(const, x.dtype), x.shape)
    else:
        seed_logabs = const + m * jnp.log(sin_theta)  # -inf at the poles -> lambda = 0
    seed_sign = 1.0 if (m % 2 == 0) else -1.0

    # carry invariant entering l: lambda_{l-1} = p * exp(s), lambda_{l-2} = q * exp(s)
    p0 = jnp.broadcast_to(jnp.asarray(seed_sign, x.dtype), x.shape)  # mantissa of lambda_mm
    q0 = jnp.zeros_like(x)  # lambda_{m-1,m} = 0
    s0 = seed_logabs
    lam_mm = p0 * jnp.exp(s0)  # true lambda_{m,m}; = 0 at poles for m >= 1

    if lmax == m:
        return lam_mm[None, :]

    a_np, b_np = legendre_recurrence_coeffs(m, lmax)
    coeffs = (jnp.asarray(a_np, x.dtype), jnp.asarray(b_np, x.dtype))

    def step(carry, coef):
        p, q, s = carry
        a_l, b_l = coef
        raw = a_l * x * p - b_l * q  # mantissa of lambda_l at scale s
        # renormalize the new pair (lambda_l, lambda_{l-1}) = (raw, p) to unit norm.
        # using the pair norm (never the single iterant) avoids dividing by a near-zero
        # when lambda_l crosses a root, which would inflate the carried lambda_{l-1}.
        r = jnp.sqrt(raw * raw + p * p)
        r = jnp.where(r == 0.0, 1.0, r)
        new_p = raw / r
        new_q = p / r
        new_s = s + jnp.log(r)
        val = new_p * jnp.exp(new_s)  # = raw * exp(s) = true lambda_l
        return (new_p, new_q, new_s), val

    _carry, lam_rest = jax.lax.scan(step, (p0, q0, s0), coeffs)  # lam_rest: (lmax-m, T)
    return jnp.concatenate([lam_mm[None, :], lam_rest], axis=0)


# --------------------------------------------------------------------------- #
# Spin-weighted (Wigner-d) recursion -- spin-0 is the m'=0 special case.
#
# The spin-weighted harmonics relate to the Wigner small-d functions by
# (libsharp Eq. 10):   {}_s lambda_{l,m}(theta) = (-1)^m sqrt((2l+1)/4pi) d^l_{-m,s}(theta).
# The d^l_{M,M'} obey a three-term recursion in l (libsharp Eq. 11 /
# Kostelec-Rockmore), solved upward for d^{l+1}:
#
#   d^{l+1} = [ (A_l x + C_l) d^l - B_l d^{l-1} ],  x = cos(theta),
#   A_l = (2l+1) l (l+1) / D_l,  C_l = -(2l+1) M M' / D_l,
#   B_l = (l+1) sqrt((l^2-M^2)(l^2-M'^2)) / D_l,
#   D_l = l sqrt(((l+1)^2-M^2)((l+1)^2-M'^2)).
#
# Same branch-free log-renorm as the scalar case.  Seeded at l_min=max(|M|,|M'|),
# where the explicit Wigner formula collapses to a single (exact) term.
# --------------------------------------------------------------------------- #
def wigner_recurrence_coeffs(M: int, Mp: int, lmin: int, lmax: int):
    """Static A_l, C_l, B_l for l = lmin .. lmax-1 (each produces d^{l+1})."""
    ell = np.arange(lmin, lmax, dtype=np.float64)
    D = ell * np.sqrt(((ell + 1.0) ** 2 - M * M) * ((ell + 1.0) ** 2 - Mp * Mp))
    A = (2.0 * ell + 1.0) * ell * (ell + 1.0) / D
    C = -(2.0 * ell + 1.0) * M * Mp / D
    B = (ell + 1.0) * np.sqrt((ell * ell - M * M) * (ell * ell - Mp * Mp)) / D
    return A, C, B


def _wigner_seed(x: jax.Array, M: int, Mp: int, lmin: int) -> tuple[float, jax.Array]:
    """Exact single-term seed d^{lmin}_{M,M'}: returns (sign, log|.|)."""
    k0 = max(0, M - Mp)  # the sole surviving summation index at l = lmin
    lg = math.lgamma
    log_fac = 0.5 * (
        lg(lmin + M + 1) + lg(lmin - M + 1) + lg(lmin + Mp + 1) + lg(lmin - Mp + 1)
    ) - (lg(k0 + 1) + lg(lmin + M - k0 + 1) + lg(lmin - Mp - k0 + 1) + lg(k0 - M + Mp + 1))
    sign = -1.0 if (k0 % 2) else 1.0
    pow_cos = 2 * lmin - 2 * k0 + M - Mp  # exponent of cos(theta/2) (>= 0)
    pow_sin = 2 * k0 - M + Mp  # exponent of sin(theta/2) (>= 0)
    cos_h = jnp.sqrt(jnp.clip((1.0 + x) / 2.0, 0.0, 1.0))
    sin_h = jnp.sqrt(jnp.clip((1.0 - x) / 2.0, 0.0, 1.0))
    log_abs = jnp.broadcast_to(jnp.asarray(log_fac, x.dtype), x.shape)
    if pow_cos > 0:  # guard 0 * log(0) = NaN at a pole
        log_abs = log_abs + pow_cos * jnp.log(cos_h)
    if pow_sin > 0:
        log_abs = log_abs + pow_sin * jnp.log(sin_h)
    return sign, log_abs


def wigner_d_column(x, M: int, Mp: int, lmax: int) -> jax.Array:
    """Wigner small-d ``d^l_{M,M'}(theta)`` for ``l = max(|M|,|M'|) .. lmax``."""
    M, Mp = int(M), int(Mp)
    lmin = max(abs(M), abs(Mp))
    if lmax < lmin:
        raise ValueError(f"lmax={lmax} < lmin={lmin} for (M,M')=({M},{Mp})")
    x = jnp.asarray(x)
    sign, seed_logabs = _wigner_seed(x, M, Mp, lmin)
    p0 = jnp.broadcast_to(jnp.asarray(sign, x.dtype), x.shape)
    q0 = jnp.zeros_like(x)
    s0 = seed_logabs
    d_min = p0 * jnp.exp(s0)
    if lmax == lmin:
        return d_min[None, :]

    A_np, C_np, B_np = wigner_recurrence_coeffs(M, Mp, lmin, lmax)
    coeffs = (jnp.asarray(A_np, x.dtype), jnp.asarray(C_np, x.dtype), jnp.asarray(B_np, x.dtype))

    def step(carry, coef):
        p, q, s = carry
        a_l, c_l, b_l = coef
        raw = (a_l * x + c_l) * p - b_l * q
        r = jnp.sqrt(raw * raw + p * p)
        r = jnp.where(r == 0.0, 1.0, r)
        new_p = raw / r
        new_q = p / r
        new_s = s + jnp.log(r)
        return (new_p, new_q, new_s), new_p * jnp.exp(new_s)

    _carry, d_rest = jax.lax.scan(step, (p0, q0, s0), coeffs)
    return jnp.concatenate([d_min[None, :], d_rest], axis=0)


def spin_weighted_lambda(x, m: int, spin: int, lmax: int) -> jax.Array:
    """``{}_s lambda_{l,m}(theta)`` for ``l = max(m,|spin|) .. lmax``.

    spin-0 (``spin=0``) returns the same functions as :func:`normalized_legendre`
    (it is the m'=0 case of the shared Wigner-d recursion).
    """
    m, spin = int(m), int(spin)
    lmin = max(m, abs(spin))
    d = wigner_d_column(x, -m, spin, lmax)  # d^l_{-m,s}, l = lmin..lmax
    ell = np.arange(lmin, lmax + 1, dtype=np.float64)
    pref = ((-1.0) ** m) * np.sqrt((2.0 * ell + 1.0) / (4.0 * np.pi))
    return jnp.asarray(pref, d.dtype)[:, None] * d
