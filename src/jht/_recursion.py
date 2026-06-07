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
from typing import NamedTuple

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


# --------------------------------------------------------------------------- #
# Vectorized all-m recursion + fused contraction (Phase 1 performance).
#
# The per-m functions above are the *validated reference*.  For production we run
# every m through a single ``lax.scan`` over l (one fused, jit/vmap-clean kernel
# instead of lmax+1 eager scans), and **fuse the l-contraction into the scan** so
# the full lambda table is never materialized -- peak memory is O(M.n_theta), not
# O(L.M.n_theta) (~16 GB/component at nside=2048).  The static recurrence/seed
# tables are built once in numpy by reusing the exact coefficient helpers above,
# so the numerics are identical to the reference by construction:
#
#   spin-0 -> Legendre 3-term (a_l, b_l) + sectoral seed   (pref == 1)
#   spin-2 -> Wigner-d 3-term (A_l, C_l, B_l) + exact seed  (pref = (-1)^m * sqrt((2l+1)/4pi))
#
# Tables are stored l-major ((lmax+1, M)) so the scan iterates over the leading
# axis directly.  lambda_{l,m} carries an O(1) mantissa + a log-scale (the same
# branch-free log-renorm as the reference); inactive (l<lmin(m)) and seed
# (l==lmin(m)) steps are handled branch-free via per-(l,m) masks.
# --------------------------------------------------------------------------- #
class RecursionPlan(NamedTuple):
    """Static (numpy) tables driving the vectorized all-m recursion for one spin.

    Grids are ``(lmax+1, M)`` (l-major, M = lmax+1 orders); ``seed_log`` is
    ``(M, n_theta)``; ``seed_sign`` is ``(M,)``.  ``spin`` and ``lmax`` are kept
    for shape/spin bookkeeping by callers.
    """

    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    pref: np.ndarray
    seed_sign: np.ndarray
    seed_log: np.ndarray
    is_seed: np.ndarray
    is_active: np.ndarray
    spin: int
    lmax: int


def build_recursion_plan(x, spin: int, lmax: int) -> RecursionPlan:
    """Precompute the static all-m recursion tables at colatitudes ``x=cos(theta)``.

    ``spin`` is one of ``0, +2, -2``.  ``x`` is a concrete ``(n_theta,)`` array of
    ring colatitudes (HEALPix geometry is static), so the whole plan is numpy and
    becomes compile-time constants inside the jitted transform.
    """
    spin = int(spin)
    if spin not in (0, 2, -2):
        raise NotImplementedError(f"spin={spin} unsupported (only 0, +2, -2)")
    x = np.asarray(x, dtype=np.float64)
    M = lmax + 1
    L1 = lmax + 1
    abs_s = abs(spin)
    lmin = np.maximum(np.arange(M), abs_s)  # max(m, |spin|)

    ell_idx = np.arange(L1)[:, None]  # (L1, 1)
    is_active = ell_idx >= lmin[None, :]  # (L1, M)
    is_seed = ell_idx == lmin[None, :]

    A = np.zeros((L1, M), dtype=np.float64)
    B = np.zeros((L1, M), dtype=np.float64)
    C = np.zeros((L1, M), dtype=np.float64)
    seed_sign = np.zeros(M, dtype=np.float64)
    seed_log = np.zeros((M, x.shape[0]), dtype=np.float64)
    pref: np.ndarray

    if spin == 0:
        pref = np.ones((L1, M), dtype=np.float64)
        sin_t = np.sqrt(np.clip(1.0 - x * x, 0.0, 1.0))
        with np.errstate(divide="ignore"):
            log_sin = np.log(sin_t)
        for m in range(M):
            a, b = legendre_recurrence_coeffs(m, lmax)  # produce lambda_{m+1..lmax}
            if a.size:
                A[m + 1 : lmax + 1, m] = a
                B[m + 1 : lmax + 1, m] = b
            seed_sign[m] = 1.0 if (m % 2 == 0) else -1.0
            const = _sectoral_seed_logabs(m)
            seed_log[m] = const if m == 0 else const + m * log_sin
    else:
        ell = np.arange(L1, dtype=np.float64)
        pref = (((-1.0) ** np.arange(M))[None, :]) * np.sqrt((2.0 * ell + 1.0) / (4.0 * np.pi))[
            :, None
        ]
        xj = jnp.asarray(x)
        for m in range(M):
            lm = int(lmin[m])
            if lm < lmax:
                Ac, Cc, Bc = wigner_recurrence_coeffs(-m, spin, lm, lmax)  # produce d^{lm+1..lmax}
                A[lm + 1 : lmax + 1, m] = Ac
                C[lm + 1 : lmax + 1, m] = Cc
                B[lm + 1 : lmax + 1, m] = Bc
            sign, logabs = _wigner_seed(xj, -m, spin, lm)
            seed_sign[m] = sign
            seed_log[m] = np.asarray(logabs)

    return RecursionPlan(A, B, C, pref, seed_sign, seed_log, is_seed, is_active, spin, lmax)


def _plan_arrays(plan: RecursionPlan):
    """Convert the static plan grids to jnp once (compile-time constants)."""
    return (
        jnp.asarray(plan.A),
        jnp.asarray(plan.B),
        jnp.asarray(plan.C),
        jnp.asarray(plan.pref),
        jnp.asarray(plan.seed_sign),
        jnp.asarray(plan.seed_log),
        jnp.asarray(plan.is_seed),
        jnp.asarray(plan.is_active),
    )


def _rec_step(p, q, s, A_c, B_c, C_c, is_seed_c, is_active_c, x, seed_sign, seed_log):
    """One l-step of the all-m recursion; returns the new carry and lambda row.

    ``p, q, s`` are ``(M, T)`` (mantissa of lambda_{l-1}, lambda_{l-2}, and the
    log-scale).  ``A_c, B_c, C_c, is_*_c`` are the ``(M,)`` column for this l.
    """
    raw = (A_c[:, None] * x[None, :] + C_c[:, None]) * p - B_c[:, None] * q
    r = jnp.sqrt(raw * raw + p * p)
    r = jnp.where(r == 0.0, 1.0, r)
    rp, rq, rs = raw / r, p / r, s + jnp.log(r)
    is_recur = is_active_c & ~is_seed_c  # l > lmin
    sd = is_seed_c[:, None]
    rc = is_recur[:, None]
    new_p = jnp.where(sd, seed_sign[:, None], jnp.where(rc, rp, p))
    new_q = jnp.where(sd, 0.0, jnp.where(rc, rq, q))
    new_s = jnp.where(sd, seed_log, jnp.where(rc, rs, s))
    lam = jnp.where(is_active_c[:, None], new_p * jnp.exp(new_s), 0.0)
    return new_p, new_q, new_s, lam


def synth_contract(plan: RecursionPlan, x, alm_dense) -> jax.Array:
    """``F_m(theta) = sum_l alm_dense[m,l] * lambda^s_{l,m}(theta)`` for all m.

    ``alm_dense`` is the dense ``(M, lmax+1)`` coefficient grid (``a_{l,m}``, zero
    where ``l < lmin(m)``).  Returns ``F`` of shape ``(M, n_theta)`` complex --
    the per-m ring coefficients before the phi-FFT.
    """
    A, B, C, pref, seed_sign, seed_log, is_seed, is_active = _plan_arrays(plan)
    x = jnp.asarray(x)
    M, T = seed_log.shape
    alm_T = jnp.asarray(alm_dense).T  # (L1, M)

    def body(carry, xs):
        p, q, s, F = carry
        A_c, B_c, C_c, pref_c, alm_c, seed_c, act_c = xs
        new_p, new_q, new_s, lam = _rec_step(
            p, q, s, A_c, B_c, C_c, seed_c, act_c, x, seed_sign, seed_log
        )
        F = F + (alm_c * pref_c)[:, None] * lam
        return (new_p, new_q, new_s, F), None

    z = jnp.zeros((M, T), dtype=jnp.float64)
    init = (z, z, z, jnp.zeros((M, T), dtype=jnp.complex128))
    (_, _, _, F), _ = jax.lax.scan(body, init, (A, B, C, pref, alm_T, is_seed, is_active))
    return F


def adjoint_contract(plan: RecursionPlan, x, V) -> jax.Array:
    """``b_{l,m} = sum_theta lambda^s_{l,m}(theta) * V[m,theta]`` for all (l, m).

    ``V`` is the ``(M, n_theta)`` per-m ring data (post phi-FFT).  Returns the
    dense ``(M, lmax+1)`` coefficient grid ``b_{l,m}`` (zero where ``l<lmin(m)``).
    This is the l-contraction of the exact adjoint; the FFT side lives in
    ``healpix.py``.
    """
    A, B, C, pref, seed_sign, seed_log, is_seed, is_active = _plan_arrays(plan)
    x = jnp.asarray(x)
    V = jnp.asarray(V)
    M, T = seed_log.shape

    def body(carry, xs):
        p, q, s = carry
        A_c, B_c, C_c, pref_c, seed_c, act_c = xs
        new_p, new_q, new_s, lam = _rec_step(
            p, q, s, A_c, B_c, C_c, seed_c, act_c, x, seed_sign, seed_log
        )
        b_col = pref_c * jnp.sum(lam * V, axis=1)  # (M,)
        return (new_p, new_q, new_s), b_col

    z = jnp.zeros((M, T), dtype=jnp.float64)
    _, b = jax.lax.scan(body, (z, z, z), (A, B, C, pref, is_seed, is_active))  # b: (L1, M)
    return b.T
