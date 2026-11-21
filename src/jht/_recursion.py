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
log-scale.  This avoids the under/overflow that kills a naive recursion near the
poles (where the sectoral seed carries ``sin(theta)**m``); float64 roundoff grows
as ~eps*sqrt(l) and is measured flat to l = 32000.

The spin-weighted functions use the Wigner-d three-term recursion with the same
renormalization (:func:`wigner_d_column`, :func:`spin_weighted_lambda`).

Library code does **not** enable x64; callers opt in per entry point via
``jax.config.update("jax_enable_x64", True)``.  Accuracy work needs float64.
"""

from __future__ import annotations

import math
from functools import lru_cache
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


def _wigner_seed_np(x: np.ndarray, M: int, Mp: int, lmin: int) -> tuple[float, np.ndarray]:
    """Pure-numpy ``_wigner_seed`` for the static plan build (trace-safe; identical formula).

    ``build_recursion_plan`` must produce numpy tables even when first called *inside*
    a ``jit`` trace (cold ``lru_cache``); the jax :func:`_wigner_seed` would leak a tracer.
    """
    k0 = max(0, M - Mp)
    lg = math.lgamma
    log_fac = 0.5 * (
        lg(lmin + M + 1) + lg(lmin - M + 1) + lg(lmin + Mp + 1) + lg(lmin - Mp + 1)
    ) - (lg(k0 + 1) + lg(lmin + M - k0 + 1) + lg(lmin - Mp - k0 + 1) + lg(k0 - M + Mp + 1))
    sign = -1.0 if (k0 % 2) else 1.0
    pow_cos = 2 * lmin - 2 * k0 + M - Mp
    pow_sin = 2 * k0 - M + Mp
    cos_h = np.sqrt(np.clip((1.0 + x) / 2.0, 0.0, 1.0))
    sin_h = np.sqrt(np.clip((1.0 - x) / 2.0, 0.0, 1.0))
    log_abs = np.broadcast_to(np.float64(log_fac), x.shape).astype(np.float64).copy()
    with np.errstate(divide="ignore"):  # log(0) -> -inf at a pole -> exp = 0 (correct)
        if pow_cos > 0:
            log_abs = log_abs + pow_cos * np.log(cos_h)
        if pow_sin > 0:
            log_abs = log_abs + pow_sin * np.log(sin_h)
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
# Vectorized all-m recursion + fused contraction.
#
# The per-m functions above are the reference.  The transforms run every m through
# a single ``lax.scan`` over l (one fused, jit/vmap-clean kernel
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
    """Recursion inputs for one spin: colatitudes ``x = cos(theta)``, ``spin``, ``lmax``.

    The per-(l, m) tables are built from these inside each kernel's trace by
    :func:`_plan_arrays`; :func:`recursion_tables_np` is the NumPy reference.
    """

    x: np.ndarray
    spin: int
    lmax: int


def build_recursion_plan(x, spin: int, lmax: int) -> RecursionPlan:
    """Plan for the vectorized all-m recursion at colatitudes ``x = cos(theta)``.

    ``spin`` is an integer with ``|spin| <= 3`` (on-grid uses 0, +-2; the off-grid
    NUFFT path also uses +-1, +-3 for pointing-derivative templates).  ``x`` is a
    concrete ``(n_theta,)`` array (geometry is static).
    """
    spin = int(spin)
    if abs(spin) > 3:
        raise NotImplementedError(f"spin={spin} unsupported (only |spin| <= 3)")
    return RecursionPlan(np.asarray(x, dtype=np.float64), spin, int(lmax))


def recursion_tables_np(x, spin: int, lmax: int):
    """NumPy reference for the tables :func:`_plan_arrays` builds in-trace.

    Returns ``(A, B, C, pref, seed_sign, seed_log, is_seed, is_active)``: grids are
    ``(lmax+1, M)`` (l-major, ``M = lmax+1`` orders), ``seed_log`` is ``(M, n_theta)``
    and ``seed_sign`` is ``(M,)``.
    """
    spin = int(spin)
    if abs(spin) > 3:
        raise NotImplementedError(f"spin={spin} unsupported (only |spin| <= 3)")
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
        pref = (
            (((-1.0) ** np.arange(M))[None, :])
            * np.sqrt((2.0 * ell + 1.0) / (4.0 * np.pi))[:, None]
        )
        for m in range(M):
            lm = int(lmin[m])
            if lm < lmax:
                Ac, Cc, Bc = wigner_recurrence_coeffs(-m, spin, lm, lmax)  # produce d^{lm+1..lmax}
                A[lm + 1 : lmax + 1, m] = Ac
                C[lm + 1 : lmax + 1, m] = Cc
                B[lm + 1 : lmax + 1, m] = Bc
            sign, logabs = _wigner_seed_np(x, -m, spin, lm)  # numpy: trace-safe static build
            seed_sign[m] = sign
            seed_log[m] = logabs

    return A, B, C, pref, seed_sign, seed_log, is_seed, is_active


@lru_cache(maxsize=None)
def _seed_vectors_np(spin: int, lmax: int):
    """Per-m seed constants as ``(M,)`` NumPy arrays: ``(sign, log_const, pow_cos, pow_sin)``.

    ``log_const`` is the theta-independent part of the seed's log-magnitude;
    ``pow_cos`` / ``pow_sin`` are the exponents of ``cos(theta/2)`` / ``sin(theta/2)``
    (spin != 0 only; zero for spin 0). Same formulas as :func:`_sectoral_seed_logabs`
    and :func:`_wigner_seed_np`.
    """
    M = lmax + 1
    sign = np.empty(M, dtype=np.float64)
    log_const = np.empty(M, dtype=np.float64)
    pow_cos = np.zeros(M, dtype=np.int64)
    pow_sin = np.zeros(M, dtype=np.int64)
    lg = math.lgamma
    for m in range(M):
        if spin == 0:
            sign[m] = 1.0 if (m % 2 == 0) else -1.0
            log_const[m] = _sectoral_seed_logabs(m)
            continue
        Mo, Mp, lm = -m, spin, max(m, abs(spin))
        k0 = max(0, Mo - Mp)
        log_const[m] = 0.5 * (
            lg(lm + Mo + 1) + lg(lm - Mo + 1) + lg(lm + Mp + 1) + lg(lm - Mp + 1)
        ) - (lg(k0 + 1) + lg(lm + Mo - k0 + 1) + lg(lm - Mp - k0 + 1) + lg(k0 - Mo + Mp + 1))
        sign[m] = -1.0 if (k0 % 2) else 1.0
        pow_cos[m] = 2 * lm - 2 * k0 + Mo - Mp
        pow_sin[m] = 2 * k0 - Mo + Mp
    for arr in (sign, log_const, pow_cos, pow_sin):
        arr.setflags(write=False)
    return sign, log_const, pow_cos, pow_sin


def _plan_arrays(plan: RecursionPlan):
    """Build the recursion tables inside the trace (reference: :func:`recursion_tables_np`).

    Only ``(M,)`` and ``(n_theta,)`` vectors enter as constants. The ``(lmax+1, M)`` and
    ``(M, n_theta)`` grids are computed in-graph and passed through
    ``optimization_barrier``, so XLA neither folds them back into embedded constants nor
    keeps a copy per use site. Every table matches the reference exactly except
    ``seed_log``, which differs by <= 1 ulp under jit (XLA contracts multiply-add); through
    the recursion that is ~5e-15 relative at lmax=192 and ~2e-13 at lmax=4000.
    """
    spin, lmax = plan.spin, plan.lmax
    M = lmax + 1
    x = plan.x
    m_np = np.arange(M)
    sign, log_const, pow_cos, pow_sin = _seed_vectors_np(spin, lmax)

    ell = jnp.arange(M, dtype=jnp.float64)[:, None]  # (L1, 1); L1 == M
    l_idx = jnp.arange(M)[:, None]
    lmin = jnp.asarray(np.maximum(m_np, abs(spin)))[None, :]
    is_active = l_idx >= lmin
    is_seed = l_idx == lmin
    seed_log = jnp.broadcast_to(jnp.asarray(log_const)[:, None], (M, x.shape[0]))

    if spin == 0:
        m = jnp.asarray(m_np, dtype=jnp.float64)[None, :]
        rec = l_idx > jnp.asarray(m_np)[None, :]  # rows m+1 .. lmax
        den = jnp.where(rec, (ell - m) * (ell + m), 1.0)
        A = jnp.where(rec, jnp.sqrt((2.0 * ell - 1.0) * (2.0 * ell + 1.0) / den), 0.0)
        bden = jnp.where(rec, (2.0 * ell - 3.0) * (ell - m) * (ell + m), 1.0)
        B = jnp.where(rec, jnp.sqrt((2.0 * ell + 1.0) * ((ell - 1.0) ** 2 - m * m) / bden), 0.0)
        C = jnp.zeros((M, M), dtype=jnp.float64)
        pref = jnp.ones((M, M), dtype=jnp.float64)
        with np.errstate(divide="ignore"):
            log_sin = np.log(np.sqrt(np.clip(1.0 - x * x, 0.0, 1.0)))
        m_col = jnp.asarray(m_np, dtype=jnp.float64)[:, None]
        # m=0 has no sin factor (avoids 0 * log(0) at a pole)
        seed_log = jnp.where(m_col == 0.0, seed_log, seed_log + m_col * jnp.asarray(log_sin)[None, :])
    else:
        par_m = np.where(m_np % 2 == 0, 1.0, -1.0)
        norm_l = np.sqrt((2.0 * np.arange(M) + 1.0) / (4.0 * np.pi))  # NumPy factor: exact pref
        pref = jnp.asarray(par_m)[None, :] * jnp.asarray(norm_l)[:, None]
        valid = l_idx > lmin  # row l+1 holds the step from l, for l >= lmin(m)
        e = jnp.where(valid, ell - 1.0, float(lmax + 4))  # off-mask filler keeps D > 0
        mo = -jnp.asarray(m_np, dtype=jnp.float64)[None, :]  # M = -m
        mp = float(spin)
        D = e * jnp.sqrt(((e + 1.0) ** 2 - mo * mo) * ((e + 1.0) ** 2 - mp * mp))
        A = jnp.where(valid, (2.0 * e + 1.0) * e * (e + 1.0) / D, 0.0)
        C = jnp.where(valid, -(2.0 * e + 1.0) * mo * mp / D, 0.0)
        B = jnp.where(valid, (e + 1.0) * jnp.sqrt((e * e - mo * mo) * (e * e - mp * mp)) / D, 0.0)
        with np.errstate(divide="ignore"):
            log_cos_h = np.log(np.sqrt(np.clip((1.0 + x) / 2.0, 0.0, 1.0)))
            log_sin_h = np.log(np.sqrt(np.clip((1.0 - x) / 2.0, 0.0, 1.0)))
        pc = jnp.asarray(pow_cos)[:, None]
        ps = jnp.asarray(pow_sin)[:, None]
        seed_log = jnp.where(pc > 0, seed_log + pc * jnp.asarray(log_cos_h)[None, :], seed_log)
        seed_log = jnp.where(ps > 0, seed_log + ps * jnp.asarray(log_sin_h)[None, :], seed_log)

    return jax.lax.optimization_barrier(
        (A, B, C, pref, jnp.asarray(sign), seed_log, is_seed, is_active)
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


# --------------------------------------------------------------------------- #
# l-blocked contraction.
#
# The recursion state (p, q, s) does not depend on the data, so under ``vmap`` over
# maps it is computed once and shared; only the contraction scales with the batch.
# Contracting one l-row at a time streams the whole (batch, M, T) accumulator (or,
# for the adjoint, the batched data) through memory once per l, which is
# bandwidth-bound.  Every contraction below instead runs the unchanged per-l
# recursion over a block of K steps (an inner ``lax.scan`` emitting a (K, R, T)
# lambda stack over the R folded rows, see "Triangle folding") and contracts the
# block as one batched-over-rows matrix product.  lambda is bitwise unchanged; only
# the order of the l-sum differs.  Peak extra memory is the lambda stack, O(K.R.T),
# never the full (L, M, T) table.
# --------------------------------------------------------------------------- #
_ELL_BLOCK: int | str = "auto"
_ELL_BLOCK_MAX = 32
_ELL_BLOCK_BUDGET_BYTES = 512 * 2**20  # lambda-stack budget for the "auto" rule


def set_ell_block(k: int | str) -> None:
    """Set the l-block size ``K`` of the Legendre contraction for subsequent transforms.

    ``"auto"`` (default) picks ``K = clamp(budget // (n_stacks * R * T * 8), 1, 32)``
    (``R = ceil((lmax+1)/2)`` folded rows, ``T`` colatitudes) with a 512 MiB budget for
    the per-block lambda stack(s): ``K = 32`` to beyond nside 512, smaller at high nside.  An integer ``k >= 1`` pins it; ``k = 1``
    contracts one l-row at a time.  Results agree to rounding for every ``K``; the
    choice trades memory (O(K M T) per stack) for fewer passes over the
    accumulators, which matters for batched (``vmap``) transforms.

    Process-global, like :func:`jht.set_azimuth_fft_mode`; it is part of the
    transform cache key, so changing it compiles fresh kernels.
    """
    if isinstance(k, bool) or not (k == "auto" or (isinstance(k, int) and k >= 1)):
        raise ValueError(f"ell block must be 'auto' or an int >= 1, got {k!r}")
    global _ELL_BLOCK
    _ELL_BLOCK = k


def get_ell_block() -> int | str:
    """Return the current l-block setting (see :func:`set_ell_block`)."""
    return _ELL_BLOCK


def resolve_ell_block(setting: int | str, rows: int, T: int, n_stacks: int = 1) -> int:
    """Block size ``K`` for a ``(rows, T)`` recursion with ``n_stacks`` lambda stacks."""
    if setting == "auto":
        k = _ELL_BLOCK_BUDGET_BYTES // max(1, n_stacks * rows * T * 8)
        return int(min(_ELL_BLOCK_MAX, max(1, k)))
    return int(setting)


def _blocked(rows, K: int):
    """Pad l-major arrays (leading axis L1) to ``nb*K`` rows and reshape to ``(nb, K, ...)``.

    Padding is zero / ``False``: a row with ``is_active = is_seed = False`` is a no-op
    step of :func:`_rec_step` (carry unchanged, lambda = 0) with zero coefficients.
    """
    L1 = rows[0].shape[0]
    nb = -(-L1 // K)
    n_pad = nb * K - L1
    out = []
    for a in rows:
        a = jnp.asarray(a)
        if n_pad:
            a = jnp.pad(a, [(0, n_pad)] + [(0, 0)] * (a.ndim - 1))
        out.append(a.reshape((nb, K) + a.shape[1:]))
    return tuple(out)


def _synth_gemm(c, lam):
    """``sum_k c[s,k,m] lam[k,m,t]`` -> ``(S, M, T)`` complex; ``c`` complex, ``lam`` real.

    Real and imaginary parts go through one real contraction, so ``lam`` is never
    promoted to complex.
    """
    S = c.shape[0]
    cr = jnp.concatenate([jnp.real(c), jnp.imag(c)], axis=0)  # (2S, K, M)
    out = jnp.einsum("skm,kmt->smt", cr, lam)
    return jax.lax.complex(out[:S], out[S:])


def _adj_gemm(lam, V):
    """``sum_t lam[k,m,t] V[s,m,t]`` -> ``(S, K, M)`` complex; ``lam`` real, ``V`` complex."""
    S = V.shape[0]
    Vr = jnp.concatenate([jnp.real(V), jnp.imag(V)], axis=0)  # (2S, M, T)
    out = jnp.einsum("kmt,smt->skm", lam, Vr)
    return jax.lax.complex(out[:S], out[S:])


# --------------------------------------------------------------------------- #
# Triangle folding.
#
# Order m is active only for l >= lmin(m) = max(m, |spin|), so a rectangular (l, m)
# sweep spends about half its work on inactive rows.  Row r of the folded sweep runs
# order m_A = r from its seed to lmax, then restarts at the seed of its partner
# m_B = M-1-r (when m_B > m_A) and runs that to lmax: ~lmax+2 steps over ceil(M/2)
# rows instead of lmax+1 steps over M rows.  Every per-(l, m) table is gathered into
# the (step, row) layout; the per-step arithmetic is the unchanged _rec_step, so each
# lambda value is bitwise the same as the rectangular sweep's.  Contractions keep
# separate A / B outputs per row (phase-masked coefficients or data) and unfold at
# the end.
# --------------------------------------------------------------------------- #
class _Fold(NamedTuple):
    """Static (step, row) layout of the folded sweep for one (lmax, |spin|)."""

    R: int  # rows = ceil(M / 2)
    S: int  # steps
    n_b: int  # rows with a partner (m_B > m_A) = M - R
    ell: np.ndarray  # (S, R) l of each cell (0 when inactive)
    m: np.ndarray  # (S, R) m of each cell (0 when inactive)
    active: np.ndarray  # (S, R) bool
    seed: np.ndarray  # (S, R) bool, l == lmin(m)
    phase_b: np.ndarray  # (S, R) bool, the cell belongs to the partner m_B
    m_a: np.ndarray  # (R,)
    m_b: np.ndarray  # (R,) partner order (== m_a when there is none)
    inv: np.ndarray  # (M * (lmax+1),) dense (m, l) -> flat step*R + row; S*R = zero slot


@lru_cache(maxsize=None)
def _fold_np(lmax: int, abs_spin: int) -> _Fold:
    M = L1 = lmax + 1
    lmin = np.maximum(np.arange(M), abs_spin)
    n = np.clip(L1 - lmin, 0, None)  # active steps of each order
    R = (M + 1) // 2
    m_a = np.arange(R)
    m_b = M - 1 - m_a
    has_b = m_b > m_a
    m_b = np.where(has_b, m_b, m_a)
    S = max(1, int(np.max(n[m_a] + np.where(has_b, n[m_b], 0))))
    j = np.arange(S)[:, None]
    in_a = j < n[m_a][None, :]
    in_b = has_b[None, :] & ~in_a & (j < (n[m_a] + n[m_b])[None, :])
    active = in_a | in_b
    m = np.where(in_a, m_a[None, :], np.where(in_b, m_b[None, :], 0))
    ell = np.where(
        in_a,
        lmin[m_a][None, :] + j,
        np.where(in_b, lmin[m_b][None, :] + j - n[m_a][None, :], 0),
    )
    seed = active & (ell == lmin[m])
    inv = np.full(M * L1, S * R, dtype=np.int64)
    jj, rr = np.nonzero(active)
    inv[m[jj, rr] * L1 + ell[jj, rr]] = jj * R + rr
    out = _Fold(R, S, int(has_b.sum()), ell, m, active, seed, in_b, m_a, m_b, inv)
    for a in (ell, m, active, seed, in_b, m_a, m_b, inv):
        a.setflags(write=False)
    return out


def _fold_tables(plan: RecursionPlan, fold: _Fold):
    """Recursion tables of ``plan`` gathered into the folded (step, row) layout.

    Returns ``(A, B, C, pref, seed_sign_a, seed_sign_b, seed_log_a, seed_log_b)`` with the
    per-cell grids ``(S, R)`` and the per-row seeds ``(R,)`` / ``(R, T)`` for orders
    m_A and m_B.
    """
    A, B, C, pref, seed_sign, seed_log, _is, _ia = _plan_arrays(plan)
    ell, m = fold.ell, fold.m
    return (
        A[ell, m],
        B[ell, m],
        C[ell, m],
        pref[ell, m],
        seed_sign[fold.m_a],
        seed_sign[fold.m_b],
        seed_log[fold.m_a],
        seed_log[fold.m_b],
    )


def _fold_coef(fold: _Fold, dense):
    """Dense ``(M, lmax+1)`` coefficients -> ``(S, R)`` in the folded layout (0 off-mask)."""
    return jnp.where(fold.active, jnp.asarray(dense)[fold.m, fold.ell], 0.0)


def _unfold_rows(fold: _Fold, Fa, Fb):
    """Per-row A / B outputs ``(R, ...)`` -> ``(M, ...)`` in m order."""
    return jnp.concatenate([Fa, Fb[: fold.n_b][::-1]], axis=0)


def _unfold_cells(fold: _Fold, b_folded, lmax: int):
    """Folded per-cell outputs ``(S', R)`` (``S' >= S``) -> dense ``(M, lmax+1)``.

    A gather through the static inverse map (never a scatter); off-mask entries read
    the appended zero.
    """
    flat = b_folded[: fold.S].reshape(-1)
    flat = jnp.concatenate([flat, jnp.zeros((1,), flat.dtype)])
    return flat[fold.inv].reshape(lmax + 1, lmax + 1)


def _parity_cells(fold: _Fold):
    return jnp.asarray(np.where(fold.ell % 2 == 0, 1.0, -1.0))


def _sel(phb, b, a):
    """Per-cell select of the B / A contraction output (``phb``: (K, R))."""
    return jnp.where(phb, b, a)


def _one_state_step(x, ss_a, ss_b, sl_a, sl_b):
    def step(carry, xs):
        p, q, s = carry
        A_c, B_c, C_c, seed_c, act_c, phb_c = xs
        ss = jnp.where(phb_c, ss_b, ss_a)
        sl = jnp.where(phb_c[:, None], sl_b, sl_a)
        p, q, s, lam = _rec_step(p, q, s, A_c, B_c, C_c, seed_c, act_c, x, ss, sl)
        return (p, q, s), lam

    return step


def _two_state_step(x, seeds_p, seeds_m):
    ssp_a, ssp_b, slp_a, slp_b = seeds_p
    ssm_a, ssm_b, slm_a, slm_b = seeds_m

    def step(carry, xs):
        pp, qp, sp, pm, qm, sm = carry
        Ap_c, Bp_c, Cp_c, Am_c, Bm_c, Cm_c, sd_c, ac_c, phb_c = xs
        pb = phb_c[:, None]
        ssp = jnp.where(phb_c, ssp_b, ssp_a)
        ssm = jnp.where(phb_c, ssm_b, ssm_a)
        slp = jnp.where(pb, slp_b, slp_a)
        slm = jnp.where(pb, slm_b, slm_a)
        pp, qp, sp, lam_p = _rec_step(pp, qp, sp, Ap_c, Bp_c, Cp_c, sd_c, ac_c, x, ssp, slp)
        pm, qm, sm, lam_m = _rec_step(pm, qm, sm, Am_c, Bm_c, Cm_c, sd_c, ac_c, x, ssm, slm)
        return (pp, qp, sp, pm, qm, sm), (lam_p, lam_m)

    return step


def _ab(c, phb):
    """Split folded coefficients ``(K, R)`` into phase-A and phase-B parts."""
    return jnp.where(phb, 0.0, c), jnp.where(phb, c, 0.0)


def synth_contract(plan: RecursionPlan, x, alm_dense, ell_block: int | str = "auto") -> jax.Array:
    """``F_m(theta) = sum_l alm_dense[m,l] * lambda^s_{l,m}(theta)`` for all m.

    ``alm_dense`` is the dense ``(M, lmax+1)`` coefficient grid (``a_{l,m}``, zero
    where ``l < lmin(m)``).  Returns ``F`` of shape ``(M, n_theta)`` complex --
    the per-m ring coefficients before the phi-FFT.  ``ell_block``: see
    :func:`set_ell_block`.
    """
    fold = _fold_np(plan.lmax, abs(plan.spin))
    A, B, C, pref, ss_a, ss_b, sl_a, sl_b = _fold_tables(plan, fold)
    x = jnp.asarray(x)
    R, T = fold.R, x.shape[0]
    K = resolve_ell_block(ell_block, R, T, 1)
    coef = _fold_coef(fold, alm_dense) * pref
    step = _one_state_step(x, ss_a, ss_b, sl_a, sl_b)

    def block(carry, xs):
        rec, F = carry
        A_k, B_k, C_k, c_k, sd_k, ac_k, pb_k = xs
        rec, lam = jax.lax.scan(step, rec, (A_k, B_k, C_k, sd_k, ac_k, pb_k))
        return (rec, F + _synth_gemm(jnp.stack(_ab(c_k, pb_k)), lam)), None

    z = jnp.zeros((R, T), dtype=jnp.float64)
    init = ((z, z, z), jnp.zeros((2, R, T), dtype=jnp.complex128))
    xs = _blocked((A, B, C, coef, fold.seed, fold.active, fold.phase_b), K)
    (_, F), _ = jax.lax.scan(block, init, xs)
    return _unfold_rows(fold, F[0], F[1])


def adjoint_contract(plan: RecursionPlan, x, V, ell_block: int | str = "auto") -> jax.Array:
    """``b_{l,m} = sum_theta lambda^s_{l,m}(theta) * V[m,theta]`` for all (l, m).

    ``V`` is the ``(M, n_theta)`` per-m ring data (post phi-FFT).  Returns the
    dense ``(M, lmax+1)`` coefficient grid ``b_{l,m}`` (zero where ``l<lmin(m)``).
    This is the l-contraction of the exact adjoint; the FFT side lives in
    ``healpix.py``.  ``ell_block``: see :func:`set_ell_block`.
    """
    fold = _fold_np(plan.lmax, abs(plan.spin))
    A, B, C, pref, ss_a, ss_b, sl_a, sl_b = _fold_tables(plan, fold)
    x = jnp.asarray(x)
    V = jnp.asarray(V)
    R, T = fold.R, x.shape[0]
    K = resolve_ell_block(ell_block, R, T, 1)
    Vab = jnp.stack([V[fold.m_a], V[fold.m_b]])  # (2, R, T)
    step = _one_state_step(x, ss_a, ss_b, sl_a, sl_b)

    def block(rec, xs):
        A_k, B_k, C_k, pref_k, sd_k, ac_k, pb_k = xs
        rec, lam = jax.lax.scan(step, rec, (A_k, B_k, C_k, sd_k, ac_k, pb_k))
        g = _adj_gemm(lam, Vab)  # (2, K, R)
        return rec, pref_k * _sel(pb_k, g[1], g[0])

    z = jnp.zeros((R, T), dtype=jnp.float64)
    xs = _blocked((A, B, C, pref, fold.seed, fold.active, fold.phase_b), K)
    _, b = jax.lax.scan(block, (z, z, z), xs)
    return _unfold_cells(fold, b.reshape(-1, R), plan.lmax)


# --------------------------------------------------------------------------- #
# North/South (equatorial) symmetry.
#
# HEALPix rings come in reflected pairs (z, -z) with equal length and phi0.  The
# recursion -- ~98% of the transform cost -- need only run over the north half +
# equator (~2.nside colatitudes) because of the reflection parity
#
#     lambda^s_{l,m}(pi - theta) = (-1)^{l+m} lambda^{-s}_{l,m}(theta).
#
# spin-0 (s = -s): one half-grid scan gives both hemispheres from a "total" and a
# "signed" accumulator (F_north = sum_l c, F_south = (-1)^m sum_l (-1)^l c).  The
# +-2 coupling means spin-2 needs the d^{+2} and d^{-2} recursions advanced
# *together* over the half grid (still one half-grid pass each -> the 2x win), but
# with the south hemisphere of each channel built from the *other* spin's column.
# --------------------------------------------------------------------------- #
def synth_contract_eo(
    plan: RecursionPlan, x, alm_dense, ell_block: int | str = "auto", lam=None
):
    """Half-grid synthesis (spin-0): returns ``(F_total, F_signed)`` of shape ``(M, T)``.

    ``F_total = sum_l alm pref lambda``  and  ``F_signed = sum_l (-1)^l alm pref lambda``
    over the (north-half) colatitudes ``x``.  The caller forms
    ``F_north = F_total`` and ``F_south = (-1)^m F_signed``.  ``lam``: optional stored
    ``(nb, K, R, T)`` table from :func:`lambda_blocks_one`, read in place of the
    recursion (its ``K`` overrides ``ell_block``).
    """
    fold = _fold_np(plan.lmax, abs(plan.spin))
    A, B, C, pref, ss_a, ss_b, sl_a, sl_b = _fold_tables(plan, fold)
    x = jnp.asarray(x)
    R, T = fold.R, x.shape[0]
    coef = _fold_coef(fold, alm_dense) * pref
    par = _parity_cells(fold)

    if lam is not None:
        K = _check_lam(lam, fold, T)

        def block_t(F, xs):
            c_k, pb_k, par_k, lam_k = xs
            ca, cb = _ab(c_k, pb_k)
            return F + _synth_gemm(jnp.stack([ca, par_k * ca, cb, par_k * cb]), lam_k), None

        c_b, pb_b, par_b = _blocked((coef, fold.phase_b, par), K)
        F, _ = jax.lax.scan(block_t, jnp.zeros((4, R, T), jnp.complex128), (c_b, pb_b, par_b, lam))
        return _unfold_rows(fold, F[0], F[2]), _unfold_rows(fold, F[1], F[3])

    K = resolve_ell_block(ell_block, R, T, 1)
    step = _one_state_step(x, ss_a, ss_b, sl_a, sl_b)

    def block(carry, xs):
        rec, F = carry
        A_k, B_k, C_k, c_k, sd_k, ac_k, pb_k, par_k = xs
        rec, lam = jax.lax.scan(step, rec, (A_k, B_k, C_k, sd_k, ac_k, pb_k))
        ca, cb = _ab(c_k, pb_k)
        return (rec, F + _synth_gemm(jnp.stack([ca, par_k * ca, cb, par_k * cb]), lam)), None

    z = jnp.zeros((R, T), dtype=jnp.float64)
    init = ((z, z, z), jnp.zeros((4, R, T), dtype=jnp.complex128))
    xs = _blocked((A, B, C, coef, fold.seed, fold.active, fold.phase_b, par), K)
    (_, F), _ = jax.lax.scan(block, init, xs)
    return _unfold_rows(fold, F[0], F[2]), _unfold_rows(fold, F[1], F[3])


def adjoint_contract_eo(
    plan: RecursionPlan, x, Vn, Vs, ell_block: int | str = "auto", lam=None
):
    """Half-grid adjoint (spin-0): ``b_{l,m} = sum_theta lambda (Vn + (-1)^l Vs)``.

    ``Vn`` is the north-half per-m data, ``Vs = (-1)^m V_south`` (the reflected
    south data with the m-parity folded in; equator column zero).  Returns the
    dense ``(M, lmax+1)`` coefficients.  ``lam``: as in :func:`synth_contract_eo`.
    """
    fold = _fold_np(plan.lmax, abs(plan.spin))
    A, B, C, pref, ss_a, ss_b, sl_a, sl_b = _fold_tables(plan, fold)
    x = jnp.asarray(x)
    Vn, Vs = jnp.asarray(Vn), jnp.asarray(Vs)
    R, T = fold.R, x.shape[0]
    V = jnp.stack([Vn[fold.m_a], Vs[fold.m_a], Vn[fold.m_b], Vs[fold.m_b]])  # (4, R, T)
    par = _parity_cells(fold)

    def contract(lam_k, pref_k, pb_k, par_k):
        g = _adj_gemm(lam_k, V)  # (4, K, R)
        return pref_k * _sel(pb_k, g[2] + par_k * g[3], g[0] + par_k * g[1])

    if lam is not None:
        K = _check_lam(lam, fold, T)
        xs = (*_blocked((pref, fold.phase_b, par), K), lam)
        _, b = jax.lax.scan(lambda c, xs: (c, contract(xs[3], xs[0], xs[1], xs[2])), 0, xs)
        return _unfold_cells(fold, b.reshape(-1, R), plan.lmax)

    K = resolve_ell_block(ell_block, R, T, 1)
    step = _one_state_step(x, ss_a, ss_b, sl_a, sl_b)

    def block(rec, xs):
        A_k, B_k, C_k, pref_k, sd_k, ac_k, pb_k, par_k = xs
        rec, lam = jax.lax.scan(step, rec, (A_k, B_k, C_k, sd_k, ac_k, pb_k))
        return rec, contract(lam, pref_k, pb_k, par_k)

    z = jnp.zeros((R, T), dtype=jnp.float64)
    xs = _blocked((A, B, C, pref, fold.seed, fold.active, fold.phase_b, par), K)
    _, b = jax.lax.scan(block, (z, z, z), xs)
    return _unfold_cells(fold, b.reshape(-1, R), plan.lmax)


def _two_state_arrays(plan_p: RecursionPlan, plan_m: RecursionPlan, fold: _Fold):
    """Folded tables to drive the d^{+2} and d^{-2} recursions together over the half grid.

    ``plan_p`` / ``plan_m`` share ``pref`` and the active / seed masks (same |spin|) and
    differ only in the recurrence coeffs and the seeds.
    """
    Ap, Bp, Cp, pref, *seeds_p = _fold_tables(plan_p, fold)
    Am, Bm, Cm, _pf, *seeds_m = _fold_tables(plan_m, fold)
    return Ap, Bp, Cp, Am, Bm, Cm, pref, tuple(seeds_p), tuple(seeds_m)


def synth_contract_spin2_ns(
    plan_p, plan_m, x, bp_dense, bm_dense, ell_block: int | str = "auto", lam=None
):
    """Half-grid spin-2 synthesis.  Returns ``(Fp_north, Fp_south, Fm_north, Fm_south)``.

    ``bp = -(aE + i aB)``, ``bm = -(aE - i aB)`` (dense ``(M, lmax+1)``).  The
    south halves still carry the ``(-1)^m`` factor for the caller to apply.  The
    south of each channel uses the *other* spin's column.  ``lam``: optional stored
    ``(lam_p, lam_m)`` pair of ``(nb, K, R, T)`` tables from :func:`lambda_blocks_two`,
    read in place of the recursion (its ``K`` overrides ``ell_block``).
    """
    fold = _fold_np(plan_p.lmax, abs(plan_p.spin))
    Ap, Bp, Cp, Am, Bm, Cm, pref, seeds_p, seeds_m = _two_state_arrays(plan_p, plan_m, fold)
    x = jnp.asarray(x)
    R, T = fold.R, x.shape[0]
    cp = _fold_coef(fold, bp_dense) * pref
    cm = _fold_coef(fold, bm_dense) * pref
    par = _parity_cells(fold)

    def contract(Fp, Fm, cp_k, cm_k, pb_k, par_k, lam_p, lam_m):
        # Fp: (FpN, FmS) A/B from lambda_+; Fm: (FmN, FpS) A/B from lambda_-
        cpa, cpb = _ab(cp_k, pb_k)
        cma, cmb = _ab(cm_k, pb_k)
        Fp = Fp + _synth_gemm(jnp.stack([cpa, par_k * cma, cpb, par_k * cmb]), lam_p)
        Fm = Fm + _synth_gemm(jnp.stack([cma, par_k * cpa, cmb, par_k * cpb]), lam_m)
        return Fp, Fm

    zc = jnp.zeros((4, R, T), dtype=jnp.complex128)
    if lam is not None:
        lam_p_t, lam_m_t = lam
        K = _check_lam(lam_p_t, fold, T)
        _check_lam(lam_m_t, fold, T)

        def block_t(carry, xs):
            return contract(*carry, *xs), None

        xs = (*_blocked((cp, cm, fold.phase_b, par), K), lam_p_t, lam_m_t)
        (Fp, Fm), _ = jax.lax.scan(block_t, (zc, zc), xs)
    else:
        K = resolve_ell_block(ell_block, R, T, 2)
        step = _two_state_step(x, seeds_p, seeds_m)

        def block(carry, xs):
            rec, Fp, Fm = carry
            Ap_k, Bp_k, Cp_k, Am_k, Bm_k, Cm_k, cp_k, cm_k, sd_k, ac_k, pb_k, par_k = xs
            rec, (lam_p, lam_m) = jax.lax.scan(
                step, rec, (Ap_k, Bp_k, Cp_k, Am_k, Bm_k, Cm_k, sd_k, ac_k, pb_k)
            )
            Fp, Fm = contract(Fp, Fm, cp_k, cm_k, pb_k, par_k, lam_p, lam_m)
            return (rec, Fp, Fm), None

        z = jnp.zeros((R, T), dtype=jnp.float64)
        init = ((z, z, z, z, z, z), zc, zc)
        xs = _blocked(
            (Ap, Bp, Cp, Am, Bm, Cm, cp, cm, fold.seed, fold.active, fold.phase_b, par), K
        )
        (_, Fp, Fm), _ = jax.lax.scan(block, init, xs)
    FpN, FmS = _unfold_rows(fold, Fp[0], Fp[2]), _unfold_rows(fold, Fp[1], Fp[3])
    FmN, FpS = _unfold_rows(fold, Fm[0], Fm[2]), _unfold_rows(fold, Fm[1], Fm[3])
    return FpN, FpS, FmN, FmS


def adjoint_contract_spin2_ns(
    plan_p, plan_m, x, Vpn, Sp, Vmn, Sm, ell_block: int | str = "auto", lam=None
):
    """Half-grid spin-2 adjoint.  Returns ``(p2, m2)`` dense ``(M, lmax+1)``.

    ``Vpn / Vmn`` are the north-half +/- channel data; ``Sp = (-1)^m Vp_south``,
    ``Sm = (-1)^m Vm_south`` (reflected south with the m-parity folded in).
    ``p2`` contracts the +2 column with the north + the -2 column with the south;
    ``m2`` the mirror.  ``lam``: as in :func:`synth_contract_spin2_ns`.
    """
    fold = _fold_np(plan_p.lmax, abs(plan_p.spin))
    Ap, Bp, Cp, Am, Bm, Cm, pref, seeds_p, seeds_m = _two_state_arrays(plan_p, plan_m, fold)
    x = jnp.asarray(x)
    R, T = fold.R, x.shape[0]
    Vpn, Sp, Vmn, Sm = (jnp.asarray(a) for a in (Vpn, Sp, Vmn, Sm))
    ma, mb = fold.m_a, fold.m_b
    Vp = jnp.stack([Vpn[ma], Sm[ma], Vpn[mb], Sm[mb]])  # contracted with lambda_+
    Vm = jnp.stack([Vmn[ma], Sp[ma], Vmn[mb], Sp[mb]])  # contracted with lambda_-
    par = _parity_cells(fold)

    def contract(lam_p, lam_m, pref_k, pb_k, par_k):
        gp = _adj_gemm(lam_p, Vp)  # (4, K, R): lam_+ . [Vpn_A, Sm_A, Vpn_B, Sm_B]
        gm = _adj_gemm(lam_m, Vm)  # (4, K, R): lam_- . [Vmn_A, Sp_A, Vmn_B, Sp_B]
        p2 = _sel(pb_k, gp[2] + par_k * gm[3], gp[0] + par_k * gm[1])
        m2 = _sel(pb_k, gm[2] + par_k * gp[3], gm[0] + par_k * gp[1])
        return pref_k * p2, pref_k * m2

    if lam is not None:
        lam_p_t, lam_m_t = lam
        K = _check_lam(lam_p_t, fold, T)
        _check_lam(lam_m_t, fold, T)

        def block_t(c, xs):
            pref_k, pb_k, par_k, lp, lm = xs
            return c, contract(lp, lm, pref_k, pb_k, par_k)

        xs = (*_blocked((pref, fold.phase_b, par), K), lam_p_t, lam_m_t)
        _, (p2, m2) = jax.lax.scan(block_t, 0, xs)
    else:
        K = resolve_ell_block(ell_block, R, T, 2)
        step = _two_state_step(x, seeds_p, seeds_m)

        def block(rec, xs):
            Ap_k, Bp_k, Cp_k, Am_k, Bm_k, Cm_k, pref_k, sd_k, ac_k, pb_k, par_k = xs
            rec, (lam_p, lam_m) = jax.lax.scan(
                step, rec, (Ap_k, Bp_k, Cp_k, Am_k, Bm_k, Cm_k, sd_k, ac_k, pb_k)
            )
            return rec, contract(lam_p, lam_m, pref_k, pb_k, par_k)

        z = jnp.zeros((R, T), dtype=jnp.float64)
        xs = _blocked(
            (Ap, Bp, Cp, Am, Bm, Cm, pref, fold.seed, fold.active, fold.phase_b, par), K
        )
        _, (p2, m2) = jax.lax.scan(block, (z, z, z, z, z, z), xs)
    lmax = plan_p.lmax
    return (
        _unfold_cells(fold, p2.reshape(-1, R), lmax),
        _unfold_cells(fold, m2.reshape(-1, R), lmax),
    )


# --------------------------------------------------------------------------- #
# Stored lambda tables.
#
# The half-grid contractions above can read lambda from a stored (nb, K, R, T) table
# instead of running the recursion.  The builders below run the same folded sweep
# (_fold_tables + _one_state_step / _two_state_step, same blocking) and keep every
# block's lambda stack, so a stored table holds the values the recursion path computes
# per call; the contraction then repeats the same GEMMs in the same order.  Size is
# 8 * nb * K * R * T bytes per table (R = ceil((lmax+1)/2), T = 2*nside).
# --------------------------------------------------------------------------- #
def _check_lam(lam, fold: _Fold, T: int) -> int:
    """Validate a stored ``(nb, K, R, T)`` table against the fold; return its ``K``."""
    nb, K, R, T_l = lam.shape
    if R != fold.R or T_l != T or nb * K < fold.S or (nb - 1) * K >= fold.S:
        raise ValueError(
            f"stored lambda table shape {lam.shape} does not match this transform "
            f"(R={fold.R}, T={T}, {fold.S} steps)"
        )
    return int(K)


def lambda_blocks_one(plan: RecursionPlan, x, K: int) -> jax.Array:
    """Stored ``(nb, K, R, T)`` lambda table of the folded one-state sweep (spin 0)."""
    fold = _fold_np(plan.lmax, abs(plan.spin))
    A, B, C, _pref, ss_a, ss_b, sl_a, sl_b = _fold_tables(plan, fold)
    x = jnp.asarray(x)
    R, T = fold.R, x.shape[0]
    step = _one_state_step(x, ss_a, ss_b, sl_a, sl_b)

    def block(rec, xs):
        return jax.lax.scan(step, rec, xs)

    z = jnp.zeros((R, T), dtype=jnp.float64)
    xs = _blocked((A, B, C, fold.seed, fold.active, fold.phase_b), K)
    _, lam = jax.lax.scan(block, (z, z, z), xs)
    return lam


def lambda_blocks_two(plan_p: RecursionPlan, plan_m: RecursionPlan, x, K: int):
    """Stored ``(lam_p, lam_m)`` tables, each ``(nb, K, R, T)``, of the two-state sweep (spin 2)."""
    fold = _fold_np(plan_p.lmax, abs(plan_p.spin))
    Ap, Bp, Cp, Am, Bm, Cm, _pref, seeds_p, seeds_m = _two_state_arrays(plan_p, plan_m, fold)
    x = jnp.asarray(x)
    R, T = fold.R, x.shape[0]
    step = _two_state_step(x, seeds_p, seeds_m)

    def block(rec, xs):
        return jax.lax.scan(step, rec, xs)

    z = jnp.zeros((R, T), dtype=jnp.float64)
    xs = _blocked((Ap, Bp, Cp, Am, Bm, Cm, fold.seed, fold.active, fold.phase_b), K)
    _, (lam_p, lam_m) = jax.lax.scan(block, (z, z, z, z, z, z), xs)
    return lam_p, lam_m
