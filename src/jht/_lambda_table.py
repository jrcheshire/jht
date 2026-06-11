"""fp64 spin-weighted lambda-table builder -- the fp64 -> fp32 hand-off for MLX.

The lambda recursion is the numerically dangerous part of an SHT: it underflows near
the poles (the sectoral seed carries ``sin(theta)**m``) and is exactly where s2fft's
spin-2 path fails.  jht solves it with a fp64 libsharp/Kostelec-Rockmore three-term
recursion + branch-free log-renorm.  This module runs that recursion in **fp64 numpy**,
vectorized over ``m`` and looping over ``l``, reusing the validated coefficients and
seed from :func:`jht._recursion.build_recursion_plan` and the exact ``_rec_step``
reconstruction, and materializes the dense table ``lambda^s_{l,m}(theta)``.

The table may be downcast to fp32 (``dtype=np.float32``) for the MLX (Apple-GPU) apply
**while the recursion itself never leaves fp64** -- this is what lets a fp32 backend hit
fp32 machine precision instead of garbage (see ``docs/mlx.md``).  By construction the
table equals :func:`jht._recursion.normalized_legendre` (spin 0) /
:func:`jht._recursion.spin_weighted_lambda` (spin +-2), with the ``pref`` folded in.

The dense table is ``O(M * (lmax+1) * T)`` -- caps the practical nside of the fp32 tier
(``tests/test_mlx.py`` documents the ceiling).  The JAX path never materializes it (it
fuses the contraction into the recursion ``lax.scan``); this table form is specific to
the separate-backend apply.
"""

from __future__ import annotations

import numpy as np

from ._recursion import build_recursion_plan


def lambda_table(x, spin: int, lmax: int, *, dtype=np.float64) -> np.ndarray:
    """Dense spin-weighted lambda table ``lambda^s_{l,m}(theta)``, shape ``(M, lmax+1, T)``.

    Parameters
    ----------
    x : array_like, shape ``(T,)``
        ``cos(theta)`` at the colatitudes (HEALPix rings, or a Clenshaw-Curtis grid).
    spin, lmax : int
        ``|spin| <= 3``; ``M = lmax + 1`` orders.
    dtype : numpy dtype
        Output dtype.  The recursion carry is always fp64; only the stored table is cast
        (use ``np.float32`` for the MLX fp32 tier).

    Returns
    -------
    table : np.ndarray, shape ``(M, lmax+1, T)``
        ``table[m, l]`` is ``lambda^s_{l,m}(theta)`` (``pref`` folded in), zero where
        ``l < lmin(m) = max(m, |spin|)``.
    """
    x = np.asarray(x, dtype=np.float64)
    plan = build_recursion_plan(x, spin, lmax)
    A, B, C, pref = plan.A, plan.B, plan.C, plan.pref
    seed_sign, seed_log, is_seed, is_active = (
        plan.seed_sign,
        plan.seed_log,
        plan.is_seed,
        plan.is_active,
    )
    M = lmax + 1
    T = x.shape[0]
    p = np.zeros((M, T))
    q = np.zeros((M, T))
    s = np.zeros((M, T))
    table = np.zeros((M, lmax + 1, T), dtype=dtype)
    for ell in range(lmax + 1):
        A_c, B_c, C_c = A[ell][:, None], B[ell][:, None], C[ell][:, None]  # (M, 1)
        seed_c, act_c = is_seed[ell][:, None], is_active[ell][:, None]
        raw = (A_c * x[None, :] + C_c) * p - B_c * q
        r = np.sqrt(raw * raw + p * p)
        r = np.where(r == 0.0, 1.0, r)
        rp, rq, rs = raw / r, p / r, s + np.log(r)
        is_recur = act_c & ~seed_c
        new_p = np.where(seed_c, seed_sign[:, None], np.where(is_recur, rp, p))
        new_q = np.where(seed_c, 0.0, np.where(is_recur, rq, q))
        new_s = np.where(seed_c, seed_log, np.where(is_recur, rs, s))
        with np.errstate(over="ignore", invalid="ignore"):
            lam = np.where(act_c, new_p * np.exp(new_s), 0.0)
        table[:, ell, :] = (pref[ell][:, None] * lam).astype(dtype)
        p, q, s = new_p, new_q, new_s
    return table
