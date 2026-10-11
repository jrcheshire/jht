"""Partial-sky / masked HEALPix analysis: the cut-sky map -> a_lm operators.

Three operators for a per-pixel sky ``mask`` (RING order, binary 0/1 or apodized
in ``[0,1]``):

1. :func:`pseudo_alm` -- the **masked pseudo-a_lm** (zero-fill): mask the *map*,
   keep the *full* quadrature weights, run the existing full-residual Jacobi
   (:func:`jht.analysis`).  This is the standard CMB pseudo-a_lm; with
   uniform weights (``use_weights=False``) it is exactly
   ``(4pi/Npix) S^T (M m)`` -- the weight-unambiguous estimator that healpy/ducc
   compute.  It is *biased* by mode-coupling (the cut mixes ell, m): it does not
   recover the true a_lm, and is not meant to.

2. :func:`deconvolve` -- the **cut-sky deconvolution**: solve the masked normal
   equations ``A a = b`` with conjugate gradient, where

       A = S^T (M W) S,    b = S^T (M W) m .

   ``A`` is the masked least-squares operator; it reduces to ``S^T W S ~ I`` as
   ``M -> 1`` (so CG -> full-sky :func:`jht.analysis` in ~1 step).  For noiseless
   band-limited data ``m = S a_true`` it recovers ``a = a_true`` *exactly* and
   *independently of W* wherever the cut leaves the modes constrained (where ``A``
   restricted to the active band is well-conditioned).  Near-null modes
   (supported almost entirely under the cut -- e.g. spin-2 E/B ambiguous modes)
   are not recoverable from the data; CG from ``x0=0`` returns the minimum-norm
   solution, and an optional Tikhonov ``reg`` stabilises hard masks.

3. :func:`wiener` (+ :func:`constrained_realization`) -- the **Wiener filter** /
   MUSE inner solve: the noise-aware posterior for ``m = S a + n`` with per-pixel
   inverse-noise ``N^-1`` and a signal prior ``a ~ N(0, C)``, ``C_{lm} = Cl``.  It
   solves ``(S^T N^-1 S + C^-1) a = S^T N^-1 m`` -- the *same* operator family as
   ``deconvolve``, but the scalar ``reg`` becomes the Cl-informed prior, which in
   the x-space below is the *exact* diagonal ``D = diag(1/Cl)``.  The prior bounds
   the near-null / E-B ambiguous modes that the reg=0 deconvolution amplifies (a
   bias-for-variance trade), and :func:`constrained_realization` draws posterior
   samples for the full MUSE gradient.  Both run CG in **prior-whitened**
   coordinates ``y = P^-1 x``, ``P = diag(sqrt(Cl))``: the operator becomes
   ``P A_x P + I`` (eigenvalues >= 1), and zero-power multipoles (``Cl = 0``,
   e.g. monopole/dipole) have ``P = 0`` exactly -- they are pinned to 0 rather
   than carried as a huge finite ``1/Cl``, which would break the CG
   relative-residual stopping rule.  :func:`wiener_preconditioner` builds an
   optional preconditioner for this system, and ``return_info=True`` reports the
   iteration count and final residual (:class:`SolveInfo`).

The inner-product subtlety (the ``2*conj`` convention): ``S``/``S^T`` are
adjoint in the ``(2 - delta_{m0})``-weighted a_lm inner product, not the
Euclidean one (see :func:`jht.healpix.adjoint_synthesis`).  So ``A`` is
Hermitian-PD under ``<.,.>_w``, not Euclidean.  It is handled once by a
real-DOF **isometry** ``T: a <-> x`` (:func:`alm_to_real` / :func:`real_to_alm`)
with ``||x||_2 = ||a||_w``: ``a_{l0} -> a_{l0}`` (real), ``a_{lm>0} -> [sqrt2 Re,
sqrt2 Im]``, dropping the structurally-zero ``l < |spin|`` modes.  In ``x``-space
``A_x = T A T^-1`` is real-symmetric-PD in plain Euclidean, so plain CG
(:func:`jax.scipy.sparse.linalg.cg`, or jht's preconditioned CG) is exactly
correct.

Library code does not enable x64; callers opt in per entry point (x64 is needed
for the deep accuracy here, as elsewhere in jht).
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg

from ._analysis import analysis
from .healpix import adjoint_synthesis, alm_size, synthesis
from .weights import pixel_weights

_SQRT2 = float(np.sqrt(2.0))


# --------------------------------------------------------------------------- #
# per-pixel weight (inverse-noise N^-1, or the mask * quadrature default)
# --------------------------------------------------------------------------- #
def _weight_pix(nside: int, spin: int, *, inv_noise=None, mask=None, use_weights: bool = True) -> jax.Array:
    """Per-pixel weight for the masked normal operator, broadcastable to the map.

    ``inv_noise`` (the per-pixel inverse-noise variance ``N^-1``, with the mask
    folded in -- masked pixels are 0) takes precedence; otherwise the weight is
    ``mask * W`` (the mask times the ring quadrature weight -- the
    :func:`deconvolve` default).  ``mask``/``inv_noise`` are one sky array
    ``(Npix,)`` (same for Q and U); returns ``(Npix,)`` for spin 0, ``(1, Npix)``
    for spin 2.
    """
    if inv_noise is not None:
        w = jnp.asarray(inv_noise)
    elif mask is not None:
        w = jnp.asarray(mask) * jnp.asarray(pixel_weights(nside, use_weights))
    else:
        raise ValueError("provide either inv_noise (per-pixel N^-1) or mask")
    return w if spin == 0 else w[None, :]


# --------------------------------------------------------------------------- #
# masked pseudo-a_lm (zero-fill)
# --------------------------------------------------------------------------- #
def pseudo_alm(maps, mask, nside: int, lmax: int, spin: int = 0, niter: int = 3, use_weights: bool = True) -> jax.Array:
    """Masked pseudo-a_lm: ``analysis(M * map)`` (zero-fill, healpy/ducc-style).

    With ``use_weights=False, niter=0`` this is the canonical uniform pseudo-a_lm
    ``(4pi/Npix) S^T (M m)``; with ring weights / iteration it is jht's own
    (weight-conditioned) pseudo-a_lm.  Biased by mode-coupling -- a building
    block (e.g. for pseudo-C_ell), not the true-a_lm recovery (use
    :func:`deconvolve` for that).
    """
    maps = jnp.asarray(maps)
    msk = jnp.asarray(mask)
    masked = (msk if spin == 0 else msk[None, :]) * maps
    return analysis(masked, nside, lmax, spin=spin, niter=niter, use_weights=use_weights)


# --------------------------------------------------------------------------- #
# the real-DOF isometry  T : a (complex, healpy-packed) <-> x (real)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _dof_layout(lmax: int, spin: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Static index arrays for ``T``: ``(k_m0, k_mpos, K, nx_per_channel)``.

    ``k_m0`` are the flat a_lm indices with ``m=0`` and ``l>=|spin|``; ``k_mpos``
    those with ``m>0`` and ``l>=|spin|``.  The structurally-zero ``l<|spin|``
    modes are excluded (they are null directions of ``A``).
    """
    ms = np.concatenate([np.full(lmax + 1 - m, m) for m in range(lmax + 1)])
    ls = np.concatenate([np.arange(m, lmax + 1) for m in range(lmax + 1)])
    active = ls >= abs(spin)
    idx = np.arange(alm_size(lmax))
    k_m0 = idx[active & (ms == 0)]
    k_mpos = idx[active & (ms > 0)]
    nx = int(k_m0.size + 2 * k_mpos.size)
    # NumPy (not jnp): these are static gather indices (lmax/spin only). An
    # lru_cache that returns a jnp array caches a *tracer* if first called under a
    # jit/grad/scan trace, which then leaks (UnexpectedTracerError) -- the same
    # cold-cache-under-trace hazard guarded in _recursion._wigner_seed. NumPy
    # indices gather into jnp arrays unchanged (jnp[np]), so callers are unaffected.
    return k_m0, k_mpos, alm_size(lmax), nx


def n_dof(lmax: int, spin: int = 0) -> int:
    """Total length of the real DOF vector ``x`` (both E,B channels for spin 2)."""
    nx = _dof_layout(lmax, spin)[3]
    return nx if spin == 0 else 2 * nx


def _to_real_1(a: jax.Array, k_m0: np.ndarray, k_mpos: np.ndarray) -> jax.Array:
    return jnp.concatenate([a[k_m0].real, _SQRT2 * a[k_mpos].real, _SQRT2 * a[k_mpos].imag])


def _to_alm_1(x: jax.Array, k_m0: np.ndarray, k_mpos: np.ndarray, k: int) -> jax.Array:
    n0 = k_m0.shape[0]
    npp = k_mpos.shape[0]
    a = jnp.zeros(k, dtype=jnp.complex128)
    a = a.at[k_m0].set(x[:n0].astype(jnp.complex128))
    re = x[n0 : n0 + npp]
    im = x[n0 + npp : n0 + 2 * npp]
    return a.at[k_mpos].set((re + 1j * im) / _SQRT2)


def alm_to_real(alm, lmax: int, spin: int = 0) -> jax.Array:
    """``T``: healpy-packed a_lm -> real DOF vector ``x`` with ``||x||_2 = ||a||_w``."""
    k_m0, k_mpos, _, _ = _dof_layout(lmax, spin)
    a = jnp.asarray(alm)
    if spin == 0:
        return _to_real_1(a, k_m0, k_mpos)
    return jnp.concatenate([_to_real_1(a[0], k_m0, k_mpos), _to_real_1(a[1], k_m0, k_mpos)])


def real_to_alm(x, lmax: int, spin: int = 0) -> jax.Array:
    """``T^-1``: real DOF vector ``x`` -> healpy-packed a_lm (``l<|spin|`` modes zero)."""
    k_m0, k_mpos, k, nx = _dof_layout(lmax, spin)
    xv = jnp.asarray(x)
    if spin == 0:
        return _to_alm_1(xv, k_m0, k_mpos, k)
    return jnp.stack([_to_alm_1(xv[:nx], k_m0, k_mpos, k), _to_alm_1(xv[nx:], k_m0, k_mpos, k)])


# --------------------------------------------------------------------------- #
# Cl signal prior: the x-space whitening  P = T C^1/2 T^-1 = diag(sqrt(Cl))
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _prior_ell_index(lmax: int, spin: int) -> np.ndarray:
    """Per-DOF multipole ``l`` for one channel, in the ``_to_real_1`` layout.

    ``T`` sends ``a_{l0} -> x`` and ``a_{lm>0} -> [sqrt2 Re, sqrt2 Im]``, so the
    real DOF vector for one channel is ``[ (m=0 block), (Re block), (Im block) ]``;
    this returns the multipole ``l`` of each entry (length ``nx``).  Because ``C``
    is diagonal in ``(l, m)`` with ``C_{lm} = Cl`` (constant over ``m``), the prior
    ``C^-1`` is *exactly diagonal* in ``x``-space -- ``D[i] = 1/Cl(l(i))``.
    """
    ms = np.concatenate([np.full(lmax + 1 - m, m) for m in range(lmax + 1)])
    ls = np.concatenate([np.arange(m, lmax + 1) for m in range(lmax + 1)])
    active = ls >= abs(spin)
    l_m0 = ls[active & (ms == 0)]
    l_mpos = ls[active & (ms > 0)]
    # NumPy (not jnp): static per-DOF multipole index, lru_cached -- see
    # _dof_layout on why a cached jnp array would leak a tracer under a trace.
    return np.concatenate([l_m0, l_mpos, l_mpos])


def _prior_sqrt(signal_cl, lmax: int, spin: int) -> jax.Array:
    """x-space prior whitening ``pw = sqrt(Cl)`` per DOF (length :func:`n_dof`).

    ``signal_cl`` is the signal power spectrum: a ``(lmax+1,)`` array for spin 0, or
    ``(C_EE, C_BB)`` (two ``(lmax+1,)`` arrays, the E and B priors) for spin 2.
    ``P = diag(pw)`` is the change of variables ``x = P y`` that turns the prior
    ``C^-1`` into the identity.  Zero-power multipoles get ``pw = 0`` *exactly* --
    the infinite-prior limit handled by exclusion, not by a large finite ``1/Cl``
    (which would pollute the CG stopping rule, catastrophically so through the
    constrained-realization noise source).  The sqrt is guarded so it is grad-safe
    in ``Cl``.
    """
    ell = _prior_ell_index(lmax, spin)

    def one(cl) -> jax.Array:
        clv = jnp.asarray(cl)
        if clv.shape != (lmax + 1,):
            raise ValueError(
                f"signal_cl channel must have shape (lmax+1,) = ({lmax + 1},), got {clv.shape}"
            )
        clv = clv[ell]
        pos = clv > 0
        return jnp.where(pos, jnp.sqrt(jnp.where(pos, clv, 1.0)), 0.0)

    if spin == 0:
        return one(signal_cl)
    c_ee, c_bb = signal_cl
    return jnp.concatenate([one(c_ee), one(c_bb)])


# --------------------------------------------------------------------------- #
# the masked normal operator A_x + D  and its RHS (shared by deconvolve/wiener)
# --------------------------------------------------------------------------- #
def _normal_op(x, w_pix, d_diag, nside: int, lmax: int, spin: int) -> jax.Array:
    """``(A_x + D) x = T S^T diag(w) S T^-1 x + D x`` -- real-symmetric-PD in ``x``.

    ``w_pix`` is the per-pixel weight (``N^-1`` or ``M W``, broadcastable to the
    map); ``d_diag`` is the added scalar Tikhonov ``reg`` (the :func:`deconvolve`
    path; the Cl-prior solves use the whitened :func:`_whitened_op` instead).
    """
    a = real_to_alm(x, lmax, spin)
    wmp = w_pix * synthesis(a, nside, lmax, spin)
    ax = alm_to_real(adjoint_synthesis(wmp, nside, lmax, spin), lmax, spin)
    return ax + d_diag * x


def _whitened_op(y, w_pix, pw, nside: int, lmax: int, spin: int) -> jax.Array:
    """``(P A_x P + I) y`` with ``P = diag(pw)`` -- the prior-whitened normal operator.

    The substitution ``x = P y`` turns ``(A_x + diag(1/Cl)) x = b`` into
    ``(P A_x P + I) y = P b``: ``P diag(1/Cl) P = I`` on the physical (``Cl > 0``)
    modes, and on zero-power modes (``pw = 0``) the operator is the identity, so
    the zeroed RHS pins them to ``y = 0`` (hence ``x = P y = 0``) *exactly*.
    Eigenvalues are ``>= 1``, so CG is well conditioned and the relative-residual
    stopping rule is meaningful for every mode -- no ``1/Cl -> inf`` scale ever
    enters the system.
    """
    a = real_to_alm(pw * y, lmax, spin)
    wmp = w_pix * synthesis(a, nside, lmax, spin)
    ax = alm_to_real(adjoint_synthesis(wmp, nside, lmax, spin), lmax, spin)
    return pw * ax + y


def _whiten_x0(x0, pw, lmax: int, spin: int) -> jax.Array:
    """Warm-start a_lm -> whitened coordinates ``y0 = P^+ T(x0)`` (0 on pinned modes)."""
    xv = alm_to_real(x0, lmax, spin)
    return jnp.where(pw > 0, xv / jnp.where(pw > 0, pw, 1.0), 0.0)


def _rhs_x(maps, w_pix, nside: int, lmax: int, spin: int) -> jax.Array:
    """``b_x = T S^T (w * m)`` -- the (weighted) adjoint of the data in x-space."""
    return alm_to_real(adjoint_synthesis(w_pix * jnp.asarray(maps), nside, lmax, spin), lmax, spin)


# --------------------------------------------------------------------------- #
# preconditioned CG with convergence diagnostics, and the Wiener preconditioner
# --------------------------------------------------------------------------- #
class SolveInfo(NamedTuple):
    """CG convergence diagnostics: iterations run and the final ``||r|| / ||b||``.

    ``rel_residual > tol`` means the solve stopped at ``max_iter`` unconverged.  Both are
    arrays (per lane under ``vmap``).
    """

    n_iter: jax.Array
    rel_residual: jax.Array


class WienerPreconditioner(NamedTuple):
    """``M^-1`` for the whitened Wiener operator ``P A_x P + I``, from :func:`wiener_preconditioner`.

    ``M`` is diagonal, ``1 + Cl * nbar``, except on the DOFs ``low_idx`` (``l <=
    ell_split``), where it is the exact operator block; ``diag_inv`` and ``low_inv`` hold
    the inverses.
    """

    diag_inv: jax.Array
    low_idx: jax.Array
    low_inv: jax.Array


def _apply_precond(p: WienerPreconditioner, r: jax.Array) -> jax.Array:
    """``M^-1 r``: diagonal everywhere, then the exact low-l block on ``low_idx``."""
    return (r * p.diag_inv).at[p.low_idx].set(p.low_inv @ r[p.low_idx])


def _pcg(a_op, b, minv, x0, tol: float, max_iter: int):
    """Preconditioned CG (``lax.while_loop``) -> ``(x, n_iter, ||r||/||b||)``.

    Same stopping rule as :func:`jax.scipy.sparse.linalg.cg`: stop once
    ``||r||^2 <= tol^2 ||b||^2`` or after ``max_iter`` iterations.
    """
    x = jnp.zeros_like(b) if x0 is None else x0
    r = b if x0 is None else b - a_op(x0)
    z = minv(r)
    bs = jnp.vdot(b, b)
    thresh = tol**2 * bs

    def cond(s):
        k, _, _, _, _, _, rs = s
        return (k < max_iter) & (rs > thresh)

    def body(s):
        k, x, r, z, d, rz, _ = s
        ad = a_op(d)
        alpha = rz / jnp.vdot(d, ad)
        x, r = x + alpha * d, r - alpha * ad
        z = minv(r)
        rz_new = jnp.vdot(r, z)
        return k + 1, x, r, z, z + (rz_new / rz) * d, rz_new, jnp.vdot(r, r)

    k, x, *_, rs = jax.lax.while_loop(cond, body, (0, x, r, z, z, jnp.vdot(r, z), jnp.vdot(r, r)))
    rel = jnp.where(bs > 0, jnp.sqrt(rs / jnp.where(bs > 0, bs, 1.0)), 0.0)
    return x, k, rel


def _solve(a_op, b, minv, x0, tol: float, max_iter: int):
    """:func:`_pcg` under ``lax.custom_linear_solve`` -> ``(x, SolveInfo)``.

    Gradients are implicit through ``a_op`` (symmetric), as with jax's ``cg``; the
    preconditioner is never differentiated.  ``x0`` warm-starts the primal and the
    transpose solve alike (as jax's ``_isolve`` does).
    """

    def solve(matvec, rhs):
        x, k, rel = _pcg(matvec, rhs, minv, x0, tol, max_iter)
        return x, SolveInfo(k, rel)

    return jax.lax.custom_linear_solve(a_op, b, solve, symmetric=True, has_aux=True)


def _precond_fn(precond: WienerPreconditioner | None, lmax: int, spin: int):
    """The ``M^-1`` callable for :func:`_solve` (identity when ``precond`` is None)."""
    if precond is None:
        return lambda r: r
    nx = n_dof(lmax, spin)
    if tuple(precond.diag_inv.shape) != (nx,):
        raise ValueError(
            f"precond has {precond.diag_inv.shape[0]} DOFs; lmax={lmax}, spin={spin} needs {nx} "
            "(build it with wiener_preconditioner at the same lmax/spin)"
        )
    return partial(_apply_precond, precond)


def wiener_preconditioner(
    signal_cl,
    nside: int,
    lmax: int,
    spin: int = 0,
    *,
    inv_noise=None,
    mask=None,
    use_weights: bool = True,
    ell_split: int | None = 32,
) -> WienerPreconditioner:
    """Preconditioner for :func:`wiener` / :func:`constrained_realization` at fixed ``N^-1``, ``Cl``.

    ``M = 1 + Cl * nbar`` per DOF, with ``nbar = mean(N^-1) * Npix / 4pi`` the isotropic
    level of ``S^T N^-1 S`` (the cut enters through the mean), except that the DOFs with
    ``l <= ell_split`` get the **exact** block of ``P A_x P + I``.  The diagonal alone
    mis-scales the large-scale modes under a large cut (they see only the prior, not
    ``nbar``) and converges no faster than plain CG there; the exact low-l block fixes
    that.  The solution does not depend on the preconditioner, only the iteration count.

    Build it eagerly, once, and reuse it for every solve with the same ``inv_noise`` /
    ``mask`` and ``signal_cl``.  The low block costs one application of the Wiener
    operator per DOF: ``(ell_split+1)^2`` for spin 0, ``2((ell_split+1)^2 - 4)`` for spin 2
    (2170 at the default 32), plus a dense inverse of that size.  ``ell_split=None`` gives
    the diagonal alone.  The default was tuned on Galactic masks at nside 128 / lmax 192
    (see ``docs/masked.md``).

    Parameters mirror :func:`wiener`.  Returns a :class:`WienerPreconditioner`.
    """
    w_pix = _weight_pix(nside, spin, inv_noise=inv_noise, mask=mask, use_weights=use_weights)
    pw = _prior_sqrt(signal_cl, lmax, spin)
    nbar = jnp.mean(w_pix) * (12 * nside * nside) / (4.0 * np.pi)
    diag_inv = 1.0 / (1.0 + pw**2 * nbar)

    ell = _prior_ell_index(lmax, spin)
    ell = ell if spin == 0 else np.concatenate([ell, ell])
    low = np.flatnonzero(ell <= ell_split) if ell_split is not None else np.zeros(0, dtype=int)
    if low.size == 0:
        return WienerPreconditioner(diag_inv, jnp.asarray(low, dtype=jnp.int32), jnp.zeros((0, 0), pw.dtype))

    nx = n_dof(lmax, spin)

    @jax.jit
    def column(j):
        e = jnp.zeros(nx, dtype=pw.dtype).at[j].set(1.0)
        return _whitened_op(e, w_pix, pw, nside, lmax, spin)[low]

    g = np.stack([np.asarray(column(int(j)), dtype=np.float64) for j in low], axis=1)
    low_inv = np.linalg.inv(0.5 * (g + g.T))
    return WienerPreconditioner(diag_inv, jnp.asarray(low, dtype=jnp.int32), jnp.asarray(low_inv, dtype=pw.dtype))


# --------------------------------------------------------------------------- #
# cut-sky deconvolution (CG on the masked normal equations)
# --------------------------------------------------------------------------- #
def deconvolve(
    maps,
    mask,
    nside: int,
    lmax: int,
    spin: int = 0,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
    reg: float = 0.0,
    use_weights: bool = True,
    x0=None,
    return_info: bool = False,
):
    """Recover the true a_lm from a cut sky by CG on ``S^T(MW)S a = S^T(MW)m``.

    Parameters
    ----------
    maps, mask : the observed map and the per-pixel sky mask (RING order). Pixels
        with ``mask=0`` are ignored; apodized ``mask in (0,1)`` are down-weighted.
    max_iter, tol : CG budget and relative residual tolerance.
    reg : optional Tikhonov ``A -> A + reg*I`` (in the real-DOF metric) for
        ill-conditioned masks; ``reg=0`` gives the minimum-norm CG solution.  (A
        Cl-informed prior instead of a scalar -- the Wiener filter -- is
        :func:`wiener`.)
    x0 : optional warm-start a_lm.
    return_info : also return a :class:`SolveInfo` (iterations, final relative
        residual) -- the only way to see that ``max_iter`` was hit unconverged.

    Returns the recovered a_lm (``(K,)`` spin-0, ``(2,K)`` spin-2), or ``(a_lm,
    SolveInfo)`` with ``return_info``.
    """
    w_pix = _weight_pix(nside, spin, mask=mask, use_weights=use_weights)
    b_x = _rhs_x(maps, w_pix, nside, lmax, spin)

    def a_op(x: jax.Array) -> jax.Array:
        return _normal_op(x, w_pix, reg, nside, lmax, spin)

    x_init = None if x0 is None else alm_to_real(x0, lmax, spin)
    if not return_info:
        x_sol, _ = cg(a_op, b_x, x0=x_init, tol=tol, maxiter=max_iter)
        return real_to_alm(x_sol, lmax, spin)
    x_sol, info = _solve(a_op, b_x, lambda r: r, x_init, tol, max_iter)
    return real_to_alm(x_sol, lmax, spin), info


# --------------------------------------------------------------------------- #
# Wiener filter (MUSE inner solve): Cl prior + per-pixel inverse-noise N^-1
# --------------------------------------------------------------------------- #
def wiener(
    maps,
    signal_cl,
    nside: int,
    lmax: int,
    spin: int = 0,
    *,
    inv_noise=None,
    mask=None,
    max_iter: int = 200,
    tol: float = 1e-8,
    use_weights: bool = True,
    x0=None,
    precond: WienerPreconditioner | None = None,
    return_info: bool = False,
):
    """Wiener-filter / MAP a_lm: solve ``(S^T N^-1 S + C^-1) a = S^T N^-1 m``.

    The noise-aware posterior mean for the data model ``m = S a + n`` with
    per-pixel inverse-noise ``N^-1`` and a diagonal signal prior ``a ~ N(0, C)``,
    ``C_{lm} = Cl``.  In the real-DOF ``x``-space (isometry ``T``) the prior is the
    *exact* diagonal ``D = diag(1/Cl)``; the system ``(A_x + D) x = b_x`` is solved
    by stock CG in **prior-whitened** coordinates ``x = P y``, ``P = diag(sqrt(Cl))``
    -- i.e. ``(P A_x P + I) y = P b_x`` (:func:`_whitened_op`), which is well
    conditioned and handles zero-power multipoles (``Cl = 0``) *exactly* (they are
    pinned to 0, never carried as a huge ``1/Cl``).  Same operator family as
    :func:`deconvolve` (``reg`` is the special case ``Cl = 1/reg`` constant).  It
    is the MUSE inner solve; for a posterior *draw* (constrained realization) see
    :func:`constrained_realization`.

    Parameters
    ----------
    maps : the observed map (RING order; ``(Npix,)`` spin 0, ``(2, Npix)`` spin 2).
    signal_cl : the signal power spectrum -- ``(lmax+1,)`` for spin 0, or
        ``(C_EE, C_BB)`` (two ``(lmax+1,)`` arrays) for spin 2; same orthonormal
        convention as :func:`jht.diff.bandpower` / ``healpy.alm2cl``.
    inv_noise : the per-pixel inverse-noise ``N^-1`` (``(Npix,)``, mask folded in).
        If omitted, falls back to ``mask * W`` (the :func:`deconvolve` weighting) --
        then ``mask`` is required.
    max_iter, tol : CG budget and relative residual tolerance.
    x0 : optional warm-start a_lm.
    precond : a :class:`WienerPreconditioner` from :func:`wiener_preconditioner` built
        for the same ``inv_noise``/``mask``, ``signal_cl``, ``lmax`` and ``spin``.  It
        changes the iteration count, not the solution.
    return_info : also return a :class:`SolveInfo` (iterations, final relative
        residual) -- the only way to see that ``max_iter`` was hit unconverged.

    With neither ``precond`` nor ``return_info`` the solve is stock
    :func:`jax.scipy.sparse.linalg.cg`; otherwise it is jht's preconditioned CG under the
    same implicit-gradient rule.  Returns the Wiener-mean a_lm (``(K,)`` spin-0,
    ``(2,K)`` spin-2), or ``(a_lm, SolveInfo)`` with ``return_info``.
    """
    w_pix = _weight_pix(nside, spin, inv_noise=inv_noise, mask=mask, use_weights=use_weights)
    pw = _prior_sqrt(signal_cl, lmax, spin)
    b_y = pw * _rhs_x(maps, w_pix, nside, lmax, spin)

    def a_op(y: jax.Array) -> jax.Array:
        return _whitened_op(y, w_pix, pw, nside, lmax, spin)

    y_init = None if x0 is None else _whiten_x0(x0, pw, lmax, spin)
    if precond is None and not return_info:
        y_sol, _ = cg(a_op, b_y, x0=y_init, tol=tol, maxiter=max_iter)
        return real_to_alm(pw * y_sol, lmax, spin)
    y_sol, info = _solve(a_op, b_y, _precond_fn(precond, lmax, spin), y_init, tol, max_iter)
    alm = real_to_alm(pw * y_sol, lmax, spin)
    return (alm, info) if return_info else alm


def constrained_realization(
    maps,
    signal_cl,
    nside: int,
    lmax: int,
    key,
    spin: int = 0,
    *,
    inv_noise=None,
    mask=None,
    max_iter: int = 200,
    tol: float = 1e-8,
    use_weights: bool = True,
    x0=None,
    precond: WienerPreconditioner | None = None,
    return_info: bool = False,
):
    """One posterior draw ``a ~ N(a_wiener, (S^T N^-1 S + C^-1)^-1)`` (a constrained realization).

    Same system as :func:`wiener` -- solved in the same prior-whitened coordinates
    ``x = P y``, ``P = diag(sqrt(Cl))`` -- with a stochastic source added to the
    whitened RHS::

        y_sample = (P A_x P + I)^-1 (P (b_x + s1) + xi_phys),
        s1 = T S^T(sqrt(N^-1) w1),

    with ``w1 ~ N(0, I)`` in pixel space and ``xi ~ N(0, I)`` in x-space (masked to
    the physical ``Cl > 0`` modes -- on pinned modes RHS and source are 0, so the
    draw is exactly 0 there, the infinite-prior limit).  ``Cov(P s1 + xi_phys) =
    P A_x P + I_phys``, so ``x = P y`` has mean ``a_wiener`` and covariance
    ``(A_x + D)^-1`` on the physical modes -- the posterior.  (The ``s1``
    factorization is exact because ``T S^T`` is the Euclidean transpose of
    ``S T^-1`` -- the gated operator symmetry.  In unwhitened coordinates the source
    would carry ``sqrt(1/Cl)``-scaled noise, which for ``Cl = 0`` multipoles inflates
    ``||b||`` enough to break the relative-residual stopping rule; whitening avoids
    that.)  ``key`` is a ``jax.random`` PRNG key; one call returns one draw.

    Parameters (including ``precond`` / ``return_info``) mirror :func:`wiener`.  Returns
    a posterior a_lm sample, or ``(sample, SolveInfo)`` with ``return_info``.

    .. note::
       This is a genuine posterior draw only when ``inv_noise`` (a real noise
       model) is given: the source ``s1 = T S^T(sqrt(w_pix) * omega)`` has
       covariance ``A_x`` only if ``w_pix`` is the inverse-noise weight.  The
       ``mask``-only fallback (inherited from :func:`wiener`'s shared signature)
       makes ``sqrt(mask * W)`` stand in for a noise model, so the draw's
       covariance is **not** the posterior -- pass ``inv_noise`` for a
       constrained realization.
    """
    w_pix = _weight_pix(nside, spin, inv_noise=inv_noise, mask=mask, use_weights=use_weights)
    pw = _prior_sqrt(signal_cl, lmax, spin)
    b_x = _rhs_x(maps, w_pix, nside, lmax, spin)

    npix = 12 * nside * nside
    omega_shape = (npix,) if spin == 0 else (2, npix)
    k1, k2 = jax.random.split(key)
    omega1 = jax.random.normal(k1, omega_shape, dtype=b_x.dtype)
    s1 = alm_to_real(adjoint_synthesis(jnp.sqrt(w_pix) * omega1, nside, lmax, spin), lmax, spin)
    xi = jax.random.normal(k2, (n_dof(lmax, spin),), dtype=b_x.dtype)
    rhs = pw * (b_x + s1) + jnp.where(pw > 0, xi, 0.0)

    def a_op(y: jax.Array) -> jax.Array:
        return _whitened_op(y, w_pix, pw, nside, lmax, spin)

    y_init = None if x0 is None else _whiten_x0(x0, pw, lmax, spin)
    if precond is None and not return_info:
        y_sol, _ = cg(a_op, rhs, x0=y_init, tol=tol, maxiter=max_iter)
        return real_to_alm(pw * y_sol, lmax, spin)
    y_sol, info = _solve(a_op, rhs, _precond_fn(precond, lmax, spin), y_init, tol, max_iter)
    sample = real_to_alm(pw * y_sol, lmax, spin)
    return (sample, info) if return_info else sample
