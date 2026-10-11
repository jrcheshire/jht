"""Gates for the Wiener filter / MUSE inner solve (:func:`jht.masked.wiener`,
:func:`jht.masked.constrained_realization`).

The Wiener solve is ``(S^T N^-1 S + C^-1) a = S^T N^-1 m`` with per-pixel
inverse-noise ``N^-1`` and a diagonal signal prior ``C_{lm} = Cl``.  In the
real-DOF x-space (isometry ``T``) the prior is the *exact* diagonal
``D = diag(1/Cl)``; the solve runs in prior-whitened coordinates ``x = P y``,
``P = diag(sqrt(Cl))`` (operator ``P A_x P + I``), so zero-power multipoles are
pinned to 0 exactly -- gated here including the zero-Cl regression (the old
``1/Cl = 1e30`` path broke the CG stopping rule through the sampler's RHS noise
and returned garbage for the physical modes).  The gate philosophy mirrors
``tests/test_masked.py``: **tight, oracle-backed deterministic gates** for the
operator, the mean, and the limits; the constrained-realization *covariance* is
checked by Monte Carlo at an **a-priori budgeted** tolerance (k-sigma on the known
sampling distribution of a sample covariance), never hand-tuned.

The constrained-realization source ``s1 = T S^T(sqrt(N^-1) w1)`` has covariance
exactly ``T S^T diag(N^-1) S T^-1`` *because* ``alm_to_real . adjoint_synthesis``
is the Euclidean transpose of ``synthesis . real_to_alm`` -- gated deterministically
by ``test_realdof_adjoint_identity`` (no Monte Carlo needed for the factorization).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jht.healpix import adjoint_synthesis, alm_size, synthesis  # noqa: E402
from jht.masked import (  # noqa: E402
    _prior_sqrt,  # private: gated against an independent numpy reference
    _whitened_op,  # private: gated against the dense P A P + I
    alm_to_real,
    constrained_realization,
    deconvolve,
    n_dof,
    real_to_alm,
    wiener,
    wiener_preconditioner,
)

PRIOR_TOL = 1e-12  # _prior_sqrt == independent ref (layout)
OP_TOL = 1e-10  # operator A_x+D == dense; real-DOF adjoint identity; symmetry
SOLVE_TOL = 1e-8  # wiener mean == dense solve (CG tol-limited)
LIMIT_TOL = 1e-7  # wiener(Cl->inf)==deconvolve(reg=0); (Cl=1/reg)==deconvolve(reg)
FD_TOL = 1e-2  # finite-difference grad agreement (FD + CG-tol limited)


# --------------------------------------------------------------------------- #
# helpers (mirroring tests/test_masked.py)
# --------------------------------------------------------------------------- #
def _lvals(lmax: int) -> np.ndarray:
    return np.concatenate([np.arange(m, lmax + 1) for m in range(lmax + 1)])


def _mvals(lmax: int) -> np.ndarray:
    return np.concatenate([np.full(lmax + 1 - m, m) for m in range(lmax + 1)])


def _rand_alm(lmax: int, spin: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lv = _lvals(lmax)

    def one() -> np.ndarray:
        a = (rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))).astype(complex)
        a[: lmax + 1] = a[: lmax + 1].real
        a[lv < abs(spin)] = 0.0
        return a

    return one() if spin == 0 else np.stack([one(), one()])


def _theta(nside: int) -> np.ndarray:
    import healpy as hp

    return hp.pix2ang(nside, np.arange(12 * nside**2))[0]


def _binary_cap(nside: int, th_cut: float) -> np.ndarray:
    m = np.ones(12 * nside**2)
    m[_theta(nside) < th_cut] = 0.0
    return m


def _apod_cap(nside: int, th_cut: float, width: float) -> np.ndarray:
    th = _theta(nside)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.clip((th - th_cut) / width, 0.0, 1.0)))
    return np.where(th < th_cut, 0.0, ramp)


def _signal_cl(lmax: int, spin: int):
    """A positive, decaying signal spectrum (tuple ``(C_EE, C_BB)`` for spin 2)."""
    base = 1.0 / (np.arange(lmax + 1) + 1.0) ** 2 + 1e-3
    return base if spin == 0 else (base, 0.5 * base)


def _signal_cl_zero(lmax: int, spin: int):
    """A spectrum with zero-power modes -- the zero-Cl regression case.

    spin 0: zeroed monopole + dipole (the standard CMB convention).  spin 2: a
    B-free prior ``C_BB = 0`` (every B DOF pinned; zeroing l<2 alone would touch
    no DOFs -- those modes are already structurally absent for spin 2).
    """
    base = 1.0 / (np.arange(lmax + 1) + 1.0) ** 2 + 1e-3
    if spin == 0:
        c = base.copy()
        c[:2] = 0.0
        return c
    return (base, np.zeros(lmax + 1))


def _prior_diag_ref(cl, lmax: int, spin: int) -> np.ndarray:
    """Independent numpy build of the x-space prior diagonal ``diag(1/Cl)`` (positive Cl)."""
    ms, ls = _mvals(lmax), _lvals(lmax)
    active = ls >= abs(spin)
    ell = np.concatenate([ls[active & (ms == 0)], ls[active & (ms > 0)], ls[active & (ms > 0)]])

    def one(c) -> np.ndarray:
        clv = np.asarray(c)[ell]
        return np.where(clv > 0, 1.0 / np.where(clv > 0, clv, 1.0), 1.0e30)

    return one(cl) if spin == 0 else np.concatenate([one(cl[0]), one(cl[1])])


def _prior_sqrt_ref(cl, lmax: int, spin: int) -> np.ndarray:
    """Independent numpy build of the whitening ``pw = sqrt(Cl)`` (0 where Cl <= 0)."""
    ms, ls = _mvals(lmax), _lvals(lmax)
    active = ls >= abs(spin)
    ell = np.concatenate([ls[active & (ms == 0)], ls[active & (ms > 0)], ls[active & (ms > 0)]])

    def one(c) -> np.ndarray:
        clv = np.asarray(c)[ell]
        return np.where(clv > 0, np.sqrt(np.maximum(clv, 0.0)), 0.0)

    return one(cl) if spin == 0 else np.concatenate([one(cl[0]), one(cl[1])])


def _ninv(nside: int, mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """A positive per-pixel inverse-noise map with the mask folded in (0 under the cut)."""
    rng = np.random.default_rng(seed)
    return mask * (0.5 + rng.random(12 * nside**2))


def _dense_S(nside: int, lmax: int, spin: int) -> np.ndarray:
    """Explicit synthesis matrix in the real-DOF basis: columns ``S(T^-1 e_k)``."""
    nx = n_dof(lmax, spin)
    cols = []
    for k in range(nx):
        e = np.zeros(nx)
        e[k] = 1.0
        cols.append(np.asarray(synthesis(real_to_alm(e, lmax, spin), nside, lmax, spin)).ravel())
    return np.stack(cols, axis=1)


def _ninv_bcast(ninv: np.ndarray, spin: int):
    return jnp.asarray(ninv) if spin == 0 else jnp.asarray(ninv)[None, :]


def _ninv_flat(ninv: np.ndarray, spin: int) -> np.ndarray:
    return ninv if spin == 0 else np.tile(ninv, 2)


# --------------------------------------------------------------------------- #
# W-prior: the Cl prior maps to the right x-space diagonal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
def test_prior_sqrt_matches_ref(spin):
    """``_prior_sqrt(Cl)`` == the independent ``sqrt(Cl)`` build; zero-Cl -> exactly 0."""
    lmax = 10
    for cl in (_signal_cl(lmax, spin), _signal_cl_zero(lmax, spin)):
        cl_j = jnp.asarray(cl) if spin == 0 else (jnp.asarray(cl[0]), jnp.asarray(cl[1]))
        pw = np.asarray(_prior_sqrt(cl_j, lmax, spin))
        ref = _prior_sqrt_ref(cl, lmax, spin)
        assert pw.shape == (n_dof(lmax, spin),)
        assert np.max(np.abs(pw - ref)) <= PRIOR_TOL * (1.0 + np.max(np.abs(ref)))
        assert np.all(pw[ref == 0.0] == 0.0)  # pinned modes are exact zeros, not tiny


# --------------------------------------------------------------------------- #
# W-operator: the whitened  P (S^T diag(N^-1) S) P + I == dense;  symmetric; >= I
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("which", ["binary", "apod"])
def test_whitened_operator_matches_dense(spin, which):
    """``_whitened_op`` == dense ``P A_x P + I``, for positive AND zero-power Cl."""
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4) if which == "binary" else _apod_cap(nside, 0.4, 0.4)
    ninv = _ninv(nside, mask, seed=1)
    S = _dense_S(nside, lmax, spin)
    A0 = S.T @ (_ninv_flat(ninv, spin)[:, None] * S)
    nb = _ninv_bcast(ninv, spin)
    rng = np.random.default_rng(0)
    for cl in (_signal_cl(lmax, spin), _signal_cl_zero(lmax, spin)):
        cl_j = jnp.asarray(cl) if spin == 0 else (jnp.asarray(cl[0]), jnp.asarray(cl[1]))
        pw_ref = _prior_sqrt_ref(cl, lmax, spin)
        A_dense = pw_ref[:, None] * A0 * pw_ref[None, :] + np.eye(pw_ref.size)
        pw = _prior_sqrt(cl_j, lmax, spin)
        for _ in range(3):
            x = rng.standard_normal(n_dof(lmax, spin))
            got = np.asarray(_whitened_op(jnp.asarray(x), nb, pw, nside, lmax, spin))
            ref = A_dense @ x
            assert np.max(np.abs(got - ref)) <= OP_TOL * (1.0 + np.max(np.abs(ref)))


@pytest.mark.parametrize("spin", [0, 2])
def test_whitened_operator_symmetric_and_pd(spin):
    """``P A_x P + I`` is symmetric and ``>= I`` (eigenvalues >= 1 -- the whitening point)."""
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4)
    ninv = _ninv(nside, mask, seed=2)
    cl = _signal_cl_zero(lmax, spin)  # the harder case: pinned modes present
    cl_j = jnp.asarray(cl) if spin == 0 else (jnp.asarray(cl[0]), jnp.asarray(cl[1]))
    nb = _ninv_bcast(ninv, spin)
    pw = _prior_sqrt(cl_j, lmax, spin)

    def aop(x):
        return np.asarray(_whitened_op(jnp.asarray(x), nb, pw, nside, lmax, spin))

    rng = np.random.default_rng(3)
    u, v = rng.standard_normal(n_dof(lmax, spin)), rng.standard_normal(n_dof(lmax, spin))
    assert abs(float(u @ aop(v)) - float(v @ aop(u))) <= OP_TOL * (1.0 + abs(float(u @ aop(v))))
    # >= I: x^T (P A P + I) x >= ||x||^2 (A PSD), so CG sees a well-conditioned system
    assert float(u @ aop(u)) >= (1.0 - 1e-10) * float(u @ u)


# --------------------------------------------------------------------------- #
# W-realdof-adjoint: alm_to_real . S^T is the Euclidean transpose of S . T^-1
# (this is what makes the constrained-realization source covariance exact)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
def test_realdof_adjoint_identity(spin):
    nside, lmax = 8, 8
    npix = 12 * nside**2
    rng = np.random.default_rng(4)
    x = rng.standard_normal(n_dof(lmax, spin))
    p = rng.standard_normal(npix if spin == 0 else (2, npix))
    sx = np.asarray(synthesis(real_to_alm(jnp.asarray(x), lmax, spin), nside, lmax, spin)).ravel()
    stp = np.asarray(alm_to_real(adjoint_synthesis(jnp.asarray(p), nside, lmax, spin), lmax, spin))
    lhs, rhs = float(sx @ p.ravel()), float(x @ stp)
    assert abs(lhs - rhs) <= OP_TOL * (1.0 + abs(lhs))


# --------------------------------------------------------------------------- #
# W-mean: wiener == explicit dense (A_x + D)^-1 b_x
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
def test_wiener_mean_matches_dense_solve(spin):
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4)
    ninv = _ninv(nside, mask, seed=5)
    cl = _signal_cl(lmax, spin)
    cl_j = jnp.asarray(cl) if spin == 0 else (jnp.asarray(cl[0]), jnp.asarray(cl[1]))

    a_true = _rand_alm(lmax, spin, 6)
    m = np.asarray(synthesis(jnp.asarray(a_true), nside, lmax, spin))

    S = _dense_S(nside, lmax, spin)
    A_dense = S.T @ (_ninv_flat(ninv, spin)[:, None] * S) + np.diag(_prior_diag_ref(cl, lmax, spin))
    nb = _ninv_bcast(ninv, spin)
    b = np.asarray(alm_to_real(adjoint_synthesis(nb * jnp.asarray(m), nside, lmax, spin), lmax, spin))
    a_dense = np.asarray(real_to_alm(jnp.asarray(np.linalg.solve(A_dense, b)), lmax, spin))

    a_w = np.asarray(wiener(m, cl_j, nside, lmax, spin=spin, inv_noise=ninv, max_iter=800, tol=1e-12))
    assert np.max(np.abs(a_w - a_dense)) <= SOLVE_TOL * (1.0 + np.max(np.abs(a_dense)))


# --------------------------------------------------------------------------- #
# W-zero-Cl: zero-power multipoles -- the regression the 1/Cl=1e30 path failed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
def test_wiener_zero_cl_matches_dense(spin):
    """Zero-power Cl: pinned modes come back exactly 0, physical modes == dense solve."""
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4)
    ninv = _ninv(nside, mask, seed=11)
    cl = _signal_cl_zero(lmax, spin)
    cl_j = jnp.asarray(cl) if spin == 0 else (jnp.asarray(cl[0]), jnp.asarray(cl[1]))

    a_true = _rand_alm(lmax, spin, 12)
    m = np.asarray(synthesis(jnp.asarray(a_true), nside, lmax, spin))

    pw = _prior_sqrt_ref(cl, lmax, spin)
    phys = pw > 0
    S = _dense_S(nside, lmax, spin)
    A0 = S.T @ (_ninv_flat(ninv, spin)[:, None] * S)
    nb = _ninv_bcast(ninv, spin)
    b = np.asarray(alm_to_real(adjoint_synthesis(nb * jnp.asarray(m), nside, lmax, spin), lmax, spin))
    # dense reference: the infinite-prior limit = solve the physical block, pin the rest
    A_pp = A0[np.ix_(phys, phys)] + np.diag(1.0 / pw[phys] ** 2)
    x_ref = np.linalg.solve(A_pp, b[phys])

    a_w = wiener(m, cl_j, nside, lmax, spin=spin, inv_noise=ninv, max_iter=800, tol=1e-12)
    x_w = np.asarray(alm_to_real(a_w, lmax, spin))
    assert np.max(np.abs(x_w[~phys])) == 0.0  # pinned exactly, not ~b/1e30
    assert np.max(np.abs(x_w[phys] - x_ref)) <= SOLVE_TOL * (1.0 + np.max(np.abs(x_ref)))


@pytest.mark.parametrize("spin", [0, 2])
def test_constrained_realization_zero_cl_regression(spin):
    """The zero-Cl posterior draw solves the physical block correctly.

    Regression: the old sampler added ``sqrt(1/Cl) xi ~ 1e15`` to the CG RHS for
    zero-power multipoles, inflating ``||b||`` so the relative-residual stop fired
    with the physical modes unsolved (measured rel. err ~1 vs the dense solve).
    The RHS is replicated here (deterministic given the key) and dense-solved.
    """
    nside, lmax = 8, 8
    npix = 12 * nside**2
    ninv = np.ones(npix)
    cl = _signal_cl_zero(lmax, spin)
    cl_j = jnp.asarray(cl) if spin == 0 else (jnp.asarray(cl[0]), jnp.asarray(cl[1]))
    a_true = _rand_alm(lmax, spin, 13)
    m = jnp.asarray(np.asarray(synthesis(jnp.asarray(a_true), nside, lmax, spin)))
    key = jax.random.PRNGKey(7)

    # replicate the sampler's stochastic RHS (same key -> same split -> same draws)
    nb = _ninv_bcast(ninv, spin)
    b = np.asarray(alm_to_real(adjoint_synthesis(nb * m, nside, lmax, spin), lmax, spin))
    k1, k2 = jax.random.split(key)
    omega_shape = (npix,) if spin == 0 else (2, npix)
    omega1 = jax.random.normal(k1, omega_shape, dtype=jnp.float64)
    s1 = np.asarray(alm_to_real(adjoint_synthesis(jnp.sqrt(nb) * omega1, nside, lmax, spin), lmax, spin))
    xi = np.asarray(jax.random.normal(k2, (n_dof(lmax, spin),), dtype=jnp.float64))

    pw = _prior_sqrt_ref(cl, lmax, spin)
    phys = pw > 0
    S = _dense_S(nside, lmax, spin)
    A0 = S.T @ (_ninv_flat(ninv, spin)[:, None] * S)
    A_pp = A0[np.ix_(phys, phys)] + np.diag(1.0 / pw[phys] ** 2)
    x_ref = np.linalg.solve(A_pp, (b + s1)[phys] + xi[phys] / pw[phys])

    a_cr = constrained_realization(
        m, cl_j, nside, lmax, key, spin=spin, inv_noise=ninv, max_iter=800, tol=1e-12
    )
    x_cr = np.asarray(alm_to_real(a_cr, lmax, spin))
    assert np.max(np.abs(x_cr[~phys])) == 0.0  # pinned modes: deterministically 0
    assert np.max(np.abs(x_cr[phys] - x_ref)) <= SOLVE_TOL * (1.0 + np.max(np.abs(x_ref)))


# --------------------------------------------------------------------------- #
# W-limits: wiener reduces to deconvolve (the prior is the generalized reg)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
def test_wiener_limits_to_deconvolve(spin):
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4)
    a_true = _rand_alm(lmax, spin, 7)
    m = np.asarray(synthesis(jnp.asarray(a_true), nside, lmax, spin))
    mm = m * (mask if spin == 0 else mask[None, :])

    def const_cl(val):
        c = np.full(lmax + 1, val)
        return jnp.asarray(c) if spin == 0 else (jnp.asarray(c), jnp.asarray(c))

    # Cl -> inf  (D -> 0)  ==  deconvolve(reg=0) (min-norm), both with the MW weighting
    a_w = np.asarray(wiener(mm, const_cl(1e14), nside, lmax, spin=spin, mask=mask, max_iter=800, tol=1e-12))
    a_d = np.asarray(deconvolve(mm, mask, nside, lmax, spin=spin, reg=0.0, max_iter=800, tol=1e-12))
    assert np.max(np.abs(a_w - a_d)) <= LIMIT_TOL * (1.0 + np.max(np.abs(a_d)))

    # Cl = 1/reg const  ==  deconvolve(reg)   (the scalar-Tikhonov special case)
    reg = 0.5
    a_w2 = np.asarray(wiener(mm, const_cl(1.0 / reg), nside, lmax, spin=spin, mask=mask, max_iter=800, tol=1e-12))
    a_d2 = np.asarray(deconvolve(mm, mask, nside, lmax, spin=spin, reg=reg, max_iter=800, tol=1e-12))
    assert np.max(np.abs(a_w2 - a_d2)) <= SOLVE_TOL * (1.0 + np.max(np.abs(a_d2)))


# --------------------------------------------------------------------------- #
# W-CR: constrained realizations have the posterior mean and covariance
# (Monte Carlo, a-priori k-sigma budget on the sample-covariance distribution)
# --------------------------------------------------------------------------- #
def test_constrained_realization_posterior_moments():
    """Sample mean -> a_wiener and sample cov -> (A_x + D)^-1, within a k-sigma MC budget.

    For N Gaussian draws in d dims: E||mean - mu||^2 = tr(Sigma)/N and
    E||Cov_hat - Sigma||_F^2 = ((tr Sigma)^2 + ||Sigma||_F^2)/N.  We gate the
    measured deviations at ~4 sigma of those (a-priori) distributions -- the
    *correctness* of the sampler is established deterministically by the operator,
    adjoint-identity and mean gates above; this is the end-to-end wiring check.
    """
    nside, lmax, spin = 8, 4, 0
    n_draws = 600
    mask = _binary_cap(nside, 0.4)
    ninv = mask * 1.0  # explicit inverse-noise (unit variance under the observed sky)
    cl = _signal_cl(lmax, spin)
    cl_j = jnp.asarray(cl)
    a_true = _rand_alm(lmax, spin, 8)
    m = jnp.asarray(np.asarray(synthesis(jnp.asarray(a_true), nside, lmax, spin)))

    # dense posterior covariance + mean (x-space)
    S = _dense_S(nside, lmax, spin)
    A = S.T @ (ninv[:, None] * S) + np.diag(_prior_diag_ref(cl, lmax, spin))
    Sigma = np.linalg.inv(A)
    nb = jnp.asarray(ninv)
    b = np.asarray(alm_to_real(adjoint_synthesis(nb * m, nside, lmax, spin), lmax, spin))
    x_hat = np.linalg.solve(A, b)

    def draw(key):
        a = constrained_realization(m, cl_j, nside, lmax, key, spin=spin, inv_noise=ninv, max_iter=200, tol=1e-10)
        return alm_to_real(a, lmax, spin)

    draw_j = jax.jit(draw)
    draw_j(jax.random.PRNGKey(0))  # pre-warm the static-layout caches before the loop
    keys = jax.random.split(jax.random.PRNGKey(1234), n_draws)
    X = np.stack([np.asarray(draw_j(k)) for k in keys])  # (n_draws, n_dof)

    mean_err = float(np.linalg.norm(X.mean(0) - x_hat))
    mean_budget = 4.0 * np.sqrt(np.trace(Sigma) / n_draws)
    assert mean_err <= mean_budget, f"CR mean {mean_err:.3e} > 4-sigma budget {mean_budget:.3e}"

    cov = np.cov(X.T)
    cov_err = float(np.linalg.norm(cov - Sigma, "fro"))
    cov_budget = 4.0 * np.sqrt((np.trace(Sigma) ** 2 + np.linalg.norm(Sigma, "fro") ** 2) / n_draws)
    assert cov_err <= cov_budget, f"CR cov {cov_err:.3e} > 4-sigma budget {cov_budget:.3e}"


# --------------------------------------------------------------------------- #
# W-grad: the Wiener solve is differentiable (grad through CG) wrt Cl and the data
# --------------------------------------------------------------------------- #
def test_wiener_differentiable():
    nside, lmax, spin = 8, 6, 0
    mask = _binary_cap(nside, 0.4)
    ninv = jnp.asarray(_ninv(nside, mask, seed=9))
    a_true = _rand_alm(lmax, spin, 10)
    m = jnp.asarray(np.asarray(synthesis(jnp.asarray(a_true), nside, lmax, spin)))
    cl0 = jnp.asarray(_signal_cl(lmax, spin))

    def loss(cl):
        a = wiener(m, cl, nside, lmax, spin=spin, inv_noise=ninv, max_iter=200, tol=1e-11)
        return jnp.real(jnp.sum(jnp.abs(a) ** 2))

    loss(cl0)  # pre-warm caches before grad
    g = np.asarray(jax.grad(loss)(cl0))
    assert np.all(np.isfinite(g))

    # finite-difference check on a couple of multipoles
    for i in (1, 3):
        h = 1e-4 * float(cl0[i])
        fd = (float(loss(cl0.at[i].add(h))) - float(loss(cl0.at[i].add(-h)))) / (2.0 * h)
        assert abs(g[i] - fd) <= FD_TOL * abs(fd) + 1e-6

    # jit(grad) agrees; grad wrt the data map is finite
    gj = np.asarray(jax.jit(jax.grad(loss))(cl0))
    assert np.max(np.abs(g - gj)) <= 1e-9 * (1.0 + np.max(np.abs(g)))
    gm = jax.grad(lambda mm: jnp.real(jnp.sum(jnp.abs(wiener(mm, cl0, nside, lmax, spin=spin, inv_noise=ninv, max_iter=200, tol=1e-10)) ** 2)))(m)
    assert np.all(np.isfinite(np.asarray(gm)))


# --------------------------------------------------------------------------- #
# W-precond: wiener_preconditioner + jht's preconditioned CG (SolveInfo diagnostics).
# The preconditioner changes the iteration count, never the solution, so every gate
# here is against the dense solve or the unpreconditioned path.
# --------------------------------------------------------------------------- #
def _cl_jax(cl, spin: int):
    return jnp.asarray(cl) if spin == 0 else (jnp.asarray(cl[0]), jnp.asarray(cl[1]))


def _ell_ref(lmax: int, spin: int) -> np.ndarray:
    """Independent per-DOF multipole (both channels for spin 2)."""
    ms, ls = _mvals(lmax), _lvals(lmax)
    active = ls >= abs(spin)
    ell = np.concatenate([ls[active & (ms == 0)], ls[active & (ms > 0)], ls[active & (ms > 0)]])
    return ell if spin == 0 else np.concatenate([ell, ell])


def _dense_whitened(nside: int, lmax: int, spin: int, ninv: np.ndarray, cl):
    """Dense ``P A_x P + I`` and ``pw`` from the explicit synthesis matrix."""
    S = _dense_S(nside, lmax, spin)
    A0 = S.T @ (_ninv_flat(ninv, spin)[:, None] * S)
    pw = _prior_sqrt_ref(cl, lmax, spin)
    return pw[:, None] * A0 * pw[None, :] + np.eye(pw.size), pw


def _dense_wiener_x(nside: int, lmax: int, spin: int, ninv: np.ndarray, cl, m) -> np.ndarray:
    """``x = P (P A_x P + I)^-1 P b_x`` (pinned modes come out 0)."""
    M, pw = _dense_whitened(nside, lmax, spin, ninv, cl)
    nb = _ninv_bcast(ninv, spin)
    b = np.asarray(alm_to_real(adjoint_synthesis(nb * jnp.asarray(m), nside, lmax, spin), lmax, spin))
    return pw * np.linalg.solve(M, pw * b)


@pytest.mark.parametrize("spin", [0, 2])
def test_precond_diagonal_matches_ref(spin):
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4)
    ninv = _ninv(nside, mask, seed=21)
    cl = _signal_cl(lmax, spin)
    p = wiener_preconditioner(_cl_jax(cl, spin), nside, lmax, spin, inv_noise=ninv, ell_split=None)
    nbar = ninv.mean() * 12 * nside**2 / (4 * np.pi)
    ref = 1.0 / (1.0 + _prior_sqrt_ref(cl, lmax, spin) ** 2 * nbar)
    assert np.max(np.abs(np.asarray(p.diag_inv) - ref)) <= PRIOR_TOL
    assert p.low_idx.shape == (0,) and p.low_inv.shape == (0, 0)


@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("which", ["binary", "apod"])
def test_precond_low_block_matches_dense(spin, which):
    """The low-l block is the exact inverse of the dense ``(P A P + I)[low, low]``."""
    nside, lmax, ell_split = 8, 8, 4
    mask = _binary_cap(nside, 0.4) if which == "binary" else _apod_cap(nside, 0.4, 0.4)
    ninv = _ninv(nside, mask, seed=22)
    cl = _signal_cl(lmax, spin)
    p = wiener_preconditioner(_cl_jax(cl, spin), nside, lmax, spin, inv_noise=ninv, ell_split=ell_split)
    low = np.flatnonzero(_ell_ref(lmax, spin) <= ell_split)
    assert np.array_equal(np.asarray(p.low_idx), low)
    M, _ = _dense_whitened(nside, lmax, spin, ninv, cl)
    ref = np.linalg.inv(M[np.ix_(low, low)])
    assert np.max(np.abs(np.asarray(p.low_inv) - ref)) <= OP_TOL * (1.0 + np.max(np.abs(ref)))


@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("ell_split", [None, 4, 8])
@pytest.mark.parametrize("zero_cl", [False, True])
def test_precond_wiener_matches_dense(spin, ell_split, zero_cl):
    """Preconditioned wiener == dense solve, incl. pinned zero-Cl modes (returned exactly 0)."""
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4)
    ninv = _ninv(nside, mask, seed=23)
    cl = _signal_cl_zero(lmax, spin) if zero_cl else _signal_cl(lmax, spin)
    cl_j = _cl_jax(cl, spin)
    m = np.asarray(synthesis(jnp.asarray(_rand_alm(lmax, spin, 24)), nside, lmax, spin))
    p = wiener_preconditioner(cl_j, nside, lmax, spin, inv_noise=ninv, ell_split=ell_split)
    a = wiener(m, cl_j, nside, lmax, spin=spin, inv_noise=ninv, precond=p, max_iter=800, tol=1e-12)
    x = np.asarray(alm_to_real(a, lmax, spin))
    x_ref = _dense_wiener_x(nside, lmax, spin, ninv, cl, m)
    assert np.max(np.abs(x - x_ref)) <= SOLVE_TOL * (1.0 + np.max(np.abs(x_ref)))
    pinned = _prior_sqrt_ref(cl, lmax, spin) == 0
    assert np.all(x[pinned] == 0.0)


@pytest.mark.parametrize("spin", [0, 2])
def test_pcg_unpreconditioned_matches_dense_and_default(spin):
    """jht's CG (``return_info`` without a preconditioner) == dense == the stock jax-cg path."""
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4)
    ninv = _ninv(nside, mask, seed=25)
    cl = _signal_cl(lmax, spin)
    cl_j = _cl_jax(cl, spin)
    m = np.asarray(synthesis(jnp.asarray(_rand_alm(lmax, spin, 26)), nside, lmax, spin))
    a, info = wiener(m, cl_j, nside, lmax, spin=spin, inv_noise=ninv, return_info=True, max_iter=800, tol=1e-12)
    a_def = wiener(m, cl_j, nside, lmax, spin=spin, inv_noise=ninv, max_iter=800, tol=1e-12)
    x = np.asarray(alm_to_real(a, lmax, spin))
    x_ref = _dense_wiener_x(nside, lmax, spin, ninv, cl, m)
    scale = 1.0 + np.max(np.abs(x_ref))
    assert np.max(np.abs(x - x_ref)) <= SOLVE_TOL * scale
    assert np.max(np.abs(np.asarray(a) - np.asarray(a_def))) <= SOLVE_TOL * scale
    assert float(info.rel_residual) <= 1e-12 and 0 < int(info.n_iter) < 800


@pytest.mark.parametrize("spin", [0, 2])
def test_precond_exact_block_converges_immediately(spin):
    """With the block covering every DOF, M is the operator itself: one CG step (two with roundoff)."""
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4)
    ninv = _ninv(nside, mask, seed=27)
    cl_j = _cl_jax(_signal_cl(lmax, spin), spin)
    m = np.asarray(synthesis(jnp.asarray(_rand_alm(lmax, spin, 28)), nside, lmax, spin))
    p = wiener_preconditioner(cl_j, nside, lmax, spin, inv_noise=ninv, ell_split=lmax)
    assert p.low_idx.shape[0] == n_dof(lmax, spin)
    _, info = wiener(m, cl_j, nside, lmax, spin=spin, inv_noise=ninv, precond=p, return_info=True, max_iter=800, tol=1e-12)
    assert int(info.n_iter) <= 2


@pytest.mark.parametrize("spin", [0, 2])
def test_solve_info_reports_true_residual(spin):
    """``rel_residual`` is the true ``||Pb - (PAP+I)y|| / ||Pb||``, converged or capped."""
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4)
    ninv = _ninv(nside, mask, seed=29)
    cl = _signal_cl(lmax, spin)
    cl_j = _cl_jax(cl, spin)
    m = np.asarray(synthesis(jnp.asarray(_rand_alm(lmax, spin, 30)), nside, lmax, spin))
    p = wiener_preconditioner(cl_j, nside, lmax, spin, inv_noise=ninv, ell_split=4)
    nb = _ninv_bcast(ninv, spin)
    pw = jnp.asarray(_prior_sqrt_ref(cl, lmax, spin))
    rhs = pw * alm_to_real(adjoint_synthesis(nb * jnp.asarray(m), nside, lmax, spin), lmax, spin)

    def true_rel(a):
        y = alm_to_real(a, lmax, spin) / pw
        r = rhs - _whitened_op(y, nb, pw, nside, lmax, spin)
        return float(jnp.linalg.norm(r) / jnp.linalg.norm(rhs))

    tol = 1e-10
    a, info = wiener(m, cl_j, nside, lmax, spin=spin, inv_noise=ninv, precond=p, return_info=True, max_iter=800, tol=tol)
    assert float(info.rel_residual) <= tol
    assert abs(float(info.rel_residual) - true_rel(a)) <= 1e-12

    a3, info3 = wiener(m, cl_j, nside, lmax, spin=spin, inv_noise=ninv, precond=p, return_info=True, max_iter=3, tol=tol)
    assert int(info3.n_iter) == 3 and float(info3.rel_residual) > tol
    assert abs(float(info3.rel_residual) - true_rel(a3)) <= 1e-8 * true_rel(a3)


def test_precond_shape_mismatch_raises():
    nside, lmax, spin = 8, 8, 2
    ninv = _ninv(nside, _binary_cap(nside, 0.4), seed=31)
    cl_j = _cl_jax(_signal_cl(lmax, spin), spin)
    p = wiener_preconditioner(_cl_jax(_signal_cl(6, spin), spin), nside, 6, spin, inv_noise=ninv, ell_split=None)
    m = np.zeros((2, 12 * nside**2))
    with pytest.raises(ValueError, match="DOFs"):
        wiener(m, cl_j, nside, lmax, spin=spin, inv_noise=ninv, precond=p)


@pytest.mark.parametrize("spin", [0, 2])
def test_precond_constrained_realization_matches_unpreconditioned(spin):
    """Same key -> same stochastic RHS -> the same draw with or without the preconditioner."""
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4)
    ninv = _ninv(nside, mask, seed=32)
    cl_j = _cl_jax(_signal_cl(lmax, spin), spin)
    m = jnp.asarray(np.asarray(synthesis(jnp.asarray(_rand_alm(lmax, spin, 33)), nside, lmax, spin)))
    key = jax.random.PRNGKey(5)
    p = wiener_preconditioner(cl_j, nside, lmax, spin, inv_noise=ninv, ell_split=4)
    a_p, info = constrained_realization(
        m, cl_j, nside, lmax, key, spin=spin, inv_noise=ninv, precond=p, return_info=True, max_iter=800, tol=1e-12
    )
    a_0 = constrained_realization(m, cl_j, nside, lmax, key, spin=spin, inv_noise=ninv, max_iter=800, tol=1e-12)
    a_0 = np.asarray(a_0)
    assert np.max(np.abs(np.asarray(a_p) - a_0)) <= SOLVE_TOL * (1.0 + np.max(np.abs(a_0)))
    assert float(info.rel_residual) <= 1e-12


def test_precond_gradients_match_unpreconditioned():
    """Implicit gradients (wrt Cl and the data) are those of the solution, not of the path."""
    nside, lmax, spin = 8, 6, 0
    mask = _binary_cap(nside, 0.4)
    ninv = jnp.asarray(_ninv(nside, mask, seed=34))
    m = jnp.asarray(np.asarray(synthesis(jnp.asarray(_rand_alm(lmax, spin, 35)), nside, lmax, spin)))
    cl0 = jnp.asarray(_signal_cl(lmax, spin))
    p = wiener_preconditioner(cl0, nside, lmax, spin, inv_noise=ninv, ell_split=3)

    def loss(cl, mm, pc):
        a = wiener(mm, cl, nside, lmax, spin=spin, inv_noise=ninv, precond=pc, max_iter=400, tol=1e-12)
        return jnp.real(jnp.sum(jnp.abs(a) ** 2))

    loss(cl0, m, None)  # pre-warm caches before grad
    g_p = np.asarray(jax.grad(loss)(cl0, m, p))
    g_0 = np.asarray(jax.grad(loss)(cl0, m, None))
    assert np.all(np.isfinite(g_p))
    assert np.max(np.abs(g_p - g_0)) <= 1e-6 * np.max(np.abs(g_0))
    g_pj = np.asarray(jax.jit(jax.grad(loss))(cl0, m, p))
    assert np.max(np.abs(g_pj - g_p)) <= 1e-9 * (1.0 + np.max(np.abs(g_p)))
    gm_p = np.asarray(jax.grad(loss, argnums=1)(cl0, m, p))
    gm_0 = np.asarray(jax.grad(loss, argnums=1)(cl0, m, None))
    assert np.max(np.abs(gm_p - gm_0)) <= 1e-6 * np.max(np.abs(gm_0))


def test_precond_vmap_matches_per_sample():
    nside, lmax, spin = 8, 8, 2
    mask = _binary_cap(nside, 0.4)
    ninv = _ninv(nside, mask, seed=36)
    cl_j = _cl_jax(_signal_cl(lmax, spin), spin)
    maps = jnp.stack([jnp.asarray(np.asarray(synthesis(jnp.asarray(_rand_alm(lmax, spin, s)), nside, lmax, spin))) for s in (37, 38, 39)])
    p = wiener_preconditioner(cl_j, nside, lmax, spin, inv_noise=ninv, ell_split=4)

    def one(mm):
        return wiener(mm, cl_j, nside, lmax, spin=spin, inv_noise=ninv, precond=p, return_info=True, max_iter=800, tol=1e-12)

    one(maps[0])  # pre-warm the static-layout caches before vmap
    a_v, info_v = jax.vmap(one)(maps)
    assert info_v.n_iter.shape == (3,) and np.all(np.asarray(info_v.rel_residual) <= 1e-12)
    for i in range(3):
        a_i = np.asarray(one(maps[i])[0])
        assert np.max(np.abs(np.asarray(a_v[i]) - a_i)) <= SOLVE_TOL * (1.0 + np.max(np.abs(a_i)))


def test_precond_cuts_iterations_on_irregular_mask():
    """Two-sided band cut + holes, red prior, S/N >> 1 at low l (spin 2, nside 32, lmax 48).

    Measured in this configuration: 454 iterations to 1e-8 unpreconditioned, 241 with the
    diagonal alone, 123 with the l <= 16 block.  Gated with margin: diagonal < none, and
    the block at most half of none.
    """
    import healpy as hp

    nside, lmax, spin = 32, 48, 2
    npix = 12 * nside**2
    th = _theta(nside)
    mask = (np.abs(th - np.pi / 2) > np.radians(20)).astype(float)
    rng = np.random.default_rng(3)
    for c in rng.integers(0, npix, 12):
        mask[hp.query_disc(nside, hp.pix2vec(nside, int(c)), np.radians(rng.uniform(3, 8)))] = 0.0
    ell = np.arange(lmax + 1)
    cl_ee = np.where(ell >= 2, 1e2 / (ell + 1.0) ** 2, 0.0)
    cl_j = (jnp.asarray(cl_ee), jnp.asarray(0.05 * cl_ee))
    alm = np.stack([hp.synalm(cl_ee, lmax=lmax, new=True), hp.synalm(0.05 * cl_ee, lmax=lmax, new=True)])
    m = np.asarray(synthesis(jnp.asarray(alm), nside, lmax, spin)) + np.random.default_rng(4).standard_normal((2, npix))

    def iters(pc):
        _, info = wiener(m, cl_j, nside, lmax, spin=spin, inv_noise=mask, precond=pc, return_info=True, max_iter=2000, tol=1e-8)
        assert float(info.rel_residual) <= 1e-8
        return int(info.n_iter)

    n_none = iters(None)
    n_diag = iters(wiener_preconditioner(cl_j, nside, lmax, spin, inv_noise=mask, ell_split=None))
    n_block = iters(wiener_preconditioner(cl_j, nside, lmax, spin, inv_noise=mask, ell_split=16))
    assert n_diag < n_none
    assert n_block <= 0.5 * n_none, (n_none, n_diag, n_block)


def test_deconvolve_reports_solve_info():
    nside, lmax, spin = 8, 8, 2
    mask = _binary_cap(nside, 0.3)
    m = np.asarray(synthesis(jnp.asarray(_rand_alm(lmax, spin, 40)), nside, lmax, spin))
    a, info = deconvolve(m, mask, nside, lmax, spin=spin, reg=0.1, max_iter=800, tol=1e-12, return_info=True)
    a_def = np.asarray(deconvolve(m, mask, nside, lmax, spin=spin, reg=0.1, max_iter=800, tol=1e-12))
    assert float(info.rel_residual) <= 1e-12
    assert np.max(np.abs(np.asarray(a) - a_def)) <= SOLVE_TOL * (1.0 + np.max(np.abs(a_def)))
