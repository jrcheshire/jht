"""Gates for the Wiener filter / MUSE inner solve (:func:`jht.masked.wiener`,
:func:`jht.masked.constrained_realization`).

The Wiener solve is ``(S^T N^-1 S + C^-1) a = S^T N^-1 m`` with per-pixel
inverse-noise ``N^-1`` and a diagonal signal prior ``C_{lm} = Cl``.  In the
real-DOF x-space (isometry ``T``) the prior is the *exact* diagonal
``D = diag(1/Cl)``, so this is the same operator family as :func:`deconvolve`
with the scalar ``reg`` replaced by ``D``.  The gate philosophy mirrors
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
    _prior_diag,  # private: gated against an independent numpy reference
    alm_to_real,
    constrained_realization,
    deconvolve,
    n_dof,
    real_to_alm,
    wiener,
)

PRIOR_TOL = 1e-12  # _prior_diag == independent ref (layout)
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


def _prior_diag_ref(cl, lmax: int, spin: int) -> np.ndarray:
    """Independent numpy build of the x-space prior diagonal ``diag(1/Cl)``."""
    ms, ls = _mvals(lmax), _lvals(lmax)
    active = ls >= abs(spin)
    ell = np.concatenate([ls[active & (ms == 0)], ls[active & (ms > 0)], ls[active & (ms > 0)]])

    def one(c) -> np.ndarray:
        clv = np.asarray(c)[ell]
        return np.where(clv > 0, 1.0 / np.where(clv > 0, clv, 1.0), 1.0e30)

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
def test_prior_diag_matches_ref(spin):
    """``_prior_diag(Cl)`` == the independent ``diag(1/Cl)`` build, in the T layout."""
    lmax = 10
    cl = _signal_cl(lmax, spin)
    cl_j = jnp.asarray(cl) if spin == 0 else (jnp.asarray(cl[0]), jnp.asarray(cl[1]))
    d = np.asarray(_prior_diag(cl_j, lmax, spin))
    ref = _prior_diag_ref(cl, lmax, spin)
    assert d.shape == (n_dof(lmax, spin),)
    assert np.max(np.abs(d - ref)) <= PRIOR_TOL * (1.0 + np.max(np.abs(ref)))


# --------------------------------------------------------------------------- #
# W-operator: A_x + D == dense  S^T diag(N^-1) S + diag(1/Cl);  symmetric; strictly PD
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("which", ["binary", "apod"])
def test_wiener_operator_matches_dense(spin, which):
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4) if which == "binary" else _apod_cap(nside, 0.4, 0.4)
    ninv = _ninv(nside, mask, seed=1)
    cl = _signal_cl(lmax, spin)
    cl_j = jnp.asarray(cl) if spin == 0 else (jnp.asarray(cl[0]), jnp.asarray(cl[1]))

    S = _dense_S(nside, lmax, spin)
    D = _prior_diag_ref(cl, lmax, spin)
    A_dense = S.T @ (_ninv_flat(ninv, spin)[:, None] * S) + np.diag(D)

    nb = _ninv_bcast(ninv, spin)
    d_diag = np.asarray(_prior_diag(cl_j, lmax, spin))
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = rng.standard_normal(n_dof(lmax, spin))
        a = real_to_alm(jnp.asarray(x), lmax, spin)
        wmp = nb * synthesis(a, nside, lmax, spin)
        ax = np.asarray(alm_to_real(adjoint_synthesis(wmp, nside, lmax, spin), lmax, spin)) + d_diag * x
        ref = A_dense @ x
        assert np.max(np.abs(ax - ref)) <= OP_TOL * (1.0 + np.max(np.abs(ref)))


@pytest.mark.parametrize("spin", [0, 2])
def test_wiener_operator_symmetric_and_pd(spin):
    """A_x + D is symmetric and *strictly* PD (the prior lifts the near-null modes)."""
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4)
    ninv = _ninv(nside, mask, seed=2)
    cl = _signal_cl(lmax, spin)
    cl_j = jnp.asarray(cl) if spin == 0 else (jnp.asarray(cl[0]), jnp.asarray(cl[1]))
    nb = _ninv_bcast(ninv, spin)
    d_diag = np.asarray(_prior_diag(cl_j, lmax, spin))

    def aop(x):
        a = real_to_alm(jnp.asarray(x), lmax, spin)
        wmp = nb * synthesis(a, nside, lmax, spin)
        return np.asarray(alm_to_real(adjoint_synthesis(wmp, nside, lmax, spin), lmax, spin)) + d_diag * x

    rng = np.random.default_rng(3)
    u, v = rng.standard_normal(n_dof(lmax, spin)), rng.standard_normal(n_dof(lmax, spin))
    assert abs(float(u @ aop(v)) - float(v @ aop(u))) <= OP_TOL * (1.0 + abs(float(u @ aop(v))))
    # strictly PD: x^T(A+D)x >= ||x||^2 / max(Cl) > 0
    assert float(u @ aop(u)) > 0.0


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
