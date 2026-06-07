"""Gates for partial-sky / masked analysis (:mod:`jht.masked`).

Two estimators, with distinct, non-conflated contracts:

* :func:`jht.masked.pseudo_alm` -- the masked pseudo-a_lm.  The **unweighted**
  (uniform) pseudo-a_lm ``(4pi/Npix) S^T(M m)`` is the canonical,
  weight-unambiguous quantity, gated == healpy/ducc to machine precision
  (``M-pseudo``).  (The ring-weighted/iterated pseudo-a_lm is jht's own
  conditioned estimator -- a different weight solver from healpy's -- so it is
  only *comparable*, not identical; that is characterised in ``scripts/masked_sweep.py``,
  not gated here, exactly as the full-sky ``test_vs_healpy`` is.)
* :func:`jht.masked.deconvolve` -- the cut-sky CG deconvolution.  For noiseless
  band-limited data it recovers the **true a_lm** wherever the cut leaves the
  modes constrained (the operator is well-conditioned on the active band),
  *independently of the weights*; gated == truth and == the explicit dense
  normal-equations solve (``M-deconv``).  Recovery degrades as the cut grows
  (near-null modes) -- characterised, not gated tight, per the project's
  "never conflate cut-information-loss with quadrature" rule.

Plus the machinery the above rests on: the ``(2-delta_{m0})``-weighted isometry
``T`` (``M-isometry``) and the masked normal operator (``M-operator``: matches
the explicit dense matrix, symmetric, PSD).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jht.analysis import map2alm  # noqa: E402
from jht.healpix import adjoint_synthesis, alm_size, synthesis  # noqa: E402
from jht.masked import alm_to_real, deconvolve, n_dof, pseudo_alm, real_to_alm  # noqa: E402
from jht.weights import pixel_weights  # noqa: E402

ISO_TOL = 1e-12  # isometry round-trip / norm (measured ~2e-16)
OP_TOL = 1e-10  # masked operator: dense match, symmetry (measured ~4e-15)
PSEUDO_TOL = 1e-10  # unweighted pseudo-a_lm vs healpy/ducc (measured ~3e-15)
RECOVER_TOL = 1e-6  # deconvolution recovers truth where well-posed (measured ~1e-13)
DENSE_TOL = 1e-8  # CG == explicit dense solve (CG tol-limited; measured ~1e-13)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _lvals(lmax: int) -> np.ndarray:
    return np.concatenate([np.arange(m, lmax + 1) for m in range(lmax + 1)])


def _mvals(lmax: int) -> np.ndarray:
    return np.concatenate([np.full(lmax + 1 - m, m) for m in range(lmax + 1)])


def _alm_weight(lmax: int) -> np.ndarray:
    """The ``(2 - delta_{m0})`` a_lm inner-product weight."""
    return np.where(_mvals(lmax) == 0, 1.0, 2.0)


def _rand_alm(lmax: int, spin: int, seed: int = 0) -> np.ndarray:
    """Random broadband a_lm for a real spin-``spin`` field (m=0 real, no l<|spin|)."""
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
    """Observed sky = everything except a polar cap of radius ``th_cut``."""
    m = np.ones(12 * nside**2)
    m[_theta(nside) < th_cut] = 0.0
    return m


def _apod_cap(nside: int, th_cut: float, width: float) -> np.ndarray:
    """Cosine-apodized polar-cap mask (0 in the cap, smooth ramp to 1)."""
    th = _theta(nside)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.clip((th - th_cut) / width, 0.0, 1.0)))
    return np.where(th < th_cut, 0.0, ramp)


def _apply_mask(maps: np.ndarray, mask: np.ndarray, spin: int) -> np.ndarray:
    return maps * (mask if spin == 0 else mask[None, :])


def _dense_S(nside: int, lmax: int, spin: int) -> np.ndarray:
    """Explicit synthesis matrix in the real-DOF basis: columns ``S(T^-1 e_k)``."""
    nx = n_dof(lmax, spin)
    cols = []
    for k in range(nx):
        e = np.zeros(nx)
        e[k] = 1.0
        cols.append(np.asarray(synthesis(real_to_alm(e, lmax, spin), nside, lmax, spin)).ravel())
    return np.stack(cols, axis=1)


def _MW_flat(nside: int, mask: np.ndarray, spin: int) -> np.ndarray:
    mw = mask * pixel_weights(nside, True)
    return mw if spin == 0 else np.tile(mw, 2)


def _premul(maps, nside: int, mask: np.ndarray, spin: int):
    """Per-pixel ``(M W) * maps`` (the masked-weight premultiply)."""
    mw = mask * pixel_weights(nside, True)
    return maps * (mw if spin == 0 else mw[None, :])


# --------------------------------------------------------------------------- #
# M-isometry: T is an isometry  (alm, <.,.>_w)  <->  (x, Euclidean)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
def test_isometry_roundtrip(spin):
    """``T^-1 T = I`` on healpy-packed a_lm."""
    lmax = 10
    a = _rand_alm(lmax, spin, 1)
    a2 = np.asarray(real_to_alm(alm_to_real(a, lmax, spin), lmax, spin))
    assert np.max(np.abs(a2 - a)) <= ISO_TOL


@pytest.mark.parametrize("spin", [0, 2])
def test_isometry_preserves_weighted_norm(spin):
    """``||T a||_2^2 == ||a||_w^2`` (the ``(2-delta_{m0})`` norm)."""
    lmax = 10
    a = _rand_alm(lmax, spin, 2)
    x = np.asarray(alm_to_real(a, lmax, spin))
    w = _alm_weight(lmax)
    a2d = np.atleast_2d(a)
    wnorm2 = float(np.sum(w * np.abs(a2d) ** 2))
    assert abs(float(np.sum(x**2)) - wnorm2) <= ISO_TOL * wnorm2


# --------------------------------------------------------------------------- #
# M-operator: the masked normal operator A_x = T S^T(MW)S T^-1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("which", ["binary", "apod"])
def test_operator_matches_dense(spin, which):
    """Matrix-free ``A_x`` equals the explicit dense ``S^T diag(MW) S`` (nside=8)."""
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4) if which == "binary" else _apod_cap(nside, 0.4, 0.4)
    S = _dense_S(nside, lmax, spin)
    A_dense = S.T @ (_MW_flat(nside, mask, spin)[:, None] * S)
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = rng.standard_normal(n_dof(lmax, spin))
        a = real_to_alm(x, lmax, spin)
        wmp = _premul(np.asarray(synthesis(a, nside, lmax, spin)), nside, mask, spin)
        ax = np.asarray(alm_to_real(adjoint_synthesis(wmp, nside, lmax, spin), lmax, spin))
        ref = A_dense @ x
        assert np.max(np.abs(ax - ref)) <= OP_TOL * (1.0 + np.max(np.abs(ref)))


@pytest.mark.parametrize("spin", [0, 2])
def test_operator_symmetric_and_psd(spin):
    """``A_x`` is symmetric and positive semi-definite (the inner-product handling)."""
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4)

    def aop(x):
        a = real_to_alm(x, lmax, spin)
        wmp = _premul(np.asarray(synthesis(a, nside, lmax, spin)), nside, mask, spin)
        return np.asarray(alm_to_real(adjoint_synthesis(wmp, nside, lmax, spin), lmax, spin))

    rng = np.random.default_rng(1)
    u, v = rng.standard_normal(n_dof(lmax, spin)), rng.standard_normal(n_dof(lmax, spin))
    assert abs(float(u @ aop(v)) - float(v @ aop(u))) <= OP_TOL * (1.0 + abs(float(u @ aop(v))))
    assert float(u @ aop(u)) >= -OP_TOL


# --------------------------------------------------------------------------- #
# M-pseudo: the canonical (unweighted) pseudo-a_lm == healpy/ducc
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nside,lmax", [(16, 16), (32, 32)])
@pytest.mark.parametrize("which", ["binary", "apod"])
def test_pseudo_spin0_vs_healpy(nside, lmax, which):
    import healpy as hp

    mask = _binary_cap(nside, 0.5) if which == "binary" else _apod_cap(nside, 0.5, 0.4)
    m = np.asarray(synthesis(_rand_alm(lmax, 0, 3), nside, lmax, spin=0))
    jht_ps = np.asarray(pseudo_alm(m, mask, nside, lmax, spin=0, niter=0, use_weights=False))
    hp_ps = hp.map2alm(mask * m, lmax=lmax, mmax=lmax, iter=0, use_weights=False)
    assert np.max(np.abs(jht_ps - hp_ps)) <= PSEUDO_TOL * np.max(np.abs(hp_ps))


@pytest.mark.parametrize("nside,lmax", [(16, 16), (32, 32)])
@pytest.mark.parametrize("which", ["binary", "apod"])
def test_pseudo_spin2_vs_healpy(nside, lmax, which):
    import healpy as hp

    mask = _binary_cap(nside, 0.5) if which == "binary" else _apod_cap(nside, 0.5, 0.4)
    QU = np.asarray(synthesis(_rand_alm(lmax, 2, 4), nside, lmax, spin=2))
    QUm = _apply_mask(QU, mask, 2)
    jht_ps = np.asarray(pseudo_alm(QU, mask, nside, lmax, spin=2, niter=0, use_weights=False))
    bE, bB = hp.map2alm_spin([QUm[0], QUm[1]], 2, lmax=lmax, mmax=lmax)
    scale = max(np.max(np.abs(bE)), np.max(np.abs(bB)))
    assert np.max(np.abs(jht_ps[0] - bE)) <= PSEUDO_TOL * scale
    assert np.max(np.abs(jht_ps[1] - bB)) <= PSEUDO_TOL * scale


# --------------------------------------------------------------------------- #
# M-fsky->1: mask == 1 reduces to the full-sky transforms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
def test_mask_ones_reduces_to_fullsky(spin):
    nside, lmax = 16, 12
    ones = np.ones(12 * nside**2)
    a = _rand_alm(lmax, spin, 5)
    m = np.asarray(synthesis(a, nside, lmax, spin))
    # pseudo-a_lm with mask=1 is exactly the full-sky map2alm
    ps = np.asarray(pseudo_alm(m, ones, nside, lmax, spin=spin, niter=3))
    full = np.asarray(map2alm(m, nside, lmax, spin=spin, niter=3))
    assert np.max(np.abs(ps - full)) <= ISO_TOL
    # deconvolution with mask=1 recovers the truth (A -> S^T W S ~ I)
    rec = np.asarray(deconvolve(m, ones, nside, lmax, spin=spin, max_iter=50, tol=1e-12))
    assert np.max(np.abs(rec - a)) <= RECOVER_TOL


# --------------------------------------------------------------------------- #
# M-deconv: CG recovers the true a_lm where well-posed, == dense solve
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
def test_deconvolve_recovers_truth_and_dense_solve(spin):
    """Mild cut: CG deconvolution == truth == explicit dense normal-equations solve."""
    nside, lmax = 8, 8
    mask = _binary_cap(nside, 0.4)  # large fsky -> well-conditioned active band
    a = _rand_alm(lmax, spin, 6)
    m = _apply_mask(np.asarray(synthesis(a, nside, lmax, spin)), mask, spin)
    rec = np.asarray(deconvolve(m, mask, nside, lmax, spin=spin, max_iter=800, tol=1e-12))
    assert np.max(np.abs(rec - a)) <= RECOVER_TOL  # recovers truth (weight-independent)
    # == explicit dense solve of S^T(MW)S a = S^T(MW)m
    S = _dense_S(nside, lmax, spin)
    A_dense = S.T @ (_MW_flat(nside, mask, spin)[:, None] * S)
    b = np.asarray(alm_to_real(adjoint_synthesis(_premul(m, nside, mask, spin), nside, lmax, spin), lmax, spin))
    a_dense = np.asarray(real_to_alm(np.linalg.solve(A_dense, b), lmax, spin))
    assert np.max(np.abs(rec - a_dense)) <= DENSE_TOL


@pytest.mark.parametrize("spin", [0, 2])
def test_deconvolve_beats_pseudo_on_mild_cut(spin):
    """Deconvolution removes the pseudo-a_lm mode-coupling bias on a recoverable cut."""
    nside, lmax = 16, 12
    mask = _binary_cap(nside, 0.3)
    a = _rand_alm(lmax, spin, 7)
    m = np.asarray(synthesis(a, nside, lmax, spin))
    pseudo_err = float(np.max(np.abs(np.asarray(pseudo_alm(m, mask, nside, lmax, spin=spin, niter=3)) - a)))
    deconv_err = float(np.max(np.abs(np.asarray(deconvolve(_apply_mask(m, mask, spin), mask, nside, lmax, spin=spin, max_iter=600, tol=1e-12)) - a)))
    assert deconv_err < pseudo_err  # the cut is real: pseudo is biased, deconv recovers
    assert deconv_err <= RECOVER_TOL
