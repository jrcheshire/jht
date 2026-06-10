"""Phase-4 off-grid NUFFT gates: ``synthesis_general`` + its exact adjoint.

* forward vs the **exact direct sum** ``sum_lm a_lm {}_sY_lm(theta_j, phi_j)`` (the
  perfect oracle) at the off-grid tier <= 1e-9;
* forward / adjoint vs ``ducc0.synthesis_general`` (cross-check, conventions);
* the inner-product adjoint identity ``<S_g a, v> == <a, S_g^T v>_G``;
* the alm-AD bridge ``jax.vjp(synthesis_general)(v) == G * conj(adjoint(v))``;
* **loc/pointing differentiability** (the beyond-ducc capability): ``d field / d loc``
  via native AD == the analytic field derivative, and ``jvp == vjp``.

spin-0 and spin-2 (the make-or-break gate -- spin-2 must sit at the spin-0 tier,
no s2fft-style m-leakage).  The off-grid NUFFT is a deliberately new accuracy tier
set by ``epsilon`` (default 1e-10, matching ducc) -- *not* the on-grid 1e-13.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jht._recursion import normalized_legendre, spin_weighted_lambda  # noqa: E402
from jht.healpix import alm_column_base, alm_metric_weight, alm_size  # noqa: E402
from jht.offgrid import adjoint_synthesis_general, synthesis_general  # noqa: E402

# The off-grid suite is the heaviest in the project (~60% of total wall time -- direct-sum
# and ducc oracles + analytic pointing-derivative checks, spin 0-3). Mark the whole module
# `slow` so it runs in the full suite (`pixi run test`) + nightly, not the fast per-push gate.
pytestmark = pytest.mark.slow

FWD_TOL = 1e-9  # a-priori off-grid tier (epsilon=1e-10 + headroom); see ROADMAP Phase 4
ADJ_TOL = 1e-12  # algebraic adjoint identity (epsilon-independent)
BRIDGE_TOL = 1e-12  # native VJP == G * conj(adjoint)
DUCC_TOL = 1e-9  # cross-check vs ducc (both approximate the same field to their eps)
FD_TOL = 1e-6  # finite-difference gradient
LOC_TOL = 1e-8  # loc-gradient vs analytic derivative (a-priori; measured ~1e-12)
LOC_REV_TOL = 1e-10  # forward-mode (jvp) == reverse-mode (vjp) on the loc path


def _rand_alm(lmax, seed, spin):
    """Random healpy-packed alm: spin-0 ``(K,)`` real m=0; spin-2 ``(2,K)`` E/B, l<2 zeroed."""
    rng = np.random.default_rng(seed)

    def one(s):
        a = (rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax)))
        a[: lmax + 1] = a[: lmax + 1].real  # m=0 column real
        for m in range(min(abs(spin), lmax + 1)):  # l < |spin| modes vanish
            base = alm_column_base(m, lmax)
            a[base : base + (abs(spin) - m)] = 0.0
        return a.astype(np.complex128)

    return one(0) if spin == 0 else np.stack([one(0), one(1)])


def _rand_loc(n, seed):
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.uniform(0.02, np.pi - 0.02, n), rng.uniform(0.0, 2.0 * np.pi, n)], axis=1
    )


def _direct(alm, loc, lmax, spin):
    """Exact oracle ``sum_lm a_lm {}_sY_lm`` via the per-m field coefficients."""
    th, ph = jnp.asarray(loc[:, 0]), jnp.asarray(loc[:, 1])
    x = jnp.cos(th)
    if spin == 0:
        f = jnp.zeros(x.shape, dtype=jnp.float64)
        for m in range(lmax + 1):
            lam = normalized_legendre(x, m, lmax)
            base = alm_column_base(m, lmax)
            a_m = jnp.asarray(alm)[base : base + (lmax - m + 1)]
            Fm = jnp.tensordot(a_m, lam, axes=([0], [0]))
            f = f + (jnp.real(Fm) if m == 0 else 2.0 * jnp.real(Fm * jnp.exp(1j * m * ph)))
        return f
    aE, aB = jnp.asarray(alm[0]), jnp.asarray(alm[1])
    s_sign = (-1.0) ** spin  # -spin channel carries (-1)^s
    qpiu = jnp.zeros(x.shape, dtype=jnp.complex128)
    for m in range(lmax + 1):
        lmin = max(m, spin)
        lp = spin_weighted_lambda(x, m, spin, lmax)
        lm = spin_weighted_lambda(x, m, -spin, lmax)
        base, off = alm_column_base(m, lmax), lmin - m
        aEc, aBc = aE[base + off : base + (lmax - m + 1)], aB[base + off : base + (lmax - m + 1)]
        fp = jnp.tensordot(-(aEc + 1j * aBc), lp, axes=([0], [0]))
        fm = jnp.tensordot(-s_sign * (aEc - 1j * aBc), lm, axes=([0], [0]))
        qpiu = qpiu + fp * jnp.exp(1j * m * ph)
        if m >= 1:
            qpiu = qpiu + jnp.conj(fm) * jnp.exp(-1j * m * ph)
    return jnp.stack([jnp.real(qpiu), jnp.imag(qpiu)])


def _ducc_synth(alm, loc, spin, lmax, epsilon=1e-12):
    ducc0 = pytest.importorskip("ducc0")
    a2d = alm[None, :] if spin == 0 else alm
    return np.asarray(
        ducc0.sht.experimental.synthesis_general(
            alm=a2d, loc=loc, spin=spin, lmax=lmax, mmax=lmax, epsilon=epsilon, nthreads=1
        )
    )


def _ducc_adjoint(field, loc, spin, lmax, epsilon=1e-12):
    ducc0 = pytest.importorskip("ducc0")
    f2d = field[None, :] if spin == 0 else field
    return np.asarray(
        ducc0.sht.experimental.adjoint_synthesis_general(
            map=f2d, loc=loc, spin=spin, lmax=lmax, mmax=lmax, epsilon=epsilon, nthreads=1
        )
    )


def _inner_alm(a, b, lmax):
    """``<a, b>_G`` with the (2 - delta_m0) metric, summed over components."""
    G = alm_metric_weight(lmax)
    a2, b2 = np.atleast_2d(a), np.atleast_2d(b)
    return float(sum(np.sum(G * np.real(np.conj(a2[c]) * b2[c])) for c in range(a2.shape[0])))


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 1, 2, 3])
@pytest.mark.parametrize("lmax", [64, 128])
def test_forward_vs_direct(spin, lmax):
    alm, loc = _rand_alm(lmax, lmax, spin), _rand_loc(4000, lmax + 1)
    f = np.asarray(synthesis_general(alm, loc, spin=spin, lmax=lmax))
    ref = np.asarray(_direct(alm, loc, lmax, spin))
    assert np.max(np.abs(f - ref)) <= FWD_TOL


@pytest.mark.parametrize("spin", [0, 1, 2, 3])
@pytest.mark.parametrize("lmax", [96, 200])
def test_forward_vs_ducc(spin, lmax):
    alm, loc = _rand_alm(lmax, lmax, spin), _rand_loc(5000, lmax + 7)
    f = np.asarray(synthesis_general(alm, loc, spin=spin, lmax=lmax))
    d = _ducc_synth(alm, loc, spin, lmax)
    d = d[0] if spin == 0 else d
    assert np.max(np.abs(f - d)) <= DUCC_TOL


@pytest.mark.parametrize("spin", [0, 1, 2, 3])
def test_adjoint_identity(spin):
    lmax = 128
    alm, loc = _rand_alm(lmax, 1, spin), _rand_loc(3000, 2)
    rng = np.random.default_rng(3)
    v = rng.standard_normal(loc.shape[0]) if spin == 0 else rng.standard_normal((2, loc.shape[0]))
    f = np.asarray(synthesis_general(alm, loc, spin=spin, lmax=lmax))
    b = np.asarray(adjoint_synthesis_general(v, loc, spin=spin, lmax=lmax))
    lhs = float(np.sum(f * v))
    rhs = _inner_alm(alm, b, lmax)
    assert abs(lhs - rhs) / abs(lhs) <= ADJ_TOL


@pytest.mark.parametrize("spin", [0, 1, 2, 3])
def test_adjoint_vs_ducc(spin):
    pytest.importorskip("ducc0")
    lmax = 128
    loc = _rand_loc(5000, 11)
    rng = np.random.default_rng(5)
    v = rng.standard_normal(loc.shape[0]) if spin == 0 else rng.standard_normal((2, loc.shape[0]))
    b = np.atleast_2d(np.asarray(adjoint_synthesis_general(v, loc, spin=spin, lmax=lmax)))
    db = np.atleast_2d(_ducc_adjoint(v, loc, spin, lmax))
    # spin-2 m=0 carries the unphysical Im(a_l0) phantom (synthesis keeps Re->Q AND
    # Im->U at m=0; see docs/design.md / DISCREPANCIES) -- jht's strict adjoint carries
    # it (identity holds to 1e-16), ducc zeros it.  Compare on the physical m>=1 modes.
    keep = np.arange(b.shape[1]) >= (lmax + 1) if spin != 0 else np.ones(b.shape[1], bool)
    rel = max(
        np.max(np.abs(b[c][keep] - db[c][keep])) / np.max(np.abs(db[c][keep]))
        for c in range(b.shape[0])
    )
    assert rel <= DUCC_TOL


@pytest.mark.parametrize("spin", [0, 1, 2, 3])
def test_vjp_bridge(spin):
    lmax = 128
    alm, loc = _rand_alm(lmax, 7, spin), _rand_loc(3000, 8)
    rng = np.random.default_rng(9)
    v = rng.standard_normal(loc.shape[0]) if spin == 0 else rng.standard_normal((2, loc.shape[0]))
    b = np.asarray(adjoint_synthesis_general(v, loc, spin=spin, lmax=lmax))
    _, vjp_fn = jax.vjp(lambda a: synthesis_general(a, loc, spin=spin, lmax=lmax), jnp.asarray(alm))
    vjp = np.asarray(vjp_fn(jnp.asarray(v))[0])
    bridge = alm_metric_weight(lmax) * np.conj(b)
    assert np.max(np.abs(vjp - bridge)) <= BRIDGE_TOL


# --------------------------------------------------------------------------- #
# loc / pointing differentiability (the beyond-ducc capability) -- native AD
# through the smooth ES kernel (the window index is stop_gradient-frozen).
# --------------------------------------------------------------------------- #
def _loc_jac_diag(fn, loc, axis):
    """Diagonal loc-Jacobian d f_j / d loc[j, axis] via a single jvp."""
    col = jnp.zeros(loc.shape).at[:, axis].set(1.0)
    return jax.jvp(fn, (loc,), (col,))[1]


@pytest.mark.parametrize("spin", [0, 1, 2, 3])
@pytest.mark.parametrize("axis", [0, 1])  # 0 = d/dtheta, 1 = d/dphi
def test_loc_grad_vs_analytic(spin, axis):
    lmax = 128
    alm, loc = _rand_alm(lmax, 1, spin), jnp.asarray(_rand_loc(2000, 2))
    nufft = _loc_jac_diag(lambda L: synthesis_general(alm, L, spin=spin, lmax=lmax), loc, axis)
    analytic = _loc_jac_diag(lambda L: _direct(alm, L, lmax, spin), loc, axis)
    rel = float(jnp.max(jnp.abs(nufft - analytic)) / jnp.max(jnp.abs(analytic)))
    assert rel <= LOC_TOL


@pytest.mark.parametrize("spin", [0, 1, 2, 3])
def test_loc_grad_fwd_eq_rev(spin):
    """Forward-mode (jvp) and reverse-mode (grad) agree on a scalar loc loss."""
    lmax = 96
    alm, loc = _rand_alm(lmax, 3, spin), jnp.asarray(_rand_loc(1500, 4))
    rng = np.random.default_rng(5)
    w = jnp.asarray(
        rng.standard_normal(loc.shape[0]) if spin == 0 else rng.standard_normal((2, loc.shape[0]))
    )
    loss = lambda L: jnp.sum(w * synthesis_general(alm, L, spin=spin, lmax=lmax))  # noqa: E731
    t = jnp.asarray(rng.standard_normal(loc.shape))
    d_fwd = jax.jvp(loss, (loc,), (t,))[1]  # directional derivative (forward)
    d_rev = jnp.sum(jax.grad(loss)(loc) * t)  # <grad, t> (reverse)
    assert abs(float(d_fwd - d_rev)) / (abs(float(d_fwd)) + 1e-12) <= LOC_REV_TOL


def test_grad_finite_and_fd_spin0():
    """alm-AD: grad of a scalar loss wrt alm is finite and matches finite differences."""
    lmax = 24
    alm, loc = _rand_alm(lmax, 2, 0), _rand_loc(500, 3)
    w = jnp.asarray(np.random.default_rng(4).standard_normal(loc.shape[0]))

    def loss(a):
        return jnp.sum(w * synthesis_general(a, loc, spin=0, lmax=lmax))

    g = jax.grad(loss)(jnp.asarray(alm))
    assert np.all(np.isfinite(np.asarray(g)))
    k = alm_column_base(5, lmax) + 2
    eps = 1e-4
    ap, am = alm.copy(), alm.copy()
    ap[k] += eps
    am[k] -= eps
    fd = float((loss(jnp.asarray(ap)) - loss(jnp.asarray(am))) / (2 * eps))
    ana = float(np.real(g[k]))  # JAX: grad = dL/dRe - i dL/dIm
    assert abs(fd - ana) / (abs(fd) + 1e-12) <= FD_TOL


def test_joint_alm_loc_grad_finite():
    """Differentiable through BOTH alm and pointing simultaneously (+ jit-clean)."""
    lmax = 32
    alm = jnp.asarray(_rand_alm(lmax, 6, 2))
    loc = jnp.asarray(_rand_loc(400, 7))
    w = jnp.asarray(np.random.default_rng(8).standard_normal((2, 400)))

    def loss(a, L):
        return jnp.sum(w * synthesis_general(a, L, spin=2, lmax=lmax))

    ga, gl = jax.jit(jax.grad(loss, argnums=(0, 1)))(alm, loc)
    assert np.all(np.isfinite(np.asarray(ga))) and np.all(np.isfinite(np.asarray(gl)))
