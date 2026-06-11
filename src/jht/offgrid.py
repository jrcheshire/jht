"""Off-grid spherical harmonic synthesis -- ``synthesis_general`` / its adjoint.

Evaluate a band-limited field at *arbitrary* points (detector pointings) on the
sphere, the JAX-native, differentiable replacement for ducc0's
``synthesis_general`` / ``adjoint_synthesis_general`` (the bk-jax sim-forward TOD
path -- the last ducc0 capability bk-jax depends on).

Algorithm (matches ducc's ``sphere_interpol``): **Double-Fourier-Sphere + 2D
type-2 NUFFT**.  ``alm`` --(Legendre step at a Clenshaw-Curtis theta grid; reuses
``synth_contract``)--> per-m field coefficients --(DFS theta-extension + theta-FFT)-->
2D Fourier coefficients ``F(k,m)`` --(:mod:`jht._nufft`)--> field at the points.
The adjoint is the exact transpose (type-1 NUFFT + transposed DFS + ``adjoint_contract``).

Conventions are inherited from the on-grid core (healpy m-major triangular alm,
orthonormal Y, the ``(2 - delta_m0)`` metric and the E,B<->Q,U sign -- see
:mod:`jht.healpix`).  Accuracy is a deliberately new tier set by the NUFFT
``epsilon`` (default 1e-10, matching ducc) -- *not* the on-grid 1e-13.  Both the
alm gradient (native AD == ``G*conj(adjoint)``) and the **pointing gradient**
(``d field / d loc``, the capability ducc's FFI cannot provide) are exact under
JAX's native autodiff.  spin 0, 1, 2, 3 (the -spin channel carries (-1)^s).
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ._nufft import NufftPlan, _next_size, nufft2d1, nufft2d2, nufft_plan
from ._recursion import RecursionPlan, adjoint_contract, build_recursion_plan, synth_contract
from .healpix import _tri_dense_maps, alm_metric_weight, alm_size

TWO_PI = 2.0 * np.pi


class _OffgridPrep(NamedTuple):
    lmax: int
    spin: int
    ncomp: int  # 1 (spin-0) or 2 (spin>0)
    ntheta_s: int
    ntheta_d: int
    x_cc: np.ndarray  # cos(theta) at the Clenshaw-Curtis rings (n_theta_s,)
    plan_pos: RecursionPlan  # +spin recursion plan at the CC grid
    plan_neg: RecursionPlan  # -spin recursion plan (== plan_pos for spin 0)
    nplan: NufftPlan
    sign_m: np.ndarray  # (2lmax+1,) DFS parity (-1)^m * (-1)^s over m=-lmax..lmax
    south_i: np.ndarray  # (ntheta_d - ntheta_s,) doubled-grid south indices
    refl: np.ndarray  # their reflected north indices
    gather: np.ndarray  # tri<->dense index maps
    valid: np.ndarray
    pack: np.ndarray  # dense.ravel() -> tri gather index (dense_to_tri)
    gmetric: np.ndarray  # (alm_size,) the (2 - delta_m0) weight


@lru_cache(maxsize=None)
def _prepare(lmax: int, spin: int, epsilon: float) -> _OffgridPrep:
    if spin not in (0, 1, 2, 3):
        raise NotImplementedError(f"spin={spin} unsupported (0, 1, 2, 3)")
    # lru_cache => warns once per (lmax, spin, epsilon); getattr: the attr is dynamic (mypy)
    if not getattr(jax.config, "jax_enable_x64", True):
        warnings.warn(
            "jht: jax_enable_x64 is OFF -- the off-grid transforms will silently run in "
            "float32, far below the requested NUFFT epsilon. Enable it before creating "
            'any array: jax.config.update("jax_enable_x64", True).',
            stacklevel=2,
        )
    ntheta_s = _next_size(lmax + 1) + 1  # CC rings on [0, pi], incl. both poles
    theta_cc = np.arange(ntheta_s) * np.pi / (ntheta_s - 1)
    x_cc = np.cos(theta_cc)
    plan_pos = build_recursion_plan(x_cc, spin, lmax)
    plan_neg = plan_pos if spin == 0 else build_recursion_plan(x_cc, -spin, lmax)
    ntheta_d = 2 * ntheta_s - 2  # DFS period
    nplan = nufft_plan(ntheta_d, 2 * lmax + 1, epsilon)
    mvals = np.arange(-lmax, lmax + 1)
    sign_m = ((-1.0) ** np.abs(mvals)) * ((-1.0) ** spin)
    south_i = np.arange(ntheta_s, ntheta_d)
    refl = ntheta_d - south_i
    gather, valid, pack = _tri_dense_maps(lmax)
    return _OffgridPrep(
        lmax, spin, 1 if spin == 0 else 2, ntheta_s, ntheta_d, x_cc,
        plan_pos, plan_neg, nplan, sign_m, south_i, refl,
        gather, valid, pack, alm_metric_weight(lmax),
    )


def _tri_to_dense(p: _OffgridPrep, alm):
    return jnp.where(jnp.asarray(p.valid), jnp.asarray(alm)[jnp.asarray(p.gather)], 0.0 + 0.0j)


def _dense_to_tri(p: _OffgridPrep, dense):
    return dense.ravel()[jnp.asarray(p.pack)]  # gather, not scatter (fp64 GPU; see healpix)


def _channels(p: _OffgridPrep, alm):
    """alm -> (alpha, beta) dense inputs for the +spin / -spin Legendre channels.

    spin-0: a single real field, both channels are the same a_lm.
    spin-2: the +-2 fields, ``+2a = -(aE + i aB)``, ``-2a = -(aE - i aB)``.
    """
    if p.spin == 0:
        d = _tri_to_dense(p, alm)
        return d, d
    aE, aB = _tri_to_dense(p, alm[0]), _tri_to_dense(p, alm[1])
    s_sign = (-1.0) ** p.spin  # the -spin channel carries (-1)^s (hidden for even spin)
    return -(aE + 1j * aB), -s_sign * (aE - 1j * aB)


def _channels_adjoint(p: _OffgridPrep, p_cot, n_cot):
    """Transpose of :func:`_channels`: (alpha_cot, beta_cot) dense -> alm cotangent."""
    if p.spin == 0:
        return _dense_to_tri(p, p_cot + n_cot)
    # (aE,aB) -> (-(aE+i aB), -(-1)^s (aE-i aB)) has Hermitian adjoint
    #   aE_cot = -(p_cot + s n_cot),  aB_cot = i (p_cot - s n_cot),  s = (-1)^spin.
    s_sign = (-1.0) ** p.spin
    aE = _dense_to_tri(p, -(p_cot + s_sign * n_cot))
    aB = _dense_to_tri(p, 1j * (p_cot - s_sign * n_cot))
    return jnp.stack([aE, aB])


def _coeffs_to_Fkm(p: _OffgridPrep, Cpos, Cneg):
    """+spin / -spin per-m field coeffs (m>=0) -> DFS-extended 2D Fourier coeffs F(k,m)."""
    lmax = p.lmax
    Cfull = jnp.zeros((2 * lmax + 1, p.ntheta_s), dtype=jnp.complex128)
    Cfull = Cfull.at[lmax:].set(Cpos)  # m = 0..lmax (the +spin field)
    Cfull = Cfull.at[:lmax].set(jnp.conj(Cneg[1:][::-1]))  # m = -lmax..-1 = conj(-spin field)
    Dd = jnp.zeros((p.ntheta_d, 2 * lmax + 1), dtype=jnp.complex128)
    Dd = Dd.at[: p.ntheta_s].set(Cfull.T)  # north hemisphere [0, pi]
    Dd = Dd.at[jnp.asarray(p.south_i)].set(
        jnp.asarray(p.sign_m)[None, :] * Cfull.T[jnp.asarray(p.refl)]
    )
    return jnp.fft.fftshift(jnp.fft.fft(Dd, axis=0), axes=0) / p.ntheta_d


def _Fkm_to_coeffs(p: _OffgridPrep, G):
    """Hermitian adjoint of :func:`_coeffs_to_Fkm`: F(k,m) cotangent -> (Cpos_cot, Cneg_cot)."""
    lmax = p.lmax
    dDd = jnp.fft.ifft(jnp.fft.ifftshift(G, axes=0), axis=0)  # adj of fftshift(fft)/Nd
    dCfullT = jnp.zeros((p.ntheta_s, 2 * lmax + 1), dtype=jnp.complex128)
    dCfullT = dCfullT.at[:].add(dDd[: p.ntheta_s])
    dCfullT = dCfullT.at[jnp.asarray(p.refl)].add(
        jnp.asarray(p.sign_m)[None, :] * dDd[jnp.asarray(p.south_i)]
    )
    dCfull = dCfullT.T  # (2lmax+1, ntheta_s)
    Cpos_cot = dCfull[lmax:]  # m = 0..lmax
    Cneg_cot = jnp.zeros((lmax + 1, p.ntheta_s), dtype=jnp.complex128)
    Cneg_cot = Cneg_cot.at[1:].set(jnp.conj(dCfull[:lmax][::-1]))  # m = 1..lmax
    return Cpos_cot, Cneg_cot


def _alm_to_Fkm(p: _OffgridPrep, alm):
    a, b = _channels(p, alm)
    Cpos = synth_contract(p.plan_pos, jnp.asarray(p.x_cc), a)  # (lmax+1, ntheta_s)
    # spin-0: plan_neg is plan_pos and a == b, so the -spin recursion is identical -- skip it.
    Cneg = Cpos if p.spin == 0 else synth_contract(p.plan_neg, jnp.asarray(p.x_cc), b)
    return _coeffs_to_Fkm(p, Cpos, Cneg)


def _Fkm_to_alm(p: _OffgridPrep, G):
    Cpos_cot, Cneg_cot = _Fkm_to_coeffs(p, G)
    p_cot = adjoint_contract(p.plan_pos, jnp.asarray(p.x_cc), Cpos_cot)
    n_cot = adjoint_contract(p.plan_neg, jnp.asarray(p.x_cc), Cneg_cot)
    return _channels_adjoint(p, p_cot, n_cot)


# --------------------------------------------------------------------------- #
# public transforms
# --------------------------------------------------------------------------- #
def synthesis_general(alm, loc, *, spin: int = 0, lmax: int, epsilon: float = 1e-10):
    """``a_lm -> field`` at arbitrary points ``loc`` (the off-grid synthesis).

    Parameters
    ----------
    alm : complex array
        healpy-packed coefficients: ``(alm_size(lmax),)`` for ``spin=0``;
        ``(2, alm_size(lmax))`` (E, B) for ``spin=2``.
    loc : real array ``(npts, 2)``
        ``loc[:,0] = theta`` (colatitude in ``[0, pi]``), ``loc[:,1] = phi`` (in
        ``[0, 2pi)``).  Out-of-range angles are not validated: both coordinates
        are evaluated on the smooth period-2pi torus extension (theta through the
        double-Fourier-sphere identity ``F(-theta, phi) = (-1)^spin F(theta,
        phi + pi)``), so e.g. ``theta = -0.1`` returns the field at the reflected
        point, not an error.
    spin : int
        ``0`` (temperature) or ``2`` (polarization Q/U).
    lmax : int
    epsilon : float
        NUFFT accuracy (default 1e-10, matching ducc's production setting).

    Returns
    -------
    field : real array ``(npts,)`` for ``spin=0``, or ``(2, npts)`` (Q, U) for ``spin=2``.
    """
    p = _prepare(lmax, spin, epsilon)
    loc = jnp.asarray(loc)
    if loc.ndim != 2 or loc.shape[-1] != 2:
        raise ValueError(f"loc must have shape (npts, 2) [theta, phi]; got {loc.shape}")
    alm = jnp.asarray(alm)
    expect = (alm_size(lmax),) if spin == 0 else (2, alm_size(lmax))
    if alm.shape != expect:
        raise ValueError(
            f"alm shape {alm.shape} != expected {expect} for lmax={lmax}, spin={spin} "
            "(a wrong-size alm would otherwise be silently clamped by the gather)"
        )
    theta, phi = loc[:, 0], loc[:, 1]
    field = nufft2d2(p.nplan, _alm_to_Fkm(p, alm), theta, phi)
    if p.spin == 0:
        return jnp.real(field)
    return jnp.stack([jnp.real(field), jnp.imag(field)])  # (Q, U)


def adjoint_synthesis_general(field, loc, *, spin: int = 0, lmax: int, epsilon: float = 1e-10):
    """Exact transpose of :func:`synthesis_general` (``field -> a_lm``).

    Satisfies ``<S_g a, v>_pts == <a, S_g^T v>_G`` with the ``(2 - delta_m0)``
    metric ``G`` -- the same strict-adjoint / VJP-bridge convention as the on-grid
    :func:`jht.healpix.adjoint_synthesis`.  ``field`` is ``(npts,)`` for ``spin=0``
    or ``(2, npts)`` (Q, U) for ``spin=2``.
    """
    p = _prepare(lmax, spin, epsilon)
    loc = jnp.asarray(loc)
    if loc.ndim != 2 or loc.shape[-1] != 2:
        raise ValueError(f"loc must have shape (npts, 2) [theta, phi]; got {loc.shape}")
    field = jnp.asarray(field)
    if jnp.iscomplexobj(field):
        raise TypeError(
            "field must be real ((npts,) for spin=0; (2, npts) real Q/U planes for spin>0); "
            "a complex field would be silently mis-folded"
        )
    npts = loc.shape[0]
    expect = (npts,) if spin == 0 else (2, npts)
    if field.shape != expect:
        raise ValueError(f"field shape {field.shape} != expected {expect} for {npts} points")
    theta, phi = loc[:, 0], loc[:, 1]
    # transpose of the (Re) / (Re, Im) field reduction
    fc = field.astype(jnp.complex128) if p.spin == 0 else (field[0] + 1j * field[1])
    fkm_cot = nufft2d1(p.nplan, fc, theta, phi)
    # _Fkm_to_alm is the raw C^K Hermitian transpose (== conj(vjp)); the strict
    # adjoint in jht's (2 - delta_m0) metric divides by G (m<0 modes are implicit).
    return _Fkm_to_alm(p, fkm_cot) / jnp.asarray(p.gmetric)
