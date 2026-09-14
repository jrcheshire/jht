"""Reverse-mode memory of the on-grid transforms (GitHub #4).

* ``synthesis_vjp`` / ``adjoint_synthesis_vjp`` return native AD's values and
  cotangents exactly (non-physical inputs included) and block forward mode;
* their gradient graphs keep no ``(lmax+1, lmax+1, 2 nside)`` recursion table, also
  inside a scanned body;
* the default save-nothing checkpoint around each kernel keeps forward mode working
  and stops a scanned body from hoisting extra recursion tables on the native path.
"""

from __future__ import annotations

import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from jht.diff import adjoint_synthesis_vjp, synthesis_vjp  # noqa: E402
from jht.healpix import adjoint_synthesis, alm_size, synthesis  # noqa: E402

ALG_TOL = 1e-12  # a-priori, as tests/test_grad.py
NSIDE, LMAX, NSIM = 16, 24, 4


def _rand_alm(rng, spin, n=None):
    shape = (alm_size(LMAX),) if spin == 0 else (2, alm_size(LMAX))
    if n is not None:
        shape = (n, *shape)
    return jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def _rand_map(rng, spin):
    npix = 12 * NSIDE**2
    return jnp.asarray(rng.standard_normal(npix if spin == 0 else (2, npix)))


def _rel(x, ref):
    return float(jnp.max(jnp.abs(x - ref)) / jnp.max(jnp.abs(ref)))


# --------------------------------------------------------------------------- #
# the wrappers are native AD, numerically
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spin", [0, 2])
def test_synthesis_vjp_matches_native(spin):
    rng = np.random.default_rng(100 + spin)
    a, v = _rand_alm(rng, spin), _rand_map(rng, spin)
    y_w, vjp_w = jax.vjp(lambda z: synthesis_vjp(z, NSIDE, LMAX, spin), a)
    y_n, vjp_n = jax.vjp(lambda z: synthesis(z, NSIDE, LMAX, spin), a)
    assert np.array_equal(np.asarray(y_w), np.asarray(y_n))
    assert _rel(vjp_w(v)[0], vjp_n(v)[0]) <= ALG_TOL


@pytest.mark.parametrize("spin", [0, 2])
def test_adjoint_synthesis_vjp_matches_native(spin):
    rng = np.random.default_rng(110 + spin)
    m, c = _rand_map(rng, spin), _rand_alm(rng, spin)
    y_w, vjp_w = jax.vjp(lambda z: adjoint_synthesis_vjp(z, NSIDE, LMAX, spin), m)
    y_n, vjp_n = jax.vjp(lambda z: adjoint_synthesis(z, NSIDE, LMAX, spin), m)
    assert np.array_equal(np.asarray(y_w), np.asarray(y_n))
    assert _rel(vjp_w(c)[0], vjp_n(c)[0]) <= ALG_TOL


# --------------------------------------------------------------------------- #
# forward mode: blocked on the wrappers, intact on the (checkpointed) default path
# --------------------------------------------------------------------------- #
def test_vjp_wrappers_block_forward_mode():
    rng = np.random.default_rng(120)
    a, m = _rand_alm(rng, 0), _rand_map(rng, 0)
    with pytest.raises(TypeError):
        jax.jvp(lambda z: synthesis_vjp(z, NSIDE, LMAX, 0), (a,), (a,))
    with pytest.raises(TypeError):
        jax.jvp(lambda z: adjoint_synthesis_vjp(z, NSIDE, LMAX, 0), (m,), (m,))


@pytest.mark.parametrize("spin", [0, 2])
def test_native_forward_mode_survives_checkpoint(spin):
    rng = np.random.default_rng(130 + spin)
    a, t = _rand_alm(rng, spin), _rand_alm(rng, spin)
    _, tangent = jax.jvp(lambda z: synthesis(z, NSIDE, LMAX, spin), (a,), (t,))
    assert _rel(tangent, synthesis(t, NSIDE, LMAX, spin)) <= ALG_TOL  # linear map


# --------------------------------------------------------------------------- #
# memory gates: (lmax+1, lmax+1, 2 nside) recursion tables in the gradient graph
# --------------------------------------------------------------------------- #
def _table_count(fn, *args) -> int:
    """Distinct HLO values shaped ``(lmax+1, lmax+1, 2 nside)`` in the compiled graph."""
    pat = re.compile(rf"(c128|f64)\[{LMAX + 1},{LMAX + 1},{2 * NSIDE}\]")
    txt = jax.jit(fn).lower(*args).compile().as_text()
    names = set()
    for line in txt.splitlines():
        if pat.search(line):
            m = re.match(r"\s*%?([A-Za-z_0-9.-]+) =", line)
            if m:
                names.add(m.group(1))
    return len(names)


def _synth_loss(transform):
    return lambda a: jnp.sum(transform(a, NSIDE, LMAX, 2) ** 2)


def _scan_loss(transform):
    # the knob enters after the transform, which is what invites JAX to hoist it
    def loss(s, A):
        return jnp.sum(jax.lax.map(lambda a: jnp.sum((transform(a, NSIDE, LMAX, 2) * s) ** 2), A))

    return loss


def test_synthesis_vjp_grad_keeps_no_recursion_table():
    a = _rand_alm(np.random.default_rng(140), 2)
    loss = _synth_loss(synthesis_vjp)
    assert _table_count(lambda z: jax.grad(loss)(z), a) == 0


def test_adjoint_synthesis_vjp_grad_keeps_no_recursion_table():
    m = _rand_map(np.random.default_rng(141), 2)
    loss = lambda z: jnp.sum(jnp.abs(adjoint_synthesis_vjp(z, NSIDE, LMAX, 2)) ** 2)  # noqa: E731
    assert _table_count(lambda z: jax.grad(loss)(z), m) == 0


def test_synthesis_vjp_in_sim_scan_hoists_no_recursion_table():
    A = _rand_alm(np.random.default_rng(142), 2, NSIM)
    loss = _scan_loss(synthesis_vjp)
    assert _table_count(lambda s, X: jax.grad(loss)(s, X), 1.0, A) == 0


def test_native_grad_in_sim_scan_hoists_no_recursion_table():
    """The default checkpoint stops a scanned body hoisting the recursion out of the loop.

    Measured at this fixture: 0 with the checkpoint, 44 with ``healpix._opaque`` removed
    (the wrapper scan gate above also reads 44 without it)."""
    A = _rand_alm(np.random.default_rng(143), 2, NSIM)
    loss = _scan_loss(synthesis)
    assert _table_count(lambda s, X: jax.grad(loss)(s, X), 1.0, A) == 0


def test_native_grad_wrt_alm_keeps_recursion_tape():
    """Native AD w.r.t. the alm does keep tables: the zero counts above come from a live
    pattern, and the wrappers' zero is the custom_vjp's doing.

    Measured at this fixture: 36, with or without the per-kernel checkpoint."""
    a = _rand_alm(np.random.default_rng(144), 2)
    loss = _synth_loss(synthesis)
    assert _table_count(lambda z: jax.grad(loss)(z), a) > 0
