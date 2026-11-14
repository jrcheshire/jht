"""GPU parity gate: the transforms produce the same fp64 result on GPU and CPU.

jht is pure JAX (the static recursion plan is host-side numpy; the jitted
contraction + ring FFTs are device-agnostic), so a CUDA GPU must reproduce the
CPU result to ~machine precision -- it is the same XLA program. This module makes
that a suite gate; it **skips cleanly when no GPU is visible** (osx-arm64 dev,
CPU CI), and runs for real under ``pixi run -e gpu test`` on an NVIDIA box.

Parity mechanism: ``jht.healpix._prepare`` is ``lru_cache``-d (its large tables are
built inside each kernel's trace), so we clear the caches and rebuild under
``jax.default_device(dev)`` to genuinely run on each device.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import jht  # noqa: E402
from jht import _analysis  # noqa: E402  (the analysis impl module, for _wvec cache)
from jht import healpix as _healpix  # noqa: E402


def _gpu():
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        return None


pytestmark = pytest.mark.skipif(_gpu() is None, reason="no GPU visible (CPU-only host)")

TOL = 1e-12  # a-priori: identical XLA program in fp64, only device differs


def _clear():
    _healpix._prepare.cache_clear()
    _analysis._wvec.cache_clear()


def _rand_alm(lmax, rng, lmin=0):
    a = (rng.standard_normal(jht.alm_size(lmax)) + 1j * rng.standard_normal(jht.alm_size(lmax))) / np.sqrt(2)
    a[: lmax + 1] = a[: lmax + 1].real
    a[:lmin] = 0.0
    return a.astype(np.complex128)


def _on(dev, fn):
    _clear()
    with jax.default_device(dev):
        return np.asarray(fn())


def _relerr(a, b):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(np.max(np.abs(np.asarray(b))), 1e-300)


@pytest.mark.parametrize("nside,lmax", [(16, 24), (32, 48)])
@pytest.mark.parametrize("spin", [0, 2])
def test_synthesis_gpu_matches_cpu(nside, lmax, spin):
    cpu, gpu = jax.devices("cpu")[0], _gpu()
    rng = np.random.default_rng(nside + spin)
    alm = _rand_alm(lmax, rng) if spin == 0 else np.stack([_rand_alm(lmax, rng, 2), _rand_alm(lmax, rng, 2)])
    g = _on(gpu, lambda: jht.synthesis(jax.device_put(alm, gpu), nside, lmax, spin))
    c = _on(cpu, lambda: jht.synthesis(jax.device_put(alm, cpu), nside, lmax, spin))
    assert _relerr(g, c) <= TOL


@pytest.mark.parametrize("nside,lmax", [(16, 24), (32, 48)])
@pytest.mark.parametrize("spin", [0, 2])
def test_map2alm_gpu_matches_cpu(nside, lmax, spin):
    cpu, gpu = jax.devices("cpu")[0], _gpu()
    rng = np.random.default_rng(nside + spin + 7)
    alm = _rand_alm(lmax, rng) if spin == 0 else np.stack([_rand_alm(lmax, rng, 2), _rand_alm(lmax, rng, 2)])
    _clear()
    m0 = np.asarray(jht.synthesis(alm, nside, lmax, spin))
    g = _on(gpu, lambda: jht.map2alm(jax.device_put(m0, gpu), nside, lmax, spin, niter=3))
    c = _on(cpu, lambda: jht.map2alm(jax.device_put(m0, cpu), nside, lmax, spin, niter=3))
    assert _relerr(g, c) <= TOL


# --------------------------------------------------------------------------- #
# off-grid (NUFFT): per-point window spread, prepared points, grouped spread
# --------------------------------------------------------------------------- #
def _offgrid_loc(lmax, n, seed):
    """Uniform points, a dense clump (grouped spreader), grid nodes and the poles."""
    rng = np.random.default_rng(seed)
    nk, nm = jht.offgrid_grid_shape(lmax)
    uni = np.stack([np.arccos(rng.uniform(-1, 1, n)), rng.uniform(0, 2 * np.pi, n)], 1)
    clump = np.stack([1.1 + 0.004 * rng.random(n), 4.0 + 0.004 * rng.random(n)], 1)
    nodes = np.stack([2 * np.pi * rng.integers(1, nk // 2, n // 4) / nk,
                      2 * np.pi * rng.integers(0, nm, n // 4) / nm], 1)
    return np.concatenate([uni, clump, nodes, [[0.0, 0.0], [np.pi, 0.0]]])


def _offgrid_alm(lmax, spin, rng):
    return _rand_alm(lmax, rng) if spin == 0 else np.stack([_rand_alm(lmax, rng, 2), _rand_alm(lmax, rng, 2)])


@pytest.mark.parametrize("spin", [0, 2])
def test_offgrid_gpu_matches_cpu(spin):
    cpu, gpu = jax.devices("cpu")[0], _gpu()
    lmax = 48
    rng = np.random.default_rng(40 + spin)
    loc = _offgrid_loc(lmax, 2000, 41)
    alm = _offgrid_alm(lmax, spin, rng)
    v = rng.standard_normal(loc.shape[0]) if spin == 0 else rng.standard_normal((2, loc.shape[0]))

    def fwd(dev):
        return _on(dev, lambda: jax.jit(lambda a, x: jht.synthesis_general(a, x, spin=spin, lmax=lmax))(
            jax.device_put(alm, dev), jax.device_put(loc, dev)))

    def adj(dev):
        return _on(dev, lambda: jax.jit(lambda u, x: jht.adjoint_synthesis_general(u, x, spin=spin, lmax=lmax))(
            jax.device_put(v, dev), jax.device_put(loc, dev)))

    assert _relerr(fwd(gpu), fwd(cpu)) <= TOL
    assert _relerr(adj(gpu), adj(cpu)) <= TOL


@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("store_weights", [False, True])
def test_prepared_points_gpu_matches_cpu(spin, store_weights):
    cpu, gpu = jax.devices("cpu")[0], _gpu()
    lmax = 48
    rng = np.random.default_rng(50 + spin)
    loc = _offgrid_loc(lmax, 2000, 51)
    alm = _offgrid_alm(lmax, spin, rng)
    v = rng.standard_normal(loc.shape[0]) if spin == 0 else rng.standard_normal((2, loc.shape[0]))
    lc = None if store_weights else loc

    def fwd(a, p, x):
        return jht.grid_to_points(jht.sky_to_grid(a, spin=spin, lmax=lmax), p, x, spin=spin)

    def adj(u, p, x):
        return jht.grid_to_sky(jht.points_to_grid(u, p, x, spin=spin), spin=spin, lmax=lmax)

    def run(dev, f, data):
        def call():
            pts = jax.device_put(jht.prepare_points(loc, lmax=lmax, store_weights=store_weights), dev)
            x = None if lc is None else jax.device_put(lc, dev)
            return jax.jit(f)(jax.device_put(data, dev), pts, x)
        return _on(dev, call)

    assert _relerr(run(gpu, fwd, alm), run(cpu, fwd, alm)) <= TOL
    assert _relerr(run(gpu, adj, v), run(cpu, adj, v)) <= TOL
