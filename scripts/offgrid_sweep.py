"""Off-grid NUFFT characterization: accuracy vs (epsilon) and timing/memory.

Exercises the production ``jht.synthesis_general`` / ``adjoint_synthesis_general``
(spin 0-3) against the exact direct sum (the perfect oracle) and ducc0 (if present),
across the kernel-DB epsilon tiers, plus a synthesis/adjoint timing pass.

Run:  pixi run python scripts/offgrid_sweep.py [--max LMAX] [--npts N]
"""

from __future__ import annotations

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jht._recursion import normalized_legendre, spin_weighted_lambda  # noqa: E402
from jht.healpix import alm_column_base, alm_size  # noqa: E402
from jht.offgrid import adjoint_synthesis_general, synthesis_general  # noqa: E402

TWO_PI = 2.0 * np.pi


def _rand_alm(lmax, seed, spin):
    rng = np.random.default_rng(seed)

    def one():
        a = rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))
        a[: lmax + 1] = a[: lmax + 1].real
        for m in range(min(abs(spin), lmax + 1)):
            base = alm_column_base(m, lmax)
            a[base : base + (abs(spin) - m)] = 0.0
        return a.astype(np.complex128)

    return one() if spin == 0 else np.stack([one(), one()])


def _direct(alm, loc, lmax, spin):
    th, ph = jnp.asarray(loc[:, 0]), jnp.asarray(loc[:, 1])
    x = jnp.cos(th)
    if spin == 0:
        f = jnp.zeros(x.shape)
        for m in range(lmax + 1):
            lam = normalized_legendre(x, m, lmax)
            base = alm_column_base(m, lmax)
            Fm = jnp.tensordot(jnp.asarray(alm)[base : base + (lmax - m + 1)], lam, ([0], [0]))
            f = f + (jnp.real(Fm) if m == 0 else 2.0 * jnp.real(Fm * jnp.exp(1j * m * ph)))
        return f
    aE, aB, s = jnp.asarray(alm[0]), jnp.asarray(alm[1]), (-1.0) ** spin
    qpiu = jnp.zeros(x.shape, dtype=jnp.complex128)
    for m in range(lmax + 1):
        off = max(m, spin) - m
        lp, lm = spin_weighted_lambda(x, m, spin, lmax), spin_weighted_lambda(x, m, -spin, lmax)
        base = alm_column_base(m, lmax)
        aEc, aBc = aE[base + off : base + (lmax - m + 1)], aB[base + off : base + (lmax - m + 1)]
        fp = jnp.tensordot(-(aEc + 1j * aBc), lp, ([0], [0]))
        fm = jnp.tensordot(-s * (aEc - 1j * aBc), lm, ([0], [0]))
        qpiu = qpiu + fp * jnp.exp(1j * m * ph)
        if m >= 1:
            qpiu = qpiu + jnp.conj(fm) * jnp.exp(-1j * m * ph)
    return jnp.stack([jnp.real(qpiu), jnp.imag(qpiu)])


def _ducc(alm, loc, spin, lmax, eps):
    import ducc0

    a2d = alm[None, :] if spin == 0 else alm
    out = ducc0.sht.experimental.synthesis_general(
        alm=a2d, loc=loc, spin=spin, lmax=lmax, mmax=lmax, epsilon=eps, nthreads=0
    )
    return np.asarray(out)[0] if spin == 0 else np.asarray(out)


def accuracy(lmax, npts):
    try:
        import ducc0  # noqa: F401

        have_ducc = True
    except ImportError:
        have_ducc = False
    rng = np.random.default_rng(0)
    loc = np.stack([rng.uniform(0.02, np.pi - 0.02, npts), rng.uniform(0, TWO_PI, npts)], 1)
    print(f"\naccuracy (lmax={lmax}, npts={npts}): max-abs vs exact direct sum"
          + ("  /  vs ducc0" if have_ducc else "  (ducc0 absent)"))
    for spin in (0, 1, 2, 3):
        alm = _rand_alm(lmax, lmax + spin, spin)
        ref = np.asarray(_direct(alm, loc, lmax, spin))
        row = [f"  spin={spin}"]
        for eps in (1e-6, 1e-8, 1e-10):
            f = np.asarray(synthesis_general(alm, loc, spin=spin, lmax=lmax, epsilon=eps))
            e_dir = np.max(np.abs(f - ref))
            tail = ""
            if have_ducc:
                d = _ducc(alm, loc, spin, lmax, eps)
                tail = f"/{np.max(np.abs(f - d)):.0e}"
            row.append(f"eps{eps:.0e}: {e_dir:.1e}{tail}")
        print("   ".join(row))


def timing(lmax, npts):
    print(f"\ntiming (lmax={lmax}, npts={npts}, eps=1e-10, jitted):")
    rng = np.random.default_rng(1)
    loc = jnp.asarray(np.stack([rng.uniform(0.02, np.pi - 0.02, npts), rng.uniform(0, TWO_PI, npts)], 1))
    for spin in (0, 2):
        alm = jnp.asarray(_rand_alm(lmax, spin, spin))
        syn = jax.jit(lambda a, L: synthesis_general(a, L, spin=spin, lmax=lmax))
        f = syn(alm, loc).block_until_ready()
        t0 = time.time()
        syn(alm, loc).block_until_ready()
        t_syn = time.time() - t0
        adj = jax.jit(lambda v, L: adjoint_synthesis_general(v, L, spin=spin, lmax=lmax))
        adj(f, loc).block_until_ready()
        t0 = time.time()
        adj(f, loc).block_until_ready()
        t_adj = time.time() - t0
        print(f"  spin={spin}: synthesis {1e3 * t_syn:6.0f} ms   adjoint {1e3 * t_adj:6.0f} ms")
    # the oversampled DFS grid is ~sigma^2 (2 lmax)^2 complex; ~0.5 GB at lmax=1000, sigma=2
    g = (2.0 ** 2) * (2 * lmax) ** 2 * 16 / 1e9
    print(f"  (oversampled grid ~{g:.2f} GB at sigma=2)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=256, help="lmax for the accuracy sweep")
    ap.add_argument("--npts", type=int, default=4000)
    args = ap.parse_args()
    accuracy(args.max, args.npts)
    timing(min(args.max, 512), 50_000)


if __name__ == "__main__":
    main()
