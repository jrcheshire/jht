"""Locate the off-grid *forward* synthesis GPU phantom by decomposing it (+ test a fix).

`jht.synthesis_general` takes ~35 s at lmax=1000 on GPU and the cost is
*independent of N* (the number of evaluation points) -- so it is NOT the O(N) NUFFT
interpolation, it is one of the N-independent stages.  The full forward is

    _channels -> synth_contract(+spin) -> synth_contract(-spin) -> _coeffs_to_Fkm
      -> nufft2d2[ C.at[].set (grid build) -> ifft2 -> stencil gather+sum ]

This script times every stage so they *sum* to the real forward and the phantom is
unambiguous.  The leading suspect (by analogy to the on-grid adjoint `dense_to_tri`
fix) is the fp64/complex128 SCATTER `C = C.at[kbins, mbins].set(c)` in `nufft2d2`;
so it also times a precomputed-inverse-index GATHER candidate for that grid build
(`cset_gath`) and asserts it is bit-identical to the scatter.

Columns (ms):  syn  chan  sc_pos  sc_neg  c2f  cset  ifft2  gath_sum  | sum | cset_gath
  -> whichever N-independent stage carries the phantom is the rewrite target;
     cset_gath shows whether replacing the grid-build scatter with a gather fixes it.
  -> the N-independent columns should be ~constant as npts grows (re-confirms the
     35 s is N-independent); only `syn` and `gath_sum` should move with npts.

GPU-only pathology.  Run on a SLURM GPU cluster:
    pixi run -e gpu python scripts/profile_offgrid_forward.py [--spin 0|2] \
        [--lmax 1000] [--npts 200000,1000000] [--dtype fp32]
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp


def _timed(fn, n: int = 3) -> float:
    jax.block_until_ready(fn())
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts)


def _safe(fn) -> float | str:
    try:
        return _timed(fn) * 1e3
    except Exception as exc:  # noqa: BLE001 -- per-stage in-band failure reporting
        low = f"{type(exc).__name__}: {exc}".lower()
        if "ptxas" in low:
            return "ptxas-FAIL"
        if "resource_exhausted" in low or "out of memory" in low:
            return "OOM"
        return f"ERR:{type(exc).__name__}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp64", "fp32"], default="fp64")
    ap.add_argument("--spin", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--lmax", type=int, default=1000)
    ap.add_argument(
        "--npts", default="200000,1000000", help="comma list; re-confirms N-independence"
    )
    ap.add_argument("--epsilon", type=float, default=1e-10)
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", args.dtype == "fp64")

    import numpy as np  # noqa: E402

    import jht  # noqa: E402
    from jht._nufft import _stencil  # noqa: E402
    from jht._recursion import synth_contract  # noqa: E402
    from jht.healpix import alm_column_base, alm_size  # noqa: E402
    from jht.offgrid import _channels, _coeffs_to_Fkm, _prepare  # noqa: E402

    lmax, spin, eps = args.lmax, args.spin, args.epsilon
    dev = next((d for d in jax.devices() if d.platform == "gpu"), jax.devices("cpu")[0])
    print(
        f"profile_offgrid_forward  dtype={args.dtype}  spin={spin}  lmax={lmax}  "
        f"device={dev}  jax={jax.__version__}"
    )
    hdr = (
        f"{'npts':>8} {'syn':>9} {'chan':>7} {'sc_pos':>8} {'sc_neg':>8} {'c2f':>8} "
        f"{'cset':>9} {'ifft2':>8} {'gath_sum':>9} {'sum':>9} {'cset_gath':>9}  (ms)"
    )
    print(hdr + "\n" + "-" * len(hdr))
    print(
        "  sum = chan+sc_pos+sc_neg+c2f+cset+ifft2+gath_sum (~= syn locates the phantom);"
        "  cset = grid-build SCATTER, cset_gath = candidate gather fix"
    )

    rng = np.random.default_rng(0)
    M = lmax + 1
    K = alm_size(lmax)

    def rand_alm():
        a = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        a[:M] = a[:M].real
        for m in range(min(abs(spin), M)):
            base = alm_column_base(m, lmax)
            a[base : base + (abs(spin) - m)] = 0.0
        return a.astype(np.complex128)

    def cplx(shape):
        return jax.device_put((rng.standard_normal(shape) + 1j * rng.standard_normal(shape)))

    def fmt(x):
        return f"{x:.1f}" if isinstance(x, float) else f"{x}"

    with jax.default_device(dev):
        p = _prepare(lmax, spin, eps)
        Nk, Nm, nk, nm = p.nplan.Nk, p.nplan.Nm, p.nplan.nk, p.nplan.nm
        W = p.nplan.W
        x_cc = jnp.asarray(p.x_cc)

        # alm on device (real-constrained); shape (K,) spin-0 else (2, K)
        alm_np = rand_alm() if spin == 0 else np.stack([rand_alm(), rand_alm()])
        alm = jax.device_put(alm_np)

        # --- the full jitted forward (the reference total) ---
        syn = jax.jit(lambda a, L: jht.synthesis_general(a, L, spin=spin, lmax=lmax, epsilon=eps))

        # --- chan: alm -> (alpha, beta) dense channels ---
        chan = jax.jit(lambda a: _channels(p, a))

        # --- sc_pos / sc_neg: the two Legendre recursions (dense (M, lmax+1) input) ---
        a_dense = cplx((M, lmax + 1))
        b_dense = cplx((M, lmax + 1))
        sc_pos = jax.jit(lambda d: synth_contract(p.plan_pos, x_cc, d))
        sc_neg = jax.jit(lambda d: synth_contract(p.plan_neg, x_cc, d))

        # --- c2f: per-m coeffs -> DFS-extended F(k,m) ---
        Cpos = cplx((M, p.ntheta_s))
        Cneg = cplx((M, p.ntheta_s))
        c2f = jax.jit(lambda cp, cn: _coeffs_to_Fkm(p, cp, cn))

        # --- nufft2d2 sub-stages ---
        kbins = jnp.asarray(p.nplan.kbins)
        mbins = jnp.asarray(p.nplan.mbins)
        c_grid = cplx((Nk, Nm))  # the (corrected) centered modes fed to the grid build

        @jax.jit
        def cset(c):  # the current fp64/complex SCATTER (nufft2d2:136-137)
            C = jnp.zeros((nk, nm), dtype=jnp.complex128)
            return C.at[kbins[:, None], mbins[None, :]].set(c)

        # candidate GATHER fix: precompute the inverse fill index (host-side, static)
        kb, mb = np.asarray(p.nplan.kbins), np.asarray(p.nplan.mbins)
        dest = (kb[:, None] * nm + mb[None, :]).ravel()  # flat (nk,nm) destination cells
        assert np.unique(dest).size == Nk * Nm, "grid-build indices collide -> gather invalid"
        flat = np.full(nk * nm, -1, dtype=np.int64)
        flat[dest] = np.arange(Nk * Nm)
        filled_j = jnp.asarray(flat >= 0)
        src_j = jnp.asarray(np.where(flat >= 0, flat, 0))

        @jax.jit
        def cset_gath(c):
            return jnp.where(filled_j, c.ravel()[src_j], 0.0 + 0.0j).reshape(nk, nm)

        # bit-identity check of the candidate fix (host-side, once)
        eq = bool(jnp.array_equal(cset(c_grid), cset_gath(c_grid)))
        print(f"  cset_gath == cset (bit-identical): {eq}")

        ifft2 = jax.jit(lambda C: jnp.fft.ifft2(C) * (nk * nm))
        g_grid = cplx((nk, nm))

        @jax.jit
        def gath_sum(g, th, ph):  # the O(N) stencil gather + weighted sum (nufft2d2:139-142)
            lk, wk = _stencil(th, nk, W, p.nplan.beta_k)
            lm, wm = _stencil(ph, nm, W, p.nplan.beta_m)
            gg = g[lk[:, :, None], lm[:, None, :]]
            return jnp.sum(wk[:, :, None] * wm[:, None, :] * gg, axis=(1, 2))

        t_chan = _safe(lambda: chan(alm))
        t_scp = _safe(lambda: sc_pos(a_dense))
        t_scn = _safe(lambda: sc_neg(b_dense))
        t_c2f = _safe(lambda: c2f(Cpos, Cneg))
        t_cset = _safe(lambda: cset(c_grid))
        t_ifft = _safe(lambda: ifft2(g_grid))
        t_cg = _safe(lambda: cset_gath(c_grid))

        for npts in [int(x) for x in args.npts.split(",")]:
            loc_np = np.stack(
                [
                    rng.uniform(0.02, np.pi - 0.02, npts),
                    rng.uniform(0.0, 2.0 * np.pi, npts),
                ],
                axis=1,
            )
            loc = jax.device_put(loc_np)
            theta, phi = loc[:, 0], loc[:, 1]
            t_syn = _safe(lambda: syn(alm, loc))
            t_gs = _safe(lambda: gath_sum(g_grid, theta, phi))

            parts = [t_chan, t_scp, t_scn, t_c2f, t_cset, t_ifft, t_gs]
            s = sum(x for x in parts) if all(isinstance(x, float) for x in parts) else "-"
            print(
                f"{npts:>8} {fmt(t_syn):>9} {fmt(t_chan):>7} {fmt(t_scp):>8} {fmt(t_scn):>8} "
                f"{fmt(t_c2f):>8} {fmt(t_cset):>9} {fmt(t_ifft):>8} {fmt(t_gs):>9} "
                f"{fmt(s):>9} {fmt(t_cg):>9}",
                flush=True,
            )


if __name__ == "__main__":
    main()
