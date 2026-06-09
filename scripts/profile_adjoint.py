"""Locate the on-grid adjoint GPU phantom by fully decomposing it (+ test a fix).

Story so far:
  * Run 2 (profile_20168749): the adjoint ring-assembly (32 ms) and the recursion
    (310 ms) are each fast in isolation, but the full fp64 adjoint is ~4400 ms at
    nside=512 -- ~4050 ms of "phantom" cost, fp64-specific, scaling as lmax^2.
  * The optimization_barrier remat fix (run profile_20179402) did NOT change it ->
    not rematerialization.

The full spin-0 adjoint is:  asm (build V) -> fold_south(V) -> adjoint_contract_eo
(recursion) -> dense_to_tri.  Only `asm` and the recursion were timed before. This
script times ALL FOUR so they *sum* to the real adjoint and the phantom is
unambiguous, and -- since synthesis uses a GATHER (`tri_to_dense`) for alm packing
while the adjoint uses a SCATTER (`dense_to_tri`, `at[...].set(mode='drop')`), the
real forward/adjoint asymmetry and a classic slow-GPU-fp64-scatter -- it also times
a GATHER-based `dense_to_tri` as the candidate fix.

Columns (ms):  adj  asm  fold  rec_adj  d2t  | sum=asm+fold+rec_adj+d2t (~=adj)  | d2t_gather
  -> whichever of asm/fold/rec_adj/d2t carries the phantom is the rewrite target;
     d2t_gather shows whether replacing the packing scatter with a gather fixes it.

GPU-only pathology; default nside 256,512 (slice-safe). Run:
    pixi run -e gpu python scripts/profile_adjoint.py [--dtype fp32] [--nsides 256,512]
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

LADDER = {256: 384, 512: 768, 1024: 1000, 2048: 1000}


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
    ap.add_argument("--nsides", default="256,512", help=">=1024 risks slow GPU compiles")
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", args.dtype == "fp64")

    import numpy as np  # noqa: E402

    import jht  # noqa: E402
    from jht.healpix import RingInfo, _ring_groups, _tri_dense_maps, alm_size  # noqa: E402
    from jht._recursion import adjoint_contract_eo, build_recursion_plan  # noqa: E402

    dev = next((d for d in jax.devices() if d.platform == "gpu"), jax.devices("cpu")[0])
    print(f"profile_adjoint  dtype={args.dtype}  device={dev}  jax={jax.__version__}")
    hdr = (
        f"{'nside':>5} {'lmax':>5} {'adj':>9} {'asm':>8} {'fold':>8} {'rec_adj':>8} "
        f"{'d2t':>9} {'sum':>9} {'d2t_gath':>9}  (ms)"
    )
    print(hdr + "\n" + "-" * len(hdr))
    print(
        "  sum = asm+fold+rec_adj+d2t (~= adj locates the phantom);  d2t = packing SCATTER,"
        " d2t_gath = candidate gather fix"
    )

    rng = np.random.default_rng(0)

    def cplx(shape):
        return jax.device_put(
            (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)
        )

    def fmt(x):
        return f"{x:.1f}" if isinstance(x, float) else f"{x}"

    for nside in [int(x) for x in args.nsides.split(",")]:
        lmax = LADDER.get(nside, min(1000, int(1.5 * nside)))
        M, K = lmax + 1, alm_size(lmax)
        with jax.default_device(dev):
            geo = RingInfo(nside)
            t_half = 2 * nside
            nrings = geo.nrings
            plan = build_recursion_plan(geo.z[:t_half], 0, lmax)
            x_half = jnp.asarray(geo.z[:t_half])
            groups = _ring_groups(geo, lmax)
            *_, pack = _tri_dense_maps(lmax)  # (gather, valid, pack); pack = dense->tri gather idx

            a = rng.standard_normal(K) + 1j * rng.standard_normal(K)
            a[:M] = a[:M].real
            alm = jax.device_put(a.astype(np.complex128))
            try:
                m0 = jax.device_put(np.asarray(jht.synthesis(alm, nside, lmax, 0)))
            except Exception:  # noqa: BLE001
                m0 = jax.device_put(np.zeros(geo.npix, dtype=np.float64))
            Vn, Vs = cplx((M, t_half)), cplx((M, t_half))
            Vfull = cplx((M, nrings))
            b_dense = cplx((M, M))

            # --- asm: build V from the map (the ring assembly) ---
            cp = jnp.asarray(np.conj(np.exp(1j * np.arange(M)[:, None] * geo.phi0[None, :])))
            Gp = [
                (jnp.asarray(g.pix_idx), jnp.asarray(g.ring_idx), jnp.asarray(g.g_plus))
                for g in groups
            ]

            @jax.jit
            def asm(m, Gp=Gp, cp=cp):
                V = jnp.zeros((M, nrings), dtype=jnp.complex128)
                for pix, ring, gp in Gp:
                    V = V.at[:, ring].set(
                        (jnp.fft.fft(m[pix], axis=1)[:, gp] * cp[:, ring].T).T, unique_indices=True
                    )
                return V

            # --- fold_south ---
            south_src = np.arange(t_half - 1)
            south_tgt = 4 * nside - 2 - south_src
            sign_m = jnp.asarray(((-1.0) ** np.arange(M))[:, None])
            ss, st = jnp.asarray(south_src), jnp.asarray(south_tgt)

            @jax.jit
            def fold(V):
                Vs_ = (
                    jnp.zeros((M, t_half), dtype=jnp.complex128)
                    .at[:, ss]
                    .set(V[:, st], unique_indices=True)
                )
                return V[:, :t_half], sign_m * Vs_

            # --- recursion adjoint ---
            rec = jax.jit(lambda vn, vs: adjoint_contract_eo(plan, x_half, vn, vs))

            # --- dense_to_tri: the OLD scatter (rebuilt from pack) vs the gather now shipped ---
            scatter_flat = np.full(M * M, K, dtype=np.int64)
            scatter_flat[pack] = np.arange(K)  # inverse of pack: dense-flat -> tri (else drop)
            sj = jnp.asarray(scatter_flat)

            @jax.jit
            def d2t_scatter(bd):
                return jnp.zeros(K, dtype=jnp.complex128).at[sj].set(bd.ravel(), mode="drop")

            fj = jnp.asarray(pack)

            @jax.jit
            def d2t_gather(bd):
                return bd.ravel()[fj]

            t_adj = _safe(lambda: jht.adjoint_synthesis(m0, nside, lmax, 0))
            t_asm = _safe(lambda: asm(m0))
            t_fold = _safe(lambda: fold(Vfull))
            t_rec = _safe(lambda: rec(Vn, Vs))
            t_d2t = _safe(lambda: d2t_scatter(b_dense))
            t_d2g = _safe(lambda: d2t_gather(b_dense))

        parts = [t_asm, t_fold, t_rec, t_d2t]
        s = (
            sum(p for p in parts if isinstance(p, float))
            if all(isinstance(p, float) for p in parts)
            else "-"
        )
        print(
            f"{nside:>5} {lmax:>5} {fmt(t_adj):>9} {fmt(t_asm):>8} {fmt(t_fold):>8} "
            f"{fmt(t_rec):>8} {fmt(t_d2t):>9} {fmt(s):>9} {fmt(t_d2g):>9}",
            flush=True,
        )


if __name__ == "__main__":
    main()
