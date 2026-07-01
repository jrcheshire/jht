"""Characterize + decompose the nside>=1024 on-grid COMPILE time (v0.1.1 item 2).

The nside=2048 ptxas-FAIL was fixed (combined-gather de-unroll, commit 83993fa), but
the *compile* is still multi-minute at nside>=1024.  Hypothesis: the per-ring-length
FFT unroll -- `synth`'s `for g in groups:` loop compiles one ifft kernel per distinct
HEALPix ring length (caps 4, 8, ..., 4*nside; the equatorial belt is one batched
group), so ~nside FFT kernels.  The per-length FFTs are unavoidable (a length-N ring
needs an exact length-N FFT; they cannot be padded to a common length), but the
*number of distinct compilations* is the suspected cost.  This probe measures HOW the
compile scales with nside and WHICH half -- the Legendre recursion or the FFT-unroll
assembly -- dominates, so the fix can target the real cost (split/cache the
sub-compiles, persistent compilation cache, or a single batched DFT matmul for the
short cap rings -- exact, O(sum N^2), one kernel instead of many tiny FFTs).

Per nside it reports:
  * structure : n_groups (= distinct ring lengths = the FFT-kernel/unroll count),
    short (N<=--short-thresh, the DFT-matmul-fallback candidates), belt (N x n_rings).
  * synth_c/synth_r : compile(+first run) / steady run of the REAL jht.synthesis.
  * asm_c /asm_r    : the FFT-unroll assembly alone (synth's `for g` loop on a
    precomputed G) -- the suspected compile dominator.
  * rec_c /rec_r    : the Legendre recursion alone (synth_contract_eo on a precomputed
    dense alm) -- the other half; a single scan, NOT unrolled, expected to compile fast.

`synth_c ~= rec_c + asm_c` attributes the compile (XLA fuses, so approximate).  In-band
ptxas/OOM reporting.  spin-0 (`synth`); spin-2 (`synth2`) is the same unroll, two
channels.  Run on a SLURM GPU cluster (a 20 GB MIG holds nside=2048):
    pixi run -e gpu python scripts/profile_compile_time.py [--nsides 256,512,1024,2048]
CPU compiles too (slower) -- fine for the scaling shape without a GPU slot; use
`--skip-synth` and small nsides for a quick local check.

The suspected fix is now SHIPPED as the opt-in looped/chirp-z azimuth mode
(`jht.set_azimuth_fft_mode("looped")`, :mod:`jht._azimuth`): it reroutes the ~nside
per-length cap FFTs through one common-length Bluestein `lax.scan` (belt kept native),
so the compiled FFT-kernel count drops from ~nside to O(1).  This probe now reports
`synth` compile+steady under BOTH modes (`(U)` unrolled default, `(L)` looped) so the
compile-size win and the per-run FLOP tax show side by side.
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


def _status(exc: Exception) -> str:
    low = f"{type(exc).__name__}: {exc}".lower()
    if "ptxas" in low:
        return "ptxas-FAIL"
    if "resource_exhausted" in low or "out of memory" in low:
        return "OOM"
    return f"ERR:{type(exc).__name__}"


def _compile_run(fn) -> tuple[float | str, float | str]:
    """(compile+first-run ms, steady-run ms); both carry the failure status on error."""
    try:
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        comp = (time.perf_counter() - t0) * 1e3
    except Exception as exc:  # noqa: BLE001 -- in-band compile-failure reporting is the point
        s = _status(exc)
        return s, s
    try:
        return comp, _timed(fn) * 1e3
    except Exception as exc:  # noqa: BLE001
        return comp, _status(exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp64", "fp32"], default="fp64")
    ap.add_argument("--nsides", default="256,512,1024,2048")
    ap.add_argument(
        "--short-thresh", type=int, default=64, help="cap rings with N<=this: DFT-matmul candidates"
    )
    ap.add_argument(
        "--skip-synth", action="store_true", help="skip the full jht.synthesis compile (slowest)"
    )
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", args.dtype == "fp64")

    import numpy as np  # noqa: E402

    import jht  # noqa: E402
    from jht import set_azimuth_fft_mode  # noqa: E402
    from jht._recursion import build_recursion_plan, synth_contract_eo  # noqa: E402
    from jht.healpix import RingInfo, _ring_groups, alm_size  # noqa: E402

    dev = next((d for d in jax.devices() if d.platform == "gpu"), jax.devices("cpu")[0])
    print(f"profile_compile_time  dtype={args.dtype}  device={dev}  jax={jax.__version__}")
    hdr = (
        f"{'nside':>5} {'lmax':>5} {'n_grp':>6} {'synthU_c':>9} {'synthU_r':>9} "
        f"{'synthL_c':>9} {'synthL_r':>9} {'asm_c':>9} {'rec_c':>9}  (ms)"
    )

    rng = np.random.default_rng(0)

    def fmt(x):
        return f"{x:.1f}" if isinstance(x, float) else f"{x}"

    def build(nside: int, lmax: int):
        """Return (groups, asm, rec, synth) with device-resident inputs captured."""
        geo = RingInfo(nside)
        M, nrings, npix = lmax + 1, geo.nrings, geo.npix
        t_half = 2 * nside
        groups = _ring_groups(geo, lmax)
        plan = build_recursion_plan(geo.z[:t_half], 0, lmax)
        x_half = jnp.asarray(geo.z[:t_half])

        Gp = [
            (int(g.N), jnp.asarray(g.ring_idx), jnp.asarray(g.k_plus), jnp.asarray(g.k_minus))
            for g in groups
        ]
        buf_pix = np.concatenate([g.pix_idx.ravel() for g in groups])  # (npix,) permutation
        pix_to_buf = np.empty(npix, dtype=np.int64)
        pix_to_buf[buf_pix] = np.arange(npix)
        pix_to_buf_j = jnp.asarray(pix_to_buf)

        G = jax.device_put(rng.standard_normal((M, nrings)) + 1j * rng.standard_normal((M, nrings)))
        dense = jax.device_put(rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M)))
        a = rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))
        a[:M] = a[:M].real
        alm = jax.device_put(a.astype(np.complex128))

        @jax.jit
        def asm(G):  # synth's FFT-unroll assembly: one length-N ifft per ring-length group
            cols = []
            for N, ring_idx, k_plus, k_minus in Gp:
                Cp = G[:, ring_idx]  # (M, n_g)
                D = jnp.zeros((ring_idx.size, N), dtype=jnp.complex128)
                D = D.at[:, k_plus].add(Cp.T)
                D = D.at[:, k_minus].add(jnp.conj(Cp[1:].T))
                cols.append(jnp.real(jnp.fft.ifft(D, axis=1) * N).ravel())  # (n_g*N,)
            return jnp.concatenate(cols)[pix_to_buf_j]  # one gather (the shipped de-unroll)

        rec = jax.jit(lambda d: synth_contract_eo(plan, x_half, d))

        return groups, lambda: asm(G), lambda: rec(dense), lambda: jht.synthesis(alm, nside, lmax, 0)

    print(hdr + "\n" + "-" * len(hdr))
    print("  *_c = compile(+first run), *_r = steady run.  U = unrolled (~nside kernels),")
    print("  L = looped/chirp-z (O(1) kernels).  synthU_c ~= rec_c + asm_c.")

    for nside in [int(x) for x in args.nsides.split(",")]:
        lmax = LADDER.get(nside, min(1000, int(1.5 * nside)))
        geo = RingInfo(nside)
        groups = _ring_groups(geo, lmax)
        n_groups = len(groups)
        lengths = sorted(int(g.N) for g in groups)
        n_short = sum(1 for g in groups if g.N <= args.short_thresh)
        belt = max(groups, key=lambda g: g.ring_idx.size)
        print(
            f"  nside={nside} lmax={lmax}: n_groups={n_groups} "
            f"(N {lengths[0]}..{lengths[-1]}), short(N<={args.short_thresh})={n_short}, "
            f"belt N={int(belt.N)} x {belt.ring_idx.size} rings",
            flush=True,
        )

        with jax.default_device(dev):
            _groups, asm_fn, rec_fn, synth_fn = build(nside, lmax)
            if args.skip_synth:
                suc = sur = slc = slr = "skip"
            else:
                set_azimuth_fft_mode("unrolled")
                suc, sur = _compile_run(synth_fn)
                set_azimuth_fft_mode("looped")
                slc, slr = _compile_run(synth_fn)
                set_azimuth_fft_mode("unrolled")
            ac, _ar = _compile_run(asm_fn)
            rc, _rr = _compile_run(rec_fn)
        print(
            f"{nside:>5} {lmax:>5} {n_groups:>6} {fmt(suc):>9} {fmt(sur):>9} "
            f"{fmt(slc):>9} {fmt(slr):>9} {fmt(ac):>9} {fmt(rc):>9}",
            flush=True,
        )


if __name__ == "__main__":
    main()
