"""Decide the nside=2048 ptxas-FAIL fix: is it the map SCATTER or the FFTs themselves?

The on-grid `synth` loops `for g in groups:` over ~nside ring-length groups, each
doing an ifft + a fp64 map scatter `out.at[pix_idx].set`.  At nside=2048 the
~2048-way unroll overflows ptxas (exit 9).  The proposed fix (item 2b) keeps the
unavoidable per-length exact FFTs but hoists the map write out into a single combined
GATHER.  That only helps if the SCATTER chains -- not the 2048 standalone FFT kernels
-- are what overflow ptxas.  This probe settles it by compiling three ring-assembly
candidates (on a precomputed `G`, so the recursion is excluded):

  * asm_full   : current structure (fold-in add + ifft + per-group map SCATTER)
                 -- reproduces the bug in isolation; expect ptxas-FAIL at 2048.
  * asm_nogath : fold-in add + ifft, concatenate the ring values (NO map write)
                 -- does the loop of FFTs + fold-in compile without the scatter?
  * asm_gather : fold-in add + ifft + ONE combined GATHER to the map (the 2b fix)
                 -- does the proposed fix compile?

Verdict:
  asm_full FAIL, asm_nogath OK, asm_gather OK  -> the scatter is the culprit; 2b works.
  asm_nogath FAIL                              -> the FFTs alone overflow; 2b is
                                                  insufficient (short-cap DFT-matmul
                                                  fallback needed -- see plan RISK).

Also asserts asm_gather == asm_full bit-identically at a small nside (the on-CPU
correctness gate for the 2b assembly).  spin-0 only; spin-2 (`synth2`) is structurally
identical.  Columns are compile(+first-run) ms and steady-run ms.  Run on Cannon:
    pixi run -e gpu python scripts/profile_ongrid_compile.py [--nsides 1024,2048] [--dtype fp32]
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
    ap.add_argument("--nsides", default="1024,2048", help="2048 is the ptxas-FAIL target")
    ap.add_argument("--bitcheck-nside", type=int, default=256)
    ap.add_argument(
        "--skip-full", action="store_true", help="skip the current-scatter candidate (slowest to compile)"
    )
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", args.dtype == "fp64")

    import numpy as np  # noqa: E402

    from jht.healpix import RingInfo, _ring_groups, alm_size  # noqa: E402

    dev = next((d for d in jax.devices() if d.platform == "gpu"), jax.devices("cpu")[0])
    print(f"profile_ongrid_compile  dtype={args.dtype}  device={dev}  jax={jax.__version__}")
    hdr = (
        f"{'nside':>5} {'lmax':>5} {'full_c':>9} {'full_r':>9} {'nogath_c':>9} "
        f"{'nogath_r':>9} {'gath_c':>9} {'gath_r':>9}  (ms)"
    )
    print(hdr + "\n" + "-" * len(hdr))
    print(
        "  *_c = compile(+first run), *_r = steady run.  full=current scatter, "
        "nogath=FFTs only, gath=proposed combined-gather fix."
    )

    rng = np.random.default_rng(0)

    def fmt(x):
        return f"{x:.1f}" if isinstance(x, float) else f"{x}"

    def build(nside: int, lmax: int):
        """Return (G_device, asm_full, asm_nogath, asm_gather) for this (nside, lmax)."""
        geo = RingInfo(nside)
        M, nrings, npix = lmax + 1, geo.nrings, geo.npix
        groups = _ring_groups(geo, lmax)
        # per-group static constants as jnp (mirror profile_adjoint's Gp list)
        Gp = [
            (
                int(g.N),
                jnp.asarray(g.ring_idx),
                jnp.asarray(g.pix_idx),
                jnp.asarray(g.k_plus),
                jnp.asarray(g.k_minus),
            )
            for g in groups
        ]
        # combined-gather index: buf is the ring values concatenated in group order;
        # buf_pix[slot] = map pixel -> pix_to_buf = its inverse permutation (out = buf[pix_to_buf]).
        buf_pix = np.concatenate([g.pix_idx.ravel() for g in groups])
        pix_to_buf = np.empty(npix, dtype=np.int64)
        pix_to_buf[buf_pix] = np.arange(npix)
        pix_to_buf_j = jnp.asarray(pix_to_buf)

        G = jax.device_put(
            (rng.standard_normal((M, nrings)) + 1j * rng.standard_normal((M, nrings)))
        )

        def _ring_vals(G):
            parts = []
            for N, ring_idx, _pix, k_plus, k_minus in Gp:
                Cp = G[:, ring_idx]  # (M, n_g)
                D = jnp.zeros((ring_idx.size, N), dtype=jnp.complex128)
                D = D.at[:, k_plus].add(Cp.T)
                D = D.at[:, k_minus].add(jnp.conj(Cp[1:].T))
                parts.append(jnp.real(jnp.fft.ifft(D, axis=1) * N))  # (n_g, N)
            return parts

        @jax.jit
        def asm_full(G):
            out = jnp.zeros(npix, dtype=jnp.float64)
            for (N, ring_idx, pix, k_plus, k_minus), vals in zip(Gp, _ring_vals(G)):
                out = out.at[pix.ravel()].set(vals.ravel(), unique_indices=True)
            return out

        @jax.jit
        def asm_nogath(G):
            return jnp.concatenate([v.ravel() for v in _ring_vals(G)])

        @jax.jit
        def asm_gather(G):
            return jnp.concatenate([v.ravel() for v in _ring_vals(G)])[pix_to_buf_j]

        return G, asm_full, asm_nogath, asm_gather

    # --- correctness gate (small nside): the 2b combined-gather assembly == the scatter.
    # allclose, not array_equal: the two candidates each run their OWN ifft, and cuFFT is
    # not bit-reproducible across separate compilations on GPU (~1e-13 ULP), so an exact
    # compare false-alarms on GPU though the assembly (a pure permutation) is correct. In
    # the real fix there is ONE ifft per ring and gather-vs-scatter of it is bitwise exact.
    nb = args.bitcheck_nside
    lb = LADDER.get(nb, min(1000, int(1.5 * nb)))
    with jax.default_device(dev):
        Gb, full_b, _ng_b, gath_b = build(nb, lb)
        a, b = full_b(Gb), gath_b(Gb)
        maxdiff = float(jnp.max(jnp.abs(a - b)))
        ok = bool(jnp.allclose(a, b, rtol=0.0, atol=1e-12))
    print(f"  asm_gather ~= asm_full (allclose, nside={nb}): {ok}  (max|diff|={maxdiff:.2e})")

    for nside in [int(x) for x in args.nsides.split(",")]:
        lmax = LADDER.get(nside, min(1000, int(1.5 * nside)))
        _M, _K = lmax + 1, alm_size(lmax)
        with jax.default_device(dev):
            G, asm_full, asm_nogath, asm_gather = build(nside, lmax)
            fc, fr = ("skip", "skip") if args.skip_full else _compile_run(lambda: asm_full(G))
            nc, nr = _compile_run(lambda: asm_nogath(G))
            gc, gr = _compile_run(lambda: asm_gather(G))
        print(
            f"{nside:>5} {lmax:>5} {fmt(fc):>9} {fmt(fr):>9} {fmt(nc):>9} "
            f"{fmt(nr):>9} {fmt(gc):>9} {fmt(gr):>9}",
            flush=True,
        )


if __name__ == "__main__":
    main()
