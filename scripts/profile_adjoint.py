"""Pin the on-grid adjoint GPU bottleneck to a specific op, before the rewrite.

Run 1 (profile_20148324) confirmed the cost is the adjoint's ring-assembly, not the
recursion (nside=512 fp64: adj 5906 ms, rec_adj 592 ms => assembly ~5314 ms; forward
assembly ~30 ms). This script sub-splits that assembly so the rewrite is targeted.

The adjoint ring-assembly (healpix._prepare's spin-0 `adj`, per ring-length group g):
    x  = m[g.pix_idx]                 # (n_g, N)  gather contiguous ring pixels from the map
    fr = fft(x, axis=1)               # per-ring FFT
    Vg = fr[:, g.g_plus] * phaseᵀ     # (n_g, M)  modular column fold/expand (m mod N) + phase
    V  = V.at[:, g.ring_idx].set(Vgᵀ) # scatter into V (M, nrings)

We time the real adjoint, the recursion alone (`adjoint_contract_eo`), and a faithful
reconstruction of the assembly under toggles, so the deltas isolate each op:
    asm        : full assembly  (should ~= adj - rec_adj -> validates the reconstruction)
    asm_belt   : assembly over the equatorial belt group only (one batched FFT)
    asm_cap    : assembly over the ~nside cap length-groups only (the unrolled part)
    no_fft     : assembly with the FFT skipped         -> FFT cost     ~= asm - no_fft
    no_scatter : assembly accumulating instead of V.set -> scatter cost ~= asm - no_scatter
                 (and gather+fold ~= no_scatter - FFT)

Default nside ladder is 256,512 -- small enough to compile fast and fit a 5 GB MIG
slice (nside>=1024 triggers the multi-minute compile / OOM we already saw, and the
signal is unambiguous at 512). Run on a GPU (the pathology is GPU-specific):
    pixi run -e gpu python scripts/profile_adjoint.py            # fp64
    pixi run -e gpu python scripts/profile_adjoint.py --dtype fp32
CPU-runnable for validation (no pathology there; assembly is cheap).
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

LADDER = {256: 384, 512: 768, 1024: 1000, 2048: 1000}


def _timed(fn, n: int = 3) -> float:
    """Best-of-n steady-state seconds (one untimed warmup captures compile)."""
    jax.block_until_ready(fn())
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts)


def _safe(fn) -> float | str:
    """ms, or a short error tag, so one stage can't abort the run."""
    try:
        return _timed(fn) * 1e3
    except Exception as exc:  # noqa: BLE001 -- per-stage in-band failure reporting
        low = f"{type(exc).__name__}: {exc}".lower()
        if "ptxas" in low:
            return "ptxas-FAIL"
        if "resource_exhausted" in low or "out of memory" in low:
            return "OOM"
        return f"ERR:{type(exc).__name__}"


def _build_assembler(groups, conj_phase, M: int, nrings: int, *, do_fft: bool, do_scatter: bool):
    """Faithful reconstruction of the spin-0 adjoint ring-assembly under op toggles."""
    cp = jnp.asarray(conj_phase)
    G = [(jnp.asarray(g.pix_idx), jnp.asarray(g.ring_idx), jnp.asarray(g.g_plus)) for g in groups]

    def _vg(m, pix, ring, gp):
        x = m[pix]
        fr = jnp.fft.fft(x, axis=1) if do_fft else x.astype(jnp.complex128)
        return fr[:, gp] * cp[:, ring].T  # (n_g, M)

    if do_scatter:

        def assemble(m):
            V = jnp.zeros((M, nrings), dtype=jnp.complex128)
            for pix, ring, gp in G:
                V = V.at[:, ring].set(_vg(m, pix, ring, gp).T, unique_indices=True)
            return V
    else:

        def assemble(m):
            acc = jnp.zeros((), dtype=jnp.complex128)
            for pix, ring, gp in G:
                acc = acc + _vg(m, pix, ring, gp).sum()
            return acc

    return jax.jit(assemble)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp64", "fp32"], default="fp64")
    ap.add_argument(
        "--nsides", default="256,512", help="comma-separated (>=1024 risks slow GPU compiles)"
    )
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", args.dtype == "fp64")

    import numpy as np  # noqa: E402

    import jht  # noqa: E402
    from jht.healpix import RingInfo, _ring_groups, alm_size  # noqa: E402
    from jht._recursion import adjoint_contract_eo, build_recursion_plan  # noqa: E402

    dev = next((d for d in jax.devices() if d.platform == "gpu"), jax.devices("cpu")[0])
    print(
        f"profile_adjoint  dtype={args.dtype}  device={dev}  "
        f"x64={args.dtype == 'fp64'}  jax={jax.__version__}"
    )
    hdr = (
        f"{'nside':>5} {'lmax':>5} {'adj':>9} {'rec_adj':>9} {'asm':>9} {'asm_belt':>9} "
        f"{'asm_cap':>9} {'no_fft':>9} {'no_scat':>9}  (ms)"
    )
    print(hdr + "\n" + "-" * len(hdr))
    print(
        "  reads: asm ~= adj-rec_adj (faithful);  FFT ~= asm-no_fft;  "
        "scatter ~= asm-no_scat;  gather+fold ~= no_scat-(asm-no_fft)"
    )

    rng = np.random.default_rng(0)

    def fmt(x):
        return f"{x:9.1f}" if isinstance(x, float) else f"{x:>9}"

    for nside in [int(x) for x in args.nsides.split(",")]:
        lmax = LADDER.get(nside, min(1000, int(1.5 * nside)))
        with jax.default_device(dev):
            a = rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))
            a[: lmax + 1] = a[: lmax + 1].real
            alm = jax.device_put(a.astype(np.complex128))
            geo = RingInfo(nside)
            t_half = 2 * nside
            plan = build_recursion_plan(geo.z[:t_half], 0, lmax)
            x_half = jnp.asarray(geo.z[:t_half])
            Vn = jax.device_put(
                (
                    rng.standard_normal((lmax + 1, t_half))
                    + 1j * rng.standard_normal((lmax + 1, t_half))
                ).astype(np.complex128)
            )
            Vs = jax.device_put(
                (
                    rng.standard_normal((lmax + 1, t_half))
                    + 1j * rng.standard_normal((lmax + 1, t_half))
                ).astype(np.complex128)
            )

            try:
                m0 = jax.device_put(np.asarray(jht.synthesis(alm, nside, lmax, 0)))
            except Exception:  # noqa: BLE001
                m0 = jax.device_put(np.zeros(geo.npix, dtype=np.float64))

            groups = _ring_groups(geo, lmax)
            belt = [g for g in groups if g.N == 4 * nside]
            cap = [g for g in groups if g.N != 4 * nside]
            m_pos = np.arange(lmax + 1)
            conj_phase = np.conj(np.exp(1j * m_pos[:, None] * geo.phi0[None, :]))
            M, nrings = lmax + 1, geo.nrings

            asm_all = _build_assembler(groups, conj_phase, M, nrings, do_fft=True, do_scatter=True)
            asm_belt = _build_assembler(belt, conj_phase, M, nrings, do_fft=True, do_scatter=True)
            asm_cap = _build_assembler(cap, conj_phase, M, nrings, do_fft=True, do_scatter=True)
            asm_nofft = _build_assembler(
                groups, conj_phase, M, nrings, do_fft=False, do_scatter=True
            )
            asm_noscat = _build_assembler(
                groups, conj_phase, M, nrings, do_fft=True, do_scatter=False
            )
            jadjc = jax.jit(lambda vn, vs: adjoint_contract_eo(plan, x_half, vn, vs))

            t_adj = _safe(lambda: jht.adjoint_synthesis(m0, nside, lmax, 0))
            t_radj = _safe(lambda: jadjc(Vn, Vs))
            t_asm = _safe(lambda: asm_all(m0))
            t_belt = _safe(lambda: asm_belt(m0))
            t_cap = _safe(lambda: asm_cap(m0))
            t_nofft = _safe(lambda: asm_nofft(m0))
            t_noscat = _safe(lambda: asm_noscat(m0))

        print(
            f"{nside:>5} {lmax:>5} {fmt(t_adj)} {fmt(t_radj)} {fmt(t_asm)} {fmt(t_belt)} "
            f"{fmt(t_cap)} {fmt(t_nofft)} {fmt(t_noscat)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
