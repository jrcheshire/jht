"""Profile-confirm the on-grid adjoint GPU bottleneck *before* the scatter refactor.

The GPU diagnostic showed the on-grid fp64 adjoint is ~10-25x slower than synthesis
(and map2alm rides on it), while the off-grid adjoint -- which shares the recursion
adjoint but NOT the per-ring-group cap-FFT scatter -- is fast. This isolates the two
stages of the on-grid adjoint to confirm the *ring assembly* (gather + per-group FFT
+ scatter into V), not the recursion adjoint, is the cost -- in fp64 AND fp32, and at
nside=2048 (where the full path ptxas-fails).

Stages timed (spin-0):
  synth      : full jht.synthesis            (the fast forward path, reference)
  adj        : full jht.adjoint_synthesis    (ring-assembly + recursion adjoint)
  rec_synth  : synth_contract_eo  alone      (forward recursion, no FFT assembly)
  rec_adj    : adjoint_contract_eo alone     (adjoint recursion, the off-grid-shared part)
  assembly   : adj - rec_adj                 (the per-ring-group gather/FFT/scatter)

Confirmation, if the hypothesis holds:
  * rec_adj is small and ~ rec_synth (recursion is symmetric and fine);
  * assembly (= adj - rec_adj) is large and carries the fp64/fp32 blowup;
  * the forward "assembly" (synth - rec_synth) is small (forward scatter is cheap);
  * at nside=2048, rec_adj/rec_synth COMPILE while synth/adj ptxas-fail -> the
    scatter loop is also the compile blocker.

The pathology is GPU-specific (on CPU all stages are comparable), so run on a GPU:
    pixi run -e gpu python scripts/profile_adjoint.py            # fp64
    pixi run -e gpu python scripts/profile_adjoint.py --dtype fp32
It runs on CPU too (validation; no pathology there).
"""

from __future__ import annotations

import argparse
import time

import jax

# nside -> lmax (BK ladder; matches the diagnostic)
LADDER = {512: 768, 1024: 1000, 2048: 1000}


def _timed(fn, n: int = 3) -> float:
    """Best-of-n steady-state seconds (one untimed warmup captures compile).

    Blocks via ``jax.block_until_ready`` so it handles pytree outputs (e.g.
    ``synth_contract_eo`` returns a ``(Ftot, Fsig)`` tuple)."""
    jax.block_until_ready(fn())
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return min(ts)


def _safe(fn) -> float | str:
    """ms, or a short error tag (e.g. ptxas failure / OOM) so one stage can't abort the run."""
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
    ap.add_argument("--nsides", default="512,1024,2048", help="comma-separated nside ladder")
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", args.dtype == "fp64")

    import jax.numpy as jnp  # noqa: E402
    import numpy as np  # noqa: E402

    import jht  # noqa: E402
    from jht.healpix import RingInfo, alm_size  # noqa: E402
    from jht._recursion import (  # noqa: E402
        adjoint_contract_eo,
        build_recursion_plan,
        synth_contract_eo,
    )

    dev = next((d for d in jax.devices() if d.platform == "gpu"), jax.devices("cpu")[0])
    print(
        f"profile_adjoint  dtype={args.dtype}  device={dev}  "
        f"x64={args.dtype == 'fp64'}  jax={jax.__version__}"
    )
    hdr = (
        f"{'nside':>5} {'lmax':>5} {'synth':>10} {'adj':>10} {'rec_synth':>10} "
        f"{'rec_adj':>10} {'assembly':>10}  (ms; assembly = adj - rec_adj)"
    )
    print(hdr + "\n" + "-" * len(hdr))

    rng = np.random.default_rng(0)

    def fmt(x):
        return f"{x:10.1f}" if isinstance(x, float) else f"{x:>10}"

    for nside in [int(x) for x in args.nsides.split(",")]:
        lmax = LADDER.get(nside, min(1000, int(1.5 * nside)))
        with jax.default_device(dev):
            # full-transform inputs
            a = rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))
            a[: lmax + 1] = a[: lmax + 1].real
            alm = jax.device_put(a.astype(np.complex128))
            # recursion-stage inputs (synthetic; timing depends on shape, not values)
            geo = RingInfo(nside)
            t_half = 2 * nside
            plan = build_recursion_plan(geo.z[:t_half], 0, lmax)
            x_half = jnp.asarray(geo.z[:t_half])
            dense = jax.device_put(
                (
                    rng.standard_normal((lmax + 1, lmax + 1))
                    + 1j * rng.standard_normal((lmax + 1, lmax + 1))
                ).astype(np.complex128)
            )
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

            t_synth = _safe(lambda: jht.synthesis(alm, nside, lmax, 0))
            # need a map for the full adjoint; if synthesis failed (e.g. nside=2048), reuse a zeros map
            try:
                m0 = jax.device_put(np.asarray(jht.synthesis(alm, nside, lmax, 0)))
            except Exception:  # noqa: BLE001
                m0 = jax.device_put(np.zeros(geo.npix, dtype=np.float64))
            t_adj = _safe(lambda: jht.adjoint_synthesis(m0, nside, lmax, 0))
            jsynthc = jax.jit(lambda d: synth_contract_eo(plan, x_half, d))
            jadjc = jax.jit(lambda vn, vs: adjoint_contract_eo(plan, x_half, vn, vs))
            t_rsynth = _safe(lambda: jsynthc(dense))
            t_radj = _safe(lambda: jadjc(Vn, Vs))

        assembly = t_adj - t_radj if isinstance(t_adj, float) and isinstance(t_radj, float) else "-"
        print(
            f"{nside:>5} {lmax:>5} {fmt(t_synth)} {fmt(t_adj)} {fmt(t_rsynth)} "
            f"{fmt(t_radj)} {fmt(assembly)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
