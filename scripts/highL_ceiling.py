"""Empirical compute ceiling of the on-grid forward transform at high nside / high l.

The recursion's fp64 accuracy is separately measured flat (~eps*sqrt(l)) to l=32000
(``scripts/exploratory/highL_recursion_growth.py`` vs mpmath), so the practical
ceiling is NOT the math -- it is the HEALPix geometry + the XLA compile (the per-ring
FFT-length unroll, ~nside distinct kernels) + memory.  This sweep finds where that
wall is: per (nside, lmax, spin) it measures, IN A CLEAN SUBPROCESS (clean peak RSS,
OOM/timeout survival), the compile(+first-run) time, steady run time, peak memory, the
ring-FFT-group count (the compile driver), and the correctness vs ducc0 AND healpy.

    pixi run python scripts/highL_ceiling.py                       # default ladder
    pixi run python scripts/highL_ceiling.py --points 4096:4000,8192:8000 --spins 0
    pixi run python scripts/highL_ceiling.py --timeout 2400        # per-point wall clock

Forward operator only (quadrature-free -> a clean recursion+assembly check vs an
independent C++ implementation); the weighted inverse / quadrature tier is separate.
Results -> runs/highL-ceiling/*.jsonl (gitignored).  Heavy: nside=8192 is ~6 GB/map
and a multi-minute compile.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Default (nside, lmax) ladder: lmax at ~nside (deep band) and near the 1.5*nside
# ceiling, climbing nside until the compile/memory wall.
DEFAULT_POINTS = "2048:2000,2048:3000,4096:4000,4096:6000,8192:8000,8192:12000"


# --------------------------------------------------------------------------- #
# worker: one (nside, lmax, spin) point, prints a single JSON line on stdout
# --------------------------------------------------------------------------- #
def _worker(nside: int, lmax: int, spin: int) -> None:
    import resource

    import jax

    jax.config.update("jax_enable_x64", True)

    import numpy as np

    import jht
    from jht.healpix import RingInfo, _ring_groups, alm_column_base, alm_size

    def peak_gb() -> float:
        m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return m / 1e9 if sys.platform == "darwin" else m / 1e6  # darwin: bytes, linux: KB

    def status(exc: Exception) -> str:
        low = f"{type(exc).__name__}: {exc}".lower()
        if "resource_exhausted" in low or "out of memory" in low or "memory" in low:
            return "OOM"
        if "ptxas" in low:
            return "ptxas-FAIL"
        return f"ERR:{type(exc).__name__}"

    def rand_alm(seed: int, lmin: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        a = (rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))) / np.sqrt(2)
        a[: lmax + 1] = a[: lmax + 1].real
        for m in range(min(lmin, lmax + 1)):
            b = alm_column_base(m, lmax)
            a[b : b + (lmin - m)] = 0.0
        return a.astype(np.complex128)

    n_groups = len(_ring_groups(RingInfo(nside), lmax))
    out = {"nside": nside, "lmax": lmax, "spin": spin, "n_groups": n_groups, "status": "OK"}

    try:
        if spin == 0:
            alm = rand_alm(0, 0)
            syn = lambda: jht.synthesis(alm, nside, lmax, spin=0)  # noqa: E731
        else:
            aE, aB = rand_alm(1, 2), rand_alm(2, 2)
            alm = np.stack([aE, aB])
            syn = lambda: jht.synthesis(alm, nside, lmax, spin=spin)  # noqa: E731

        t0 = time.perf_counter()
        m_jht = np.asarray(jax.block_until_ready(syn()))
        out["compile_s"] = round(time.perf_counter() - t0, 2)
        ts = []
        for _ in range(2):
            t1 = time.perf_counter()
            jax.block_until_ready(syn())
            ts.append(time.perf_counter() - t1)
        out["run_s"] = round(min(ts), 3)

        import ducc0

        info = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
        alm2d = alm[None, :] if spin == 0 else alm
        m_ducc = np.asarray(
            ducc0.sht.experimental.synthesis(
                alm=alm2d, lmax=lmax, spin=spin, mmax=lmax, nthreads=0, **info
            )
        )
        if spin == 0:
            m_ducc = m_ducc[0]
        scale = float(np.max(np.abs(m_ducc)))
        out["rel_ducc"] = float(np.max(np.abs(m_jht - m_ducc)) / scale)

        import healpy as hp

        if spin == 0:
            m_hp = hp.alm2map(alm, nside, lmax=lmax, mmax=lmax, pol=False)
        else:
            m_hp = np.asarray(hp.alm2map_spin([alm[0], alm[1]], nside, spin, lmax, mmax=lmax))
        out["rel_hp"] = float(np.max(np.abs(m_jht - m_hp)) / scale)
    except Exception as exc:  # noqa: BLE001 -- in-band failure reporting is the point
        out["status"] = status(exc)
        out["err"] = f"{type(exc).__name__}: {exc}"[:200]
    out["peak_gb"] = round(peak_gb(), 2)
    print("JSON " + json.dumps(out), flush=True)


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default=DEFAULT_POINTS, help="nside:lmax,nside:lmax,...")
    ap.add_argument("--spins", default="0,2")
    ap.add_argument("--timeout", type=int, default=2400, help="per-point wall-clock seconds")
    ap.add_argument("--worker", nargs=3, type=int, metavar=("NSIDE", "LMAX", "SPIN"))
    args = ap.parse_args()

    if args.worker:
        _worker(*args.worker)
        return

    points = [(int(n), int(lm)) for n, lm in (p.split(":") for p in args.points.split(","))]
    spins = [int(s) for s in args.spins.split(",")]
    outdir = Path("runs/highL-ceiling")
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl = outdir / f"sweep-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"

    hdr = (
        f"{'nside':>6} {'lmax':>6} {'spin':>4} {'n_grp':>6} {'compile_s':>10} "
        f"{'run_s':>8} {'peak_gb':>8} {'rel_ducc':>10} {'rel_hp':>10}  status"
    )
    print(f"highL_ceiling  per-point timeout={args.timeout}s  -> {jsonl}\n")
    print(hdr + "\n" + "-" * len(hdr))

    # spin-0 full ladder first (cheaper -> finds the wall), then spin-2.
    for spin in spins:
        for nside, lmax in points:
            t0 = time.perf_counter()
            try:
                r = subprocess.run(
                    [sys.executable, __file__, "--worker", str(nside), str(lmax), str(spin)],
                    capture_output=True, text=True, timeout=args.timeout,
                )
                line = next((ln for ln in r.stdout.splitlines() if ln.startswith("JSON ")), None)
                if line is None:
                    rec = {"nside": nside, "lmax": lmax, "spin": spin, "status": "NO-OUTPUT",
                           "err": (r.stderr or r.stdout)[-200:]}
                else:
                    rec = json.loads(line[5:])
            except subprocess.TimeoutExpired:
                rec = {"nside": nside, "lmax": lmax, "spin": spin,
                       "status": f"TIMEOUT>{args.timeout}s", "wall_s": round(time.perf_counter() - t0)}

            with jsonl.open("a") as f:
                f.write(json.dumps(rec) + "\n")

            def g(k, fmt="{}"):
                return fmt.format(rec[k]) if k in rec and rec[k] is not None else "--"

            print(
                f"{nside:>6} {lmax:>6} {spin:>4} {g('n_groups'):>6} {g('compile_s'):>10} "
                f"{g('run_s'):>8} {g('peak_gb'):>8} {g('rel_ducc','{:.2e}'):>10} "
                f"{g('rel_hp','{:.2e}'):>10}  {rec['status']}",
                flush=True,
            )
            if rec["status"].startswith("TIMEOUT") or rec["status"] == "OOM":
                print(f"  -> wall hit at nside={nside} lmax={lmax} spin={spin} "
                      f"({rec['status']}); higher points for this spin will likely also fail.")


if __name__ == "__main__":
    main()
