"""Off-grid NUFFT stage timings: ns per point, cores used, peak memory.

One configuration per process, so the peak RSS reported is that configuration's own::

    pixi run python scripts/bench_offgrid_spread.py --points patch --n 65536
    pixi run python scripts/bench_offgrid_spread.py --sweep      # standard table, one process at a time

Stages timed (all compiled, median of ``--repeat`` calls after one warm-up):

* ``legendre``: ``alm -> F(k,m)`` and its transpose; independent of the points.
* ``gather``: the forward stencil step, values out of a grid at the points.
* ``scatter-0.4.1``: the transposed stencil step as in jaxht 0.4.1 (per-cell
  ``.at[].add``), kept here as the fixed reference.
* ``spread``: the library's transposed stencil step (real values for spin 0).
* ``grouped`` / ``interp-prepared``: ``points_to_grid`` / ``grid_to_points`` on a set
  prepared with ``--group-size`` (add ``stored`` to the stages for stored weights);
  the host-side ``prepare_points`` time is printed once.
* ``adjoint`` / ``synthesis``: the public end-to-end transforms.

Stencil steps take positions and values and compute the kernel weights inside the timed
call. Point sets: ``patch`` (a dense 27 x 8 degree scan-like patch, ~100+ points per grid
cell at 2^18) and ``sphere`` (uniform over the sphere, ~1 point per cell at 2^18).

Memory: the 0.4.1 stencil steps hold ``(N, W, W)`` complex arrays, ~3-4 kB per point.
The script prints its estimate and refuses ``scatter-0.4.1`` above 2^18 points without
``--force``.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jht import _nufft, offgrid  # noqa: E402
from jht.healpix import alm_size  # noqa: E402

SCATTER_041_MAX = 1 << 18


def point_set(kind: str, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    if kind == "patch":
        th = np.repeat(np.deg2rad(147.5 + rng.uniform(-4, 4, 64)), n // 64)
        th = th + 1e-3 * rng.standard_normal(th.size)
        ph = (np.deg2rad(rng.uniform(0, 5, (64, 1)) + np.linspace(0, 27, n // 64)[None, :])).ravel()
    elif kind == "sphere":
        th, ph = np.arccos(rng.uniform(-1, 1, n)), rng.uniform(0, 2 * np.pi, n)
    else:
        raise ValueError(kind)
    return th, ph


def gather_041(plan, g, x, y):
    lk, wk = _nufft._stencil(x, plan.nk, plan.W, plan.beta_k)
    lm, wm = _nufft._stencil(y, plan.nm, plan.W, plan.beta_m)
    return jnp.sum(wk[:, :, None] * wm[:, None, :] * g[lk[:, :, None], lm[:, None, :]], axis=(1, 2))


def scatter_041(plan, v, x, y):
    lk, wk = _nufft._stencil(x, plan.nk, plan.W, plan.beta_k)
    lm, wm = _nufft._stencil(y, plan.nm, plan.W, plan.beta_m)
    contrib = (wk[:, :, None] * wm[:, None, :]) * v[:, None, None]
    g = jnp.zeros((plan.nk, plan.nm), dtype=jnp.complex128)
    return g.at[lk[:, :, None], lm[:, None, :]].add(contrib)


def timed(f, *args, repeat: int):
    jax.block_until_ready(f(*args))
    walls, cpus = [], []
    for _ in range(repeat):
        w0, c0 = time.perf_counter(), time.process_time()
        jax.block_until_ready(f(*args))
        walls.append(time.perf_counter() - w0)
        cpus.append(time.process_time() - c0)
    i = int(np.argsort(walls)[len(walls) // 2])
    return walls[i], cpus[i]


def peak_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6  # bytes on macOS, kB on Linux


def run_one(a) -> dict:
    p = offgrid._prepare(a.lmax, a.spin, a.eps)
    plan = p.nplan
    W = plan.W
    est = a.n * W * W * 16 * 2 / 1e9
    print(f"# {a.points} n={a.n} lmax={a.lmax} eps={a.eps:g} spin={a.spin} W={W} "
          f"grid={plan.nk}x{plan.nm}; 0.4.1 stencil step estimate ~{est:.1f} GB", flush=True)
    th, ph = point_set(a.points, a.n)
    x, y = jnp.asarray(th), jnp.asarray(ph)
    rng = np.random.default_rng(1)
    g = jnp.asarray(rng.standard_normal((plan.nk, plan.nm)) + 1j * rng.standard_normal((plan.nk, plan.nm)))
    if a.spin == 0:
        v = jnp.asarray(rng.standard_normal(a.n)).astype(jnp.complex128)
    else:
        v = jnp.asarray(rng.standard_normal(a.n) + 1j * rng.standard_normal(a.n))
    cells = np.unique(
        np.asarray(jnp.ceil(x * (plan.nk / _nufft.TWO_PI))).astype(np.int64) * plan.nm
        + np.asarray(jnp.ceil(y * (plan.nm / _nufft.TWO_PI))).astype(np.int64)
    ).size
    out = {"points": a.points, "n": a.n, "lmax": a.lmax, "eps": a.eps, "spin": a.spin,
           "W": W, "points_per_cell": a.n / cells, "stages": {}}

    def record(name, t, c, per_point=True):
        out["stages"][name] = {"ms": 1e3 * t, "ns_per_pt": 1e9 * t / a.n if per_point else None,
                               "cores": c / t, "peak_gb": peak_gb()}
        npp = f"{1e9 * t / a.n:7.1f} ns/pt" if per_point else " " * 14
        print(f"  {name:22s} {1e3 * t:9.1f} ms  {npp}  cores {c / t:4.1f}  peak {peak_gb():5.1f} GB",
              flush=True)

    stages = a.stages.split(",")
    if "legendre" in stages:
        alm = jnp.asarray(rng.standard_normal(alm_size(a.lmax)) + 0j)
        if a.spin:
            alm = jnp.stack([alm, alm])
        fkm = jax.jit(lambda al: offgrid._alm_to_Fkm(p, al))
        t, c = timed(fkm, alm, repeat=a.repeat)
        record("legendre fwd", t, c, per_point=False)
        G = fkm(alm)
        t, c = timed(jax.jit(lambda G: offgrid._Fkm_to_alm(p, G)), G, repeat=a.repeat)
        record("legendre transposed", t, c, per_point=False)
    if "gather" in stages:
        t, c = timed(jax.jit(lambda g, x, y: gather_041(plan, g, x, y)), g, x, y, repeat=a.repeat)
        record("gather-0.4.1", t, c)
    if "scatter-0.4.1" in stages:
        if a.n > SCATTER_041_MAX and not a.force:
            print(f"  scatter-0.4.1 skipped above {SCATTER_041_MAX} points (--force)", flush=True)
        else:
            t, c = timed(jax.jit(lambda v, x, y: scatter_041(plan, v, x, y)), v, x, y, repeat=a.repeat)
            record("scatter-0.4.1", t, c)
    if "spread" in stages:
        vs = jnp.real(v) if a.spin == 0 else v
        t, c = timed(jax.jit(lambda v, x, y: _nufft._spread_points(plan, v, x, y)), vs, x, y,
                     repeat=a.repeat)
        record("spread (library)", t, c)
    loc = jnp.stack([x, y], axis=1)
    field = jnp.real(v) if a.spin == 0 else jnp.stack([jnp.real(v), jnp.imag(v)])
    if any(s in stages for s in ("grouped", "interp-prepared")):
        store = "stored" in stages
        t0 = time.perf_counter()
        pts = offgrid.prepare_points(np.stack([th, ph], 1), lmax=a.lmax, epsilon=a.eps,
                                     group_size=a.group_size, store_weights=store)
        tp = time.perf_counter() - t0
        slots = sum(s.size for s, _ in pts.buckets)
        print(f"  prepare (host)        {1e3 * tp:9.1f} ms  {1e9 * tp / a.n:7.1f} ns/pt  "
              f"[group_size {a.group_size}: {pts}; padding {slots / a.n - 1:.0%}]", flush=True)
        out["prepare_ns_per_pt"] = 1e9 * tp / a.n
        lc = None if store else loc
        gs = jnp.real(g) if a.spin == 0 else g
        if "grouped" in stages:
            t, c = timed(jax.jit(lambda fl, p, lo: offgrid.points_to_grid(fl, p, lo, spin=a.spin)),
                         field, pts, lc, repeat=a.repeat)
            record(f"points_to_grid K={a.group_size}" + (" st" if store else ""), t, c)
        if "interp-prepared" in stages:
            t, c = timed(jax.jit(lambda gr, p, lo: offgrid.grid_to_points(gr, p, lo, spin=a.spin)),
                         gs, pts, lc, repeat=a.repeat)
            record("grid_to_points" + (" st" if store else ""), t, c)
    if "adjoint" in stages and (a.n <= SCATTER_041_MAX or a.force):
        f = jax.jit(lambda fl, lc: offgrid.adjoint_synthesis_general(
            fl, lc, spin=a.spin, lmax=a.lmax, epsilon=a.eps))
        t, c = timed(f, field, loc, repeat=a.repeat)
        record("adjoint (public)", t, c)
    if "synthesis" in stages:
        alm = jnp.asarray(rng.standard_normal(alm_size(a.lmax)) + 0j)
        if a.spin:
            alm = jnp.stack([alm, alm])
        f = jax.jit(lambda al, lc: offgrid.synthesis_general(al, lc, spin=a.spin, lmax=a.lmax, epsilon=a.eps))
        t, c = timed(f, alm, loc, repeat=a.repeat)
        record("synthesis (public)", t, c)
    return out


SWEEP = [("patch", 1 << 16), ("sphere", 1 << 16), ("patch", 1 << 18), ("sphere", 1 << 18)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", choices=["patch", "sphere"], default="patch")
    ap.add_argument("--n", type=int, default=1 << 16)
    ap.add_argument("--lmax", type=int, default=256)
    ap.add_argument("--eps", type=float, default=1e-10)
    ap.add_argument("--spin", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--stages", default="legendre,gather,scatter-0.4.1,spread,adjoint,synthesis")
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--force", action="store_true", help="allow scatter-0.4.1 above 2^18 points")
    ap.add_argument("--json", help="append the result as one JSON line to this file")
    ap.add_argument("--sweep", action="store_true", help="run the standard table, one subprocess each")
    a = ap.parse_args()
    if a.sweep:
        for kind, n in SWEEP:
            cmd = [sys.executable, __file__, "--points", kind, "--n", str(n), "--lmax", str(a.lmax),
                   "--eps", str(a.eps), "--spin", str(a.spin), "--repeat", str(a.repeat),
                   "--stages", a.stages, "--group-size", str(a.group_size)] + (["--json", a.json] if a.json else [])
            subprocess.run(cmd, check=True)
        return
    out = run_one(a)
    if a.json:
        with open(a.json, "a") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
