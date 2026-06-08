"""Single-shot large-scale GPU performance diagnostic for jht.

Designed to wring the maximum diagnostic value out of **one** moderate-scale
``gpu_requeue`` slot on Cannon (A100 40/80 GB, or a MIG slice). jht's GPU numbers
have so far been entirely *deferred* -- never measured. This run captures, in a
single job:

  * fp64 GPU-vs-CPU speedup envelope, and the card's **fp64/fp32 throttle factor**
    -- the same op timed in both dtypes (fp32 is THROUGHPUT-ONLY here, never an
    accuracy claim: without x64 the transform silently drops to ~1e-3);
  * fp64 **GPU==CPU parity** (relerr ~1e-12) measured on-device (so far only
    asserted by the skipped ``tests/test_gpu.py`` gate);
  * the on-device **memory ceiling** per (nside, lmax, spin), incl. nside=2048;
  * **batched / vmap throughput** vs batch size (the forecast-sweep use case --
    its per-realization knee is the whole motivation for the GPU tier);
  * **off-grid NUFFT at production scale** (lmax~1000, N up to 1e6), with measured
    vs *predicted* footprint -- the ``(Npts, W, W)`` ES-stencil intermediate
    (``_nufft.py``) dominates, not the oversampled grid;
  * the off-grid **pointing-gradient** cost (the capability ducc's FFI raises
    ``NotImplementedError`` on -- its GPU cost is completely unknown);
  * **compile time**, reported separately from steady-state everywhere.

Design -- why subprocess-per-point.  Each measurement runs in a FRESH SUBPROCESS.
This is deliberate and solves four problems at once: (a) a clean per-point device
peak (``peak_bytes_in_use`` is a cumulative process counter -- a single laddered
process only ever reports the monotonic high-water); (b) **OOM survival** -- a
child that OOMs (likely at nside=2048 or N=1e6 on a MIG slice) is recorded as OOM
and the ladder continues, so one OOM never wastes the slot; (c) per-point float64
toggle (``jax_enable_x64`` is process-global and effectively one-way); (d) compile
isolation (no persisted XLA cache between points -- so ``compile_s`` is honest).

The parent auto-sizes the ladder to the detected device memory, writes a human
table progressively AND a machine-readable JSONL (so the run can be re-plotted
without a second job), and prints a derived summary (median speedup, throttle
ratio, largest size that fit, vmap per-realization floor).

Usage (on the GPU box):
    pixi run -e gpu python scripts/gpu_diagnostic.py             # auto-size + run
    pixi run -e gpu python scripts/gpu_diagnostic.py --dry-run   # print the ladder, run nothing
    pixi run -e gpu python scripts/gpu_diagnostic.py --max-wall 1800   # cap total wall time
    # preview what a given slice would run, from anywhere (even a laptop):
    python scripts/gpu_diagnostic.py --dry-run --limit-gb 10     # a 10 GB MIG slice
    python scripts/gpu_diagnostic.py --dry-run --limit-gb 80     # a full A100

Built for ``gpu_requeue``, which hands you *whatever is free* -- often a 10-20 GB
A100 MIG slice, occasionally a whole 80 GB card. The ladder is **memory-driven**:
each heavy point is gated by a predicted device footprint against the detected (or
``--limit-gb``) memory, so the run fills whatever it lands on; the per-point OOM
guard is the real boundary, and ``--max-wall`` ensures a short slot still saves
every completed point. On a CPU-only box it runs a small self-test ladder
(device='cpu') exercising every path, so it is never dead code. See ``docs/gpu.md``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import numpy as np

RESULT_PREFIX = "RESULT_JSON "  # child emits exactly one line with this prefix
GB = 1e9


# --------------------------------------------------------------------------- #
# shared (parent + child) device helpers
# --------------------------------------------------------------------------- #
def _target_device():
    """The GPU if visible, else the CPU (so the script always runs)."""
    import jax

    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        return jax.devices("cpu")[0]


def _device_limit_gb(dev) -> float:
    try:
        stats = dev.memory_stats() or {}
    except Exception:
        return float("nan")
    lim = stats.get("bytes_limit") or stats.get("bytes_reservable_limit")
    return lim / GB if lim else float("nan")


def _device_peak_gb(dev) -> float:
    try:
        stats = dev.memory_stats() or {}
    except Exception:
        return float("nan")
    peak = stats.get("peak_bytes_in_use")
    return peak / GB if peak else float("nan")


# --------------------------------------------------------------------------- #
# child-side measurement (one point per process)
# --------------------------------------------------------------------------- #
def _timed(fn, n: int) -> tuple[float, float]:
    """(first-call wall incl. compile, best-of-n steady-state)."""
    t0 = time.perf_counter()
    fn().block_until_ready()
    first = time.perf_counter() - t0
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn().block_until_ready()
        ts.append(time.perf_counter() - t0)
    return first, min(ts)


def _relerr(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    denom = max(float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(a - b))) / denom


def _rand_alm(lmax, rng, lmin=0):
    import jht

    a = (
        rng.standard_normal(jht.alm_size(lmax)) + 1j * rng.standard_normal(jht.alm_size(lmax))
    ) / np.sqrt(2)
    a[: lmax + 1] = a[: lmax + 1].real  # real a_{l,0}
    a[:lmin] = 0.0
    return a.astype(np.complex128)


def _make_alm(lmax, spin, rng):
    if spin == 0:
        return _rand_alm(lmax, rng)
    return np.stack([_rand_alm(lmax, rng, 2), _rand_alm(lmax, rng, 2)])


def _clear_caches():
    from jht import analysis as _analysis
    from jht import healpix as _healpix

    _healpix._prepare.cache_clear()
    _analysis._wvec.cache_clear()


def _synth_on(dev, alm, nside, lmax, spin):
    import jax

    import jht

    _clear_caches()
    with jax.default_device(dev):
        return jht.synthesis(jax.device_put(alm, dev), nside, lmax, spin)


def measure_ongrid(spec: dict) -> dict:
    import jax

    import jht

    nside, lmax, spin = spec["nside"], spec["lmax"], spec["spin"]
    n = spec.get("best_of", 3)
    dev = _target_device()
    rng = np.random.default_rng(0)
    alm = _make_alm(lmax, spin, rng)

    _clear_caches()
    with jax.default_device(dev):
        alm_d = jax.device_put(alm, dev)
        syn_gpu = jht.synthesis(alm_d, nside, lmax, spin)
        m0 = np.asarray(syn_gpu)
        m0_d = jax.device_put(m0, dev)
        c_syn, r_syn = _timed(lambda: jht.synthesis(alm_d, nside, lmax, spin), n)
        c_adj, r_adj = _timed(lambda: jht.adjoint_synthesis(m0_d, nside, lmax, spin), n)
        r_m2a = float("nan")
        if spec.get("do_m2a"):
            _, r_m2a = _timed(lambda: jht.map2alm(m0_d, nside, lmax, spin, niter=3), max(2, n - 1))

    res = dict(
        compile_s=max(c_syn, c_adj),
        synth_ms=r_syn * 1e3,
        adj_ms=r_adj * 1e3,
        m2a_ms=r_m2a * 1e3,
        device_peak_GB=_device_peak_gb(dev),
        device_limit_GB=_device_limit_gb(dev),
        device=str(dev),
    )

    # fp64 GPU-vs-CPU: speedup + parity (skip when already on CPU, or in fp32)
    if spec.get("with_cpu") and dev.platform != "cpu":
        cpu = jax.devices("cpu")[0]
        _clear_caches()
        with jax.default_device(cpu):
            alm_c = jax.device_put(alm, cpu)
            _, r_syn_cpu = _timed(lambda: jht.synthesis(alm_c, nside, lmax, spin), n)
        syn_cpu = _synth_on(cpu, alm, nside, lmax, spin)
        res["cpu_synth_ms"] = r_syn_cpu * 1e3
        res["speedup"] = r_syn_cpu / r_syn
        res["synth_relerr"] = _relerr(np.asarray(syn_gpu), np.asarray(syn_cpu))
    return res


def measure_vmap(spec: dict) -> dict:
    import jax

    import jht

    nside, lmax, spin, batch = spec["nside"], spec["lmax"], spec["spin"], spec["batch"]
    n = spec.get("best_of", 3)
    dev = _target_device()
    rng = np.random.default_rng(0)
    alm = _make_alm(lmax, spin, rng)
    _clear_caches()
    with jax.default_device(dev):
        batch_alm = jax.device_put(np.stack([alm] * batch), dev)
        vsyn = jax.vmap(lambda a: jht.synthesis(a, nside, lmax, spin))
        c_v, r_v = _timed(lambda: vsyn(batch_alm), max(2, n - 1))
    return dict(
        compile_s=c_v,
        vsynth_total_ms=r_v * 1e3,
        per_real_ms=r_v * 1e3 / batch,
        device_peak_GB=_device_peak_gb(dev),
        device_limit_GB=_device_limit_gb(dev),
        device=str(dev),
    )


def _rand_alm_offgrid(lmax, spin, seed):
    import jht
    from jht.healpix import alm_column_base

    rng = np.random.default_rng(seed)

    def one():
        a = rng.standard_normal(jht.alm_size(lmax)) + 1j * rng.standard_normal(jht.alm_size(lmax))
        a[: lmax + 1] = a[: lmax + 1].real
        for m in range(min(abs(spin), lmax + 1)):
            base = alm_column_base(m, lmax)
            a[base : base + (abs(spin) - m)] = 0.0
        return a.astype(np.complex128)

    return one() if spin == 0 else np.stack([one(), one()])


def measure_offgrid(spec: dict) -> dict:
    import jax

    import jht
    from jht.offgrid import _prepare as _off_prepare

    lmax, npts, spin, eps = spec["lmax"], spec["npts"], spec["spin"], spec["epsilon"]
    n = spec.get("best_of", 3)
    dev = _target_device()
    rng = np.random.default_rng(1)
    alm = _rand_alm_offgrid(lmax, spin, lmax + spin)
    loc = np.stack(
        [rng.uniform(0.02, np.pi - 0.02, npts), rng.uniform(0.0, 2.0 * np.pi, npts)], axis=1
    )

    # predicted footprint from the ACTUAL plan (honest, not a formula guess)
    p = _off_prepare(lmax, spin, eps)
    grid_gb = p.nplan.nk * p.nplan.nm * 16 / GB
    stencil_gb = npts * p.nplan.W * p.nplan.W * 16 / GB

    with jax.default_device(dev):
        alm_d, loc_d = jax.device_put(alm, dev), jax.device_put(loc, dev)
        syn = jax.jit(lambda a, L: jht.synthesis_general(a, L, spin=spin, lmax=lmax, epsilon=eps))
        c_syn, r_syn = _timed(lambda: syn(alm_d, loc_d), n)
        field = syn(alm_d, loc_d).block_until_ready()
        adj = jax.jit(
            lambda v, L: jht.adjoint_synthesis_general(v, L, spin=spin, lmax=lmax, epsilon=eps)
        )
        _, r_adj = _timed(lambda: adj(field, loc_d), n)
        r_grad = float("nan")
        if spec.get("do_grad"):
            # cost of d(field)/d(loc): the ducc-NotImplementedError capability.
            def scalar(L):
                f = jht.synthesis_general(alm_d, L, spin=spin, lmax=lmax, epsilon=eps)
                return jax.numpy.sum(jax.numpy.real(f) ** 2)

            g = jax.jit(jax.grad(scalar))
            _, r_grad = _timed(lambda: g(loc_d), max(2, n - 1))

    return dict(
        compile_s=c_syn,
        syn_ms=r_syn * 1e3,
        adj_ms=r_adj * 1e3,
        grad_ms=r_grad * 1e3,
        W=p.nplan.W,
        grid_GB=grid_gb,
        stencil_GB=stencil_gb,
        pred_min_GB=grid_gb + stencil_gb,
        device_peak_GB=_device_peak_gb(dev),
        device_limit_GB=_device_limit_gb(dev),
        device=str(dev),
    )


_MEASURE = {"ongrid": measure_ongrid, "vmap": measure_vmap, "offgrid": measure_offgrid}


def run_child(spec: dict) -> None:
    """Run one measurement and emit exactly one ``RESULT_JSON`` line. Always exit 0
    (OOM / errors are reported in-band so the parent's ladder survives)."""
    import jax

    jax.config.update("jax_enable_x64", spec.get("dtype", "fp64") == "fp64")
    out = dict(spec)
    try:
        if spec.get("dtype", "fp64") == "fp64":
            assert jax.config.jax_enable_x64, "x64 not enabled for an fp64 point"  # type: ignore[attr-defined]
        out.update(_MEASURE[spec["kind"]](spec))
        out["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 -- in-band failure reporting is the point
        msg = f"{type(exc).__name__}: {exc}"
        low = msg.lower()
        out["status"] = (
            "oom" if ("resource_exhausted" in low or "out of memory" in low) else "error"
        )
        out["error"] = msg[:400]
    print(RESULT_PREFIX + json.dumps(out), flush=True)


# --------------------------------------------------------------------------- #
# parent-side: ladder, orchestration, reporting
# --------------------------------------------------------------------------- #
ONGRID = [(256, 384), (512, 768), (1024, 1000), (2048, 1000)]
VMAP_SIZES = [(512, 768), (1024, 1000)]
VMAP_BATCHES = [1, 4, 16, 64]
# off-grid candidates (lmax, npts, eps). lmax=1000 (production) is included on ANY
# slice that fits it -- off-grid is memory-light (grid + Npts*W^2), unlike large
# nside on-grid, so even a 10-20 GB MIG can reach the production point.
OFFGRID = [
    (256, 100_000, 1e-10),
    (512, 200_000, 1e-10),
    (1000, 200_000, 1e-10),
    (1000, 500_000, 1e-10),
    (1000, 1_000_000, 1e-10),
    (1000, 1_000_000, 1e-8),  # cheaper kernel (W=10): the W->memory trade at scale
]
_W_OF_EPS = {1e-6: 8, 1e-8: 10, 1e-10: 14, 1e-12: 16}
# device-memory safety factors. Predictions are rough -- the on-grid one is derived
# from the isolated host-RSS ceilings in docs/performance.md -- so gate generously
# and let the per-point OOM guard find the true boundary (small points run first,
# so an over-reach costs only its own compile, never the whole slot).
SAFETY_ONGRID = 1.5
SAFETY_OFFGRID = 1.3
# fallback budget when the device memory limit can't be read (used only for sizing
# a --dry-run / --tier preview off-device; a real run reads the actual limit).
_TIER_BUDGET_GB = {"mig": 12.0, "a100-40": 40.0, "a100-80": 80.0, "cpu": float("nan")}


def _tier_for(limit_gb: float, gpu: bool) -> str:
    if not gpu or np.isnan(limit_gb):
        return "cpu"
    if limit_gb < 24:  # a gpu_requeue MIG slice (typically 10-20 GB)
        return "mig"
    if limit_gb < 56:
        return "a100-40"
    return "a100-80"


def _ongrid_peak_gb(nside: int, spin: int, batch: int = 1) -> float:
    """Rough device-peak estimate from the isolated host-RSS ceilings in
    docs/performance.md (nside=2048: ~10.8 GB spin-0 / ~13.2 GB spin-2)."""
    mpix = 12 * nside * nside / 1e6
    per_mpix = 0.26 if spin == 2 else 0.22  # GB / Mpix
    return per_mpix * mpix * batch


def _offgrid_peak_gb(lmax: int, npts: int, eps: float) -> float:
    """DFS grid (sigma=2 => ~ (4 lmax)^2 complex) + ~2x the (Npts, W, W) stencil
    (the forward gather plus the adjoint's transpose intermediate); docs/offgrid.md."""
    grid = 16.0 * lmax * lmax * 16 / 1e9
    w = _W_OF_EPS.get(eps, 14)
    return grid + 2.0 * npts * w * w * 16 / 1e9


def _fits(pred_gb: float, budget_gb: float, safety: float) -> bool:
    return bool(np.isnan(budget_gb)) or pred_gb * safety <= budget_gb


def build_ladder(
    tier: str, budget_gb: float, kinds: tuple[str, ...] = ("ongrid", "offgrid", "vmap")
) -> list[dict]:
    """The memory-driven list of measurement specs: each heavy point is gated by a
    predicted device footprint vs ``budget_gb``, so a 10 GB MIG slice and an 80 GB
    A100 each get exactly as much as fits -- with the per-point OOM guard as the
    real boundary. ``kinds`` selects the measurement families to include (a focused
    re-run can do, e.g., just off-grid + vmap). ``--dry-run`` prints this for review."""
    cpu = tier == "cpu"
    batches = [1, 4] if cpu else ([1, 4, 16] if tier == "mig" else VMAP_BATCHES)
    dtypes = ["fp64"] if cpu else ["fp64", "fp32"]  # fp32 = throttle probe only

    specs: list[dict] = []
    for nside, lmax in ONGRID if "ongrid" in kinds else ():
        if cpu and nside > 512:  # keep the self-test quick
            continue
        for spin in (0, 2):
            if not _fits(_ongrid_peak_gb(nside, spin), budget_gb, SAFETY_ONGRID):
                continue
            for dt in dtypes:
                best = 3 if nside <= 1024 else (2 if not cpu else 1)
                specs.append(
                    dict(
                        kind="ongrid",
                        nside=nside,
                        lmax=lmax,
                        spin=spin,
                        dtype=dt,
                        do_m2a=(dt == "fp64" and nside <= 1024),
                        with_cpu=(dt == "fp64" and nside <= 1024),
                        best_of=best,
                    )
                )
    # off-grid before vmap: gpu_requeue is preemptible, so order by value -- the
    # never-measured off-grid (ducc-replacement) capability should land before the
    # more-inferable vmap throughput sweep if the slot is cut short.
    for lmax, npts, eps in OFFGRID if "offgrid" in kinds else ():
        if cpu and (lmax > 256 or npts > 100_000 or eps != 1e-10):
            continue
        if not _fits(_offgrid_peak_gb(lmax, npts, eps), budget_gb, SAFETY_OFFGRID):
            continue
        for spin in (0, 2) if eps == 1e-10 else (0,):
            specs.append(
                dict(
                    kind="offgrid",
                    lmax=lmax,
                    npts=npts,
                    spin=spin,
                    epsilon=eps,
                    dtype="fp64",
                    do_grad=(spin == 0 and lmax in (256, 512)),
                    best_of=3,
                )
            )
    for nside, lmax in VMAP_SIZES if "vmap" in kinds else ():
        if cpu and nside > 512:
            continue
        for batch in batches:
            if not _fits(_ongrid_peak_gb(nside, 0, batch), budget_gb, SAFETY_ONGRID):
                continue
            specs.append(
                dict(
                    kind="vmap",
                    nside=nside,
                    lmax=lmax,
                    spin=0,
                    batch=batch,
                    dtype="fp64",
                    best_of=2,
                )
            )
    return specs


def _label(s: dict) -> str:
    if s["kind"] == "ongrid":
        return f"ongrid  nside={s['nside']:>4} lmax={s['lmax']:>4} spin={s['spin']} {s['dtype']}"
    if s["kind"] == "vmap":
        return (
            f"vmap    nside={s['nside']:>4} lmax={s['lmax']:>4} batch={s['batch']:>3} {s['dtype']}"
        )
    return (
        f"offgrid lmax={s['lmax']:>4} npts={s['npts']:>9} spin={s['spin']} eps={s['epsilon']:.0e}"
    )


def run_point(spec: dict, timeout: float) -> dict:
    """Run one child subprocess; parse its RESULT_JSON line. A timeout or a hard
    crash (OOM-kill / segfault, no JSON) is recorded as a failure, not raised."""
    spec = dict(spec, best_of=spec.get("best_of", 3))
    try:
        proc = subprocess.run(
            [sys.executable, __file__, "--one", json.dumps(spec)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return dict(spec, status="timeout")
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    # no result line: classify by the child's stderr (XLA OOM-kills land here)
    tail = (proc.stderr or "")[-300:]
    low = tail.lower()
    status = (
        "oom"
        if ("resource_exhausted" in low or "out of memory" in low or proc.returncode < 0)
        else "crash"
    )
    return dict(spec, status=status, error=tail)


def _fmt(x, w=8, p=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return " " * (w - 3) + "n/a"
    return f"{x:>{w}.{p}f}"


def print_row(r: dict) -> None:
    st = r.get("status", "?")
    if st != "ok":
        print(f"  {_label(r):<46}  [{st.upper()}]", flush=True)
        return
    if r["kind"] == "ongrid":
        extra = ""
        if "speedup" in r:
            extra = f"  speedup={_fmt(r['speedup'], 5, 1)}x  relerr={r.get('synth_relerr', float('nan')):.1e}"
        print(
            f"  {_label(r):<46}  comp={_fmt(r['compile_s'], 6, 1)}s "
            f"syn={_fmt(r['synth_ms'])}ms adj={_fmt(r['adj_ms'])}ms "
            f"m2a={_fmt(r['m2a_ms'], 8, 0)}ms  peak={_fmt(r['device_peak_GB'], 5, 2)}GB{extra}",
            flush=True,
        )
    elif r["kind"] == "vmap":
        print(
            f"  {_label(r):<46}  comp={_fmt(r['compile_s'], 6, 1)}s "
            f"total={_fmt(r['vsynth_total_ms'])}ms per-real={_fmt(r['per_real_ms'])}ms "
            f"peak={_fmt(r['device_peak_GB'], 5, 2)}GB",
            flush=True,
        )
    else:
        print(
            f"  {_label(r):<46}  comp={_fmt(r['compile_s'], 6, 1)}s "
            f"syn={_fmt(r['syn_ms'])}ms adj={_fmt(r['adj_ms'])}ms grad={_fmt(r['grad_ms'])}ms  "
            f"peak={_fmt(r['device_peak_GB'], 5, 2)}GB (pred>={_fmt(r['pred_min_GB'], 5, 2)})",
            flush=True,
        )


def summarize(results: list[dict]) -> None:
    ok = [r for r in results if r.get("status") == "ok"]
    print("\n== derived summary ==")
    speedups = [r["speedup"] for r in ok if r["kind"] == "ongrid" and "speedup" in r]
    if speedups:
        print(
            f"  GPU/CPU synth speedup (fp64): median {np.median(speedups):.1f}x  "
            f"range {min(speedups):.1f}-{max(speedups):.1f}x"
        )
    # fp64/fp32 throttle: pair ongrid points by (nside,lmax,spin)
    by_key: dict[tuple, dict] = {}
    for r in ok:
        if r["kind"] == "ongrid":
            by_key.setdefault((r["nside"], r["lmax"], r["spin"]), {})[r["dtype"]] = r["synth_ms"]
    ratios = [v["fp64"] / v["fp32"] for v in by_key.values() if "fp64" in v and "fp32" in v]
    if ratios:
        print(
            f"  fp64/fp32 throttle (synth time ratio): median {np.median(ratios):.2f}x  "
            f"(A100 expect ~2x; throttled cards >>; <1.3x => fp64 not the bottleneck)"
        )
    relerrs = [
        r["synth_relerr"] for r in ok if r.get("synth_relerr") is not None and "synth_relerr" in r
    ]
    if relerrs:
        print(f"  fp64 GPU==CPU parity: max relerr {max(relerrs):.1e} (gate ~1e-12)")
    big = [r for r in ok if r["kind"] == "ongrid"]
    if big:
        b = max(big, key=lambda r: r["nside"])
        print(
            f"  largest on-grid that fit: nside={b['nside']} "
            f"(peak {b['device_peak_GB']:.1f} GB of {b.get('device_limit_GB', float('nan')):.0f})"
        )
    offok = [r for r in ok if r["kind"] == "offgrid"]
    if offok:
        b = max(offok, key=lambda r: r["npts"])
        print(
            f"  largest off-grid that fit: N={b['npts']:,} lmax={b['lmax']} "
            f"(peak {b['device_peak_GB']:.1f} GB; predicted >= {b['pred_min_GB']:.1f})"
        )
    vmap = [r for r in ok if r["kind"] == "vmap"]
    if vmap:
        floor = min(r["per_real_ms"] for r in vmap)
        b1 = [r["per_real_ms"] for r in vmap if r["batch"] == 1]
        speed = f" ({b1[0] / floor:.1f}x vs batch-1)" if b1 else ""
        print(f"  vmap per-realization floor: {floor:.1f} ms{speed}")
    fails = [r for r in results if r.get("status") != "ok"]
    if fails:
        print(
            f"  {len(fails)} point(s) did not complete: "
            + ", ".join(f"{_label(r)} [{r['status']}]" for r in fails)
        )


def _provenance(dev, limit_gb, tier) -> dict:
    import jax

    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        head = "unknown"
    return dict(
        _meta=True,
        git=head,
        jax=jax.__version__,
        device=str(dev),
        platform=dev.platform,
        limit_GB=limit_gb,
        tier=tier,
        x64_default=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--one", metavar="SPEC_JSON", help="(internal) run a single measurement child")
    ap.add_argument(
        "--tier", choices=["cpu", "mig", "a100-40", "a100-80"], help="override auto-detect"
    )
    ap.add_argument(
        "--limit-gb",
        type=float,
        default=0.0,
        help="override detected device memory (GB) for ladder sizing -- e.g. preview "
        "a 10 GB MIG slice's ladder with `--dry-run --limit-gb 10`",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the planned ladder and exit")
    ap.add_argument("--timeout", type=float, default=900.0, help="per-point wall-clock budget (s)")
    ap.add_argument(
        "--max-wall",
        type=float,
        default=0.0,
        help="stop launching new points after this many seconds total (0 = no limit); "
        "a short requeue slot still saves every completed point to the JSONL",
    )
    ap.add_argument(
        "--kinds",
        default="ongrid,offgrid,vmap",
        help="comma-separated measurement families to run (ongrid,offgrid,vmap); "
        "e.g. `--kinds offgrid,vmap` for a focused re-run that skips the slow on-grid block",
    )
    ap.add_argument("--out", default="gpu_diag_results.jsonl", help="JSONL output path")
    args = ap.parse_args()

    if args.one is not None:
        run_child(json.loads(args.one))
        return

    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    bad = set(kinds) - {"ongrid", "offgrid", "vmap"}
    if bad:
        ap.error(f"--kinds: unknown {sorted(bad)} (allowed: ongrid, offgrid, vmap)")

    import jax

    jax.config.update("jax_enable_x64", True)
    dev = _target_device()
    gpu = dev.platform == "gpu"
    limit_gb = args.limit_gb if args.limit_gb else _device_limit_gb(dev)
    if args.tier:
        tier = args.tier
    elif args.limit_gb:  # explicit override: size as a GPU slice even off-device (preview)
        tier = _tier_for(args.limit_gb, gpu=True)
    else:
        tier = _tier_for(limit_gb, gpu)
    budget_gb = limit_gb if not np.isnan(limit_gb) else _TIER_BUDGET_GB[tier]
    ladder = build_ladder(tier, budget_gb, kinds)

    print(f"jht GPU diagnostic  (jax {jax.__version__})")
    print(
        f"device: {dev}  platform={dev.platform}  mem-limit={limit_gb:.1f} GB  "
        f"-> tier '{tier}'  (sizing budget {budget_gb:.0f} GB)"
    )
    wall = f"max-wall {args.max_wall:.0f}s  |  " if args.max_wall else ""
    print(
        f"ladder: {len(ladder)} points  |  kinds {','.join(kinds)}  |  "
        f"per-point timeout {args.timeout:.0f}s  |  {wall}out: {args.out}"
    )
    if not gpu:
        print("** no GPU visible -- CPU self-test (exercises every path; not the perf run) **")

    if args.dry_run:
        print("\n-- planned ladder (dry run, nothing executed) --")
        for s in ladder:
            print(f"  {_label(s)}")
        return

    prov = _provenance(dev, limit_gb, tier)
    prov["budget_GB"] = budget_gb
    results: list[dict] = []
    t_start = time.perf_counter()
    with open(args.out, "w") as fh:
        fh.write(json.dumps(prov) + "\n")
        last_kind = None
        for i, spec in enumerate(ladder, 1):
            if args.max_wall and time.perf_counter() - t_start > args.max_wall:
                for rest in ladder[i - 1 :]:
                    rec = dict(rest, status="skipped-wall")
                    results.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(
                    f"\n** wall budget {args.max_wall:.0f}s exceeded -- skipped {len(ladder) - i + 1} point(s) **"
                )
                break
            if spec["kind"] != last_kind:
                print(f"\n== {spec['kind']} ==")
                last_kind = spec["kind"]
            r = run_point(spec, args.timeout)
            results.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print_row(r)
            sys.stderr.write(f"[{i}/{len(ladder)}] {_label(spec)} -> {r.get('status')}\n")

    summarize(results)
    print(f"\nfull machine-readable results: {args.out}")


if __name__ == "__main__":
    main()
