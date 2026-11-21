"""Benchmark the l-block size K of the Legendre contraction on batched transforms.

Times ``vmap``-batched ``synthesis`` / ``adjoint_synthesis`` (and optionally their
gradients) on the default device for a grid of batch sizes and block sizes, which
is the regime the blocked contraction targets: under ``vmap`` the recursion is shared
across the batch while the contraction scales with it. Run::

    pixi run python scripts/bench_contract.py --nside 128 --lmax 192 --batch 1 10 \\
        --blocks 1 8 32 --spin 0 2
    pixi run -e gpu python scripts/bench_contract.py --nside 512 --lmax 768 \\
        --batch 1 10 80 --grad --out bench_contract.jsonl
    pixi run python scripts/bench_contract.py --blocks auto --tables off on \\
        --fft-mode looped --cap-chunk 1 32

``--tables on`` passes ``jht.legendre_tables`` (built once per point, outside the timing)
as an argument; ``--fft-mode`` / ``--cap-chunk`` set the azimuth path.

Per point it prints the steady-state time per call (best of ``--reps`` after one
warm-up), the compile time, and the compiled program's temp + output bytes from
``memory_analysis()`` (per program, so points do not contaminate each other the way
the process-cumulative ``peak_bytes_in_use`` does).
"""

from __future__ import annotations

import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import jht  # noqa: E402


def _alm(rng, lmax, spin, batch):
    K = jht.alm_size(lmax)
    shape = (batch, K) if spin == 0 else (batch, 2, K)
    a = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
    a[..., : lmax + 1] = a[..., : lmax + 1].real
    return jnp.asarray(a)


def _program(kind, nside, lmax, spin, grad):
    """jit(program)(x, tables); ``tables`` None runs the recursion."""
    if kind == "synth":
        f = jax.vmap(lambda a, t: jht.synthesis(a, nside, lmax, spin, **_kw(t)), in_axes=(0, None))
    else:
        f = jax.vmap(lambda m, t: jht.adjoint_synthesis(m, nside, lmax, spin, **_kw(t)), in_axes=(0, None))
    if grad:
        return jax.jit(jax.grad(lambda x, t: jnp.sum(jnp.abs(f(x, t)) ** 2)))
    return jax.jit(f)


def _kw(t):
    return {} if t is None else {"tables": t}


HAS_BLOCK = hasattr(jht, "set_ell_block")  # False on jht < 0.6 (per-row contraction)
HAS_TABLES = hasattr(jht, "legendre_tables")
HAS_CAP_CHUNK = hasattr(jht, "set_cap_chunk")


def bench_point(kind, nside, lmax, spin, batch, block, grad, reps, rng, tables=False, cap_chunk=None):
    if HAS_BLOCK:
        jht.set_ell_block(block)
    if HAS_CAP_CHUNK and cap_chunk is not None:
        jht.set_cap_chunk(cap_chunk)
    t = jht.legendre_tables(nside, lmax, spin) if tables else None
    a = _alm(rng, lmax, spin, batch)
    x = a if kind == "synth" else jax.vmap(lambda z: jht.synthesis(z, nside, lmax, spin))(a)
    fn = _program(kind, nside, lmax, spin, grad)
    t0 = time.perf_counter()
    compiled = fn.lower(x, t).compile()
    compile_s = time.perf_counter() - t0
    mem = compiled.memory_analysis()
    mem_gb = (mem.temp_size_in_bytes + mem.output_size_in_bytes) / 1e9 if mem is not None else float("nan")
    jax.block_until_ready(compiled(x, t))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(x, t))
        ts.append(time.perf_counter() - t0)
    return dict(
        kind=kind + ("_grad" if grad else ""),
        nside=nside,
        lmax=lmax,
        spin=spin,
        batch=batch,
        block=block,
        K=jht._recursion.resolve_ell_block(block, (lmax + 2) // 2, 2 * nside, 1 if spin == 0 else 2)
        if HAS_BLOCK
        else 0,
        jht_version=jht.__version__,
        tables=bool(tables),
        table_gb=jht.legendre_tables_nbytes(nside, lmax, spin) / 1e9 if tables else 0.0,
        fft_mode=jht.get_azimuth_fft_mode(),
        cap_chunk=cap_chunk,
        steady_ms=1e3 * min(ts),
        per_map_ms=1e3 * min(ts) / batch,
        compile_s=compile_s,
        mem_gb=mem_gb,
        device=str(jax.devices()[0].device_kind),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nside", type=int, default=128)
    ap.add_argument("--lmax", type=int, default=192)
    ap.add_argument("--spin", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--blocks", nargs="+", default=["1", "8", "32"])
    ap.add_argument("--kinds", nargs="+", default=["synth", "adj"], choices=["synth", "adj"])
    ap.add_argument("--grad", action="store_true", help="also time the gradient of each kind")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tables", nargs="+", default=["off"], choices=["off", "on"])
    ap.add_argument("--fft-mode", default=None, choices=["unrolled", "looped"])
    ap.add_argument("--cap-chunk", nargs="+", default=[None], help="looped-mode ring pairs per step")
    ap.add_argument("--out", default=None, help="append JSON lines here")
    args = ap.parse_args()
    blocks = [b if b == "auto" else int(b) for b in args.blocks]
    chunks = [c if c in (None, "auto") else int(c) for c in args.cap_chunk]
    if args.fft_mode:
        jht.set_azimuth_fft_mode(args.fft_mode)
    if "on" in args.tables and not HAS_TABLES:
        raise SystemExit("this jht has no legendre_tables")
    rng = np.random.default_rng(0)
    print(
        f"jht {jht.__version__} bench_contract: jax {jax.__version__}, "
        f"device {jax.devices()[0].device_kind} (K=0: per-row contraction, jht < 0.6)",
        flush=True,
    )
    if not HAS_BLOCK:
        blocks = blocks[:1]
    hdr = (
        f"{'kind':>10} {'spin':>4} {'batch':>5} {'K':>4} {'tab':>3} {'chunk':>5} "
        f"{'steady_ms':>10} {'per_map_ms':>10} {'compile_s':>9} {'mem_GB':>7}"
    )
    print(hdr)
    print("-" * len(hdr), flush=True)
    for spin in args.spin:
        for kind in args.kinds:
            for grad in [False, True] if args.grad else [False]:
                for batch in args.batch:
                    for block in blocks:
                        for tab in args.tables:
                            for chunk in chunks:
                                r = bench_point(
                                    kind, args.nside, args.lmax, spin, batch, block, grad, args.reps, rng,
                                    tables=tab == "on", cap_chunk=chunk,
                                )
                                print(
                                    f"{r['kind']:>10} {spin:>4} {batch:>5} {r['K']:>4} {tab:>3} {str(chunk):>5} "
                                    f"{r['steady_ms']:>10.1f} {r['per_map_ms']:>10.2f} {r['compile_s']:>9.1f} "
                                    f"{r['mem_gb']:>7.2f}",
                                    flush=True,
                                )
                                if args.out:
                                    with open(args.out, "a") as fh:
                                        fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
