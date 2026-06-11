"""Phase-0 MLX feasibility spike: does fp64-table / fp32-apply hold the tier?

The make-or-break question for an MLX (Apple-GPU, fp32) backend: Apple Silicon has
no fp64, but the numerically dangerous part of an SHT -- the lambda recursion (the
thing s2fft got wrong) -- is already a fp64 numpy precompute here.  This spike keeps
that recursion at **fp64 on the CPU**, materializes the spin-weighted lambda table,
casts it to **fp32**, and runs the accuracy-critical **contraction**
``F_m(theta) = sum_l a_{l,m} lambda^s_{l,m}(theta)`` on the **MLX GPU in fp32**.
It then measures the error vs the JAX-fp64 oracle (`jht.synthesis`), both end-to-end
(the map) and per-m (the ring coefficients F -- to confirm the spin-2 error is flat
across m, i.e. no s2fft-style m<l leakage).

The phi-FFT + ring fold is done in numpy here (it is mechanical and not where the
fp32 risk lives -- the contraction is); Phase 1 moves it into MLX.  The verdict here
is GO/NO-GO on the contraction tier and sets the a-priori MLX gate.

Run:  pixi run python scripts/mlx_spike.py [--max-nside 1024]
"""

from __future__ import annotations

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import jht  # noqa: E402
from jht._recursion import build_recursion_plan, spin_weighted_lambda  # noqa: E402
from jht.healpix import RingInfo, alm_column_base, alm_size  # noqa: E402

# (nside, lmax) ladder; lmax kept <= ~1.5*nside (the band-limit ceiling)
LADDER = [(64, 96), (128, 192), (256, 384), (512, 768), (768, 1000), (1024, 1000)]


# --------------------------------------------------------------------------- #
# fp64 lambda-table builder (the reusable Phase-1 piece, prototyped here).
# Vectorized over m, looping l, reusing the validated build_recursion_plan
# coefficients + the exact _rec_step reconstruction.  Returns the *spin-weighted*
# lambda (pref folded in), matching normalized_legendre / spin_weighted_lambda.
# --------------------------------------------------------------------------- #
def lambda_table(x: np.ndarray, spin: int, lmax: int) -> np.ndarray:
    """Dense ``lambda^s_{l,m}(theta)`` table, shape ``(M, lmax+1, T)`` fp64.

    Zero where ``l < lmin(m) = max(m, |spin|)``.  Recursion carry stays fp64.
    """
    plan = build_recursion_plan(x, spin, lmax)
    A, B, C, pref = plan.A, plan.B, plan.C, plan.pref
    seed_sign, seed_log, is_seed, is_active = (
        plan.seed_sign,
        plan.seed_log,
        plan.is_seed,
        plan.is_active,
    )
    M = lmax + 1
    T = x.shape[0]
    x = np.asarray(x, dtype=np.float64)
    p = np.zeros((M, T))
    q = np.zeros((M, T))
    s = np.zeros((M, T))
    table = np.zeros((M, lmax + 1, T), dtype=np.float64)
    for ell in range(lmax + 1):
        A_c, B_c, C_c = A[ell][:, None], B[ell][:, None], C[ell][:, None]  # (M,1)
        seed_c, act_c = is_seed[ell][:, None], is_active[ell][:, None]
        raw = (A_c * x[None, :] + C_c) * p - B_c * q
        r = np.sqrt(raw * raw + p * p)
        r = np.where(r == 0.0, 1.0, r)
        rp, rq, rs = raw / r, p / r, s + np.log(r)
        is_recur = act_c & ~seed_c
        new_p = np.where(seed_c, seed_sign[:, None], np.where(is_recur, rp, p))
        new_q = np.where(seed_c, 0.0, np.where(is_recur, rq, q))
        new_s = np.where(seed_c, seed_log, np.where(is_recur, rs, s))
        with np.errstate(over="ignore", invalid="ignore"):
            lam = np.where(act_c, new_p * np.exp(new_s), 0.0)
        table[:, ell, :] = pref[ell][:, None] * lam
        p, q, s = new_p, new_q, new_s
    return table


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _lvals(lmax: int) -> np.ndarray:
    return np.concatenate([np.arange(m, lmax + 1) for m in range(lmax + 1)])


def rand_alm(lmax: int, spin: int, seed: int = 0) -> np.ndarray:
    """Random valid band-limited a_lm (m=0 real; no l<|spin| modes)."""
    rng = np.random.default_rng(seed)
    lv = _lvals(lmax)

    def one() -> np.ndarray:
        a = (rng.standard_normal(alm_size(lmax)) + 1j * rng.standard_normal(alm_size(lmax))).astype(
            np.complex128
        )
        a[: lmax + 1] = a[: lmax + 1].real
        a[lv < abs(spin)] = 0.0
        return a

    return one() if spin == 0 else np.stack([one(), one()])


def dense_grid(alm_1d: np.ndarray, lmax: int) -> np.ndarray:
    """Triangular a_lm -> dense (M, lmax+1) grid (zero where l<m)."""
    M = lmax + 1
    g = np.zeros((M, M), dtype=np.complex128)
    for m in range(M):
        base = alm_column_base(m, lmax)
        g[m, m:] = alm_1d[base : base + (lmax - m + 1)]
    return g


def mlx_contract(table_f32: mx.array, c_dense_f32_re, c_dense_f32_im) -> np.ndarray:
    """F[m,t] = sum_l c[m,l] * lambda[m,l,t], fp32 on the MLX GPU; -> host complex64."""
    fr = np.asarray(mx.einsum("mlt,ml->mt", table_f32, c_dense_f32_re))
    fi = np.asarray(mx.einsum("mlt,ml->mt", table_f32, c_dense_f32_im))
    return (fr + 1j * fi).astype(np.complex64)


def _to_mlx_table(table: np.ndarray) -> mx.array:
    return mx.array(table.astype(np.float32))


def _to_mlx_re_im(c_dense: np.ndarray):
    return (
        mx.array(np.ascontiguousarray(c_dense.real, dtype=np.float32)),
        mx.array(np.ascontiguousarray(c_dense.imag, dtype=np.float32)),
    )


def fold_ifft_spin0(F: np.ndarray, geo: RingInfo, lmax: int) -> np.ndarray:
    m_pos = np.arange(lmax + 1)
    out = np.empty(geo.npix, dtype=np.float32)
    for r in range(geo.nrings):
        N = int(geo.npix_ring[r])
        sidx = int(geo.startpix[r])
        G = (F[:, r] * np.exp(1j * m_pos * geo.phi0[r])).astype(np.complex64)
        D = np.zeros(N, dtype=np.complex64)
        np.add.at(D, m_pos % N, G)
        np.add.at(D, (-m_pos[1:]) % N, np.conj(G[1:]))
        out[sidx : sidx + N] = np.real(np.fft.ifft(D) * N).astype(np.float32)
    return out


def fold_ifft_spin2(Fp: np.ndarray, Fm: np.ndarray, geo: RingInfo, lmax: int) -> np.ndarray:
    m_pos = np.arange(lmax + 1)
    Q = np.empty(geo.npix, dtype=np.float32)
    U = np.empty(geo.npix, dtype=np.float32)
    for r in range(geo.nrings):
        N = int(geo.npix_ring[r])
        sidx = int(geo.startpix[r])
        ph = np.exp(1j * m_pos * geo.phi0[r])
        D = np.zeros(N, dtype=np.complex64)
        np.add.at(D, m_pos % N, (Fp[:, r] * ph).astype(np.complex64))
        np.add.at(D, (-m_pos[1:]) % N, np.conj((Fm[1:, r] * ph[1:]).astype(np.complex64)))
        qu = np.fft.ifft(D) * N
        Q[sidx : sidx + N] = np.real(qu).astype(np.float32)
        U[sidx : sidx + N] = np.imag(qu).astype(np.float32)
    return np.stack([Q, U])


def relerr(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / denom


def per_m_relerr(Fm32: np.ndarray, Fm64: np.ndarray) -> np.ndarray:
    denom = max(float(np.max(np.abs(Fm64))), 1e-300)
    return np.max(np.abs(Fm32 - Fm64), axis=1) / denom  # (M,)


# --------------------------------------------------------------------------- #
def run_case(nside: int, lmax: int, spin: int) -> dict:
    geo = RingInfo(nside)
    z = geo.z  # full grid (north+south); spike does not exploit N/S symmetry
    alm = rand_alm(lmax, spin, seed=nside + spin)

    # fp64 oracle map
    map_ref = np.asarray(jht.synthesis(jnp.asarray(alm), nside, lmax, spin))

    t0 = time.perf_counter()
    if spin == 0:
        table = lambda_table(z, 0, lmax)  # (M, L, T)
        tmlx = _to_mlx_table(table)
        cd = dense_grid(alm, lmax)
        cre, cim = _to_mlx_re_im(cd)
        F32 = mlx_contract(tmlx, cre, cim)  # (M, nrings)
        # fp64 oracle ring coeffs
        F64 = np.einsum("mlt,ml->mt", table, cd)
        pm = per_m_relerr(F32, F64)
        map32 = fold_ifft_spin0(F32, geo, lmax)
    else:
        tp = lambda_table(z, 2, lmax)
        tm = lambda_table(z, -2, lmax)
        aE, aB = dense_grid(alm[0], lmax), dense_grid(alm[1], lmax)
        cp = -(aE + 1j * aB)
        cm = -(aE - 1j * aB)
        Fp32 = mlx_contract(_to_mlx_table(tp), *_to_mlx_re_im(cp))
        Fm32 = mlx_contract(_to_mlx_table(tm), *_to_mlx_re_im(cm))
        Fp64 = np.einsum("mlt,ml->mt", tp, cp)
        Fm64 = np.einsum("mlt,ml->mt", tm, cm)
        pm = np.maximum(per_m_relerr(Fp32, Fp64), per_m_relerr(Fm32, Fm64))
        map32 = fold_ifft_spin2(Fp32, Fm32, geo, lmax)
    dt = time.perf_counter() - t0

    return {
        "nside": nside,
        "lmax": lmax,
        "spin": spin,
        "map_relerr": relerr(map32, map_ref),
        "F_relerr_max": float(np.max(pm)),
        "F_relerr_argmax_m": int(np.argmax(pm)),
        "secs": dt,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-nside", type=int, default=1024)
    args = ap.parse_args()

    print(f"jht MLX feasibility spike  (mlx {mx.__version__}, device {mx.default_device()})")
    print("fp64 lambda-table -> fp32 contraction (MLX GPU) -> map; error vs jht.synthesis fp64\n")
    hdr = (
        f"{'nside':>5} {'lmax':>5} {'spin':>4} {'map_relerr':>12} "
        f"{'F_relerr':>11} {'argmax_m':>8} {'secs':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for nside, lmax in LADDER:
        if nside > args.max_nside:
            continue
        for spin in (0, 2):
            try:
                r = run_case(nside, lmax, spin)
            except Exception as exc:  # noqa: BLE001
                print(f"{nside:>5} {lmax:>5} {spin:>4}  ERROR {type(exc).__name__}: {exc}")
                continue
            print(
                f"{r['nside']:>5} {r['lmax']:>5} {r['spin']:>4} "
                f"{r['map_relerr']:>12.2e} {r['F_relerr_max']:>11.2e} "
                f"{r['F_relerr_argmax_m']:>8d} {r['secs']:>6.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
