"""fp64 roundoff-growth of the jht recursion at high l, vs the mpmath reference.

Loads ``_mpmath_ref.npz`` (produced by ``highL_mpmath_ref.py`` in an mpmath env)
and compares jht's fp64 ``normalized_legendre`` (spin-0) and
``spin_weighted_lambda(spin=2)`` against it, reporting the worst relative error
(normalized by the addition-theorem amplitude ``sqrt((2l+1)/4pi)``, root-insensitive)
over (m, theta) for each band limit l.  The l=500 row is the convention anchor; the
l>1000 rows are the answer to "is fp64 enough, or does the log-renorm recursion need
two-part X-number scaling above the design ceiling?".

    pixi run python scripts/exploratory/highL_recursion_growth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from jht._recursion import normalized_legendre, spin_weighted_lambda  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _highL_grid import LMAX, amp_scale  # noqa: E402

HERE = Path(__file__).parent


def main() -> None:
    npz = HERE / "_mpmath_ref.npz"
    if not npz.exists():
        raise SystemExit(
            f"{npz} missing -- generate it first with:\n"
            "  pixi exec --spec mpmath --spec numpy -- "
            "python scripts/exploratory/highL_mpmath_ref.py"
        )
    d = np.load(npz)
    theta = d["theta"]
    m_list = d["m_list"].tolist()
    l_list = d["l_list"].tolist()
    ref = {0: d["ref_spin0"], 2: d["ref_spin2"]}
    x = np.cos(theta)

    # worst[spin][l] = max over (m, theta) of |jht - ref| / sqrt((2l+1)/4pi)
    worst = {0: {L: 0.0 for L in l_list}, 2: {L: 0.0 for L in l_list}}
    argworst = {0: {}, 2: {}}
    for spin in (0, 2):
        lmin0 = abs(spin)
        for im, m in enumerate(m_list):
            lmin = max(m, lmin0)
            if lmin > LMAX:
                continue
            # jht column once to LMAX; l = lmin..LMAX (spin0 lmin=m)
            col = np.asarray(
                normalized_legendre(x, m, LMAX)
                if spin == 0
                else spin_weighted_lambda(x, m, spin, LMAX)
            )  # (LMAX-lmin+1, nTheta)
            for il, L in enumerate(l_list):
                if L < lmin:
                    continue
                jht_row = col[L - lmin]  # (nTheta,)
                ref_row = ref[spin][im, il]  # (nTheta,)
                err = np.abs(jht_row - ref_row) / amp_scale(L)
                j = int(np.nanargmax(err))
                if err[j] > worst[spin][L]:
                    worst[spin][L] = float(err[j])
                    argworst[spin][L] = (m, float(theta[j]))

    print(f"high-l recursion fp64 roundoff vs mpmath (dps={int(d['dps'])}), "
          f"rel err normalized by sqrt((2l+1)/4pi):\n")
    hdr = "  l    | spin-0 worst (m,theta)        | spin-2 worst (m,theta)"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for L in l_list:
        a0, a2 = argworst[0].get(L), argworst[2].get(L)
        s0 = f"{worst[0][L]:.2e} (m={a0[0]},th={a0[1]:.2f})" if a0 else "--"
        s2 = f"{worst[2][L]:.2e} (m={a2[0]},th={a2[1]:.2f})" if a2 else "--"
        print(f"  {L:<5}| {s0:<29} | {s2}")
    print(
        "\n(l=500 = convention anchor; jht is independently known correct there to "
        "~1e-12 vs scipy.\n l>1000 rows = the fp64 roundoff-growth law of the recursion.)"
    )


if __name__ == "__main__":
    main()
