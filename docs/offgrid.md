# Off-grid synthesis (`synthesis_general`) — the NUFFT path

`jht.synthesis_general` / `jht.adjoint_synthesis_general` evaluate a band-limited
field at **arbitrary points** (θ_j, φ_j) on the sphere — detector pointings, not
the HEALPix grid. This is the JAX-native, differentiable replacement for ducc0's
`sht.experimental.synthesis_general` / `adjoint_synthesis_general` (the
synthesis-at-arbitrary-pointings path): on-grid SHT + this NUFFT together cover
ducc0's full on-/off-grid transform surface.

## Interface

```python
import jax; jax.config.update("jax_enable_x64", True)
import jht

field = jht.synthesis_general(alm, loc, *, spin, lmax, epsilon=1e-10)
alm   = jht.adjoint_synthesis_general(field, loc, *, spin, lmax, epsilon=1e-10)
```

- `alm` — healpy m-major triangular packing: `(alm_size(lmax),)` for `spin=0`,
  `(2, alm_size(lmax))` (E, B) for `spin>0`.
- `loc` — `(npts, 2)`, `loc[:,0]=θ ∈ [0,π]`, `loc[:,1]=φ ∈ [0,2π)`.
- `spin` — `0, 1, 2, 3`. spin-0 returns `(npts,)`; spin>0 returns `(2, npts)`
  (the two real field components, e.g. Q, U for spin-2).
- `epsilon` — NUFFT accuracy (default `1e-10`, ducc's production setting).

Matches ducc's convention (verified bit-for-bit on the physical modes), so it is a
drop-in: same alm packing, same `loc` layout, **no `psi`** argument (the spin
orientation rotation stays on the caller's side, as in ducc).

## Two stages and prepared point sets

The transform splits at the NUFFT's σ-oversampled grid (`jht.offgrid_grid_shape(lmax,
epsilon)`; the NUFFT's internal torus grid, not a HEALPix map):

```python
pts  = jht.prepare_points(loc, lmax=lmax, epsilon=1e-10)  # host side, once per point set
grid = jht.sky_to_grid(alm, spin=spin, lmax=lmax)         # a_lm -> grid (point-independent)
f    = jht.grid_to_points(grid, pts, loc, spin=spin)      # == synthesis_general(alm, loc, ...)
g    = jht.points_to_grid(f, pts, loc, spin=spin)         # exact transpose of grid_to_points
alm  = jht.grid_to_sky(g, spin=spin, lmax=lmax)           # == adjoint_synthesis_general(f, loc, ...)
```

- **One sky, many point sets.** `sky_to_grid` / `grid_to_sky` hold the Legendre step,
  the double-Fourier-sphere extension and the FFTs (~85–90 ms each at lmax 256), which
  do not depend on the points. Grids from several point blocks add, so a transposed pass
  over many blocks is `grid_to_sky(sum of points_to_grid(...))`, one Legendre step.
- **Grid type.** Real float64 for spin 0 (the spin-0 field is the real part and the
  kernel weights are real), complex128 for spin > 0. The stage pairs are exact
  transposes in the `(2 − δ_{m0})` metric on aₗₘ and `Re Σ conj(g₁) g₂` on the grid.
- **Prepared points.** `prepare_points` fixes each point's stencil cell and groups the
  points by cell, so `points_to_grid` sums each group's `(W, W)` block and writes it once
  (on CPU: 6–7× the per-point spread on dense sets, ~2× on sparse ones; on GPU the
  per-cell scatter-add is faster and is used instead; `performance.md`).
  Groups hold up to `group_size` points (default 16); a cell with fewer uses the next
  power of two, so padding stays under 2× at any density. Nothing is assumed about the
  pointing. Prepare the points with the grid's `lmax` and `epsilon`: the grid shape is
  checked, an `epsilon` mismatch is not.
- **`loc` and the pointing gradient.** With `loc` passed, the kernel weights are
  recomputed at the prepared cells, and the result is differentiable in `loc` exactly as
  `synthesis_general` is. Every point must still lie in its prepared cell (motion within
  the cell is exact). If not, a call with concrete `loc` raises `ValueError`, and a
  traced call returns NaN throughout. Re-prepare when the pointing moves further.
- **Stored weights.** `prepare_points(..., store_weights=True)` keeps the kernel weights
  (`2W` float64 per point, 224 B at ε = 1e-10) and the calls take `loc=None`: about 1.6×
  faster forward and 1.1× transposed, no pointing gradient.
- **Real-DOF.** `jht.sky_to_grid_real` / `jht.grid_to_sky_real` compose the stages with
  the real isometry `T`, as `synthesis_general_real` does.

## Algorithm — Double-Fourier-Sphere + 2D type-2 NUFFT

This is ducc's `sphere_interpol` method (Reinecke, Belkner & Carron 2023,
arXiv:2304.10431), clean-roomed in pure JAX. Per helicity component:

1. **Legendre step** (`alm → per-m field coefficients`): evaluate `F_m(θ_i)` on a
   Clenshaw–Curtis θ-grid (`ntheta_s = good_size(lmax+1)+1` rings on `[0,π]`,
   including the poles). This **reuses the on-grid recursion** (`synth_contract` at
   the CC colatitudes = ducc's `alm2leg`); the recursion is libsharp-style with the
   branch-free log-renorm, general in spin.
2. **DFS θ-extension** (`→ 2D Fourier coefficients F(k,m)`): extend θ from `[0,π]`
   to `[0,2π)` (the double-Fourier-sphere doubling, `2·ntheta_s−2` samples) with the
   spin parity `_sf̃(2π−θ,φ) = (−1)^s _sf(θ,φ+π)`, then FFT in θ.
3. **2D type-2 NUFFT** (`F(k,m) → field at (θ_j,φ_j)`): grid-correct (deconvolve by
   the kernel FT), zero-pad to the σ-oversampled grid, IFFT2, then **ES-kernel**
   (`exp(β(√(1−(2t/W)²)−1))`, Barnett+ 2019) separable interpolation onto the
   points. Implemented in `jht._nufft` (type-2 `nufft2d2` + its exact transpose
   type-1 `nufft2d1`). On CPU the transposed (spreading) step writes one `(W, W)`
   window per point, or per group of prepared points, onto the grid padded by `W − 1`,
   then folds the padding back onto the periodic grid; on other platforms it is a
   per-cell scatter-add (selected when the program is lowered).

The adjoint reverses each step (type-1 NUFFT → transposed DFS/FFT → `adjoint_contract`).
Code: `jht._nufft` (the 2D NUFFT) and `jht.offgrid` (the DFS + spin channels); the
θ-Legendre step is the on-grid recursion evaluated at the CC colatitudes.

### Accuracy knobs (ε → σ, W)

The kernel half-width `W` and oversampling `σ` set ε; `jht._nufft._KERNEL_DB` maps a
requested `epsilon` to `(W, σ)` (cheapest meeting the request). Pinned empirically
against the exact direct sum (`scripts/offgrid_sweep.py`):

| epsilon | (W, σ) | measured max-abs vs exact sum |
|---|---|---|
| 1e-10 (default) | (14, 2.0) | ~1e-10 (lmax 256) … ≤6e-10 (lmax 512) |
| 1e-8 | (10, 2.0) | ~3e-7 |
| 1e-6 | (8, 2.0) | ~2e-5 |

(lmax 256 unless noted; reproduce with `scripts/offgrid_sweep.py`. \|f\| ~ a few hundred,
so ε=1e-10 is ~1e-12 relative.)

This is a **separate accuracy tier** set by the NUFFT ε (matching ducc), not the
on-grid weighted ~1e-13. Do not conflate the tiers.

### Memory at scale

The dominant intermediates scale with the number of points, not with the oversampled
DFS grid (`nk·nm` complex, ~0.26 GB at lmax=1000, σ=2):

- the forward stencil gather (`synthesis_general`, `grid_to_points`) and the per-point
  spread of `adjoint_synthesis_general` hold `(Npts, W, W)` arrays, ≈ `Npts·W²·16 B`
  (complex; half for the real spin-0 spread): ~3.1 GB at N=1e6, W=14 (ε=1e-10).
  Quadratic in `W`, so a looser ε is the memory lever: ε=1e-8 (W=10) halves it.
- the grouped spread of `points_to_grid` holds one `(W, W)` block per group, so its
  footprint falls with the points per cell: 1.4 GB peak process RSS for 1M points
  in a dense patch (lmax 256, ε=1e-10, spin 2).

## Differentiability — alm **and** pointing

Both gradients are exact under JAX's **native autodiff** (no custom rule):

- **alm gradient** — `synthesis_general` is alm-linear; `jax.vjp(synthesis_general)(v)
  == G·conj(adjoint_synthesis_general(v))` with `G = jht.alm_metric_weight(lmax)`
  (the same `(2−δ_{m0})` bridge as on-grid).
- **real-DOF layer** — `synthesis_general_real` (`S_g ∘ T⁻¹`) and its exact transpose
  `adjoint_synthesis_general_real` (`T ∘ S_gᵀ`) compose the above with the real
  isometry `T` of `jht.masked` (the off-grid duals of `synthesis_real`): a plain
  real-linear `ℝⁿ→ℝᵐ`, so `jax.vjp` returns the **exact** transpose with no `2·conj`
  bridge and `jacfwd ≡ jacrev` holds exactly. The recommended gradient-based entry
  point to this path; spin 0–3.
- **pointing gradient** — `∂field/∂loc` is exact to the field's true derivative
  (~1e-12 vs the analytic derivative). The ES-kernel window index is
  `stop_gradient`-frozen (it is genuinely locally constant), so AD flows through the
  smooth kernel weights only; `jacfwd ≡ jacrev` is preserved. **This is the capability
  ducc's FFI raises `NotImplementedError` on** — a consumer can differentiate straight
  through the pointing instead of hand-building analytic position-derivative templates.

## Conventions and known differences vs ducc (see `DISCREPANCIES.md`)

- **−spin channel `(−1)^s`:** the E/B → ±spin map is `+s a = −(aE+iaB)`,
  `−s a = −(−1)^s(aE−iaB)`. The `(−1)^s` is invisible for even spin (the on-grid 0/2
  path never exposed it) and flips odd spin (1, 3); matched to ducc.
- **spin>0 m=0 phantom:** synthesis keeps both real components at m=0, so the
  unphysical `Im(a_{l,0})` matters; jht's strict adjoint carries it (the adjoint
  identity holds to ~1e-16), ducc zeros it. Harmless for physical (real-`a_{l,0}`)
  skies; the ducc cross-check compares the physical m≥1 modes.

## Validation (`tests/test_offgrid.py`, `tests/test_offgrid_prepared.py`, `tests/test_nufft_spread.py`; spin 0–3)

| gate | oracle | result |
|---|---|---|
| forward | exact direct sum `Σ a_lm {}_sY_lm` | ≤1e-9 (tier; ~1e-11 measured) |
| forward / adjoint | ducc0 `synthesis_general` | ~1.5e-11 (drop-in) |
| adjoint identity | `⟨Sa,v⟩=⟨a,Sᵀv⟩_G` | ~1e-16 |
| alm-VJP bridge | `G·conj(adjoint)` | ~1e-14 |
| loc/pointing grad | analytic field derivative | ~1e-12 |
| window / grouped spread | per-cell scatter-add | ~1e-15 |
| stage compositions | `synthesis_general` / `adjoint_synthesis_general` | ~1e-15 |
| stage adjoint identities | `Re⟨Pg,v⟩=Re⟨g,Pᵀv⟩`, `Re⟨Sa,g⟩=⟨a,Sᵀg⟩_G` | ~3e-14 |
| stages, forward | ducc0 `synthesis_general` | ~3e-11 |
| prepared-path pointing grad | through `synthesis_general` / its adjoint | ~5e-16 |

## References

- Reinecke, Belkner & Carron 2023, arXiv:2304.10431 (the ducc method).
- Barnett, Magland & af Klinteberg 2019, arXiv:1808.06736 (the ES kernel / FINUFFT).
- Townsend, Wilber & Wright 2016, SIAM J. Sci. Comput. 38(4) (double Fourier sphere).
