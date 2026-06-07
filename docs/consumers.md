# Using jht as a dependency (the downstream seam)

jht is **standalone and consumer-agnostic** — it has no knowledge of any
particular caller and contains no consumer-specific code. This note documents the
stable seam a downstream project depends on, written so jht never needs to import
or reference the consumer. The motivating consumer is `bk-jax` adopting jht *in
place of ducc0* for its GPU/differentiable tier, but nothing below is BK-specific.

> **Backend wiring lives in the consumer.** Any environment-variable / registry
> dispatch (e.g. a `BK_JAX_SHT_BACKEND=jht` switch) is implemented on the
> consumer side. jht just exposes the functions.

## What jht guarantees

- **Runtime deps = `jax` + `numpy` only.** No compiled extension; healpy/ducc0
  are validation oracles, never runtime deps.
- **Conventions** (verified vs healpy 1.19.0 / ducc0 0.41.0; see
  [`design.md`](design.md)): healpy m-major triangular aₗₘ packing, orthonormal
  Yₗₘ with the Condon–Shortley phase, HEALPix-internal (COSMO) polarization. A
  consumer storing the same conventions can pass arrays through directly.
- **float64 is opt-in per entry point.** The caller sets
  `jax.config.update("jax_enable_x64", True)` before allocating; library code
  never mutates global config.

## The operator path (the ducc-replacement seam)

| need | jht function | notes |
|------|--------------|-------|
| `aₗₘ → map` | `jht.synthesis(alm, nside, lmax, spin)` | spin ∈ {0, 2}; spin-2 takes `(E,B)`, returns `(Q,U)` |
| `map → aₗₘ`, **exact transpose** `Sᵀ` | `jht.adjoint_synthesis(m, nside, lmax, spin)` | the *unweighted* adjoint — the operator a matrix-free solver / VJP needs, **not** an inverse |
| `map → aₗₘ`, approximate **inverse** | `jht.map2alm(m, …, niter=3)` | ring weights + Jacobi iteration (the healpy-`map2alm` analogue) |
| masked analysis | `jht.pseudo_alm`, `jht.deconvolve` | zero-fill pseudo-aₗₘ; cut-sky CG deconvolution |
| Wiener / MUSE inner solve | `jht.wiener`, `jht.constrained_realization` | `(SᵀN⁻¹S + C⁻¹)⁻¹SᵀN⁻¹m` (per-pixel `N⁻¹` + Cℓ prior); posterior draws |

`adjoint_synthesis` is the strict transpose of `synthesis` in the
`(2 − δ_{m0})`-weighted aₗₘ inner product — this is the operator to drop into a
CG/Wiener solve (it is exactly what `jht.deconvolve` and `jht.wiener` build on).

## The gradient path

For autodiff / optimization / field-level inference, prefer the **real-DOF**
layer — plain ℝⁿ→ℝᵐ with no complex-conjugate convention subtlety
(`jacfwd ≡ jacrev`, finite differences unambiguous):

- `jht.synthesis_real(x, nside, lmax, spin)` — real-DOF vector → map
- `jht.analysis_real(maps, …)` — map → real-DOF vector
- `jht.alm_to_real` / `jht.real_to_alm` / `jht.n_dof` — the isometry `T`
- `jht.bandpower(alm, lmax, spin)` — angular auto-power `C_ℓ` (== `healpy.alm2cl`)

The complex transforms also differentiate under native JAX AD directly.

## The convention bridge (the bk-jax `2·conj` gotcha, resolved)

JAX's native reverse-mode returns the cotangent in the JAX convention, which
relates to the strict math adjoint by the `(2 − δ_{m0})` metric `G`:

```
jax.vjp(synthesis)(v)  ==  G * conj(adjoint_synthesis(v))
G = jht.alm_metric_weight(lmax)        # 1 at m=0, 2 at m>0
```

Exact for spin-0 (all modes) and spin-2 (m>0); the spin-2 m=0 modes carry an
extra E/B-mixing phantom term (physical fields have real a_{ℓ0}; the real-DOF
layer drops it). If you hand-write adjoints/VJPs against jht, use this identity;
if you stay on the real-DOF layer you never see it. Full derivation in
[`design.md`](design.md) §Differentiability.

## The accuracy boundary (do not conflate tiers)

jht serves the **GPU / differentiable tier** where the HEALPix ~1e-3 sampling
floor is acceptable; weights + iteration reach ~1e-13 on band-limited inputs. It
is **not** a replacement for ducc on a purity-critical (~1e-4 E→B-leakage)
production path. A consumer should route only the GPU/diff workloads through jht
and keep ducc where person-decades of tuning live. See [`accuracy.md`](accuracy.md)
and [`DISCREPANCIES.md`](../DISCREPANCIES.md) (e.g. spin-2 E/B ambiguous modes
under a cut).

## GPU

The CUDA story (the `gpu` pixi env, the x64 requirement, the parity harness) is
in [`gpu.md`](gpu.md). A consumer installing jht into its own CUDA env needs the
same x64 opt-in and a CUDA `jaxlib`.

## Public API surface

`jht.__all__` is the supported surface (`import jht; jht.<name>`). Lower-level
geometry (`jht.healpix.RingInfo`, the recursion in `jht._recursion`) is internal
and not part of the dependency contract.
