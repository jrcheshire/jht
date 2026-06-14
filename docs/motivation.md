# Why jht exists

jht is a clean-room, pure-JAX spherical harmonic transform. The established
high-accuracy libraries (ducc0, healpy) are excellent but are compiled, CPU-only
C++. That leaves a gap for three kinds of work:

- **GPU execution.** Forecasting sweeps, field-level inference (an inner
  Wiener-filter solve, run many times), and large simulation ensembles are
  accelerator-hungry; a CPU-only transform caps them at CPU throughput.
- **End-to-end differentiability.** JAX-native autodiff lets gradients flow
  through the transform — including, on the off-grid path, through the evaluation
  pointings themselves — which a compiled C++ FFI does not expose.
- **Dependency control.** Pure JAX + numpy installs with `pip` and needs no
  C++/CUDA build toolchain, removing a recurring source of CI and install
  breakage.

The leading JAX-native SHT (s2fft) handles spin-0 well, but at the time jht was
written its **spin-2 HEALPix analysis had a precision gap** at production
band-limits — and spin-2 (polarization) is exactly the case that matters here.
Rather than depend on, or fork, a large codebase that was inaccurate in the
corner we needed, jht reimplements the transform clean-room — informed by the
published methods (libsharp / Kostelec–Rockmore for the spin-2-safe ℓ-recursion;
the Price–McEwen branch-free renormalization technique) — with spin-2
correctness as an explicit validation gate.

## Scope, honestly

HEALPix has no sampling theorem, so any HEALPix SHT is approximate. jht targets
the tier where the ~1e-3 sampling floor (closed to ~1e-13 on band-limited maps
with ring weights + iteration) is acceptable — GPU-accelerated, differentiable
analysis — rather than displacing the most accuracy-critical (~1e-4) CPU
production pipelines. Owning a transform means owning its numerics, chiefly
recursion stability at high ℓ (the classic high-ℓ underflow, solved with
libsharp-style log-renorm scaling — measured exact to ℓ = 32000) and the
quadrature weights; both hold for the spin-0/2 regime jht targets (validated to
nside ≤ 4096, ℓ_max ≲ 6000).

See `docs/design.md` for the algorithm and conventions, and `docs/accuracy.md`
for the measured accuracy and the validation contract.
