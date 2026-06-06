# jht — JAX Harmonic Transforms

JAX-native spherical harmonic transforms: GPU-capable, fully differentiable,
and **dependency-controlled** (pure JAX + numpy at runtime — no compiled C++
extension). Scoped first to the BICEP/Keck regime (spin-0 + spin-2 on the
HEALPix ring pixelization, ℓ_max ≲ 1000), but written cleanly so it can serve
as a general transform dependency.

**Status (2026-06-06): scaffolding only.** No transforms are implemented yet.
This repository was initialized to carry the design forward in a later working
session. Start with:

- [`CLAUDE.md`](CLAUDE.md) — scope, conventions, decisions, how to work here.
- [`ROADMAP.md`](ROADMAP.md) — the phased plan, with hard accuracy gates.
- [`docs/motivation.md`](docs/motivation.md) — why jht exists (the decision record).
- [`docs/design.md`](docs/design.md) — technical design + conventions.

The first action for the development session is **Phase 0** in `ROADMAP.md`:
a bounded feasibility spike with go/no-go accuracy gates against ducc0 / healpy.
