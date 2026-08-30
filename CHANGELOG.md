# Changelog

All notable changes to SWEEP are documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **ViscoAcoustic: Zhu & Harris (2014) nearly constant-Q equation, per-edge
  free surface, and a CUDA backend (`impl='c'`).**  The equation now
  implements the paper's decoupled fractional Laplacians (eq. 10/11,
  doi:10.1190/geo2013-0245.1): Kjartansson power-law dispersion
  `c_p = c0*(w/w0)^gamma` (measured exponent matches theory to 0.4% over the
  band; the amplitude decay matches `exp(-pi f r/(Q vp))` to 2%) and a
  `k^(2*gbar+1)` loss filter, replacing the legacy frequency-independent
  effective-velocity coefficients.  Both terms ride on the CPML step as
  spectral corrections that vanish identically at `gamma -> 0`, so both
  switches off still reduces bit-exactly to `Acoustic`.  Heterogeneous media
  freeze the fractional exponent at the average `gbar` (the paper's
  freezing-unfreezing), including its Q-derivative, on every backend.  The
  reference frequency `omega` is a real model parameter with a genuine
  gradient.  Per-edge free surfaces: every `free_surface=` form the acoustic
  solver accepts.  The CUDA backend reuses the acoustic2d CPML kernels with
  the paper's velocity and applies both spectral terms per step through
  ATen/cuFFT with a hand-derived exact adjoint — forward, full /
  chunk-checkpoint / recursive-checkpoint backwards and RTM (closed-box
  c-vs-eager gradients ~1e-6 for wavelet/vp/Q/omega; ckpt matches full
  storage bitwise on the record).  Boundary saving is refused (the
  dissipative global-FFT term is not reverse-reconstructible from boundary
  strips): the impl='c' default memory strategy falls back to 'full', and an
  explicit boundary request raises.  Note: gradient tests reference the
  UNCOMPILED eager step — inductor's fused pow/ln backward perturbs the
  Q-gradient cotangents enough to be amplified to ~5e-3 by the
  `ln(vp/omega)`-weighted chain; the plain eager step matches the CUDA
  adjoint at ~5e-7.
- `ForwardInput/BackwardInput.eq_aux` — equation-specific auxiliary tensors
  (opaque to the shared autograd wrapper); ViscoAcoustic uses it for its |k|
  FFT grid.

### Changed
- `SecondOrderEquation._apply_free_surface` — the per-edge pressure-release
  zeroing moved from `Acoustic` to the shared base (bit-identical) so
  ViscoAcoustic reuses it.
- `ViscoAcoustic.prepare_models` maps (vp, Q, omega) -> (vp_step, A) once per
  forward (shared by the eager and CUDA paths; the eager step no longer
  recomputes the dispersion/damping coefficients every time step).

## [0.2.0] - 2026-08-24

### Added
- **Domain decomposition (`sweep.parallel`).**  `ModelParallel` splits one
  model into tiles — one GPU per tile — and exchanges a halo every time step,
  so a single shot is solved cooperatively instead of replicated.
  `MeshTopology(py, px, shot_groups=...)` describes the rank grid and composes
  with shot parallelism; `pad_to_mesh` / `unpad_from_mesh` size a model to the
  tile multiple.  Forward and backward are plain autograd, and the gradient is
  **bit-identical** to the single-domain gradient on fp32 GPU boundaries.
  Supported equations: `Acoustic` (2-D), `Acoustic3D`, `AcousticVRZ3D`,
  `Elastic` (2-D), `Elastic3D`; anything without stepped kernels is refused at
  construction.  See the [Domain decomposition](docs/user-guide/parallel.md)
  guide and notebooks 25 / 26.
- **CPU-staged boundary storage under domain decomposition.**
  `BoundaryOptions(storage="cpu")` now works for the Acoustic 2-D/3-D DD
  backward, so a tile whose boundary ring does not fit in GPU memory has a
  fallback instead of a hard stop.  The gradient is **bit-identical** to
  gpu-direct on fp32 and bf16, and within each dtype's own run-to-run floor on
  fp16/int8; it composes with `tail_steps`.  `storage="disk"` under DD, and
  cpu staging on a single-tile mesh, are still refused — by name, at the first
  backward.  Elastic DD remains gpu-direct only.  See
  [Domain decomposition](docs/user-guide/parallel.md#boundary-storage-under-dd).
- **`BoundaryOptions.tail_steps`** (dict spelling:
  `boundary_saving_config={'tail_steps': K}`): keep only the last `K` steps of
  the boundary ring and stop the reverse loop there.  For steady-state
  objectives (frequency-selection / encoded FWI) whose adjoint source is zero
  outside a probe window, the truncated gradient is the same gradient; the
  reverse pass and the ring both shrink proportionally.  Acoustic 2-D/3-D,
  boundary-saving backward only, and it composes with domain decomposition.
- **CPML aux strip allocation.**  `psi`/`zeta` (acoustic) and the elastic
  memory variables now live in per-axis slabs — the PML band plus stencil
  reach — instead of full grids, for `Acoustic`, `Acoustic3D`, `Elastic` and
  `Elastic3D` on `impl='c'`.  Gradients are bit-for-bit unchanged; only the
  allocation shrinks.  Equation authors opt in via the new `CUDALayoutSpec`
  fields `pml_slot_axes`, `checkpoint_slot_axes` and `adjoint_pml_slab`.
- Per-edge free surface (deepwave-style).  `Propagator(free_surface=...)` now
  accepts a per-edge spec — an edge-name list (`['top', 'left']`), a
  length-`2*ndim` bool mask (`[z0, z1, x0, x1]`), or a dict — in addition to the
  historical `bool` (top-only): a free surface on any subset of the domain
  faces.  `abcn` likewise accepts a per-edge list for an independent PML
  thickness per face.  The **eager** backend supports it for **Acoustic and
  Elastic 2-D** (all four edges, gradient-consistent); the compiled `impl='c'`
  backend supports it for **Acoustic and Elastic 2-D on CUDA** (all four edges,
  including z∩x corners) — bit-exact vs eager forward, adjoint-gradient cosine
  ~1.  `free_surface=True` / a scalar `abcn` stay bit-for-bit unchanged.
  On `impl='c'` all three CUDA backward memory modes — **full**
  (`use_ckpt=False`), **checkpointing** (`use_ckpt=True`), and **boundary
  saving** — are gradient-consistent for every edge and z∩x corner
  (adjoint cosine ~1 vs eager).  Other unimplemented requests raise a clear
  `NotImplementedError` pointing at `impl='eager'`: per-edge on 3-D or on
  non-migrated equations, per-edge on the CPU `impl='c'` backend, and per-edge
  PML *thickness* on `impl='c'`.
- Documentation overhaul (Phase 1, facade): rewritten landing page with
  capability cards and audience-routed navigation; README and README.zh-CN
  gained badges and a tagline block.
- `CHANGELOG.md` and `CONTRIBUTING.md` scaffolding.

### Changed
- **The gradient-memory mode is now one three-way choice** — `'full'`,
  `'boundary'` or `'ckpt'` — resolved identically for the eager and CUDA
  backends by `resolve_memory_strategy`, and selected in one place with
  `memory=MemoryOptions(strategy=...)`.  The legacy `use_ckpt` /
  `boundary_saving_config` knobs still work and resolve into the same choice.
  Three behaviour changes come with it:
  - `boundary_saving_config={'enabled': True}` now actually runs the boundary
    backward on both backends.  It used to lose silently to the `use_ckpt=True`
    default, so scripts that believed they were using boundary saving were
    checkpointing (`impl='c'`) or ignoring the dict entirely (`impl='eager'`).
  - Contradictory requests raise `ValueError` instead of one path winning
    silently — `use_ckpt=True` together with an enabled `boundary_saving_config`,
    or `memory=` contradicting a legacy knob.  Knobs that *agree*
    (`memory=MemoryOptions(strategy='boundary')` with `use_ckpt=False`) are
    accepted.
  - A dict passed without `enabled=True` (e.g. `{'storage': 'cpu'}`) selects
    `'full'`, not checkpointing.
  Unchanged on purpose: no knobs at all still means boundary saving for
  `impl='c'` and checkpointing for the eager backend, and an explicit
  off-switch (`use_ckpt=False`) still means full-wavefield storage.
- `docs/user-guide/equations.md`: summary table expanded from 3 rows to cover
  all 20+ exported equation classes, grouped by physics family. Template
  reminder at the bottom replaced with a "See Also" cross-reference block.
- `mkdocs.yml`: enabled `attr_list` and `md_in_html` Markdown extensions to
  support Material grid-card layouts.

### Fixed
- **CPML aux writes on a domain-decomposition cut tile.**  With the strip
  allocation, the per-axis PML compute band reaches columns on a cut face that
  carry no slab storage; the unclamped index produced a negative offset and the
  ungated store aliased `±0` into another row's slab cell, racing its owner.
  Writes are now gated on `stored()` and read through the clamped accessor
  (`aux_rd_*`).  Only reachable with `impl='c'` acoustic + multi-GPU DD, and
  never on a released build; single-domain runs are bit-for-bit unchanged.
- **The boundary spec can no longer be changed after construction.**
  `prop.free_surface = ...` (and `fs_faces`, `abcn`, `pad`, `pml_type`,
  `topography`) used to land on the `PropTorch` wrapper, where it shadowed the
  backend's value: the read-back reported the new setting while every kernel
  kept the old one — a script could believe it had switched a free surface on
  and quietly model without one.  The write now raises `AttributeError` and
  points at the constructor.
- **DAS Mu 2-D/3-D were non-deterministic on `impl='c'`.**  `das_mu*/kernels.cuh`
  includes the elastic kernels, which address the CPML memory variables through
  the solver's aux slabs, but the DAS drivers never installed them: the row
  stride collapsed to zero and every row aliased the first, so the same input
  gave a different answer each run (plain single-GPU forward, with or without a
  free surface).  `AcousticLSRTM` and `AcousticVRZ3D` borrow the acoustic
  kernels the same way but never launch the slab-addressed ones and were
  unaffected.  `test/test_c_aux_slab_repeatability.py` now pins the class.
- **`AcousticVRZ3D` boundary staging with `storage_dtype='fp16'`/`'int8'`.**
  The 2-D and 3-D VRZ paths now pass `boundary_tangent_pad = M` into the
  effective-boundary saver, fixing an out-of-bounds staging copy on the
  low-precision ring.

## Earlier history

Earlier release notes will be backfilled from the commit history. For now,
see the
[GitHub commit history](https://github.com/DeepWave-KAUST/sweep/commits/dev)
for changes prior to this entry.

[Unreleased]: https://github.com/DeepWave-KAUST/sweep/compare/v0.2.0...dev
[0.2.0]: https://github.com/DeepWave-KAUST/sweep/compare/v0.1.0...v0.2.0
