# Changelog

All notable changes to SWEEP are documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Smaller wheel: every CUDA kernel is compiled once.**  A kernel template
  launched from a header -- a driver hook or a `LAUNCH_*` macro -- was
  instantiated (five stencil orders x every arch of the fat binary) in every
  translation unit that included it, called or not: an equation's forward and
  backward units, and every equation sharing a kernel family (`Elastic` /
  `DASMu` / `ViscoElastic`, `Elastic3D` / `DASMu3D`, `Acoustic` /
  `ViscoAcoustic`).  Kernels defined in headers were compiled into each
  includer the same way.  Each family is now instantiated once, in its
  `kernels.cu`, and launched through a table of its stencil-order
  specializations (`launch/by_order.cuh`): the cu12 core drops from 157.3 to
  70.8 MB, the cu13 core from 189.9 to 85.8 MB and the wheel from 93.1 to
  42.4 MB (PyPI's per-file limit is 100 MB).  The machine code of every kernel
  on every arch is byte-identical to 0.3.3's.

## [0.3.3] - 2026-10-02

### Added
- **ViscoElastic: 2-D visco-elastic equation (generalized standard linear
  solid), eager and CUDA (`impl='c'`) backends.**  The rheology SPECFEM2D
  (and so SeisFlows) uses:
  `n_sls` Zener bodies (default 3) with relaxation times fixed by the band,
  memory variables advanced with the trapezoidal rule, on the `Elastic`
  staggered-grid step (shared, not copied: the velocity-gradient/CPML/free-
  surface block of the stress sub-step is factored into
  `elastic_velocity_gradients`; `Elastic` and `DASMu` stay bit-identical).
  Models are `vp, vs, rho, Qp, Qs`, velocities given at the reference
  frequency `f_ref` (SPECFEM's `READ_VELOCITIES_AT_f0`), and all five carry
  gradients.  Each mechanism's strength is the least-squares constant-Q fit,
  linear in the strengths (a rational function of 1/Q, evaluated elementwise
  with an analytic backward), so Q stays within ~5% of target over the band
  for Q >= 10 (~9% at Q = 3).  `Q = inf` reduces bit-exactly to `Elastic`.
  The flat free surface (per edge) solves the surface normal strain rate so
  the traction stays zero through the memory variables.
  `ViscoElastic.unrelaxed_velocities` gives the velocities the explicit
  update runs at, for the CFL check.  Checked against the exact 2-D
  homogeneous Green's function (P and S spectral ratios to 2e-4 / 2e-3) and
  against SPECFEM2D with a free surface (visco/elastic spectral ratio: median
  deviation 0.3-2% over 4-25 Hz; time-domain misfit no larger than the
  elastic-vs-elastic baseline).  The CUDA backend is a set of staggered-
  skeleton traits derived from elastic2d's (`visco_elastic2d/driver_traits.cuh`,
  a template on the mechanism count): Elastic's velocity kernel, source /
  receiver / rho-correction plumbing and adjoint stencil transposes, plus the
  visco stress update and its hand-derived exact adjoint (memory variables,
  surface solve and its material derivatives) with the gradient imaging fused
  into it in every mode.  `full`, chunk- and recursive-checkpoint backwards,
  all three bitwise equal.  c vs eager: records and all five gradients agree
  at the Elastic c-vs-eager floor (<= 8.8e-5 over 72 source x receiver x
  geometry x free-surface combinations with PML, <= 1.3e-5 in a closed box,
  surface rows included).
  Not supported: boundary saving (default falls back to `full`), topography,
  domain decomposition (all refused explicitly).

## [0.3.2] - 2026-10-01

### Fixed
- `impl='eager'` no longer fails when `torch.compile` cannot build the step,
  e.g. on CPU with g++ older than 10 (`InductorError: CppCompileError`, raised by
  the first forward of a plain CPU run, since `use_compile` defaults to True).
  The step runs uncompiled instead, with one `RuntimeWarning`; results are the
  same. Errors raised by the step itself still propagate.
- A `pml_type` the equation was not written for is refused with a `ValueError`
  at construction. It used to reach the kernels, which read the PML profiles
  positionally: `Elastic` with `pml_type='cpmlr'` and `impl='c'` aborted the
  Python process on a C assert. Each equation's options are in its new
  `supported_pml` property (also in `Equation.defaults()`); every equation has
  one, except `Acoustic1st`, which also takes `'spml'`. Leave `pml_type` unset.
  The README, docs, notebooks and examples no longer pass it.

## [0.3.1] - 2026-10-01

### Added
- `Full`, `BoundarySaving` and `Ckpt` are exported from `sweep.propagator`
  (only the deprecated `MemoryOptions` family was).

### Fixed
- `sweep.backend.torch.binding.diagnostics()["shim"]` read `"ctypes"` on a
  `SWEEP_BUILD_CUDA=1` install, whose compiled `sweep._C` is the pybind shim;
  it now reads `"pybind"` there, as under `SWEEP_JIT_FULL=1`.
- A typed strategy inside `cuda_options=CUDAOptions(memory=...)` did not reach
  the `impl='c'` backend: `Ckpt()` and `Full()` raised a "conflicting
  gradient-memory mode" `ValueError` at construction, and
  `BoundarySaving(storage='cpu', pinned_memory=True)` silently built the
  default GPU ring.  It is now read exactly like `memory=`.
- `AcousticCurvilinear` / `ElasticCurvilinear` counted as compiled bindings
  because their `_C()` stub (which only refuses `impl='c'`) was callable, so
  `impl=None` on a CUDA device picked `'c'` and raised `NotImplementedError`.
  A binding now needs a `C_NAME`: both report no binding (the CLI prints
  `no`), `'auto'` runs them eager, and an explicit `impl='c'` falls back with
  a `UserWarning` like any eager-only equation.
- `ModelParallel` decomposed a propagator built with `topography=` as a
  flat-top problem, silently (the tiles never carried the surface, and
  boundary saving under a surface is wrong anyway).  It now refuses it with
  `NotImplementedError`.

### Documentation
- README and the installation page are shorter; the release-core build rules
  moved to *Developer › Release cores*.
- The API reference, user guide, examples and notebooks were audited against
  the 0.3.0 code: wrong signatures and defaults, removed or renamed APIs,
  deprecated spellings shown as current, and stale JIT-era install claims
  were fixed.

## [0.3.0] - 2026-09-30

### Added
- **The wheel ships a prebuilt CUDA core; first use of `impl='c'` no longer
  needs nvcc.**  The CUDA kernels are built once, ahead of the wheel, as a
  torch-free `libsweep_core.so` (every `.cu`, nvcc) behind a plain C
  interface; the torch side binds to it without compiling (the ctypes layer
  above -- the compiled pybind shim, `module.cpp` + the CPU binding, is the
  `SWEEP_JIT_FULL=1` developer path).  It is one wheel, carrying one core per
  CUDA major as a fat binary (`sweep/lib/cu12/` and `sweep/lib/cu13/`, each a
  `libsweep_core.so` + a `core.json` sidecar with ABI version, CUDA release,
  archs, PTX and sha256; the cu13 entry below has the per-core coverage), so
  `pip install sweepx` followed by the first `impl='c'`
  call, or `sweep.precompile()`, compiles nothing: no nvcc and no toolkit.
  `sweep.backend.c.jit.core_path()` picks the shipped core when its ABI, CUDA major
  and archs fit this process (PTX makes newer cards fit; a GPU outside the
  shipped archs and older than the shipped PTX, e.g. Pascal sm_6x, does not) and
  otherwise falls back to the unchanged local core build, which is also what a
  torch built for another CUDA major, an sdist or a clone gets, since neither
  carries a core; `SWEEP_CORE=<path>` overrides the lookup,
  `sweep.backend.c.jit.shipped_core_info()` reports what was chosen and why, and
  `sweep.backend.torch.binding.diagnostics()` gains a `shipped_core` key with
  the same answer.  `can_compile()` is satisfied by a fitting core with no nvcc
  on the machine, and `python -m sweep.build --no-gpu-required` no longer
  demands `TORCH_CUDA_ARCH_LIST` when a shipped core fits (the arch list only
  names the target of a local core build).  cuFFT is linked dynamically and
  preloaded from torch's `nvidia-cufft` wheel (or `<cuda_home>/lib64`) before
  the core is loaded; the optional extra `sweep-solver[cuda12]` pulls
  `nvidia-cufft-cu12` when a torch wheel did not.
  Release side: `python -m sweep.build --core [--archs ...] [--out DIR]
  [--cuda-home DIR]` produces the core and its sidecar without a GPU (`--out`
  defaults to `sweep/lib/<tag>` inside the installed package, `src/sweep/lib/<tag>`
  in a clone).  The wheel's platform tag is the build host's glibc
  (`manylinux_<glibc>_x86_64`) and the core binds the host's `libstdc++`, so the
  release builds both cores and the wheel inside PyTorch's `manylinux2_28-builder`
  containers (`utils/build_cores_manylinux.sh`): the shipped cores need only
  `GLIBC_2.17` / `GLIBCXX_3.4.22` and the wheel is tagged `manylinux_2_28`.
  Also: the package now really imports on Python 3.9
  (`propagator/options.py` evaluated a `str | None` annotation at runtime),
  and `sweep.build --core` no longer needs torch (its lock and its arch list
  do not import it), which is what lets it run in that container.  Measured
  size (both cores, container build): the cu12 `.so` is 127 MiB, the cu13 `.so`
  155 MiB, and the two-core wheel about 76 MiB, against PyPI's 100 MB per-file
  limit -- keep a single `+PTX` entry per core, the newest arch.  A tree holding a core turns the wheel
  into a platform wheel, otherwise it stays `py3-none-any`.  `src/sweep/lib/` is
  git-ignored and pruned from the sdist, so build the core, then `python -m
  build --wheel` from the same tree: a bare `python -m build` packs the wheel
  from the pruned sdist and ships no core.  `SWEEP_REQUIRE_CORE` (the release
  script sets `cu12,cu13`) makes `setup.py` refuse such a tree instead of quietly
  producing a core-less wheel: `1` asks for at least one core, a comma list names
  the tags that must all be present.
- **A second prebuilt core, for CUDA 13 (`sweep/lib/cu13/`).**  The wheel
  ships cu12 and cu13 side by side and the loader picks the tag torch's CUDA
  major names, so a torch cu130 install compiles nothing either.  Per core:
  cu12 (nvcc 12.9) covers V100 and newer through H100/H200 as SASS
  (sm_70-sm_90) and Blackwell through its sm_90 PTX; cu13 covers T4/RTX 20 and
  newer (sm_75-sm_90) with Blackwell native (sm_100, sm_120 + PTX) -- no V100,
  since nvcc 13 dropped offline compilation for compute capability < 7.5 -- and
  needs driver >= 580, exactly what torch cu130 needs.  The core's rpath now
  reaches cuFFT in both pip layouts (`nvidia/cufft/lib` for the CUDA 12 wheels,
  `nvidia/cu13/lib` for the CUDA 13 ones), and the new extra `sweep-solver[cuda13]`
  pulls `nvidia-cufft` 12.x (the unsuffixed wheel is the CUDA 13 line; `-cu12`
  stays for CUDA 12) when a torch wheel did not.  Release side: `python -m
  sweep.build --core` files the core under the tag of the nvcc it uses
  (`--cuda-home`) and, without `--archs`, builds that major's recommended list
  (cu12 `7.0;7.5;8.0;8.6;8.9;9.0+PTX`, cu13
  `7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX`; an sm_70 entry under nvcc 13 is refused
  up front), so a release is one command per toolkit.  `SWEEP_REQUIRE_CORE`
  accepts a comma list of tags (`cu12,cu13`) that must all be present, naming
  the missing one, next to `1` for "at least one".  The cu13 core is a second
  `.so` (155 MiB against cu12's 127 MiB); the wheel with both is about 76 MiB.
- **`impl='c'` talks to the prebuilt core through a pure-Python `ctypes`
  layer; after `pip install` nothing compiles.**  `sweep.backend.c` fills the
  core's C structs from each tensor's `data_ptr()`, shape, strides and dtype
  and calls into `libsweep_core.so` directly, so the default path no longer
  builds a torch shim at all: no nvcc, no C++ compiler, no CUDA headers, no
  ninja run (the `ninja` dependency serves a local core build only), and no
  torch C++ ABI in the loop -- the same wheel works with any
  torch version by construction.  `nvcc` is needed only when no shipped core
  fits (a torch built for another CUDA major, a GPU outside the shipped archs
  and older than the shipped PTX) or for an sdist/clone install, and then only
  the CUDA core is compiled (`python -m sweep.build`, or the first `impl='c'`
  use).  That local build needs nvcc >= 12.4 (12.0-12.3 ship a broken
  `<cuda/std>` bf16 header; `SWEEP_JIT_ALLOW_OLD_CUDA=1` tries anyway), >= 12.8
  when the target list reaches Blackwell (sm_100 / sm_120), or a CUDA 13 nvcc,
  and is refused up front with the toolkit and the version named otherwise.  A
  local core built once is reused by later runs without nvcc, on the
  strength of a source stamp (`sources.sha256`: the csrc tree plus the nvcc
  flags) written beside it.  `sweep.precompile()` now just makes sure a core
  exists and loads it: a no-op with a shipped core, the local core build
  otherwise.  The compiled pybind shim survives as the
  developer path behind `SWEEP_JIT_FULL=1` (needs torch's C++ headers, the CUDA
  runtime headers -- torch's pip CUDA wheels bring them -- and a C++ compiler;
  nvcc only when no shipped core fits).  A core that fits a GPU only
  through its PTX is refused when the driver's CUDA release is older than the
  toolkit that emitted it (the driver could not JIT it), with the versions in
  the reason.
  `sweep.backend.torch.binding.diagnostics()` gains a `shim` key, `"ctypes"`
  or `"pybind"`.
  A core sitting under `sweep/lib/<cuN>/` is loaded as it is: in a clone that
  holds one (after `utils/build_cores_manylinux.sh`), edits under `csrc/` do not
  reach `impl='c'` until that core is rebuilt or removed (`SWEEP_CORE`,
  `diagnostics()['shipped_core']` show which core is in use).

- **``boundary_buffer=`` on every propagator**: sigma=0 cells between the
  physical box and the PML ramp.  ``None`` (the default) gives an equation that
  declares ``BOUNDARY_BUFFER_REACH`` its need, ``REACH * M + 1`` cells, under
  every memory strategy and backend; only ``AcousticVRZ`` / ``AcousticVRZ3D``
  declare one.  The buffer is cropped with the pad, so shapes and coordinates
  do not change; boundary saving refuses an explicit value below the need.

- **A persistent backward runner for ``AcousticVRZ3D``**
  (``acoustic_vrz3d_backward_bs_runner``), phase-aware, so the domain-decomposed
  backward no longer re-enters the per-call binding three times per step and
  rebuilds its whole setup each time: on a production-size 3-D grid (2x2 DD,
  4xH100) an iteration went from 240 s to 95.7 s, gradients and records
  bit-identical.  It owns its boundary saver, so it also serves host-staged
  boundaries.

- **A CPU-only test job** (`.github/workflows/tests-cpu.yml`).  Nothing in this
  repository ran the tests before, and nothing could: `pytest test/` was unable
  to return 0.  With that fixed, every push and pull request to `dev` runs the
  suite on a hosted runner -- 313 tests in about three minutes; the rest skip
  for want of a GPU or a compiled binding.  A skip is not a failure, so the job
  also asserts a floor on the number of tests that actually EXECUTED
  (`.github/scripts/assert_executed_floor.py`): without it, "the tests passed"
  and "the tests did not run" print the same summary.  The bit-exactness gate
  stays where the GPUs are.

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
- **Equation registry.**  `@register_equation(name=..., aliases=..., method=...)`
  co-locates an equation's public export names with its class, replacing the
  hand-maintained import wall and `globals()` scan in `sweep.equations`.
  `get_equation(name)`, `list_equations()`, `equation_classes()` and
  `equation_method(cls)` are the lookup side.  Adding an equation is now "write
  the module, decorate the class".  Declaring `C_NAME = "<binding prefix>"` is
  what gives a class its compiled bindings: the base derives `_C` from it as
  `{C_NAME}_forward` plus the four backward variants, so the per-equation `_C`
  boilerplate is gone.  Both are OPTIONAL — subclassing `SecondOrderEquation`
  and writing `_C` by hand still works, which is what notebook 18 does.
- **Persistent stepped runners** (`sweep._C.ForwardRunner` / `BackwardRunner`,
  plus `<equation>_forward_runner` / `<equation>_backward_bs_runner` factories
  for all ten templated equations).  A stepped or domain-decomposed propagation
  used to re-enter the extension every time step and rebuild the entire
  prologue; the runner constructs once per propagation and each step is a
  `run(it_begin, it_end, phase)` call.  Reuse requires gpu-direct boundary
  storage and no checkpointing; anything else keeps the per-call path.
  Measured against dev on DD (median of 5 interleaved rounds, Ada): the runner
  alone is worth 3.99x on 2-D elastic 600x900, 1.56x on 3-D elastic, and
  1.04-1.06x on acoustic, because what it removes is a roughly constant ~0.2-1.4
  ms of host cost per step — so the ratio is set by how much GPU work one step
  does, and equations with more fields to rebind gain most.
- **Typed gradient-memory strategies** — `Full()`, `BoundarySaving(...)`,
  `Ckpt(...)` under `sweep.propagator.options`, passed as `memory=`.  The old
  dict/flag spellings still work and are read exactly as before.
- **Declared capabilities instead of inferred ones.**  `C_NAME`,
  `C_HAS_RECURSIVE_CKPT`, `supports_image_topography[_c]`, the CUDA wavefield
  slot table and the DD admission / tail-truncation eligibility are now class
  attributes an equation declares.  Nothing infers behaviour from the class
  name any more.

### Changed
- **``AcousticVRZ`` / ``AcousticVRZ3D`` gradients**: the vp gradient images
  the second time difference ``U_{it+1} - 2U_it + U_{it-1}`` (the forward
  identity) instead of a second spatial derivative of the stored field, so the
  full, checkpointing and boundary-saving gradients agree to rounding; the full
  history gains one slot (6 in 2-D, 8 in 3-D) and the checkpoint replay buffer
  two rows.  With the default ``boundary_buffer`` the runtime grid of a VRZ
  propagator is ``M + 1`` cells wider per absorbing face, which moves the
  gradient near the PML by ~1e-2 relative on a small grid.  The exact CPML
  adjoint (Fixed) makes the backward ~22% slower and adds three adjoint zeta
  shadows (~0.1 GiB on a production-size 3-D grid).
- **Domain decomposition**: a multi-axis mesh ships every halo field of a step
  in ONE batched P2P across the cut axes instead of one round per axis (the
  strips are latency-bound: a six-field shipment on a production-size 3-D grid ran at
  6.5 GiB/s over NVLink).
- **Boundary saving no longer re-injects a source that sits in the restore
  strip.**  In the boundary-saving reverse pass the strip cells (the M+1 cells
  inside every non-cut physical edge, i.e. physical rows 0..M under a free
  surface) are restored to the true, source-included field, and the source was
  then added a second time; the strip's finite-difference `u_tt` also kept the
  source term, and the error leaked into the interior through the stencil.
  Any shot within M+1 cells of an edge was affected -- in particular a shallow
  source under a free surface, the common marine setup.  Measured before the
  fix: vp gradient vs `Full()` 4e-4..2e-2 of max|g| (source-row), ~1e-3 in the
  interior, source illumination 1.4e-2; after: <=1.2e-6, and vs eager (FP32)
  identical to `Full()` vs eager.  Fixed for Acoustic, Acoustic3D, AcousticLSRTM
  (2-D/3-D) and ElasticTTI2nd (whose strip is the outermost row/column only);
  VRZ, VTI, DAS, the staggered-grid family and the eager/JAX paths inject in an
  order that never had it.  Sources outside the strips are bit-identical; the
  restore kernels and the new un-injection share one strip predicate
  (`csrc/cuda/common/boundary/strip.cuh`).  `test/test_bs_strip_source.py`
  covers every restore variant (gpu/cpu/disk, fp32/fp16/bf16/int8).
- **`impl` resolution by device.**  `impl='auto'` picks the compiled backend
  on CUDA only; on a CPU it is eager, and an explicit `impl='c'` on a CPU falls
  back to eager with a warning, in every build (see Removed: the compiled CPU
  engine).  Also fixed: a device given as a string with an index
  (`dev='cuda:0'`) was read as "not CUDA", so `impl='auto'` silently resolved
  to eager there.
- **A legacy `boundary_saving_config` dict that sets staging knobs its storage
  ignores reads again.**  `{'storage': 'gpu', 'pinned_memory': True, ...}` was
  accepted (the knob ignored) until the dict route gained the typed capability
  check, which then raised `pinned_memory is only valid when storage='cpu'`.
  The check now validates only the knobs the chosen storage accepts; the typed
  `BoundarySaving(...)` stays strict.
- **Host-staged boundaries under DD no longer pay a per-step Python call.**
  The persistent stepped runners (forward and backward) now accept
  `storage='cpu'`; the core keyed its saves and flushes on the global step
  already, and the backward primes its first chunk once per propagation instead
  of once per step (which re-copied the whole chunk prefix every step).  The
  per-call path caches the host copies of the index tensors per parameter
  object, so it no longer synchronizes the device twice per step.  Measured on
  A100 vs dev: cpu-staged DD forward from 1.4-4x slower to parity, cpu-staged
  DD forward+backward 0.74x (faster), all records and gradients bit-identical.
- **The ctypes layer is the package `sweep/backend/c/`** (`loader.py`,
  `adapt.py`, `entries.py`, `runners.py`, the generated `abi.py`, and `jit.py`,
  which decides where the core comes from), replacing `sweep/_capi.py`,
  `_core_abi.py` and `_jit.py`; `sweep._C` stays the lazy entry point.  It
  lost its output mapping and per-call cache: entries and runners return
  nothing, callers read the tensors they bound.
- **`source_illumination` means the same thing under every memory strategy.**
  It was accumulated by one kernel that both backward paths call, but they
  handed it different fields: `Full()` / `Ckpt()` passed the forward store,
  which for acoustic *is* `u_tt = vp^2*Lap(u)`, while `BoundarySaving()` passed
  the reconstructed raw pressure — so one attribute returned
  `sum_t u_tt^2` or `sum_t u^2` depending on a knob that is supposed to be a
  pure space/time trade, and the two differ by ~3e10 (measured: 5.24e18 vs
  1.59e8 in 2-D, 4.94e18 vs 1.61e7 in 3-D, on a heterogeneous model).  Both are
  legitimate quantities — the Shin (2001) pseudo-Hessian and the RTM
  amplitude-compensation illumination — but nothing said which you were
  getting, and nothing said it changed with `memory=`.  All strategies now
  return the pseudo-Hessian `sum_t u_tt^2` (dimensionally paired with the
  gradient `sum_t u_tt*lambda`, which is what makes `grad/(illum+eps)` sane),
  accumulated over the physical box like the gradient kernels.  The boundary
  path forms `u_tt` from the same three reconstruction time levels
  `calculate_grad_utt_band` uses, in its **own** kernel: no gradient arithmetic
  is touched, and `test/test_illumination_pin.py` pins that enabling
  illumination cannot move the gradient.  `receiver_illumination` stays
  `sum_t lambda^2`; under `BoundarySaving()` it used to differ from `Full()` by
  ~0.5% (measured on dev) and now matches it bit for bit.  The contract is now
  written down in `_CompiledPropagator.__init__`.  Notebook 08 (RTM) shows the
  consequence: its illumination-compensated image is now the image divided by
  the illumination of the same (original-wavelet) field, a dimensionless ratio
  (colour scale ~0.05 instead of ~4e5).
- `SecondOrderEquation._apply_free_surface` — the per-edge pressure-release
  zeroing moved from `Acoustic` to the shared base (bit-identical) so
  ViscoAcoustic reuses it.
- `ViscoAcoustic.prepare_models` maps (vp, Q, omega) -> (vp_step, A) once per
  forward (shared by the eager and CUDA paths; the eager step no longer
  recomputes the dispersion/damping coefficients every time step).
- **Unknown keyword arguments are rejected instead of ignored.**  Both the
  propagator constructor and the call now raise `TypeError` naming the
  offending keywords.  A misspelled option previously did nothing at all and
  the run silently used the default, which is how `full_mode=` survived as a
  dead parameter for as long as it did.
- **Configurations that used to fail silently, late, or not at all now refuse
  at construction**, each with a message saying what is unsupported: recursive
  checkpointing when the compiled binding is absent; image-method topography on
  an equation that does not implement it; a free surface on any anisotropic
  equation (those branches were unreachable and are now an explicit refusal);
  and the legacy dict/flag memory spellings, which now go through the same
  capability check as the typed options.  Illumination/ADCIG crop failures warn
  instead of silently returning zeros.
- **The CUDA drivers are two shared skeletons instead of ten hand-written
  pairs.**  `common/eq_driver.cuh` (acoustic family) and `common/sg_driver.cuh`
  (staggered family) hold the forward and the four backward modes; each
  equation supplies a `driver_traits.cuh` of hooks and a thin forward.cu /
  backward.cu.  All ten templated equations are on it.  Contributor-facing
  only — every mode is bit-identical to the code it replaced, and each skeleton
  carries a HOOK TIMING MAP listing the hooks in call order.  See
  `docs/dev/cuda_drivers.md`.
- **Domain decomposition is declarative.**  A `DDSpec` interpreter drives the
  forward and backward loops from the equation's declared wavefield roles,
  grads layout and coupling exchange; the per-family branches and the
  name-sniffing `_FAMILIES` table are gone.
- `PropBase.__init__` is decomposed into grid geometry, IO validation,
  topography, argument resolution, boundary spec and the two DD blocks, and
  exposes `memory_strategy` and `use_ckpt` as read-only properties.
- **Kernel-level performance, all bit-exact** (measured on RTX 6000 Ada, and
  the direction re-checked on V100): imaging is computed over the physical box
  only rather than the padded box, acoustic boundary-saving imaging is fused
  into the reverse reconstruction, the band imaging kernels put the x offset
  innermost, elastic boundary-saving imaging folds into `stress_adjoint_prepare`,
  the elastic velocity carriers are captured inside the NOPML kernel, and the
  elastic3d forward kernels take `__launch_bounds__(256, 4)` (V100 stress kernel
  -13%, Ada -4%, no register spill).  Together with the runner these are worth
  1.20-1.34x on DD acoustic end-to-end, on top of what the runner itself gives.

### Deprecated
- `boundary_saving_config={...}` and `MemoryOptions(strategy=..., boundary=...,
  ckpt=...)`, in favour of `memory=Full()` / `BoundarySaving(...)` / `Ckpt(...)`.
  Both still work, are still read exactly as before, and now emit a
  `DeprecationWarning`; the removal criterion is written next to the shim.
  (`use_ckpt=` is NOT deprecated — it is still a plain supported keyword.)

### Removed
- **The compiled CPU engine** (`csrc/cpu/**`, about 20k lines), which the
  `SWEEP_JIT_FULL=1` pybind shim and an AOT-built `sweep._C` compiled in, with
  `SWEEP_SKIP_CPU` and its stub.  The wheel never shipped it, and its gradients
  disagreed with eager and with the CUDA core (adjoint-source scaling,
  checkpoint modes that crashed, stress receivers with the wrong sign, VRZ, no
  free-surface backward): the CUDA backward moved to exact discrete adjoints
  and the CPU copy never followed.  **Migration:** run CPU propagators with
  `impl='eager'`, which is what `impl='auto'` already picked there.  An
  explicit `impl='c'` on a CPU falls back to eager with a warning, and a host
  tensor handed to a compiled entry is refused with the same message by the
  ctypes layer and by the pybind shim.  The examples' legacy `--backend cpu`
  now means eager on the CPU, and `--impl c --device cpu` is refused.
  `test/cpu_binding_gradient_consistency.py` became `test/gradient_cases.py`,
  the case library of `test/backend_gradient_matrix.py`, which lost its
  `c-cpu` column.
- **`solver.rtm()` and its three compiled bindings** (`acoustic2d_rtm`,
  `acoustic3d_rtm`, `visco_acoustic2d_rtm`), together with `_C_rtm` on the
  equations and the 2-D host wrapper, the four 3-D `rtm_*_impl` wrappers and
  their dispatcher.  The entry had no caller left: the RTM notebook computes
  the image as the gradient of an inner-product loss, sweep-tasks' production
  RTM migrated to forward+backward with `compute_illumination`, and no
  bit-exact gate ever covered it.  **Migration:** run the ordinary forward and
  backward with `compute_illumination=True` and read the image off the
  illumination path — that is what `rtm()` did internally.  `RTMOutput`,
  `accumulate_rtm_image_*`, `init_rtm_output_*` and the `run_*_imaging`
  machinery are deliberately KEPT: despite the names they are the illumination
  path of the ordinary backward, which production RTM uses.  An old script
  calling `rtm()` now gets a plain `AttributeError` rather than a tombstone,
  so it cannot quietly return the wrong thing.
- **`full_mode=` propagator keyword** (and `PropagatorDefaults.full_mode`).  It
  was stored on the instance and never read by anything — a dead parameter.
  Because unknown keywords are now rejected, passing it raises `TypeError`
  instead of being silently ignored; delete it from the call.  Nothing replaces
  it; the memory strategy is chosen with `memory=Full()` / `BoundarySaving(...)`
  / `Ckpt(...)`.

### Fixed
- **Boundary saving held a whole forward history in the backward** for
  ``AcousticVRZ`` / ``AcousticVRZ3D``, ``AcousticVTI1st`` / ``AcousticVTI1st3D``,
  ``ElasticTTI2nd`` and ``DASZhao``.  Every boundary-saving backward bound a
  per-call replay buffer of ``nt`` padded grids times one to five fields that
  their drivers never read -- boundary saving still paid one to five full
  grids per time step (notebook 17 asked for 68.6 GiB and failed on a 32 GB
  GPU).  Introduced with the Python-side allocation.  Only a driver whose
  boundary-saving backward re-runs the forward into that buffer declares
  ``CUDALayoutSpec.bs_backward_replays_forward`` and gets it (``DASZhao3D``,
  through a call-time ``boundary_saving_config``); every other backward's own
  allocations no longer grow with ``nt`` (nt 150 -> 450: 3-D VTI 185 -> 547 MiB
  before, 3.2 MiB after).
- **``AcousticVRZ`` / ``AcousticVRZ3D`` imaged the wrong time step.**  The
  deferred history capture ran after the buffer rotation, so full and
  checkpointing imaged ``U_{it+1}`` while boundary saving imaged ``U_{it-1}``
  (boundary saving vs full rel 0.12); every mode now images ``U_it``
  (4e-4, the rest of which was the 2-D seed below).  Only VRZ used that hook.
- **VRZ boundary saving imaged unrestored cells.**  Its gradient reaches ``2M``,
  one ``M`` past the ``M + 1`` shell, so the outermost physical cells read what
  the reverse step left there (up to 9x the largest full-storage gradient on a
  small 3-D grid; 99.6% of the Marmousi residual in the outermost cell).  The
  sigma=0 ``boundary_buffer`` moves the shell out of the stencil's reach
  instead of widening it (peak memory 69.4 -> 54.7 GB against a ``2M + 1``
  shell on a 3-D field-data single shot).
- **2-D VRZ boundary saving started its reverse pass from a damaged state.**
  The reconstruction seed zeroed the pad ring of the two last wavefields, as
  ``Acoustic`` does (harmless there), but the VRZ restore band starts ``M``
  cells inside the pad and the first reverse step reads those cells before any
  restore.  Boundary saving vs full was 3.5e-4 to 2.6e-3 whenever the wavefield
  still reached the pad at the last step; it is now 2e-7 to 3e-6, the level of
  the 3-D runner, which never zeroed.
- **The compiled VRZ adjoint was an approximation in the PML band** (it applied
  the forward CPML operator to the adjoint field).  It is now the exact
  discrete transpose, 2-D and 3-D: compiled vs eager 1.7e-7 (2-D) and 8e-7
  (3-D) with TF32 off, finite differences to 1e-4.
- **DD sub-groups ignored the configured NCCL timeout** and fell back to the
  default one.
- **`pytest test/` could never exit 0.**  `test_import_does_not_pull_optional_deps`
  asserted `mod not in sys.modules`, which is process-global: by the time it
  runs it is a statement about everything the preceding ~600 tests imported,
  not about `sweep.datasets`.  Every recorded full-suite run ended
  `1 failed, 886 passed`, so no script, hook or CI job could gate on the suite.
  The check now runs the import in a fresh interpreter and asserts on *its*
  `sys.modules`, which is both order-independent and what the test always meant
  to say.
- **The JIT staging directory ignored edits to `csrc`.**  `_stage()` mirrors
  the CUDA tree into the torch-extension build dir with unique compiled-source
  basenames; the directory was named after the installed `sweep-solver`
  version and the staleness check was "does `.staged` exist".  Editing a kernel
  without bumping the version therefore left the previous copy in place, ninja
  compiled the OLD source, and the `.so` silently did not contain the edit --
  which is why working on `csrc` came with a "delete the extension directory
  first" ritual.  The sentinel is now a manifest of per-file SHA-256 digests:
  only files whose contents changed are re-staged (so ninja still rebuilds just
  the affected translation units, and through its header depfiles whatever
  includes a changed `.cuh`), files whose source disappeared are removed from
  the stage, and a re-staged copy is stamped with the current time so a source
  that moves BACKWARDS in time (`git checkout` of an older revision) still
  invalidates the object built from it.  The staged tree is byte-identical to
  what the previous implementation produced (150 files, the same 50 compiled
  sources); the manifest costs ~44 ms once per process.
  The staging is also taken under `torch.utils.file_baton.FileBaton`, the
  mechanism torch uses for concurrent extension builds.  `torchrun` starts
  one process per GPU and they all stage before `cpp_extension.load` takes
  its own lock, so nothing serialised them: two ranks copied the tree on
  top of each other, and on a real 150-file tree over a shared filesystem
  one lost with `FileExistsError` on the stage directory -- which is what
  killed a 2-rank domain-decomposition benchmark before it measured
  anything.  The regression test asserts on work rather than timing:
  staging from two processes must copy no more than staging from one.
- **Three test files turned a build failure into a green skip.**  They probed
  the extension as `try: import sweep._C as _C; return hasattr(_C, "sym")
  except Exception: return False`.  `sweep._C` is a lazy shim whose attribute
  access triggers the JIT, so when nvcc failed torch raised, the `except`
  swallowed it, the module-level `skipif` fired, and 38 collected items
  reported green on a build that does not exist.  `test/conftest.py` now owns
  the one decision: `requires_binding(*symbols)` answers from
  `is_torch_binding_available()`, which compiles nothing; a binding that IS
  present but lacks a named symbol raises, because that is a stale or partial
  build rather than a missing capability; and `SWEEP_TEST_REQUIRE_CUDA=1` turns
  the skip itself into a collection error, so a machine that is supposed to
  have a GPU cannot report green over half the suite.

- **The staggered family refused host-staged boundary storage under stepping
  and domain decomposition.**  `sg_check_stepped_backward` rejected
  `storage='cpu'` and `storage='disk'` for every stepped or DD `backward_bs`
  across the seven equations on that skeleton (elastic2d/3d, das_mu2d/3d,
  elastic_tti_sg2d/3d, elastic_vr2d), which capped the largest elastic 3-D DD
  tile at whatever boundary ring fits in VRAM.  Two things actually blocked it,
  neither a design limit: the skeleton built its own copy stream per call
  instead of taking the Python-owned `BoundarySession`, so under DD -- where
  every time step is a separate entry into the extension -- the stream and its
  ring events were destroyed and rebuilt every step and no transfer could stay
  in flight; and it primed the TAIL chunk on every segment
  (`prefetch_initial_backward_chunk` with no `it_hi`), so a stepped reverse
  loop fetched the wrong slabs on all but its first call.  Both are fixed and
  the refusal is narrowed to match the acoustic family: disk is still refused
  under a cut mask, cpu is not.  Gate tiers A/C/B/dd1 unchanged.

- **Every backward allocated the RTM/illumination buffers, requested or not.**
  `init_rtm_output` did three `torch::zeros_like(vp)` on the runtime-padded
  grid unconditionally, while the kernels that read them are gated on
  `compute_illumination || compute_adcig` -- and `compute_illumination` is
  itself derived from whether Python allocated a real buffer.  So a plain FWI
  gradient allocated and memset three full padded fields no kernel touched.
  The allocation now takes the same predicate the launch already takes.
  Measured backward-phase peak: `Acoustic` at 724x324 padded 0.024 -> 0.021 GB
  and `Acoustic3D` at 212^3 padded 1.222 -> 1.108 GB, i.e. exactly the 3 MB /
  114 MB the three fields occupied, with records and gradients bit-exact and
  gate tiers A/C/dd1 unchanged.

- **A demoted `impl='c'` lost the memory strategy the caller asked for.**  When
  the compiled binding is unavailable, or the equation has no `_C`, `impl='c'`
  falls back to eager and the cuda-only knobs are dropped with it.  The
  gradient-memory strategy is not a cuda-only knob, though -- it just travels
  inside `cuda_options` -- so dropping the object took the request with it and
  the eager backend fell back to its own default, checkpointing.  A caller who
  asked for boundary saving got `'ckpt'`, silently.  The request is now carried
  across the demotion (an explicit `memory=` still wins), so it reaches
  `check_memory_supported` and the backend actually runs what was asked for.

- **The staggered backward allocated a dead wavefield buffer per call.**
  `SgBackwardBsRunner` passed `{}` for the saver's `last_two`, so
  `allocate_last_two` took its self-allocating branch and built a
  `{BS_NVAR, 1, B, 1, nz[, ny], nx}` FP32 buffer -- 9 padded 3-D grids for
  `Elastic3D` and `ElasticTTI_SG3D`, 15 for `DASMu3D` -- on the GPU, or in
  PINNED HOST memory on the staged path.  Nothing touches it: `save_last_state`
  writes `last_two` in the FORWARD, and the backward seeds from
  `p.u_last_two` directly.  Binding the tensor Python already owns returns all
  of it.  Measured at a 180^3 padded grid, backward-phase peak: `Elastic3D`
  2.737 -> 2.527 GB and `DASMu3D` 4.278 -> 3.928 GB, i.e. exactly the 210 MB /
  350 MB the buffer occupied, with records and every model gradient bit-exact.
  Same defect and same fix the acoustic skeleton already had.
- **Domain decomposition re-emitted staging knobs the storage rejects.**
  `ModelParallel` rebuilt the wrapped propagator's boundary strategy as a typed
  `BoundarySaving` carrying whatever it inherited, including
  `transfer_interval` / `ring_buffers` / `pinned_memory` for `storage='gpu'`,
  which `BoundaryOptions.__post_init__` refuses -- so a configuration the legacy
  dict route accepted became a construction error, including the gpu baseline
  of `test/dd_session_bench.py` at that file's own default, which is the
  reference its bit-exactness gates compare against.  Only the knobs that apply
  to the chosen storage are passed on now.

- **3-D DAS boundary saving cost more than full storage, and was the default.**
  `das3d::backward_bs` is three lines that re-run the forward and allocate the
  entire `{nt, 3, B, nz, ny, nx}` strain history -- exactly what full storage
  holds -- while the forward writes no boundary strips at all (it returns an
  empty `last_two`).  On top of that the Python side allocated a 13-field
  boundary ring and a 13-grid `last_two` that nothing writes and nothing reads.
  Since boundary saving is the implicit `impl='c'` strategy, every plain 3-D DAS
  gradient paid strictly MORE than `'full'` for a "memory-saving" mode, and
  every gradient test passed because the answer was right and nothing asserted
  on memory.  `DASZhao3D.supports_boundary_saving_c = False` now routes the
  default to `'full'` and makes an explicit boundary request raise, the contract
  `ViscoAcoustic` already had.  This is "not implemented", not "impossible":
  `das2d` writes real strips and `DASMu` / `DASMu3D` get them from the shared
  staggered skeleton.  `test/solver_gradient_mode_suite.py` drops the 14 `bs_*`
  entries that all ran the same code as `full`.
- **A propagator could step with another propagator's CPML profiles.**
  `PropBase.init_abc` caches the profiles on the EQUATION (`equation.b`) but
  held the freshness key on the PROPAGATOR.  Sharing one equation between
  propagators is a supported pattern -- `ModelParallel` builds a second
  propagator over the wrapped one's equation -- and anything that changes the
  pad (a free surface, a different `abcn`, a DD tile) changes the profile
  length.  Two propagators therefore each started with their own `None` key,
  both built, and whichever ran last owned `equation.b` while the other's key
  still matched, so its rebuild was skipped and it handed the kernel profiles
  built for a different padded shape.  Nothing validates the length on the way
  in.  Measured on a shared `Acoustic`: `abcn=10` (padded 60x70) then
  `abcn=30` (padded 100x110), then the first propagator again -- its z profile
  stayed 100 long instead of 60.  `docs/notebooks/02_fwi_elastic_marmousi.ipynb`
  builds `solver`, then `solver_fs`, then runs `solver` again, so it hits this.
  The key now lives on the equation next to the value it describes.

- **`AcousticVTI1st` on `impl='c'` read the grid spacing with the axes
  swapped.** `PropBase` stores spacing in model-axis order `(dz, dx)` and
  `_cuda_spacing()` reverses it, so `p.spacing` reaches the kernels as
  `[dx, dz]` -- as every other driver reads it, including this equation's own
  3-D twin. `acoustic_vti_1st_2d` read it as `(dz, dx)` in all four entry
  points, which is identical whenever `dx == dz` and wrong the moment they
  differ: at `dh=(dz=10, dx=20)` the compiled record's cosine against eager was
  **-0.62** (and **0.10** at `(20, 10)`), with the model gradients scattered.
  With `dh=10` it was 1.000000, which is why nothing caught it -- the gate, the
  gradient matrix and every test pass a scalar `dh`, though `normalise_spacing`
  has always accepted a per-axis sequence. Isotropic results are unchanged (the
  two values are equal, so the swap is a no-op there).
- **`gather_record` crossed shot groups.**  The record gather ran over the
  WORLD process group, but with `shot_groups > 1` every group propagates a
  different shot through the same tile grid, so ranks sharing a tile
  coordinate carry the same global receiver indices holding different shots'
  traces.  Rank 0 wrote all of them into one array and whichever tile was
  assembled last silently won, so the "global record" was spliced from several
  shots and nothing raised.  The gather now runs over `mesh.model_pg` -- the
  `py * px` ranks that decompose ONE shot -- and each group assembles its own
  record on its own root (`shot_group * py * px`), which is rank 0 for the
  `shot_groups == 1` case the guides describe.  The collective and the
  index placement moved to `sweep.parallel.gather_tile_records` /
  `assemble_tile_records` (the inverse of `partition_global_coords`), so the
  regression test `test/test_dd_gather_record_shot_groups.py` exercises the
  real collective on gloo/CPU without a GPU.
- **Domain decomposition silently answered a geometry it cannot solve.**  A
  3-D source/receiver array means source encoding, whose leading axis is 1;
  anything longer is what a caller writes when they mean "several shots", and
  the single-domain propagator rejects it.  `ModelParallel._prepare_call` did
  not: it boolean-indexes the owned sources of EVERY leading entry into one
  flat array, and reads the receivers -- and their ownership, which
  `own_receiver_indices` publishes and the parallel guide promises is a
  partition of the global receiver list -- from index 0 alone.  So several
  shots were fired as one fused supershot, recorded at the first entry's
  receivers, and returned without an error.  Both arrays are now checked, with
  a message naming the three supported ways to run several shots.
- **`downsample=` returned a view that pinned the full-resolution array.**
  `decimate` ended in basic slicing, so the decimated model kept the array it
  came from alive through `.base` for as long as the caller held it -- a few
  megabytes pinning a few hundred on the 3-D benchmarks.  It returns a
  C-contiguous copy now, matching `read_model`; the `factor == 1` path stays
  zero-copy.
- **The Dynamo recompile-cap bump did nothing on torch < 2.6.**  The eager step
  raises Dynamo's cap before compiling, because each wavefield's
  `requires_grad` flips on first use and the many-wavefield equations exhaust
  the default of 8 before specialization settles -- at which point Dynamo falls
  back to eager silently.  The knob was renamed (`cache_size_limit` up to torch
  2.5, `recompile_limit` from 2.6) and the bump keyed on the new name behind a
  `hasattr` guard, so on an older torch it was a no-op and the fallback it
  exists to prevent happened anyway.  `torch` is unpinned here, so that is a
  shipped configuration.  Whichever name the installed Dynamo exposes is raised
  now, including the `accumulated_` secondary cap, and a Dynamo with neither
  warns once instead of passing in silence.
- **The eager wavefield-snapshot buffer was cached forever and then copied.**
  `return_wavefield=True` took its buffer from `_workspace_cache`, which has no
  eviction policy, so a buffer sized
  `nsnapshots x nwavefields x B x prod(padded_shape) x 4` -- and the default is
  a snapshot at EVERY time step -- stayed pinned to the propagator for its
  whole lifetime, long after the caller had finished with the snapshots.
  Because it was workspace, it then had to be cloned on the way out, so the
  path held two copies at once.  Measured on `Acoustic` at 136x176 padded with
  `nt=200`: 115 MB per copy.  The buffer is allocated per call now and returned
  directly.
- **The eager record was built by `nt` in-place slice writes into a live
  autograd tensor.**  That builds a chain of `nt` `CopySlices` nodes, and each
  one allocates a full-record buffer and copies the incoming gradient through
  it, so the record's own backward cost was `2 * nt * |record|` -- 64 GB per
  shot at `nt=4000` with 500 receivers, 1.44 TB at OBN shapes.  The default
  rollout collects the per-step gather and stacks once, making it
  `2 * |record|` and removing `nt` full-record allocations.  Records and
  gradients are bit-exact (`torch.equal`); the whole backward, on CPU with 500
  receivers on a 40x60 grid, goes 136 -> 129 ms at `nt=250`, 280 -> 235 ms at
  `nt=500` and 616 -> 497 ms at `nt=1000` -- the share grows with `nt` because
  the removed term is the quadratic one.  The stacked record is already fresh,
  so it is returned without the defensive clone the workspace buffer needed.
  The checkpointing and eager-boundary-saving rollouts still fill a
  preallocated record: their write is per CHUNK, not per step.
- **The source injection mask was built on paths that never read it.**
  `SourceTorch` allocated a whole padded wavefield, filled it with an
  `index_put_` and, with a spread kernel, convolved it -- on every
  construction.  Exactly one place reads it, `SourceBase.forward`, which
  `SourceTorch.forward` reaches only when neither source encoding nor adjoint
  modelling is active; both of those inject through `_add_indexed_sources` and
  never look at it.  So every encoded forward and every adjoint construction
  paid `prod(padded_shape) * 4` bytes plus the scatter and the convolution for
  a buffer nothing read.  It is built on the branch that reads it now.
- **A SEG-Y read peaked at 7.3x its payload.**  `segy_to_array` made the whole
  file's traces contiguous in one go -- the memmap rows are strided, so dropping
  the 240-byte trace headers needs a copy -- and handed the result to
  `_ibm_to_ieee`, which holds five temporaries the size of what it is given
  (sign, exponent, mantissa, the `ldexp` result and the negation).  Both scaled
  with the file rather than with a bound.  Measured on a synthetic IBM-float
  file, resident growth over the read: 60 MB payload 440 MB (7.33x), 120 MB
  payload 879 MB (7.33x).  Reading in blocks of ~16 MiB of samples caps the
  temporaries: the same files now grow 224 MB (3.73x) and 356 MB (2.96x), and
  the ratio keeps falling toward the inherent floor of the output array plus the
  mapped pages, because the temporary budget no longer scales with the file.

- **Disk-staged boundary saving reconstructed a wrong gradient.**  Since the
  persistent staging session / non-blocking copy stream (PR #81), every
  `storage='disk'` gradient was wrong: max|disk-gpu|/scale 0.2-0.5 for acoustic
  2-D/3-D and 0.8-5.0 for nvar>1 3-D (Elastic3D, DASMu3D); cpu staging with
  `ring_buffers >= 2` was off by 1.4-2.4x on the same equations.  Three defects in
  `boundary/runtime.cuh`: (1) `prefetch_next_backward_chunk_if_needed` issued
  the next chunk early on `ring_buffers >= 2`, but the synchronous disk path is
  pinned to slot 0 whatever `ring_buffers` says (and defaults to 3 / 2), so the
  early H2D overwrote the chunk still being restored -- the predicate is now the
  slot assignment; (2) the synchronous-disk enqueue never got the
  `cudaStreamWaitEvent(compute_ready_)` write-after-read fence the host-staging
  branch has; (3) the two nvar>1 restore readers kept the slot-0 override for
  cpu staging with `ring_buffers >= 2`.  Regression test
  `test/test_boundary_disk_staging_slot.py`: disk (default knobs and short
  chunks) and cpu ring 2 must be bit-exact against gpu-direct.

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

[Unreleased]: https://github.com/DeepWave-KAUST/sweep/compare/v0.3.3...dev
[0.3.3]: https://github.com/DeepWave-KAUST/sweep/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/DeepWave-KAUST/sweep/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/DeepWave-KAUST/sweep/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/DeepWave-KAUST/sweep/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/DeepWave-KAUST/sweep/compare/v0.1.0...v0.2.0
