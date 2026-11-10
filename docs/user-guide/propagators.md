# Propagators

A **propagator** wraps an equation object, grid configuration, acquisition
geometry, and model tensors into a callable solver. The two main entry points
are:

- `sweep.propagator.torch.PropTorch` — PyTorch-family runtime
- `sweep.propagator.jax.PropJax` — JAX-family runtime

## `backend` vs `impl`

`PropTorch` takes two related but distinct knobs:

- `backend` — the **array / autograd framework** carrying tensors and gradients
  (`"torch"` for PyTorch eager / `torch.compile`; `"jax"` for JAX on
  `PropJax`).
- `impl` — the **implementation path** under that framework (`"eager"` =
  pure-Python operators with PyTorch autograd; `"c"` = the prebuilt CUDA core
  (`libsweep_core.so`) driven through the pure-Python `sweep._C` ctypes layer).

The two axes are orthogonal:

| Backend | Impl | Device | What runs |
| --- | --- | --- | --- |
| `torch` | `eager` | CPU or CUDA | Pure-Python operators + PyTorch autograd |
| `torch` | `c` | CUDA | Prebuilt CUDA core via the `sweep._C` ctypes layer |
| `jax` | — | CPU or CUDA | JAX implementation via `PropJax` |

`"cuda"` is **not** a backend or `impl` value — it is a device choice driven by
the tensors you pass in. The compiled path runs on CUDA tensors only; a host
model tensor is refused with a clear error, and a CPU propagator resolves to
`impl="eager"`.

A minimal invocation for each path:

```python
from sweep.propagator.torch import PropTorch

solver_eager = PropTorch(..., backend="torch", impl="eager")
solver_c     = PropTorch(..., backend="torch", impl="c")
```

```python
from sweep.propagator.jax import PropJax

solver_jax   = PropJax(..., backend="jax")
```

## Geometry conventions

- `sources` — shape `(nshots, ndim)` for a single point source per shot, or
  `(1, nsources, ndim)` for one source-encoded supershot (all sources fire
  together; `receivers` is then `(1, nreceivers, ndim)` and the wavelet `(nt,)`
  or `(nsources, nt)`)
- `receivers` — shape `(nshots, nreceivers, ndim)`
- 2D coordinates are **`(x, z)`** in grid indices — horizontal first, depth
  last
- 3D coordinates are **`(x, y, z)`** in grid indices — horizontal axes first,
  depth last

Model and wavefield arrays themselves are stored as `(nz, nx)` / `(nz, ny, nx)`
(depth first), which is the same axis order `matplotlib.imshow` expects when
plotting with depth growing downward. The propagator reverses the user-supplied
`(x, z)` / `(x, y, z)` coordinate tuples internally before indexing into the
wavefield tensor.

## Boundaries: free surface and PML

Every propagator takes ``free_surface=`` and ``abcn=`` (PML thickness).
``free_surface`` accepts several equivalent forms, all normalised to one
canonical per-face boolean tuple (axis-major order — 2-D
``(z_lo, z_hi, x_lo, x_hi)``, 3-D ``(z_lo, z_hi, y_lo, y_hi, x_lo, x_hi)``):

```python
PropTorch(eq, ..., free_surface=False)                # absorbing everywhere (default)
PropTorch(eq, ..., free_surface=True)                 # historical top-only free surface
PropTorch(eq, ..., free_surface="top")                # single face name — same as True
PropTorch(eq, ..., free_surface=["top", "left"])      # list/set of face names
PropTorch(eq, ..., free_surface={"bottom": True,
                                 "right": True})      # dict: face name -> on/off
PropTorch(eq, ..., free_surface=(True, False,
                                 True, False))        # canonical per-face tuple: top + left
PropTorch(eq, ..., free_surface=(1, 1, 1, 1))         # same form with ints = closed box
```

Face names (also the order of the canonical tuple): 2-D
``top, bottom, left, right``; 3-D ``top, bottom, front, back, left, right``
— ``top`` is the z-min face, matching the ``(nz, nx)`` depth-first array
layout. An unknown name raises ``ValueError`` listing the valid names.

Each free face replaces its PML pad with the image-method boundary
condition; the remaining faces stay absorbing. ``abcn`` takes a scalar or a
canonical tuple of per-face PML widths (no face names or dict). Per-face
widths are eager-only -- ``impl='c'`` raises ``NotImplementedError`` for them
-- and, like per-edge free surfaces, 2-D ``Acoustic`` / ``Elastic`` /
``ViscoAcoustic`` only.

A solid refuses one family of layouts: a PML that runs between two faces of
the other axis that have no PML, one of them free -- ``["top", "bottom"]``,
``["top", "left", "right"]`` and the like, or a free face opposite an ``abcn``
0 face.  ``Elastic``, ``ElasticBorn`` and ``ViscoElastic`` raise ``ValueError``
for it.  The solid between those faces is a plate, and a PML amplifies its
backward-propagating Lamb modes instead of absorbing them (Loh et al. 2009,
*Phys. Rev. E* 79, 065601): such a run grows without bound after a few
seconds, and a thicker or retuned PML only changes how fast.  Keep one face of
the pair absorbing, or free all four faces (a closed box has no PML).  Fluids
have no backward modes, so ``Acoustic`` and ``ViscoAcoustic`` accept every
layout.

``boundary_buffer=`` (default ``None``) puts that many sigma=0 cells between
the physical box and the PML ramp on every absorbing face.  Boundary saving
stores a shell ``M + 1`` cells wide, which is all the reverse time step needs;
an equation whose gradient stencil reaches further -- ``AcousticVRZ`` and
``AcousticVRZ3D`` take a divergence of a gradient, reach ``2M`` -- would image
cells the shell never restored.  Such an equation declares
``BOUNDARY_BUFFER_REACH`` and ``None`` gives it ``REACH * M + 1`` buffer cells
under every memory strategy and backend, so full, checkpointing and boundary
saving solve the same grid; the buffer is cropped with the pad, so shapes and
coordinates do not change.  Equations with pointwise imaging get no buffer.
An explicit value is honoured, but boundary saving refuses one below the
equation's need.  A non-zero buffer cannot be combined with ``topography=``
yet (``NotImplementedError``).  On the eager and JAX backends an explicit
``boundary_buffer=M + 1`` also removes the error boundary saving leaves on the
outermost cells next to the PML; see
[Boundary saving next to the PML](#bs-near-pml).

Support matrix:

- **2-D ``Acoustic`` / ``Elastic``** — full per-edge free-surface support on
  the eager and compiled CUDA backends, with adjoint gradients across every
  backward memory mode (full / boundary-saving / checkpointing).
- **2-D ``ViscoAcoustic``** — full per-edge free-surface support on the eager
  and compiled CUDA backends (full / checkpointing gradients; boundary saving
  is refused — the dissipative spectral damping term cannot be
  reverse-time reconstructed, and the ``impl='c'`` default falls back to
  ``'full'``).
- **3-D** — top face only (``free_surface=True``); per-edge lists raise.
- **Topography** (irregular surface) — a separate top-only feature; it
  cannot be combined with per-edge faces.  The image-method staircase
  (``topo_method='image'``) is acoustic-only (``Acoustic``, ``Acoustic3D``,
  ``ViscoAcoustic``).  The elastic staircase was removed: it was unstable on
  slopes steeper than about 0.5 and, stabilised, wrong in a way that grew with
  refinement.  For an elastic surface with topography use
  ``ElasticCurvilinear`` (2-D) or ``PropSEM``.
- **Anisotropic equations** (``AcousticVTI1st``, ``AcousticVTI``,
  ``AcousticTTI``, ``ElasticTTI(SG)``, …) raise ``NotImplementedError`` for
  any free surface: the anisotropic stress-free condition couples through
  the stiffness tensor and is not the isotropic image method these solvers
  implement. Run with ``free_surface=False`` or use an isotropic equation.

See the [per-edge free surface notebook](../notebooks/24_wavefield_per_edge_free_surface.ipynb) for
snapshots of each configuration and a closed-box energy check.

## Implementation-specific options

Every option block lives in `sweep.propagator.options` and is documented in
detail on the [Propagator Options](../api/propagators/options.md) page. A
quick reference:

| Option block | Used with | Configures |
| --- | --- | --- |
| `EagerOptions` | `impl="eager"` | `torch.compile` flags, [time steps per compiled call](#steps-per-call), the [CPML strip layout](#cpml-strips) (`cpml_strips`, on by default in 3-D), [Inductor options](#inductor-options), debug knobs |
| `CUDAOptions` | `impl="c"` | Container for `impl="c"`-only options; its `memory=` is equivalent to `PropTorch(memory=...)` |
| `Full` | as `memory=` | Keep the whole forward wavefield. No parameters |
| `BoundarySaving` | as `memory=` | Boundary saving with GPU / CPU / disk storage |
| `Ckpt` | as `memory=` | Chunk- or recursive-mode checkpointing |

`MemoryOptions(strategy=..., boundary=..., ckpt=...)` is the older way to say
the same three things and is deprecated; `BoundarySaving` and `Ckpt` subclass
the `BoundaryOptions` / `CkptOptions` it wrapped, so every field, default and
validation rule is the one you already knew.

Pass each block through the matching kwarg. The memory strategy (`Full`,
`BoundarySaving`, `Ckpt`) always goes through `PropTorch`'s own `memory=`,
whichever `impl` runs:

```python
PropTorch(..., eager_options=EagerOptions(...))  # impl="eager"
PropTorch(..., cuda_options=CUDAOptions(...))    # impl="c"
PropTorch(..., memory=BoundarySaving(...))       # either impl
```

### Inductor options of the eager step {#inductor-options}

On a GPU, `impl="eager"` compiles the time step of a 2-D grid with two options
of Inductor, `torch.compile`'s compiler, that shorten the time the host spends
on each compiled call. That time sets the pace when the GPU has little to do
per step, as it has on a 2-D grid:

- `cpp_wrapper`: the host code of a compiled call (allocating its outputs,
  launching its kernels) is generated as C++ instead of Python. The kernels
  are the same, and so are the records and gradients, bit for bit.
- `combo_kernels`: independent small kernels of one compiled graph (the source
  injection, the receivers, the C-PML strips, the boundary ring) run as one
  launch. The records stay the same bit for bit; the gradients differ at the
  level of float32 round-off, as a few sums are taken in another order.

`EagerOptions(compile_cpp_wrapper=..., compile_combo_kernels=...)` sets them.
The default, `None`, turns both on when Inductor compiles the step of a 2-D
grid on a CUDA device with torch 2.9 or newer (the version they were measured
on), and `cpp_wrapper` only where torch finds the CUDA toolkit (see below). A 3-D
step keeps the GPU busy, the options gained nothing there (see below), and it
compiles as it did before, as does any step elsewhere. The default depends on
the dimension, the device, the backend, the mode, the torch version and the
toolkit, never on the size of the grid or the number of shots. `False` turns
an option off: `compile_combo_kernels=False` gives the gradients of a compile
without it bit for bit. `True` turns it on, also where the default does not
(3-D, an older torch, no toolkit found, `cpp_wrapper` on a CPU). A value that
cannot take effect raises `ValueError`: either option with `use_compile=False`
or a `compile_backend` other than Inductor, `compile_cpp_wrapper=True` with a
mode that replays CUDA graphs, and `compile_combo_kernels=True` on a CPU. On a
CUDA device the `compile_mode`s that replay CUDA graphs (`"reduce-overhead"`,
`"max-autotune"`) raise `ValueError` themselves: a replay writes over the
outputs of the replay before, which the time loop still holds (the records of
the earlier calls, the fields a call passes on), and every rollout failed
reading them back. `"max-autotune-no-cudagraphs"` keeps the autotuning.

Inductor builds a C++ wrapper with a C++ compiler
(`torch._inductor.config.cpp.cxx`, seeded from `$CXX` when torch is imported)
against the CUDA toolkit that `torch.utils.cpp_extension.CUDA_HOME` names:
`$CUDA_HOME` or `$CUDA_PATH`, else the `nvcc` on the `PATH`, else
`/usr/local/cuda`. On a cluster that provides CUDA as a module, load it (or set
`CUDA_HOME`) before Python starts. Conda's CUDA packages keep the headers and
libraries under `$CONDA_PREFIX/targets/x86_64-linux`: point `CUDA_HOME` there
(g++ 9 then builds the wrapper too). Where the wrapper does not build, the
step compiles without it and warns once, and the later compiles of the process
leave it out from the start. A step that does not compile with
`combo_kernels` compiles without it, and only a step that does not compile at
all runs uncompiled, which is several times slower. The fallback catches what
fails in the compiled call itself. A backward graph is compiled when autograd
first runs it, outside that call, and fails there; the forward graph of the
same step is compiled first, with the same options and the same toolchain.

**Measured.** RTX 6000 Ada, torch 2.9.1, a 10 Hz Ricker wavelet,
`spatial_order=4`, float32, `memory=BoundarySaving()`, random models of
1000 × 1000 cells with a C-PML of 40 cells outside them and `nt` = 1000 (one
shot, or 8 shots with `cpml_strips=True`), or of 160³ cells with a C-PML of 20
cells and `nt` = 300 (one shot). Each entry is the range over two processes,
in ms per time step: forward + backward (gradients of all models of the sum
of the squared records) / forward without gradients.

| ms per step | both off | both on (the 2-D default) | faster |
| --- | --- | --- | --- |
| 2-D acoustic, 1 shot | 1.72–1.86 / 0.45–0.53 | 1.62–1.63 / 0.42–0.49 | 1.10 / 1.09 times |
| 2-D elastic, 1 shot | 3.01–3.50 / 0.77–0.78 | 2.66–2.71 / 0.72 | 1.21 / 1.08 times |
| 2-D acoustic, 8 shots | 2.30–2.31 / 0.57 | 1.88–1.96 / 0.50–0.57 | 1.20 / 1.06 times |
| 2-D elastic, 8 shots | 5.87 / 1.15 | 5.61–5.64 / 0.92–0.93 | 1.04 / 1.24 times |
| 3-D acoustic, 1 shot | 3.60 / 0.75 | 3.60–3.61 / 0.75 | 1.00 / 1.00 times |
| 3-D elastic, 1 shot | 15.49–15.51 / 3.15–3.16 | 15.67 / 3.14–3.15 | 0.99 / 1.00 times |

Most of it is `cpp_wrapper`'s: alone, it made a gradient step of one shot 1.13
(acoustic) and 1.21 (elastic) times faster, and `combo_kernels` alone 1.02 and
1.03 times. The records were the same bit for bit and the gradients the same or
within 2e-8 relative; the gap is that of two independent compiles, which can
pick different configurations for a reduction kernel. The first compile of the
step took about 10 s longer (17 s instead of 7 s for the acoustic forward,
25 s instead of 17 s for its gradient), and the peak GPU memory of a gradient
call grew by at most 5 %. In 3-D (both options asked for: the default leaves
them off there) they gained nothing: the gradient of the elastic step was 1 %
slower, all of it `combo_kernels`' (alone 1.5 % slower; `cpp_wrapper` alone the
same), and the first compile took about 10 s longer.

## A full Option composition example

To FWI on a large model with the compiled CUDA path, boundary saving on pinned
host memory, and asynchronous disk fallback at the largest scale:

```python
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.propagator.options import Full, BoundarySaving, Ckpt

dev = torch.device("cuda")
shape = (512, 2048)
dh, dt = 10.0, 0.002

solver = PropTorch(
    Acoustic(spatial_order=8, device=dev),
    shape=shape, dh=dh, dt=dt, device=dev,
    impl="c",
    memory=BoundarySaving(storage="cpu", pinned_memory=True),
)
```

Swap in `memory=BoundarySaving(storage="disk", disk_async_read=True)` for the
disk fallback, `memory=Ckpt(mode="chunk", chunks=100)` for chunk-mode
checkpointing, or `memory=Full()` to keep the whole forward wavefield. The
strategy IS the object, so there is no separate `strategy=` string to keep in
step with it.

The validation rules are enforced in the dataclass `__post_init__` methods, so
incompatible combinations (e.g. `disk_async_read=True` with `storage="cpu"`)
fail loudly at construction time rather than during a long FWI run.

## Memory-saving features

The gradient-memory mode is a **three-way choice, identical for the eager and
CUDA backends**, and the choice is a type:

```python
memory=Full()                                     # keep the forward wavefield
memory=BoundarySaving(storage="gpu")              # reconstruct from the boundary
memory=Ckpt(mode="chunk", chunks=100)             # rematerialise from checkpoints
```

Left unset, `impl="c"` defaults to `BoundarySaving(storage="gpu")` (fp32 ring;
`Full()` for `ViscoAcoustic` and `DASZhao3D`, which have no compiled boundary
saving) and the eager backend to `Ckpt()`.  The modes are mutually exclusive,
and conflicting requests raise a `ValueError` instead of one path silently
winning.

The mode only matters for a call with gradients, one that autograd records
(grad mode on, and a model or the wavelet requiring grad). A forward-only
call, for example inside `torch.no_grad()` during a line search, has no
backward pass to prepare: under any mode it runs the plain forward steps and
saves no boundary, no checkpoint and no wavefield.

!!! warning "The older spellings still work, and now warn"
    `memory=MemoryOptions(strategy=..., boundary=..., ckpt=...)` and the flat
    `boundary_saving_config={...}` dict are read exactly as before and emit a
    `DeprecationWarning`; the removal criterion is written next to the shim.
    `use_ckpt=` is **not** deprecated — it stays a plain supported keyword.

Three rules make that resolution predictable:

* **An off-switch means `'full'`, not "the other trick".**  `use_ckpt=False`
  (or `boundary_saving_config={'enabled': False}`) with nothing else selects
  full-wavefield storage on both backends — the long-standing meaning of
  `impl='c', use_ckpt=False`.  The implicit backend default applies only when
  no gradient-memory knob is passed at all.
* **A request is honoured, not out-voted.**
  `boundary_saving_config={'enabled': True}` now really runs the boundary
  backward; it used to lose silently to the `use_ckpt=True` default, so
  scripts that thought they were measuring boundary saving were checkpointing.
* **`memory=` may sit next to a legacy knob when they agree.**
  `memory=BoundarySaving(), use_ckpt=False` states one intent twice and is
  accepted; `memory=BoundarySaving(), use_ckpt=True` contradicts it and
  raises.  Where both carry detail, `memory=` wins.

Tail truncation has a legacy dict spelling too — `boundary_saving_config=
{'enabled': True, 'tail_steps': K}` is equivalent to
`BoundarySaving(tail_steps=K)`.

| Feature | Path | Configured by |
| --- | --- | --- |
| Full storage (no reconstruction) | both | `Full()` |
| Boundary saving (GPU, CPU or disk ring) | both | `BoundarySaving(storage=..., storage_dtype=..., ...)` |
| Asynchronous disk prefetch | both (always on for `impl="eager"`) | `BoundarySaving(storage="disk", disk_async_read=True, ...)` |
| Boundary tail truncation (steady-state / freqsel objectives) | `impl="c"` acoustic | `BoundarySaving(tail_steps=...)` |
| Chunked checkpointing | both | `Ckpt(mode="chunk", chunks=...)` |
| Recursive (fixed-budget) checkpointing | `impl="c"` | `Ckpt(mode="recursive", count=...)` |
| `torch.compile` on the eager step | `impl="eager"` | `EagerOptions(use_compile=True, ...)` |
| [Several time steps per compiled call](#steps-per-call) (default 4 for 2-D gradient rollouts on a GPU) | `impl="eager"`, full tape, checkpointing and boundary saving | `EagerOptions(steps_per_call=...)` |
| [Inductor's C++ wrapper and combo kernels](#inductor-options) (default on 2-D grids on a GPU) | `impl="eager"` | `EagerOptions(compile_cpp_wrapper=..., compile_combo_kernels=...)` |

**Several forwards before one backward.** Summing the losses of several
forwards on one propagator and calling `backward()` once gives the same
gradient as a backward after each forward — on `impl="c"` in every memory
mode, and on the eager backend with `Full()` or `BoundarySaving()` (eager
`Ckpt()` still raises "modified by an inplace operation" when another forward
runs before the backward).  On `impl="c"` the boundary / checkpoint storage the
backward reads lives on the propagator: the first forward whose backward is
still pending uses it, and each further forward allocates storage of its own,
freed when its backward runs or its graph is dropped — so K pending forwards
hold K copies of that storage.  A forward → backward loop never allocates more
than one.  A call under `torch.no_grad()` saves nothing for a backward and
touches none of this storage, whatever `requires_grad` its inputs carry.
`ModelParallel` keeps one set per instance and refuses a second forward while
a backward is pending (see [Parallel](parallel.md)).

A runnable comparison of these options lives in the
[Memory · strategies notebook](../notebooks/07_memory_strategies.ipynb), which
exercises full-wavefield, boundary saving, and checkpointing on the same
Marmousi shot and prints the per-mode peak GPU / host memory.

### Boundary saving next to the PML (eager and JAX) {#bs-near-pml}

On `impl="c"` boundary saving matches `Full()` to rounding on every cell.  On
the eager and JAX backends it matches to rounding everywhere except the cells
next to an absorbing face: the outermost `M` cells (`M = spatial_order // 2`;
`M + 1` for `Acoustic1st` and `ElasticVRR`).  The backward pass re-runs each
step under autograd from the reconstructed state, which has no C-PML memory and
no PML cells beyond the saved ring, so the model derivative of those cells reads
slightly wrong values.  The CUDA backward images from the saved full-step fields
and never reads the C-PML memory.

The error is small for most equations and large for `AcousticVRR`.  Measured
against the gradient on the outermost ring of cells itself (fp64, order 4):

| Equation | Error on the outermost ring |
| --- | --- |
| `AcousticVRR` (`vp`) | 66 % |
| `ElasticVRR` (`vp`, `vs`) | 5-7 % |
| The other equations with eager boundary saving | 0.01-0.3 % |

In a 300-iteration Adam FWI of a downsampled Marmousi with absorbing faces on
all four sides, `Acoustic` with boundary saving ended at most 2.3 m/s from the
`Full()` result, with the same model error.  `AcousticVRR` drove its outermost
cells to the velocity bounds and ended with a model error about 4 % larger.

The default keeps this error.  To make eager or JAX boundary saving exact, pass
a sigma=0 buffer of `M + 1` cells:

```python
PropTorch(eq, ..., impl="eager", memory=BoundarySaving(),
          boundary_buffer=3)                      # M + 1 at spatial_order=4
```

The cells with the error then fall in the buffer, whose gradient is cropped, and
the gradient matches `Full()` to rounding on every physical cell (checked in
fp64 for every equation with eager boundary saving).  The buffer moves the PML
ramp out by `M + 1` cells, so the solution itself changes at the level of the
PML's reflection (the records by 3e-6 to 2e-3 relative in our tests), and each
absorbing axis grows by `2(M + 1)` cells.  It cannot be combined with
`topography=` yet.  `impl="c"` needs no buffer, and the eager default `Ckpt()`
is exact without one.  `AcousticVRR` has no compiled kernel: when its outermost
cells matter, use the buffer or `Ckpt()`.

### Boundary tail truncation (`BoundarySaving.tail_steps`)

For **steady-state objectives** — frequency-selection / DFT-comb FWI, where the
loss reads only the **last** `n_probe` samples of the record and the adjoint
source is therefore zero everywhere earlier — the reverse sweep does not need
to walk the whole record.  `tail_steps=K` makes the forward save only the last
`K` steps' boundary strips and stops the backward after them:

```python
memory=BoundarySaving(storage="gpu", tail_steps=n_probe + margin)
```

- **The forward physics is unchanged** — the wavefield still runs the full
  record so the steady state can ring up.  Only the saved/reconstructed step
  range shrinks, so both the backward wall time and the one-shot boundary
  buffer drop by roughly `1 - K / nt` (measured: 74 % backward time and 75 %
  buffer at `K/nt = 25 %`).
- The restore at reverse step `it` consumes the strip saved at forward step
  `it - 1`, so one saved step is spent on alignment: the **effective reverse
  depth is `K - 1`** — budget it inside `margin`.
- **`margin` is physical**: the dropped gradient term is exactly the
  adjoint × ring-up-transient correlation that steady-state methods discard,
  and it decays as the adjoint field drains through the absorbing boundary.
  Sweep the margin once per setup: on a ramped-sine test the truncated
  gradient converges monotonically to the full one
  (cos 0.992 → 1.000000 for margin 0 → 800 steps on a 140×160 grid).
- **Do not use it with impulsive-source objectives**: there the early
  adjoint–forward correlations are real gradient content and the truncated
  gradient is genuinely different (cos ≈ 0.1 in the same test).
- Scope: `impl="c"` Acoustic 2-D/3-D with the boundary-saving backward, any
  `storage`/`storage_dtype`.  Checkpointing, the eager backend and the other
  equations raise `NotImplementedError`/`ValueError` rather than silently
  ignoring the option.  Unset (`None`, the default) is bit-exact legacy
  behaviour, and `tail_steps >= nt` degenerates to it bitwise.
- **Domain decomposition composes**: `ModelParallel` inherits `tail_steps`
  from the wrapped propagator's memory config exactly like
  `storage`/`storage_dtype`, shrinks every tile's boundary ring to `K`
  steps, and stops the lockstep reverse halo loop at the same global step
  on every rank (the stop index is derived from `(nt, tail_steps)`, which
  are identical across ranks by construction, so no rank can be left
  waiting in an exchange).  The truncated DD gradient is bit-exact against
  the truncated single-domain gradient on fp32 boundaries
  (`test/test_dd_tail_two_tile.py`, `test/dd_tail_nccl_check.py`).

## Several time steps per compiled call {#steps-per-call}

`impl="eager"` compiles its time step with `torch.compile`, and on a GPU every
call of the compiled step costs the host 0.2–0.5 ms (Dynamo's guard checks, the
autograd wrappers and the kernel launches), whatever the grid size. On a 2-D
grid of 1000 × 1000 cells this is longer than the step takes on the GPU, which
then waits for the host between steps. `EagerOptions(steps_per_call=k)`
compiles `k` time steps into one call, together with their source injection
and receiver sampling, so that this cost is paid once every `k` steps:

```python
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch

solver = PropTorch(..., impl="eager",
                   eager_options=EagerOptions(steps_per_call=4))
```

The default, `steps_per_call=None`, takes 4 for a rollout that autograd
records (grad mode on, and a model or the wavelet requiring grad) on a 2-D grid
on a GPU, and 1 otherwise. An explicit value applies with and without
gradients. Forward modelling keeps one step per call by default because,
without gradients, Inductor fuses the steps of a call into one another: rather
than write out the fields of each step, the C-PML memory fields first, it
stores the spatial derivatives they are computed from and recomputes the
fields where the next step reads them. On a 2000 × 2000 acoustic grid this
doubles the GPU time of a step (from 0.14 to 0.32 ms on an A100), more than
the call saves on the host; on a 1000 × 1000 grid the fields stay in the L2
cache, and the forward step with `k = 4` is still 1.7–2.0 times faster than
with `k = 1`. With gradients, the fields of every step are saved for the
backward pass and therefore written out, and the GPU time of a step stays about
that of `k = 1`. A 3-D step keeps the GPU busy on its own and was slower
with `k > 1`, with and without gradients; on a CPU the arithmetic outweighs the
host cost.

At `k = 1` the receivers are sampled inside the call as well, except in a
rollout with gradients on the [C-PML strips](#cpml-strips) (the 3-D default):
there the step is compiled alone and the receivers are sampled after the call,
outside the graph. With the sampling inside, the backward graph also scatters
the receivers' gradient into the field, and on the strips Inductor then splits
the step's backward into more full-grid kernels: a 160³ acoustic gradient took
2.95 ms per step instead of 2.62 on an RTX 6000 Ada, and 8 shots of a 1000²
acoustic gradient with `cpml_strips=True` 1.8 ms instead of 1.6. On the full
grid both ran alike, and without gradients the sampling inside the call is the
faster one (0.15 instead of 0.21 ms per step for one shot of a 1000² forward).

Calls start at steps that are multiples of `k`. The first steps of a rollout,
up to the first multiple of `k` after the fields' `requires_grad` flags have
settled, and the last `nt % k` steps run one at a time, uncompiled, so no graph
is compiled for a remainder and a new `nt` does not recompile. Under
`Ckpt(chunks=c)` the chunk length is rounded up to a multiple of `k`: no call
crosses a checkpoint, and the backward pass recomputes each chunk with the same
calls as the forward pass. Two cases keep one step per call:

- `return_wavefield=True`, since each snapshot is the state after its step.
- `use_compile=False`, which runs the steps one at a time; an explicit
  `steps_per_call` above 1 raises `ValueError`.

Under boundary saving (`memory=BoundarySaving(...)`) the backward pass
reconstructs the inputs of a call's `k` steps backwards in time and
differentiates the `k` steps in one compiled call, each step at its own
reconstructed input, so the gradient is that of one step per call up to
float32 round-off. The forward pass of a call runs as one compiled graph when
the C-PML memory fields are kept on the absorbing strips (`cpml_strips`) or a
field has at most 2.5 million cells, and as one compiled call per step
otherwise: without gradients Inductor fuses the steps of a graph into one
another, and with the C-PML fields on the full grid it splits four acoustic
steps into six full-grid kernels. Peak memory grows with `k`, since the
backward pass holds the `k` reconstructed states and the saved intermediate
values of `k` steps: on a 1000 × 1000 acoustic grid with 8 shots it rose from
1.5 GB at `k = 1` to 2.5 GB at `k = 4`, and was 1.3 GB at `k = 4` with
`cpml_strips=True`. `steps_per_call=1` keeps the memory of one step per call.

`PropSEM` compiles one step per call and raises `NotImplementedError` for an
explicit `steps_per_call`.

**Accuracy.** With `k > 1` the records and gradients differ from those with
`k = 1` at the level of float32 round-off: in the runs below, the relative
maximum difference was 3e-7 to 9e-7 for the records and 1e-7 to 2e-6 for the
gradients, the order of the difference between the compiled and the
uncompiled step. It comes from the steps that run uncompiled and from the order
in which Inductor fuses and accumulates the operations within a call, and it is
the same from run to run. In a CPU test in which the uncompiled steps use the
same operations as the compiled ones (`aot_eager` backend), the records for any
`k` equal those for `k = 1` bit for bit. `steps_per_call=1` reproduces results
computed one step per call.

**Compile time and memory.** A call of more steps takes longer to compile: with
a cold Inductor cache, a 2-D gradient rollout compiled in 11–19 s with `k = 4`
against 5–7 s with `k = 1` (1–2 s with a warm cache). With `k = 4` the peak GPU
memory of a forward + backward call grew by 2–12 % in 2-D and 11–13 % in 3-D
under checkpointing, and by 0.3–2 % under `Full()`; the memory held between the
forward and the backward pass did not change under checkpointing.

**Measured.** One shot of a 10 Hz Ricker wavelet, `spatial_order=4`, float32,
random models of 1000², 2000² and 160³ cells with a C-PML of 40 (2-D) or 20
(3-D) cells outside them, `nt` = 1000, 600 and 300, and eager's default
`Ckpt()` (chunks of 100 steps). Each entry is the median of four processes, in
ms per time step, forward / forward + backward (gradients of all models of the
sum of the squared records). The defaults are the `k = 1` forward times and, in
2-D, the `k = 4` forward + backward times.

| ms per step | A100, `k = 1` | A100, `k = 4` | RTX 6000 Ada, `k = 1` | RTX 6000 Ada, `k = 4` |
| --- | --- | --- | --- | --- |
| 2-D acoustic, 1000² | 0.221 / 1.343 | 0.109 / 0.437 | 0.228 / 1.227 | 0.124 / 0.464 |
| 2-D acoustic, 2000² | 0.193 / 1.242 | 0.319 / 0.889 | 0.263 / 1.569 | 0.542 / 1.463 |
| 2-D elastic, 1000² | 0.287 / 1.583 | 0.165 / 0.898 | 0.336 / 1.808 | 0.173 / 0.820 |
| 2-D elastic, 2000² | 0.411 / 2.161 | 0.489 / 2.092 | 0.752 / 3.763 | 0.841 / 3.568 |
| 3-D acoustic, 160³ | 0.548 / 2.359 | 0.732 / 2.445 | 0.686 / 3.294 | 1.053 / 3.778 |
| 3-D elastic, 160³ | 1.494 / 7.521 | 2.045 / 8.320 | 2.797 / 13.833 | 3.379 / 14.769 |

With `k = 4` a 2-D forward + backward step was 1.8–3.1 times faster at 1000²
and 1.03–1.4 times faster at 2000²; a 2-D forward step was 1.7–2.0 times faster
at 1000² and 1.1–2.1 times slower at 2000²; a 3-D step was 4–15 % slower with
gradients and 1.2–1.5 times slower without. `k = 8` was faster than `k = 4` for
the 2-D gradients at 1000² except the elastic one on the RTX 6000 Ada
(acoustic: 3.6–3.8 times faster than `k = 1`), not at 2000² or in 3-D, and its
first compile took about twice as long as with `k = 4` (20–34 s in 2-D). An
explicit `steps_per_call=4` is therefore worth setting for forward modelling on
2-D grids of about 1000² cells or fewer; `steps_per_call=1` halves the first
compile, which matters for a run of a few rollouts.

**One shot in 2-D: `k = 8`.** A gradient of one shot on a 2-D grid of about
1000² cells still waits for the host at `k = 4`, and `steps_per_call=8` halves
the number of calls. With the [Inductor options](#inductor-options) the step
compiles with on a GPU, a gradient step on an RTX 6000 Ada (1000² cells,
`nt` = 1000, `memory=BoundarySaving()`) took 0.243 ms instead of 0.494 ms for
the acoustic equation and 0.643 ms instead of 0.894 ms for the elastic one,
2.0 and 1.39 times faster than `k = 4` without those options; `k = 8` alone was
1.25 and 1.08 times faster. The records and gradients differ from those with
`k = 4` at the level of float32 round-off (at most 4.5e-7 relative), the first
compile takes about 30 % longer (63 s instead of 47 s for the acoustic
gradient), and the peak GPU memory of a gradient call grows by 20–25 %. Set it
yourself: the default stays at 4, because `k = 8` has not been measured with
several shots per call, where the GPU rather than the host sets the pace.

## C-PML memory fields on the absorbing strips {#cpml-strips}

The C-PML memory fields are zero away from the absorbing layers. With
`EagerOptions(cpml_strips=True)` the eager propagator keeps each of them as its
two strips along its axis and updates only those: the step no longer reads and
writes the memory fields over the whole grid, but it launches more kernels. The
default, `cpml_strips=None`, takes the strips on a 3-D grid and the full grid
on a 2-D grid and with `topography=`; `True` or `False` sets the layout
(`True` with `topography=` raises). The records and gradients of the two
layouts agree to float32 round-off (uncompiled, bit for bit).

On a 2-D grid, whether the strips pay depends on how many cells one call
computes per field: the number of shots times the cells of the padded grid. A
call of a few shots is launch-bound, the GPU waiting for the host between
kernels, and the extra kernels make it slower; once the fields of a call no
longer fit in the GPU's L2 cache, the full-grid memory traffic sets the pace and
the strips win. Gradients on an RTX 6000 Ada (96 MB L2), compiled,
`memory=BoundarySaving()`, a C-PML of 40 cells, `spatial_order=4`, one run per
case, ms per time step with the full grid / the strips:

| Shots × grid | Cells per field | Acoustic | Elastic |
| --- | --- | --- | --- |
| 1 × 1000² | 1.2 M | 0.55 / 0.79 | 0.97 / 1.69 |
| 2 × 1000² | 2.4 M | 0.59 / 0.77 | 1.58 / 1.54 |
| 3 × 1000² | 3.5 M | 1.05 / 0.80 | 2.54 / 2.09 |
| 4 × 1000² | 4.7 M | 1.36 / 0.76 | 3.39 / 2.63 |
| 8 × 1000² | 9.4 M | 2.72 / 1.33 | 6.77 / 4.89 |
| 1 × 2000² | 4.3 M | 1.27 / 0.76 | — |

Calls of the same number of cells on 400², 1000² and 2000² grids gave the same
ratio, and an A100 (40 MB L2) crossed over at about the same size (acoustic,
2 × 1000²: 0.62 / 0.84 ms, 3 × 1000²: 0.86 / 0.73 ms). A forward without
gradients, one compiled call per step, is launch-bound longer: the strips made
it faster on the RTX 6000 Ada only at about 9 million cells (acoustic,
4 × 1000²: 0.26 / 0.33 ms, 8 × 1000²: 0.51 / 0.33 ms), and on the A100 not yet
at 8 × 1000² (0.29 / 0.33 ms). The strips also lower the peak memory of a
gradient: 2.2 to 1.3 GB for 8 acoustic shots on 1000² under boundary saving,
5.5 to 4.8 GB for 8 elastic ones.

## Environment toggles

A few knobs stay out of the API because the right value depends on the
machine, not on the problem. All are read once per run. None changes the
physics, but one changes the numbers: a lossy `SWEEP_BOUNDARY_DTYPE`
(fp16/bf16/int8).

| Variable | Effect |
| --- | --- |
| `SWEEP_DD_DISABLE_OVERLAP=1` | Domain decomposition: serial step-then-exchange instead of the overlapped forward (see [Domain decomposition](parallel.md)). |
| `SWEEP_BOUNDARY_DTYPE` | `impl="c"`: `storage_dtype` of the implicit default ring only (no memory knob passed). Any `BoundarySaving(...)` carries `storage_dtype="fp32"` unless set, so it always wins. |
| `SWEEP_DATASETS_CACHE` | Where `sweep.datasets` caches downloads (see [Datasets](datasets.md)). |
| `SWEEP_JIT_FULL=1` | Developer path: compile the pybind shim against your torch instead of using the ctypes layer over the prebuilt core. |
| `SWEEP_CORE=<path>/libsweep_core.so` | Use a custom CUDA core; its `core.json` sidecar gets the same fit check as the shipped one. |
| `SWEEP_JIT_ALLOW_OLD_CUDA=1` | Try an nvcc 12.0–12.3 for a local core build. |
| `TORCH_CUDA_ARCH_LIST` | Target arch(s) of a local core build; when set it overrides the visible GPU's arch, and it is required when no GPU is visible. |
| `TORCH_EXTENSIONS_DIR` | Where a locally built core is cached. |
| `CUDA_HOME` / `CUDA_PATH` | Toolkit for a local core build, version-checked against torch's CUDA major. |

## Consistency testing

The C memory modes are exercised by `test/solver_gradient_mode_suite.py`. The
suite compares eager gradients against compiled full-wavefield, boundary
saving, and checkpoint modes across interior, finite-difference edge, and
free-surface source placements, and writes per-mode gradient figures to
`test/test_outputs/solver_gradient_mode_suite/`.
