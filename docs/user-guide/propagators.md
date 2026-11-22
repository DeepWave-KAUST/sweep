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
  pure-Python operators with PyTorch autograd; `"c"` = the compiled core
  (`libsweep_core.so`) driven through the pure-Python `sweep._C` ctypes layer —
  the prebuilt CUDA core on a CUDA device, the host core built from the same
  sources on a CPU device).

The two axes are orthogonal:

| Backend | Impl | Device | What runs |
| --- | --- | --- | --- |
| `torch` | `eager` | CPU or CUDA | Pure-Python operators + PyTorch autograd |
| `torch` | `c` | CUDA | Prebuilt CUDA core via the `sweep._C` ctypes layer |
| `torch` | `c` (explicit) | CPU | The same compiled code as the [host core](../dev/host_core.md), when one is at hand |
| `jax` | — | CPU or CUDA | JAX implementation via `PropJax` |

`"cuda"` is **not** a backend or `impl` value — it is a device choice driven by
the tensors you pass in. The compiled path follows that device: CUDA tensors
run the CUDA core; CPU tensors run the host core when `impl="c"` is spelled out
and a host core is at hand (otherwise a warning and eager). A CPU propagator
with `impl=None` resolves to `impl="eager"`. Models on mixed devices are
refused with a clear error.

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
- **``DASZhao`` / ``DASElastic`` (2-D and 3-D)** raise ``NotImplementedError``
  for any free surface: their step has no free-surface condition.  For DAS
  under a free surface use ``DASMu`` / ``DASMu3D``.

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
| Lossless compression of the boundary ring (nvCOMP Bitcomp) | `impl="c"`, template-driver equations | `BoundarySaving(compress="bitcomp", ...)` |
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
### Second derivatives (Hessian-vector products)

A Hessian-vector product by double backward (`torch.autograd.grad(..., create_graph=True)`,
then a second `grad`) needs the first backward on the autograd tape. Only the uncompiled
eager step has that: `impl="eager"` with `memory=Full()` or `Ckpt()` and
`EagerOptions(use_compile=False)`, with `torch.backends.cudnn.allow_tf32 = False` on
Ampere and later GPUs. The tape holds every step of the forward and of the first backward,
so this route is for small models. `impl="c"`, `BoundarySaving()`, the compiled eager step,
`ModelParallel` and registered gradient-mode imaging conditions raise a `RuntimeError`
under `create_graph=True`. The qP equations (`AcousticVTI`, `AcousticTTI` and their Born
pair) refuse a second derivative in every `s_gradient` mode, because it is off the tape
(`'adjoint'`, which refuses `create_graph=True`), NaN in float32 (`'exact'`) or not a
Hessian (`'frozen'`); `'exact'` and `'frozen'` keep their `create_graph=True` gradient for
a derivative with respect to the residual, as a Gauss-Newton solver takes it (J h by a
double vjp for `'exact'`, the linearisation with S_n held fixed for `'frozen'`). A difference
of their gradients is no substitute: S_n depends on the direction of a field's own gradient,
which turns abruptly where that gradient nearly vanishes, so where epsilon differs from delta
the record and the gradient are not smooth functions of the model, and central differences
converge only at steps of about 1e-5 of the model or below, where float32 round-off
dominates.

At production size, compute the product from first derivatives, which run with every
`impl` and memory strategy:

| Product | How | Cost |
| --- | --- | --- |
| $\mathbf H\mathbf h$, to $O(\varepsilon^2)$ | `(g(m + eps*h) - g(m - eps*h)) / (2*eps)` | two gradients |
| $\mathbf J^{\mathsf T}\mathbf J\mathbf h$ (Gauss-Newton), exact | Born record of `h` (e.g. `AcousticBorn`, `mp = 2*h/vp`) as `grad_outputs` of a gradient | one Born forward, one gradient |
| $\mathbf H\mathbf h$, exact | gradient of $\langle \mathbf d(\mathbf m), \mathbf s\rangle + \langle \mathbf J(\mathbf m)\mathbf h, \mathbf r\rangle$ with $\mathbf s = \mathbf J\mathbf h$ and the residual $\mathbf r$ held fixed; `J(m) h` is the Born record with `mp = 2*h/m` computed from the model tensor | one Born forward, one forward, their gradients |

The last route is the second-order adjoint method written as first derivatives: it needs the
Born equation to return the gradient of its background model (`AcousticBorn` with
`impl="c"` in `Full()` or `BoundarySaving()`, or any eager Born equation), and it runs with
the compiled eager step as well.

### Jacobian-vector products (forward mode)

Forward-mode AD carries a direction `h` through the uncompiled eager step with
`memory=Full()`: pass the models as dual tensors, and the tangent of the record is
$\mathbf J\mathbf h$, the Born record of `h`, for any eager equation and without a Born
equation:

```python
import torch.autograd.forward_ad as fwAD

with torch.no_grad(), fwAD.dual_level():
    record = prop(wavelet, sources, receivers, models=[fwAD.make_dual(vp, h)])
    d, jh = fwAD.unpack_dual(record)          # the record and J h
```

It keeps no autograd tape, so its memory is about twice that of the forward; the
uncompiled dual step took 11 to 17 times as long as a plain forward on a V100. With the
models also requiring grad (drop `torch.no_grad()`), the gradient of
$\langle \mathbf d, \mathbf s\rangle + \langle \mathbf J(\mathbf m)\mathbf h, \mathbf r\rangle$
with $\mathbf s = \mathbf J\mathbf h$ held fixed in the first term and the residual
$\mathbf r$ in the second is the exact $\mathbf H\mathbf h$ from first derivatives. The
compiled step, `Ckpt()`, `BoundarySaving()`, `impl="c"`, `ModelParallel`, `PropSEM`,
registered gradient-mode imaging conditions, `AcousticVTI` and `AcousticTTI` in their
`'adjoint'` and `'frozen'` modes and their Born pair in every mode refuse a dual tensor with
a `RuntimeError` (the compiled steps used to return a tangent of zero; the `'frozen'` tangent
would be the linearisation with S_n held fixed; the Born pair's scattered field takes S_n
from its own gradient, so at mp = 0 a tangent would propagate it elliptically).
`AcousticVTI` and `AcousticTTI` carry J h with `s_gradient='exact'`. `ViscoAcoustic`'s Q
tangent holds the fractional exponent fixed, as its gradient does.

`sweep.autodiff` wraps these routes in one call each, `jacobian_vector_product`,
`gauss_newton_vector_product` and `hessian_vector_product`, for any twice-differentiable
loss of the record. `route="auto"` (the default) takes forward mode where the propagator
carries a tangent and a central difference of two gradients (or forwards) everywhere else,
so the same call runs with `impl="c"`, boundary saving and `ModelParallel`:

```python
from sweep.autodiff import hessian_vector_product

loss = lambda d: 0.5 * ((d - d_obs) ** 2).sum()
(hv,) = hessian_vector_product(prop, wavelet, sources, receivers, [vp], [h], loss)
```

A `None` direction holds a model fixed: it is not perturbed, no gradient is taken of it
(`impl="c"` `ElasticBorn` and the qP Born pair refuse a background model that requires one,
as `AcousticBorn` does under checkpointing) and its entry of the result is `None`.
Directions are cast to their model's dtype and device. The difference step changes no model
by more than `rel_step` (default 1e-2) of its peak, an all-zero model (LSRTM's zero
reflectivity) counting as 1; under `ModelParallel` every rank takes the step of the whole
model. `"auto"` takes forward mode, which is exact, where the propagator carries a tangent
(the uncompiled step with `Full()`), and differences everywhere else, also on the uncompiled
step with `Ckpt()`, where `route="double_backward"` is exact too. One H h on a 100 x 200 grid
with 1000 steps (4 shots, V100, `Acoustic`):

| Route | Time | Peak memory | Off the exact product |
| --- | --- | --- | --- |
| double backward, uncompiled eager, `Full()` | 7.3 s | 5.3 GiB | (reference) |
| reverse over forward, uncompiled eager, `Full()` (`"auto"`) | 27 s | 3.3 GiB | 2.8e-6 |
| difference, uncompiled eager, `Full()` | 5.4 s | 0.7 GiB | 1.6e-3 |
| difference, eager defaults: compiled step, `Ckpt()` (`"auto"`) | 0.95 s | 127 MiB | 2.1e-3 |
| difference, `impl="c"`, boundary saving (`"auto"`) | 0.13 s | 7 MiB | 1.3e-3 |

On a direction that touches the outermost ring (a gradient does), a difference is about
1e-2 off double backward whatever the step (below). The qP equations refuse the curvature
products and differences (their `curvature_refusal()` and `difference_refusal()`): where
epsilon differs from delta their record and gradient are not smooth functions of the model,
and the `'exact'` gradient, the Gauss-Newton product's J^T, moves by tens of percent between
float32 and float64. J h of `AcousticVTI` and `AcousticTTI` along a smooth direction still
holds by forward mode with `s_gradient='exact'` (float32 within 0.5% of float64); the Born
pair refuses a tangent (above), and for LSRTM its record with mp = h is the Born operator
applied to h, the mp gradient its adjoint.

Forward mode, like a difference of gradients, differentiates the gradient the propagator
returns. On the outermost ring of model cells that gradient is the cropped one (the model is
edge-replicated into the PML, `EdgePadding`, and its backward crops), so for a direction on
that ring forward mode and differences give a product that is not symmetric, while double
backward gives the symmetric one; keep directions off that ring when the routes are
compared or when a solver needs a symmetric operator.

The
[Hessian-vector products notebook](../notebooks/46_hessian_vector_product.ipynb)
compares the routes, shows how to choose `rel_step`, and assembles the full Hessian and the
Gauss-Newton matrix of a small model with `hessian_vector_products`.

**Many directions at once.** `hessian_vector_products` takes K directions stacked on a
leading axis, `(K, *model.shape)` per model, and returns the K products stacked the same
way -- what K calls of `hessian_vector_product` return. Where the equation takes a model per
shot (`supports_batched_models`: the 2-D `Acoustic`, `Elastic`, `ElasticBorn`, `ViscoAcoustic`
and `Biot`) and the models are shared, `batch` directions (all K by default) run as copies of
the survey in one call, each copy with its own models: one forward and one double backward
for the lot (one dual forward for `route="forward"`, one gradient of 2 x `batch` copies for
`"difference"`, with every impl). A copy holds what one survey holds, so `batch` bounds the
memory. Elsewhere the directions run one after another.

```python
from sweep.autodiff import hessian_vector_products

eye = torch.eye(vp.numel(), device=vp.device).reshape(-1, *vp.shape)   # every unit direction
(H,) = hessian_vector_products(prop, wavelet, sources, receivers, [vp], [eye], loss,
                               route="double_backward", batch=150)
H = H.reshape(vp.numel(), vp.numel())                                    # the full Hessian
```

All 600 columns of the Hessian of a 20 x 30 model (one shot, 400 steps, free surface, V100,
`Acoustic`):

| Route | Per column | 600 columns | Peak memory |
| --- | --- | --- | --- |
| `hessian_vector_product`, double backward, one call per column | 2.9 s | 29 min | 42 MiB |
| `torch.autograd.grad(..., is_grads_batched=True)` on the double backward | 0.58 s | 5.8 min | 110 MiB |
| `hessian_vector_products`, double backward, `batch=150` | 20 ms | 12 s | 6.0 GiB |
| the same inside `slice_stencils()`, `batch=150` | 32 ms | 19 s | 1.6 GiB |
| the same inside `slice_stencils()`, `batch=300` | 16 ms | 9.8 s | 3.1 GiB |
| `hessian_vector_products`, reverse over forward, `batch=150` | 70 ms | 42 s | 3.8 GiB |
| `hessian_vector_products`, difference, `impl="c"`, `batch=300` | 0.9 ms | 0.55 s | 78 MiB |

PyTorch's own batching of directions, `torch.autograd.grad(..., is_grads_batched=True)` and
`torch.autograd.functional.hessian(..., vectorize=True)`, does not batch the eager step: its
legacy vmap has no batching rule for the convolutions' derivatives (nor for `copy_`,
`zeros_like`, `index_select`), so it loops per direction inside them. The double backward
builds its graph with the convolutions, whose backward keeps the kernel alone, so a second
derivative is one more convolution. Inside `sweep.operators.torch.slice_stencils()` the
uncompiled step runs its stencils as the shifted slices `torch.compile` runs: the same
products (5e-7 apart), a graph three to four times smaller and 1.4 to 1.7 times the time at
the same `batch`, so the batch can grow three to four times where memory is the limit (the
`batch=300` row). Nothing switches to it by itself, and the slices carry no forward-mode
tangent.

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

### Boundary ring compression (`BoundarySaving.compress`) {#bs-compress}

`compress="bitcomp"` runs every chunk of the saved boundary through nvCOMP's
lossless Bitcomp coder on the copy stream, and keeps the packed bytes instead
of the chunk: in a device arena (`storage="gpu"`), a pinned host arena
(`storage="cpu"`) or one sparse file (`storage="disk"`).  It is lossless, so
the gradient is **bit-identical** to the same `storage_dtype` without it
(`test/test_boundary_compress.py` asserts `torch.equal` on every storage);
what changes is bytes, and it composes with `storage_dtype`:

```python
memory=BoundarySaving(storage="gpu", storage_dtype="int8",
                      compress="bitcomp", compress_budget=0.6)
```

- **What it saves.**  Packed bytes as a fraction of the ring (measured, one
  shot, `transfer_interval=16`):

  | problem | int8 | fp16 | bf16 | fp32 |
  | --- | --- | --- | --- | --- |
  | Overthrust crop, 3-D acoustic, 100x160x160, 800 steps | 52 % | 54 % | 51 % | 60 % |
  | SEG/EAGE Salt crop, 3-D acoustic, 100x160x160, 800 steps | 37 % | 43 % | 46 % | 52 % |
  | Marmousi2 at 25 m, 2-D elastic (five fields), 2000 steps | 28 % | 28 % | 29 % | 33 % |

  `storage_dtype="int8"` is itself 25 % of fp32, so int8 + compress is 7-14 %
  of an fp32 ring.  The coder is told the element width, which is why an fp32
  ring packs at all (its exponents are smooth).  The ratio of a run is
  `propagator.last_boundary_compress_ratio` (uncompressed / packed) and
  `propagator.boundary_compress_report()` gives it per face.
- **`compress_budget`** (`storage="gpu"` / `"cpu"` only): the arena is
  allocated up front at this fraction of the uncompressed ring (plus the
  int8 / fp16 scales, which ride along uncompressed), since the packed size
  is only known as the forward runs; the bytes a run saves are
  `1 - compress_budget` of the ring, not the ratio it happens to reach.  A run
  that overflows it raises, naming the live set that did not fit and the
  budget it needed.  The disk file is sparse and takes no budget.
- **What the arena holds, and how to set the budget.**  With a ring as
  long as the record every chunk stays: the arena holds the whole forward
  packed, and `compress_budget = 1 / ratio` fits.  A ring refilled per
  segment (`ModelParallel(segment_steps=K)`: the ring is `K + 1` steps and
  the forward overwrites it) holds only the chunks the ring still covers,
  at most `ceil((K + 1) / transfer_interval) + 1`; older chunks are
  evicted and the arena is allocated circularly, which can waste up to one
  packed chunk at the wrap.  The budget therefore multiplies that live
  window plus one chunk, so `1 / ratio` of the *live set* fits here too.
  Read the budget off a run instead of guessing it:
  `propagator.boundary_compress_report()` returns, besides the per-face
  bytes of the chunks in the arena *now* (`ratio`), the fullest live set the
  last call held at any time (`peak_ratio`, `peak_bytes`: for a segmented
  ring the forward's last segment or whichever refill packed worst, since
  the backward replays the segments in reverse and leaves the first,
  quietest one in the arena) and `budget_needed`, the `compress_budget`
  that set needed without margin; set about 10 % more.
  `propagator.last_boundary_compress_ratio` is `peak_ratio`.  Under
  `ModelParallel` the store lives on the captured tile params, so call
  `ddp.boundary_compress_report()` / `ddp.last_boundary_compress_ratio` on
  the wrapper (per rank; ranks differ, take the lowest ratio).  Each chunk's
  index row records the steps it was compressed with, so segment ends need
  not align with `transfer_interval`.
- **Chunking.**  The chunk is `transfer_interval` steps (the ring's slot);
  with `storage="gpu"` the knobs `transfer_interval` / `ring_buffers` apply
  as on the staged storages and default to 16 / 2.  `storage="disk"` needs
  `disk_async_read=True` (the packed chunks are fetched by the reader
  thread).  Cost: compressing a chunk is one batched kernel on the copy
  stream per `transfer_interval` steps; it overlaps the next chunk's
  compute, and the per-step save / restore kernels are untouched.
- **When it pays.**  The chunk kernels hide behind compute only when there
  is compute to hide behind.  Measured per-step cost of `compress` (forward
  with saving + backward, int8 ring, RTX 6000 Ada): a step of 0.06-0.14 ms
  (small 2-D grids, a 48x64x64 3-D box) pays 2-8 %, a 0.5 ms step
  (100x160x160) about 1 %, a step of 1.8 ms and up (160x256x256 and
  larger) 0.3 % or less.  Use it where the ring is what limits the problem
  size, not on a small model that fits anyway.
- **Dependency.**  The library is loaded at run time from the
  `nvidia-nvcomp-cu12` wheel (`import nvidia.libnvcomp`) or
  `SWEEP_NVCOMP_LIB=/path/to/libnvcomp.so.6`; nothing falls back, a missing
  library raises `ImportError`.
- Scope: `impl="c"` on a CUDA device, the equations on the shared template
  drivers (Acoustic 2-D/3-D, Elastic 2-D/3-D, the staggered TTI and VRR
  families, DASMu, AcousticVTI, ViscoElastic; `cuda_layout.supports_boundary_compress`).
  The hand-written drivers, the eager backend and the host core raise
  `NotImplementedError` rather than ignoring the option.  `ModelParallel`
  inherits it per tile like `storage_dtype`.  Worked example:
  [Compress the saved boundary with nvCOMP](../notebooks/35_boundary_ring_compression.ipynb).

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
numbers: a setting that does belongs in the propagator's config, where it is
recorded with the run. The boundary storage dtype used to be one
(`SWEEP_BOUNDARY_DTYPE`, `SWEEP_FP16_BOUNDARY`); it is
`BoundarySaving(storage_dtype=...)` now, and a set variable is an error.

| Variable | Effect |
| --- | --- |
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
