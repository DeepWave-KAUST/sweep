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
  pure-Python operators with PyTorch autograd; `"c"` = compiled C++ / CUDA
  kernels through the `sweep._C` extension).

The two axes are orthogonal:

| Backend | Impl | Device | What runs |
| --- | --- | --- | --- |
| `torch` | `eager` | CPU or CUDA | Pure-Python operators + PyTorch autograd |
| `torch` | `c` | CPU or CUDA | Compiled C++ / CUDA kernels via `sweep._C` |
| `jax` | — | CPU or CUDA | JAX implementation via `PropJax` |

`"cuda"` is **not** a backend or `impl` value — it is a device choice driven by
the tensors you pass in. The compiled extension runs C++ CPU kernels on CPU
tensors and CUDA kernels on CUDA tensors.

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
  `(nshots, nsources, ndim)` for multiple sources per shot
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
condition; the remaining faces stay absorbing. ``abcn`` accepts the same
per-edge forms (a scalar, or a canonical tuple of per-face PML widths).

Support matrix:

- **2-D ``Acoustic`` / ``Elastic``** — full per-edge support on the eager
  and compiled CUDA backends, with adjoint gradients across every backward
  memory mode (full / boundary-saving / checkpointing).
- **2-D ``ViscoAcoustic``** — full per-edge support on the eager and
  compiled CUDA backends (full / checkpointing gradients; boundary saving
  is refused — the dissipative spectral damping term cannot be
  reverse-time reconstructed, and the ``impl='c'`` default falls back to
  ``'full'``).
- **3-D** — top face only (``free_surface=True``); per-edge lists raise.
- **Topography** (irregular surface) — a separate top-only feature; it
  cannot be combined with per-edge faces.
- **Anisotropic equations** (``AcousticVTI1st``, ``AcousticVTI``,
  ``AcousticTTI``, ``ElasticTTI(SG)``, …) raise ``NotImplementedError`` for
  any free surface: the anisotropic stress-free condition couples through
  the stiffness tensor and is not the isotropic image method these solvers
  implement. Run with ``free_surface=False`` or use an isotropic equation.

See the [per-edge free surface notebook](../examples/index.md) for
snapshots of each configuration and a closed-box energy check.

## Implementation-specific options

Every option block lives in `sweep.propagator.options` and is documented in
detail on the [Propagator Options](../api/propagators/options.md) page. A
quick reference:

| Option block | Used with | Configures |
| --- | --- | --- |
| `EagerOptions` | `impl="eager"` | `torch.compile` flags, debug knobs |
| `CUDAOptions` | `impl="c"` | Container for `MemoryOptions` |
| `MemoryOptions` | inside `CUDAOptions.memory` | Picks **one** C memory-saving strategy |
| `BoundaryOptions` | inside `MemoryOptions.boundary` | Boundary saving with GPU / CPU / disk storage |
| `CkptOptions` | inside `MemoryOptions.ckpt` | Chunk- or recursive-mode checkpointing |

Pass each block through the matching kwarg:

```python
PropTorch(..., eager_options=EagerOptions(...))  # impl="eager"
PropTorch(..., cuda_options=CUDAOptions(...))    # impl="c"
```

## A full Option composition example

To FWI on a large model with the compiled CUDA path, boundary saving on pinned
host memory, and asynchronous disk fallback at the largest scale:

```python
import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.propagator.options import (
    CUDAOptions, MemoryOptions, BoundaryOptions,
)

dev = torch.device("cuda")
shape = (512, 2048)
dh, dt = 10.0, 0.002

solver = PropTorch(
    Acoustic(spatial_order=8, device=dev),
    shape=shape, dh=dh, dt=dt, dev=dev,
    impl="c",
    cuda_options=CUDAOptions(
        memory=MemoryOptions(
            strategy="boundary",
            boundary=BoundaryOptions(storage="cpu", pinned_memory=True),
        )
    ),
)
```

Replace the `BoundaryOptions(...)` block with `CkptOptions(mode="chunk",
chunks=100)` (wrapped in `MemoryOptions(strategy="ckpt", ckpt=...)`) to use
chunk-mode checkpointing instead.

The validation rules are enforced in the dataclass `__post_init__` methods, so
incompatible combinations (e.g. `disk_async_read=True` with `storage="cpu"`)
fail loudly at construction time rather than during a long FWI run.

## Memory-saving features

The gradient-memory mode is a **three-way choice — `'full'`, `'boundary'`, or
`'ckpt'` — identical for the eager and CUDA backends**, selected once via
`memory=MemoryOptions(strategy=...)`.  Left unset, `impl="c"` defaults to
`'boundary'` (GPU ring, fp32) and the eager backend to `'ckpt'`.  The modes
are mutually exclusive: conflicting requests (e.g. the legacy `use_ckpt=True`
together with an enabled `boundary_saving_config`) raise a `ValueError`
instead of one path silently winning.  The legacy `use_ckpt` /
`boundary_saving_config` knobs remain accepted and resolve into the same
three-way choice.

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
  `memory=MemoryOptions(strategy='boundary'), use_ckpt=False` states one
  intent twice and is accepted; `..., use_ckpt=True` contradicts it and
  raises.  Where both carry detail, `memory=` wins.

Tail truncation has a dict spelling too — `boundary_saving_config={'enabled':
True, 'tail_steps': K}` is equivalent to
`BoundaryOptions(tail_steps=K)`.

| Feature | Path | Configured by |
| --- | --- | --- |
| Full storage (no reconstruction) | both | `MemoryOptions(strategy="full")` |
| Boundary saving (GPU ring; + pinned CPU / disk on `impl="c"`) | both | `MemoryOptions(strategy="boundary", boundary=BoundaryOptions(storage=..., storage_dtype=..., ...))` |
| Asynchronous disk prefetch | `impl="c"` | `BoundaryOptions(storage="disk", disk_async_read=True, ...)` |
| Boundary tail truncation (steady-state / freqsel objectives) | `impl="c"` acoustic | `BoundaryOptions(tail_steps=...)` |
| Chunked checkpointing | both | `MemoryOptions(strategy="ckpt", ckpt=CkptOptions(mode="chunk", chunks=...))` |
| Recursive (fixed-budget) checkpointing | `impl="c"` | `CkptOptions(mode="recursive", count=...)` |
| `torch.compile` on the eager step | `impl="eager"` | `EagerOptions(use_compile=True, ...)` |

A runnable comparison of these options lives in the
[Memory · strategies notebook](../notebooks/07_memory_strategies.ipynb), which
exercises full-wavefield, boundary saving, and checkpointing on the same
Marmousi shot and prints the per-mode peak GPU / host memory.

### Boundary tail truncation (`BoundaryOptions.tail_steps`)

For **steady-state objectives** — frequency-selection / DFT-comb FWI, where the
loss reads only the **last** `n_probe` samples of the record and the adjoint
source is therefore zero everywhere earlier — the reverse sweep does not need
to walk the whole record.  `tail_steps=K` makes the forward save only the last
`K` steps' boundary strips and stops the backward after them:

```python
memory=MemoryOptions(
    strategy="boundary",
    boundary=BoundaryOptions(storage="gpu", tail_steps=n_probe + margin),
)
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
  `storage`/`storage_dtype`.  Checkpointing, elastic and `rtm()` raise
  `NotImplementedError`/`ValueError` rather than silently ignoring the
  option.  Unset (`None`, the default) is bit-exact legacy behaviour, and
  `tail_steps >= nt` degenerates to it bitwise.
- **Domain decomposition composes**: `ModelParallel` inherits `tail_steps`
  from the wrapped propagator's memory config exactly like
  `storage`/`storage_dtype`, shrinks every tile's boundary ring to `K`
  steps, and stops the lockstep reverse halo loop at the same global step
  on every rank (the stop index is derived from `(nt, tail_steps)`, which
  are identical across ranks by construction, so no rank can be left
  waiting in an exchange).  The truncated DD gradient is bit-exact against
  the truncated single-domain gradient on fp32 boundaries
  (`test/test_dd_tail_two_tile.py`, `test/dd_tail_nccl_check.py`).

## Environment toggles

A few knobs stay out of the API because the right value depends on the
machine, not on the problem. All are read once per run; none changes results.

| Variable | Effect |
| --- | --- |
| `SWEEP_VRZ_GRAD_SPLIT=1` | `AcousticVRZ3D` backward: force the O(M) split gradient (materialise `c_d`/`e_d`, then one divergence) instead of the fused nested-stencil kernel that `order<=4` picks by default. The crossover is GPU-dependent — fused wins on RTX 6000 Ada, split is ~12 s/iter faster on V100 at production scale. |
| `SWEEP_DD_DISABLE_OVERLAP=1` | Domain decomposition: serial step-then-exchange instead of the overlapped forward (see [Domain decomposition](parallel.md)). |
| `SWEEP_BOUNDARY_DTYPE` | Default `storage_dtype` for the boundary ring; an explicit `BoundaryOptions(storage_dtype=...)` wins. |
| `SWEEP_DATASETS_CACHE` | Where `sweep.datasets` caches downloads (see [Datasets](datasets.md)). |

## Consistency testing

The C memory modes are exercised by `test/solver_gradient_mode_suite.py`. The
suite compares eager gradients against compiled full-wavefield, boundary
saving, and checkpoint modes across interior, finite-difference edge, and
free-surface source placements, and writes per-mode gradient figures to
`test/test_outputs/solver_gradient_mode_suite/`.
