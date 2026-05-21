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

| Feature | Path | Configured by |
| --- | --- | --- |
| Eager activation checkpointing | `impl="eager"` | top-level `use_ckpt` / `ckpt_chunks` on `PropTorch` |
| `torch.compile` on the eager step | `impl="eager"` | `EagerOptions(use_compile=True, ...)` |
| Boundary saving (GPU / pinned CPU / disk) | `impl="c"` | `BoundaryOptions(storage="gpu" | "cpu" | "disk", ...)` |
| Asynchronous disk prefetch | `impl="c"` | `BoundaryOptions(storage="disk", disk_async_read=True, ...)` |
| Chunked checkpointing | `impl="c"` | `CkptOptions(mode="chunk", chunks=...)` |
| Recursive (fixed-budget) checkpointing | `impl="c"` | `CkptOptions(mode="recursive", count=...)` |

A runnable comparison of these options lives in the
[Memory · strategies notebook](../notebooks/07_memory_strategies.ipynb), which
exercises full-wavefield, boundary saving, and checkpointing on the same
Marmousi shot and prints the per-mode peak GPU / host memory.

## Consistency testing

The C memory modes are exercised by `test/solver_gradient_mode_suite.py`. The
suite compares eager gradients against compiled full-wavefield, boundary
saving, and checkpoint modes across interior, finite-difference edge, and
free-surface source placements, and writes per-mode gradient figures to
`test/test_outputs/solver_gradient_mode_suite/`.
