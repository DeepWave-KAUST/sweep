# Propagator Options

This page documents the dataclass-based option blocks that configure
`PropTorch`. They live under `sweep.propagator.options` and are passed through
the propagator constructor:

```python
from sweep.propagator.torch import PropTorch
from sweep.propagator.options import (
    EagerOptions,
    CUDAOptions,
    MemoryOptions,
    BoundaryOptions,
    CkptOptions,
)
```

Implementation: `src/sweep/propagator/options.py`.

## Quick map

| Block | Used with | Configures |
| --- | --- | --- |
| `EagerOptions` | `impl="eager"` | `torch.compile` flags, debug knobs |
| `CUDAOptions` | `impl="c"` | Compiled C++ / CUDA runtime configuration |
| `MemoryOptions` | inside `CUDAOptions.memory` | Chooses one C-memory-saving strategy |
| `BoundaryOptions` | inside `MemoryOptions.boundary` | Boundary-saving GPU / CPU / disk storage |
| `CkptOptions` | inside `MemoryOptions.ckpt` | Chunk / recursive checkpointing in the C path |

Pass an option block via the matching named kwarg:

```python
PropTorch(..., eager_options=EagerOptions(...))   # impl="eager"
PropTorch(..., cuda_options=CUDAOptions(...))     # impl="c"
```

A generic `backend_options=` slot is also accepted; see the
[backend_options](#backend_options) section at the bottom.

### Eager checkpointing today

Eager-side activation checkpointing is still controlled by the top-level
`use_ckpt` / `ckpt_chunks` arguments on `PropTorch` — *not* by
`MemoryOptions`. Only the C path (`impl="c"`) reads `MemoryOptions`.

## `EagerOptions`

```python
@dataclass
class EagerOptions:
    use_compile: bool = False
    compile_mode: str = "default"
    compile_dynamic: bool = False
    compile_backend: str | None = None
    compile_fullgraph: bool = False
    store_last_wavefield: bool = False
```

Use with `impl="eager"`:

```python
PropTorch(..., backend="torch", impl="eager",
          eager_options=EagerOptions(use_compile=True))
```

Fields:

| Field | Meaning |
| --- | --- |
| `use_compile` | Enables `torch.compile` on the eager step function. See the [`torch.compile` notes on operators](../../user-guide/backends.md) for caveats. |
| `compile_mode` | `mode` argument passed to `torch.compile`. Common values: `"default"`, `"reduce-overhead"`, `"max-autotune"`. |
| `compile_dynamic` | Allow dynamic shapes in the compiled graph. |
| `compile_backend` | Optional `backend` argument forwarded to `torch.compile`. |
| `compile_fullgraph` | Request full-graph compilation (errors on fallbacks instead of silently re-compiling). |
| `store_last_wavefield` | Keep the final wavefield tensors on the solver for inspection / debugging. |

## `CUDAOptions`

```python
@dataclass
class CUDAOptions:
    memory: MemoryOptions | None = None
```

Use with `impl="c"`:

```python
PropTorch(..., backend="torch", impl="c",
          cuda_options=CUDAOptions(memory=MemoryOptions(...)))
```

Fields:

- `memory`: a `MemoryOptions` block, described below.

## `MemoryOptions`

```python
@dataclass
class MemoryOptions:
    strategy: Literal["boundary", "ckpt"] | None = None
    boundary: BoundaryOptions | None = None
    ckpt: CkptOptions | None = None
```

Pick at most one C-side memory-saving strategy.

Validation rules (enforced in `__post_init__`):

- `strategy="boundary"` ⇒ `boundary=BoundaryOptions(...)` must be provided;
  `ckpt` must stay `None`.
- `strategy="ckpt"` ⇒ `ckpt=CkptOptions(...)` must be provided;
  `boundary` must stay `None`.
- `strategy=None` ⇒ both `boundary` and `ckpt` must stay `None`.

Common patterns:

```python
# Boundary saving on GPU (no host transfer)
CUDAOptions(memory=MemoryOptions(
    strategy="boundary",
    boundary=BoundaryOptions(storage="gpu"),
))

# Boundary saving staged on pinned host memory
CUDAOptions(memory=MemoryOptions(
    strategy="boundary",
    boundary=BoundaryOptions(storage="cpu", pinned_memory=True),
))

# Boundary saving backed by disk, with asynchronous prefetch
CUDAOptions(memory=MemoryOptions(
    strategy="boundary",
    boundary=BoundaryOptions(
        storage="disk",
        disk_dir="/scratch/sweep-boundary",
        disk_async_read=True,
    ),
))

# Chunk-mode checkpointing
CUDAOptions(memory=MemoryOptions(
    strategy="ckpt",
    ckpt=CkptOptions(mode="chunk", chunks=100),
))

# Recursive-mode checkpointing with CPU offload
CUDAOptions(memory=MemoryOptions(
    strategy="ckpt",
    ckpt=CkptOptions(mode="recursive", count=8, storage="cpu"),
))
```

## `BoundaryOptions`

```python
@dataclass
class BoundaryOptions:
    storage: Literal["gpu", "cpu", "disk"] = "gpu"
    transfer_interval: int | None = None
    pinned_memory: bool | None = None
    disk_dir: str | None = None
    ring_buffers: int | None = None
    disk_async_read: bool = False
```

Controls boundary saving in the C path. Fields:

| Field | Applies when | Meaning |
| --- | --- | --- |
| `storage` | always | `"gpu"` keeps boundary buffers on device; `"cpu"` stages them in host memory; `"disk"` writes them to local storage. |
| `transfer_interval` | `cpu` / `disk` | How many time steps between host (or disk) transfers. `None` falls back to the default per storage tier. |
| `pinned_memory` | `cpu` | Use pinned host pages for faster H2D / D2H. |
| `disk_dir` | `disk` | Directory used as a staging area. `None` uses the runtime default. |
| `ring_buffers` | `cpu` / `disk` | Number of staging buffers in the boundary ring. |
| `disk_async_read` | `disk` | Enable asynchronous disk readback during backward. |

Validation rules (enforced in `__post_init__`):

- `storage` must be `"gpu"`, `"cpu"`, or `"disk"`.
- `transfer_interval ≥ 1` when set.
- `ring_buffers ≥ 1` when set.
- When `storage="gpu"`: `transfer_interval` must stay `None`/`1`,
  `ring_buffers` must stay `None`/`1`, `pinned_memory` must be falsy, and
  `disk_async_read` must be `False`.
- `pinned_memory` is only valid with `storage="cpu"`.
- `disk_async_read` is only valid with `storage="disk"`.

## `CkptOptions`

```python
@dataclass
class CkptOptions:
    mode: Literal["chunk", "recursive"] = "chunk"
    chunks: int = 100
    count: int = 0
    storage: Literal["gpu", "cpu"] = "gpu"
    pinned_memory: bool | None = None
```

Controls activation checkpointing in the C path. Fields:

| Field | Applies when | Meaning |
| --- | --- | --- |
| `mode` | always | `"chunk"` runs a periodic chunked replay; `"recursive"` runs a fixed-budget recursive replay. |
| `chunks` | `mode="chunk"` | Number of chunks per replay. Must be `≥ 1`. |
| `count` | `mode="recursive"` | Checkpoint budget. Must be `≥ 1`. |
| `storage` | always | Where saved checkpoints live: `"gpu"` keeps them on device; `"cpu"` offloads to host memory. |
| `pinned_memory` | `storage="cpu"` | Use pinned host pages for the checkpoint pool. |

Validation rules:

- `storage` must be `"gpu"` or `"cpu"`.
- `pinned_memory` is only valid with `storage="cpu"`.
- `mode="chunk"` ⇒ `chunks ≥ 1` and `count` must remain `0`.
- `mode="recursive"` ⇒ `count ≥ 1` and `chunks` must stay at its default.

## `backend_options`

`backend_options` is a generic catch-all slot. Either `EagerOptions` or
`CUDAOptions` can be passed through it instead of through `eager_options` /
`cuda_options`:

```python
PropTorch(
    ...,
    backend="torch",
    impl="eager",
    backend_options=EagerOptions(use_compile=True),
)

PropTorch(
    ...,
    backend="torch",
    impl="c",
    backend_options=CUDAOptions(
        memory=MemoryOptions(
            strategy="boundary",
            boundary=BoundaryOptions(storage="cpu", pinned_memory=True),
        )
    ),
)
```

In application code, the explicit `eager_options=` / `cuda_options=` kwargs are
usually clearer because they make the `impl` choice obvious at the call site.
