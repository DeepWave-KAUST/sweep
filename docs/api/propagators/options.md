# Propagator Options

This page documents the dataclass-based options used by the Torch-family
propagator interface.

Implementation:

- `src/sweep/propagator/options.py`

These option blocks are primarily used through:

- `PropTorch(..., eager_options=...)`
- `PropTorch(..., cuda_options=...)`
- `PropTorch(..., backend_options=...)`

Checkpoint note:

- eager checkpointing still uses the top-level `use_ckpt` and `ckpt_chunks`
  arguments on `PropTorch`
- c checkpointing should be configured through
  `CUDAOptions(memory=MemoryOptions(strategy="ckpt", ckpt=CkptOptions(...)))`

## EagerOptions

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

Use this block with:

```python
PropTorch(..., backend="torch", impl="eager", eager_options=EagerOptions(...))
```

This block controls eager compile/runtime behavior. It does not currently
replace the top-level eager checkpoint arguments `use_ckpt` and `ckpt_chunks`.

Fields:

- `use_compile`: enables `torch.compile` on the eager implementation
- `compile_mode`: compile mode passed into `torch.compile`
- `compile_dynamic`: whether dynamic-shape behavior is allowed in the compiled graph
- `compile_backend`: optional backend argument passed into `torch.compile`
- `compile_fullgraph`: whether to request full-graph compilation
- `store_last_wavefield`: whether to keep the last wavefield state for inspection/debugging

## CUDAOptions

```python
@dataclass
class CUDAOptions:
    memory: MemoryOptions | None = None
```

Use this block with:

```python
PropTorch(..., backend="torch", impl="c", cuda_options=CUDAOptions(...))
```

Fields:

- `memory`:  memory-policy block, described in [MemoryOptions](#memoryoptions)

## MemoryOptions

```python
@dataclass
class MemoryOptions:
    strategy: Literal["boundary", "ckpt"] | None = None
  boundary: BoundaryOptions | None = None
    ckpt: CkptOptions | None = None
```

This block chooses one  memory-saving strategy.

Rules:

- if `strategy="boundary"`, you must provide `boundary=BoundaryOptions(...)`
- if `strategy="ckpt"`, you must provide `ckpt=CkptOptions(...)`
- `boundary` and `ckpt` cannot be used at the same time
- if `strategy=None`, neither `boundary` nor `ckpt` may be provided

Typical usage:

```python
CUDAOptions(
    memory=MemoryOptions(
        strategy="boundary",
        boundary=BoundaryOptions(storage="gpu"),
    )
)
```

or:

```python
CUDAOptions(
    memory=MemoryOptions(
        strategy="ckpt",
        ckpt=CkptOptions(mode="chunk", chunks=100),
    )
)
```

## BoundaryOptions

```python
@dataclass
class BoundaryOptions:
    storage: Literal["gpu", "cpu"] = "gpu"
    transfer_interval: int = 1
    pinned_memory: bool = False
```

This block controls c boundary saving.

Fields:

- `storage`: `"gpu"` keeps saved boundaries on device, and `"cpu"` stages saved boundaries on host memory
- `transfer_interval`: only meaningful when `storage="cpu"`; controls how often boundary values are transferred or staged
- `pinned_memory`: only meaningful when `storage="cpu"`; enables pinned host memory for transfers

Validation rules:

- `transfer_interval >= 1`
- if `storage="gpu"`, then `transfer_interval` must stay at `1` and `pinned_memory` must stay `False`

## CkptOptions

```python
@dataclass
class CkptOptions:
    mode: Literal["chunk", "recursive"] = "chunk"
    chunks: int = 100
    count: int = 0
```

This block controls c checkpointing.

Fields:

- `mode`: `"chunk"` means periodic chunk-based replay, and `"recursive"` means fixed checkpoint-budget replay
- `chunks`: used only when `mode="chunk"`
- `count`: used only when `mode="recursive"`

Validation rules:

- for `mode="chunk"`, `chunks >= 1` and `count` must remain `0`
- for `mode="recursive"`, `count >= 1` and `chunks` must remain at its default chunk-mode value

## backend_options

`backend_options` is the generic merged options block accepted by `PropTorch`.

It can be used when you want to pass backend-specific fields without choosing
between `eager_options` and `cuda_options` in the call site.

Example:

```python
PropTorch(
    ...,
    backend="torch",
    impl="eager",
    backend_options=EagerOptions(use_compile=True),
)
```

or:

```python
PropTorch(
    ...,
    backend="torch",
    impl="c",
    backend_options=CUDAOptions(
        memory=MemoryOptions(
            strategy="boundary",
            boundary=BoundaryOptions(storage="gpu"),
        )
    ),
)
```

In most user-facing code, `eager_options` and `cuda_options` are clearer than
`backend_options`, because they make the implementation split explicit.
