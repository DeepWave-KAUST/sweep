# Propagator options

Dataclass-based option blocks that configure the `PropTorch` execution paths.
Pass them through `backend_options=`, `eager_options=`, `cuda_options=` on the
`PropTorch` constructor, or use the top-level kwargs that mirror the dataclass
fields; `PropJax` accepts none of these blocks (it takes `memory=` and
`scan_unroll=`).  The gradient-memory mode (`full` / `boundary` / `ckpt`) is
picked once via `memory=Full()` / `memory=BoundarySaving(...)` /
`memory=Ckpt(...)`, on `PropTorch` and `PropJax` alike, and resolved
identically for every backend by `resolve_memory_strategy`.  The older
`MemoryOptions(strategy=...)` and `boundary_saving_config={...}` spellings
still work and emit a `DeprecationWarning`.

## Memory strategies (every impl)

`memory=` takes exactly one of these; `BoundarySaving` and `Ckpt` carry the
fields of `BoundaryOptions` and `CkptOptions` below.  Not every backend
implements every field, and an unsupported one raises at construction:
eager boundary saving stores to `'gpu'` or `'cpu'` and has no `tail_steps`;
`PropJax` boundary saving keeps the ring on the device (`storage='gpu'`) and
has no `tail_steps`; eager and `PropJax` checkpointing are `mode='chunk'` only.

::: sweep.propagator.options.Full

::: sweep.propagator.options.BoundarySaving

::: sweep.propagator.options.Ckpt

::: sweep.propagator.options.resolve_memory_strategy

## Eager (`impl='eager'`)

::: sweep.propagator.options.EagerOptions

## CUDA core (`impl='c'`)

Pass the memory strategy as `memory=` on `PropTorch`; `CUDAOptions(memory=...)`
is equivalent on `impl='c'`.

::: sweep.propagator.options.CUDAOptions

## Boundary saving

::: sweep.propagator.options.BoundaryOptions

## Checkpointing

::: sweep.propagator.options.CkptOptions

## Deprecated

`MemoryOptions(strategy=..., boundary=..., ckpt=...)` is the old spelling of
`memory=`. It still works, is read exactly as before, and emits a
`DeprecationWarning`; use `Full()` / `BoundarySaving(...)` / `Ckpt(...)`.

::: sweep.propagator.options.MemoryOptions

## Defaults

::: sweep.propagator.options.PropagatorDefaults

::: sweep.propagator.options.EagerDefaults

::: sweep.propagator.options.CkptDefaults

::: sweep.propagator.options.BoundaryDefaults
