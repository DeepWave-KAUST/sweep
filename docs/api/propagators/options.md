# Propagator options

Dataclass-based option blocks that configure the various `PropTorch` /
`PropJax` execution paths. Pass them through `backend_options=`,
`eager_options=`, `cuda_options=` on the propagator constructor, or use the
top-level kwargs that mirror the dataclass fields.  The gradient-memory mode
(`full` / `boundary` / `ckpt`) is picked once via
`memory=MemoryOptions(strategy=...)` and resolved identically for every
backend by `resolve_memory_strategy`.

## Eager (`impl='eager'`)

::: sweep.propagator.options.EagerOptions

## Compiled CUDA / C++ (`impl='c'`)

::: sweep.propagator.options.CUDAOptions

::: sweep.propagator.options.MemoryOptions

::: sweep.propagator.options.resolve_memory_strategy

::: sweep.propagator.options.BoundaryOptions

## Checkpointing

::: sweep.propagator.options.CkptOptions

## Defaults

::: sweep.propagator.options.PropagatorDefaults

::: sweep.propagator.options.EagerDefaults

::: sweep.propagator.options.CkptDefaults

::: sweep.propagator.options.BoundaryDefaults
