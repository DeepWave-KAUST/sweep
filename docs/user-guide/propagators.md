# Propagators

Propagators connect an equation object, grid configuration, acquisition geometry,
and model tensors into a callable solver.

## Main Propagator APIs

- `sweep.propagator.torch.PropTorch`
- `sweep.propagator.jax.PropJax`

## Recommended Entry Points

- For Torch-family workflows, prefer `PropTorch(...)`.
  - `backend="eager"` uses the Python/Torch implementation.
  - `backend="cuda"` dispatches to the compiled CUDA backend.
- `PropCUDA` remains available as the lower-level CUDA-specific class when you
  need to work directly with CUDA-only runtime behavior.
- Use `PropJax` for JAX-based propagation.

## Backend-Specific Options

- `EagerOptions`
  - groups compile-related Torch options such as `use_compile` and `compile_mode`
- `CUDAOptions`
  - groups CUDA-only runtime options
- `MemoryOptions`
  - selects one CUDA memory-saving strategy
- `BoundaryOptions`
  - controls CUDA boundary saving
- `CkptOptions`
  - controls CUDA checkpointing mode and tuning parameters

## Geometry Conventions

- `sources`: shape `(nshots, ndim)` or backend-specific batched variants
- `receivers`: shape `(nshots, nreceivers, ndim)`
- 2D coordinates use `(x, z)` in example scripts
- 3D coordinates use `(x, y, z)` in example scripts

## Memory-Saving Features

- PyTorch eager checkpointing
- `torch.compile` on the eager backend
- CUDA boundary saving
- CUDA checkpointing in `chunk` and `recursive` modes

See `examples/reducingmemory/` for runnable comparisons of these options.
