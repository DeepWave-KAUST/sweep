# Propagators

Propagators connect an equation object, grid configuration, acquisition geometry,
and model tensors into a callable solver.

## Main Propagator APIs

- `sweep.propagator.torch.PropTorch`
- `sweep.propagator.jax.PropJax`

## Recommended Entry Points

- For Torch-family workflows, prefer `PropTorch(...)`. `backend="eager"` uses the Python/Torch implementation, while `backend="cuda"` dispatches to the compiled CUDA backend.
- `PropCUDA` remains available as the lower-level CUDA-specific class when you
  need to work directly with CUDA-only runtime behavior.
- Use `PropJax` for JAX-based propagation.

## Backend-Specific Options

- `EagerOptions`: groups compile-related Torch options such as `use_compile` and `compile_mode`
- `CUDAOptions`: groups CUDA-only runtime options
- `MemoryOptions`: selects one CUDA memory-saving strategy
- `BoundaryOptions`: controls CUDA boundary saving
- `CkptOptions`: controls CUDA checkpointing mode and tuning parameters

## Geometry Conventions

- `sources`: shape `(nshots, ndim)` or backend-specific batched variants
- `receivers`: shape `(nshots, nreceivers, ndim)`
- 2D coordinates use `(x, z)` in example scripts
- 3D coordinates use `(x, y, z)` in example scripts

## Memory-Saving Features

- PyTorch eager checkpointing
- `torch.compile` on the eager backend
- CUDA boundary saving with GPU, CPU, pinned CPU, and disk-backed storage
- CUDA boundary saving disk prefetch/readback, including asynchronous disk reads
- CUDA checkpointing in `chunk` and `recursive` modes where the equation supports it

CUDA boundary saving is configured with `CUDAOptions(memory=MemoryOptions(...))`.
The most commonly tuned boundary options are:

- `BoundaryOptions.storage`: `"gpu"`, `"cpu"`, or `"disk"`
- `BoundaryOptions.transfer_interval`: number of time steps between boundary transfers
- `BoundaryOptions.pinned_memory`: use pinned host memory for CPU boundary storage
- `BoundaryOptions.disk_dir`: directory used by disk-backed boundary storage
- `BoundaryOptions.ring_buffers`: number of disk staging buffers
- `BoundaryOptions.disk_async_read`: enable asynchronous disk readback during backward

See `examples/reducingmemory/` for runnable comparisons of these options.

## Consistency Testing

The CUDA memory modes are covered by `test/solver_gradient_mode_suite.py`. The
suite compares eager gradients against CUDA full-wavefield, boundary-saving,
and checkpoint modes across interior, finite-difference-edge, and free-surface
source placements. It also saves per-mode gradient figures under
`test/test_outputs/solver_gradient_mode_suite/`.
