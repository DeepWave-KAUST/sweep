# Propagators

Propagators connect an equation object, grid configuration, acquisition geometry,
and model tensors into a callable solver.

## Main Propagator APIs

- `sweep.propagator.torch.PropTorch`
- `sweep.propagator.jax.PropJax`

## Recommended Entry Points

- For Torch-family workflows, prefer `PropTorch(...)`. `backend="torch", impl="eager"` uses the Python/Torch implementation, while `backend="torch", impl="c"` dispatches to compiled C++/CUDA extension kernels.
- Use `PropJax` for JAX-based propagation.

## Implementation-Specific Options

- `EagerOptions`: groups compile-related Torch options such as `use_compile` and `compile_mode`
- `CUDAOptions`: groups c runtime options. The class name is retained for compatibility.
- `MemoryOptions`: selects one c memory-saving strategy
- `BoundaryOptions`: controls c boundary saving
- `CkptOptions`: controls c checkpointing mode and tuning parameters

## Geometry Conventions

- `sources`: shape `(nshots, ndim)` or backend-specific batched variants
- `receivers`: shape `(nshots, nreceivers, ndim)`
- 2D coordinates use `(x, z)` in example scripts
- 3D coordinates use `(x, y, z)` in example scripts

## Memory-Saving Features

- PyTorch eager checkpointing
- `torch.compile` on the eager implementation
- c boundary saving with GPU, CPU, pinned CPU, and disk-backed storage
- c boundary saving disk prefetch/readback, including asynchronous disk reads
- c checkpointing in `chunk` and `recursive` modes where the equation supports it

C boundary saving is configured with `CUDAOptions(memory=MemoryOptions(...))`.
The most commonly tuned boundary options are:

- `BoundaryOptions.storage`: `"gpu"`, `"cpu"`, or `"disk"`
- `BoundaryOptions.transfer_interval`: number of time steps between boundary transfers
- `BoundaryOptions.pinned_memory`: use pinned host memory for CPU boundary storage
- `BoundaryOptions.disk_dir`: directory used by disk-backed boundary storage
- `BoundaryOptions.ring_buffers`: number of disk staging buffers
- `BoundaryOptions.disk_async_read`: enable asynchronous disk readback during backward

See `examples/reducingmemory/` for runnable comparisons of these options.

## Consistency Testing

The c memory modes are covered by `test/solver_gradient_mode_suite.py`.
The suite compares eager gradients against c full-wavefield,
boundary-saving, and checkpoint modes across interior, finite-difference-edge,
and free-surface source placements. It also saves per-mode gradient figures under
`test/test_outputs/solver_gradient_mode_suite/`.
