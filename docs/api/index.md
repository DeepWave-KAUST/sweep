# API Reference

This section documents SWEEP's runtime-facing APIs based on how they are
actually used in the codebase.

## Covered Modules

- `sweep.equations`
- `sweep.propagator`

## Current Pages

- [Equations](equations/index.md)
- [Propagators](propagators/index.md)

## Documentation Approach

The API Reference is written by checking the implementation directly rather than
copying constructor comments blindly. For each class, the goal is to capture:

- the actual constructor arguments
- backend-specific differences
- input and output conventions
- behaviors that are easy to misunderstand

For Torch-family propagation, read `PropTorch` as the primary API surface.
The compiled CUDA runtime is documented separately as `PropCUDA`, but it is now
typically reached through `PropTorch(..., backend="cuda")`.
