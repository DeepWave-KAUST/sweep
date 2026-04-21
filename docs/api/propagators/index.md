# Propagators

This section documents the solver classes in `sweep.propagator`.

## Overview

All propagators combine the same core pieces:

- an `equation`
- grid and boundary configuration
- source and receiver field selection
- runtime inputs such as `wavelet`, `sources`, `receivers`, and `models`

The main user-facing solver classes are:

- `PropTorch`
- `PropJax`

For most Torch-based workflows, `PropTorch` is now the main user-facing entry
point. Use:

- `PropTorch(..., backend="eager")` for the pure PyTorch implementation
- `PropTorch(..., backend="cuda")` for the compiled CUDA implementation

`PropCUDA` remains available as the lower-level CUDA-specific implementation,
but the Torch-side API is centered on `PropTorch(..., backend="cuda")`.

## API Tabs

=== "PropTorch"

    ```python
    class PropTorch(
        equation,
        shape,
        source_type=[],
        receiver_type=[],
        abcn=50,
        free_surface=False,
        dh=10.0,
        dt=0.002,
        dev=None,
        backend="eager",
        backend_options=None,
        eager_options=None,
        cuda_options=None,
        use_ckpt=True,
        ckpt_chunks=100,
        pml_type="spml",
    )
    ```

    Torch-family propagator facade. `backend="eager"` uses the Python/Torch
    implementation, while `backend="cuda"` dispatches to the compiled CUDA
    backend.

    See [PropTorch](prop_torch.md) for parameter meanings.

=== "PropJax"

    ```python
    class PropJax(
        equation,
        shape,
        source_type=[],
        receiver_type=[],
        abcn=50,
        free_surface=False,
        dh=10.0,
        dt=0.002,
        dev=None,
        use_ckpt=True,
        ckpt_chunks=100,
        pml_type="spml",
    )
    ```

    JAX propagator based on `jax.lax.scan` with chunk-style rematerialization.

    See [PropJax](prop_jax.md) for parameter meanings.

## Parameter Pages

The following pages use a class-reference style layout:

- [Propagator Options](options.md)
- [PropTorch](prop_torch.md)
- [PropJax](prop_jax.md)

For lower-level CUDA-specific runtime details, see [PropCUDA](prop_cuda.md).
