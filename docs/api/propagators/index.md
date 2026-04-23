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

## Runtime Shape Conventions

Across `PropTorch`, `PropJax`, and `PropCUDA`, runtime inputs usually follow
one of these patterns:

- Single-source batched shots:
  - `wavelet`: `(nt,)` or `(B, nt)`
  - `sources`: `(B, dim)`
  - `receivers`: `(B, nrec, dim)`
- Multi-source batched shots or blended shots:
  - `wavelet`: `(B, nsrc, nt)`
  - `sources`: `(B, nsrc, dim)`
  - `receivers`: `(B, nrec, dim)`
- Source-encoding super-shot:
  - `wavelet`: `(1, nsrc, nt)`
  - `sources`: `(1, nsrc, dim)`
  - `receivers`: `(1, nrec, dim)`

Here:

- `B` is the runtime batch size
- `nsrc` is the number of sources inside one batch element
- `nrec` is the number of receivers
- `dim` is `2` in 2D and `3` in 3D

When the inputs use the super-shot layout
`(1, nsrc, nt) / (1, nsrc, dim) / (1, nrec, dim)`, all three propagators now
auto-detect this pattern and treat it as `source_encoding=True`.

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
