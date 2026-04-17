# Propagators

This section documents the solver classes in `sweep.propagator`.

## Overview

All propagators combine the same core pieces:

- an `equation`
- grid and boundary configuration
- source and receiver field selection
- runtime inputs such as `wavelet`, `sources`, `receivers`, and `models`

The main solver classes are:

- `PropTorch`
- `PropCUDA`
- `PropJax`

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
        use_ckpt=True,
        ckpt_chunks=100,
        ckpt_mode="chunk",
        pml_type="spml",
        use_compile=False,
        compile_backend=None,
        compile_mode="default",
        compile_dynamic=False,
        compile_fullgraph=False,
    )
    ```

    Pure PyTorch propagator with optional `torch.compile` on the single-step
    update.

    See [PropTorch](prop_torch.md) for parameter meanings.

=== "PropCUDA"

    ```python
    class PropCUDA(
        equation,
        shape,
        source_type=[],
        receiver_type=[],
        abcn=50,
        free_surface=False,
        dh=10.0,
        dt=0.002,
        dev=None,
        use_ckpt=False,
        ckpt_chunks=100,
        ckpt_mode="chunk",
        ckpt_num=0,
        pml_type="spml",
        nt=-1,
        B=1,
        allow_growth=True,
        boundary_saving_config=None,
    )
    ```

    Compiled CUDA propagator with runtime buffer reuse, checkpointing, and RTM
    support.

    See [PropCUDA](prop_cuda.md) for parameter meanings.

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

The following pages use a class-reference style layout and focus on what each
parameter means for one specific propagator:

- [PropTorch](prop_torch.md)
- [PropCUDA](prop_cuda.md)
- [PropJax](prop_jax.md)
