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

- `PropTorch(..., backend="torch", impl="eager")` for the pure PyTorch implementation
- `PropTorch(..., backend="torch", impl="c")` for compiled C++/CUDA extension kernels

## Runtime Shape Conventions

`PropTorch` and `PropJax` accept **three** input modes for
`(wavelet, sources, receivers)`. The mode is **auto-detected from the array
shapes** — there is no `source_encoding` keyword argument; pass the inputs in
the shape that matches your acquisition geometry.

| Mode | `wavelet` | `sources` | `receivers` | Meaning |
|------|-----------|-----------|-------------|---------|
| **A1** Shared wavelet | `(nt,)` | `(nshots, dim)` | `(nshots, nrec, dim)` | Naive multi-shot, all shots share one wavelet |
| **A2** Per-shot wavelet | `(nshots, nt)` | `(nshots, dim)` | `(nshots, nrec, dim)` | Naive multi-shot, each shot has its own wavelet |
| **B** Source encoding | `(nt,)` or `(nsrc, nt)` | `(1, nsrc, dim)` | `(1, nrec, dim)` | One super-shot with `nsrc` superposed point sources |

- `dim` is `2` in 2D and `3` in 3D.
- **Receivers are always 3-D** `(B, nrec, dim)`. If you previously shared a
  single receiver array across shots, pre-broadcast it:
  `receivers = np.broadcast_to(rec, (nshots, *rec.shape)).copy()` (or
  `[None, ...].repeat(nshots, axis=0)`).
- In mode **B**, the leading dim of `sources`/`receivers` is the trigger
  that distinguishes encoding from a naive multi-shot run.

### Dispatch rules (precise)

1. `sources.ndim == 2` → mode **A** (`nshots == sources.shape[0]`)
   - `wavelet.ndim == 1` → **A1**
   - `wavelet.shape == (nshots, nt)` → **A2**
2. `sources.ndim == 3` and `sources.shape[0] == 1` → mode **B**
   - `wavelet.shape == (nt,)` or `(nsrc, nt)` (with `nsrc == sources.shape[1]`)
3. Anything else raises `ValueError` with a message describing the contract.

### Migration from the old API

The `source_encoding=` keyword argument has been **removed**. To migrate:

- `solver(wavelet=(1, nsrc, nt), sources=(1, nsrc, dim), receivers=(1, nrec, dim), source_encoding=True)`
  → drop the kwarg and pass `wavelet` as 2-D `(nsrc, nt)`; the encoding mode is
  inferred from the leading-`1` sources/receivers.
- `solver(wavelet=(nt,), sources=(nshots, dim), receivers=(nrec, dim))` →
  pre-broadcast receivers to `(nshots, nrec, dim)`.

### Record output layout

Every backend (`impl="eager"` and `impl="c"`) returns the receiver record in
the **canonical** shape

```
(B, nt, nrec, nfield)
```

where `nfield` is the number of recorded components for the equation
(e.g. `1` for acoustic pressure, `2` for elastic vx/vz, `5` for the Zhao
DAS receivers). This matches the layout expected by `sweep_loss` so the
output of a solver can be fed straight into a misfit:

```python
syn = solver(wavelet, sources, receivers, models=models)
loss = sweep_loss.L2()(syn, observed)  # both are (B, nt, nrec, nfield)
```

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
        backend="torch",
        impl="eager",
        backend_options=None,
        eager_options=None,
        cuda_options=None,
        use_ckpt=True,
        ckpt_chunks=100,
        pml_type="spml",
    )
    ```

    Torch-family propagator facade. `backend="torch", impl="eager"` uses the
    Python/Torch implementation, while `backend="torch", impl="c"`
    dispatches to compiled C++/CUDA extension kernels.

    !!! info "Default memory strategy by impl"
        - `impl="eager"`, `impl="jax"` → chunked checkpointing
          (`use_ckpt=True`, `ckpt_chunks=100`).
        - `impl="c"` → **boundary saving with GPU storage**
          (`use_ckpt=False`, `boundary_saving_config={'enabled': True,
          'storage': 'gpu'}`). To opt back into chunked checkpointing
          on the C backend, pass
          `cuda_options={"memory": {"strategy": "ckpt"}}`. 2-D RTM
          silently falls back to full-wavefield mode regardless of the
          configured strategy.

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
