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
- `PropTorch(..., backend="torch", impl="c")` for the prebuilt CUDA core (`sweep/lib/cu<major>/libsweep_core.so`, driven through the pure-Python ctypes layer `sweep.backend.c`; CUDA GPU required, nothing compiles after `pip install`)

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

where `nfield = len(receiver_type)`, one per recorded field (with the default
receivers: `1` for acoustic pressure `h1`, `2` for elastic `vx`/`vz`, `2` for
the Zhao DAS strain rates `exx_t`/`ezz_t`). This matches the layout expected
by `sweep_loss` so the output of a solver can be fed straight into a misfit:

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
        device=None,    # None: the equation's device (dev= is a deprecated alias)
        backend=None,   # inherits equation.backend
        impl=None,      # 'auto': 'c' when the CUDA core is usable, the equation has
                        # bindings and the device is CUDA, else 'eager'
        backend_options=None,
        eager_options=None,
        cuda_options=None,
        memory=None,    # Full() / BoundarySaving(...) / Ckpt(...); None: impl default
        use_ckpt=None,  # legacy knob; None: impl default (see below)
        ckpt_chunks=100,
        pml_type=None,  # None: equation.default_pml_type ('cpmlr' / 'cpmls')
    )
    ```

    Torch-family propagator facade. `backend="torch", impl="eager"` uses the
    Python/Torch implementation, while `backend="torch", impl="c"`
    dispatches to the prebuilt CUDA core through the `sweep.backend.c` ctypes
    layer (CUDA tensors only).

    !!! info "Default memory strategy by impl"
        - `impl="eager"` (and `PropJax`) → chunked checkpointing, i.e.
          `memory=Ckpt(mode="chunk", chunks=100)`.
        - `impl="c"` → **boundary saving with GPU storage**, i.e.
          `memory=BoundarySaving(storage="gpu")`. To opt back into chunked
          checkpointing on the C backend, pass `memory=Ckpt()` to `PropTorch`
          (or `memory=Full()` for full wavefield storage).
        - Exception: `ViscoAcoustic` and `DASZhao3D` do not support boundary
          saving on `impl="c"`; they default to `Full()` there, and an explicit
          `BoundarySaving` raises `NotImplementedError`.

        The three strategies are the types `Full`, `BoundarySaving` and `Ckpt`
        from `sweep.propagator.options`; the older
        `MemoryOptions(strategy=...)` and `boundary_saving_config={...}`
        spellings still work and emit a `DeprecationWarning`.

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
        device=None,
        backend=None,   # 'jax' or None
        memory=None,    # Full() / BoundarySaving(...) / Ckpt(...); None: checkpointing
        use_ckpt=None,  # legacy knob; None: checkpointing
        ckpt_chunks=100,
        pml_type=None,  # None: equation.default_pml_type
        scan_unroll=1,
    )
    ```

    JAX propagator based on `jax.lax.scan`. The gradient-memory strategy is
    `memory=Full()`, `BoundarySaving(...)` (on-device ring only) or
    `Ckpt(...)` (chunk mode); with none given it is chunk-style
    rematerialization.

    See [PropJax](prop_jax.md) for parameter meanings.

## Parameter Pages

The following pages use a class-reference style layout:

- [Propagator Options](options.md)
- [PropTorch](prop_torch.md)
- [PropJax](prop_jax.md)
