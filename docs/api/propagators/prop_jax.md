# PropJax

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
    backend=None,
    memory=None,
    use_ckpt=None,
    ckpt_chunks=100,
    pml_type=None,
    scan_unroll=1,
)
```

Implementation:

- `src/sweep/propagator/jax.py`

JAX propagator built around `jax.lax.scan`. The gradient-memory strategy is
chosen with `memory=`; with none given it is chunk-style rematerialization.

!!! note

    `PropJax` shares the same solver concepts as the PyTorch backend, but its
    runtime behavior is shaped by JAX transforms rather than Python-side loops.

## Parameters

- `equation` (equation instance): The equation instance to be stepped in JAX.
- `shape` (`tuple[int, ...]`): Physical model shape before absorbing
  boundaries are added. Use `(nz, nx)` in 2D and `(nz, ny, nx)` in 3D.
- `source_type` (`list[str]`, optional): Wavefield names used for source
  injection. Defaults to `equation.default_source_fields`; names (or their
  `FieldSpec` aliases) must be source-capable fields of the equation.
- `receiver_type` (`list[str]`, optional): Wavefield names sampled at receiver
  locations. Defaults to `equation.default_receiver_fields`; names or aliases
  must be receiver-capable fields.
- `abcn` (`int`, optional): Absorbing boundary width.
- `free_surface` (`bool`, optional): Whether the top boundary is treated as a
  free surface. This affects internal coordinate offsets before source
  injection and receiver sampling.
- `dh` (`float` or sequence, optional): Grid spacing: a scalar, or one value
  per axis, `(dz, dx)` in 2D and `(dz, dy, dx)` in 3D.
- `dt` (`float`, optional): Time step in seconds.
- `device` (device or context, optional): Stored device/context argument
  (`dev=` is a deprecated alias). Actual JAX execution placement is still
  driven by JAX arrays and transforms.
- `backend` (`str`, optional): `'jax'` or `None`; accepted for symmetry with
  `PropTorch`.
- `memory` (optional): The gradient-memory strategy, one of `Full()`,
  `BoundarySaving(...)` or `Ckpt(...)` from `sweep.propagator.options`.
  `Full()` differentiates through the scan tape; `Ckpt(chunks=...)` is chunked
  `jax.checkpoint` rematerialization (`mode='chunk'` only);
  `BoundarySaving(...)` reconstructs the forward wavefield in reverse time from
  saved boundaries, with the ring kept on the device (`storage='gpu'`, no
  `tail_steps`). `None` (default) means checkpointing.
- `use_ckpt` (`bool | None`, optional): Legacy switch: `True` requests chunked
  rematerialization, `False` full storage. `None` (default) leaves the choice
  to `memory=`, else checkpointing.
- `ckpt_chunks` (`int`, optional): Chunk size, in time steps, when
  checkpointing.
- `pml_type` (`str`, optional): PML formulation. `None` (default) uses
  `equation.default_pml_type` (e.g. `'cpmlr'` for `Acoustic`, `'cpmls'` for
  `Elastic`); `'spml'` is supported by `Acoustic1st` only.
- `scan_unroll` (`int`, optional): `lax.scan` unroll factor for the time loop.
  Small, launch-bound grids can gain from 2–4 (gradients bit-identical);
  large, bandwidth-bound grids lose a few percent, so the default is `1`.

## Forward Parameters

```python
forward(
    wavelet,
    sources,
    receivers,
    models=None,
    return_wavefield=False,
    adj=False,
    **kwargs,
)
```

- `wavelet`, `sources`, `receivers`: see
  [Runtime Shape Conventions](index.md#runtime-shape-conventions). Modes
  **A1**, **A2**, and **B** (source encoding) are auto-detected from the
  input shapes; the legacy `source_encoding=` kwarg has been removed.
- `models` (list of arrays, optional): List of model arrays in the exact order
  required by `equation.models`.
- `return_wavefield` (`bool`, optional): If `True`, returns an auxiliary
  wavefield output in addition to the recorded data.
- `adj` (`bool`, optional): Adjoint-style forward switch.

## Return Value

- default: `record`
- if `return_wavefield=True`: `(record, snapshots)`
