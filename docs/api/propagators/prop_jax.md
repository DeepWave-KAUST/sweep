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
    dev=None,
    use_ckpt=True,
    ckpt_chunks=100,
    pml_type="spml",
)
```

Implementation:

- `src/sweep/propagator/jax.py`

JAX propagator built around `jax.lax.scan` and chunk-style rematerialization.

!!! note

    `PropJax` shares the same solver concepts as the PyTorch backend, but its
    runtime behavior is shaped by JAX transforms rather than Python-side loops.

## Parameters

- `equation` (equation instance): The equation instance to be stepped in JAX.
- `shape` (`tuple[int, ...]`): Physical model shape before absorbing
  boundaries are added. Use `(nz, nx)` in 2D and `(nz, ny, nx)` in 3D.
- `source_type` (`list[str]`, optional): Wavefield names used for source
  injection. These names must exist in `equation.wavefields`. `PropJax` does
  not auto-fill defaults.
- `receiver_type` (`list[str]`, optional): Wavefield names sampled at receiver
  locations. These must also match `equation.wavefields`.
- `abcn` (`int`, optional): Absorbing boundary width.
- `free_surface` (`bool`, optional): Whether the top boundary is treated as a
  free surface. This affects internal coordinate offsets before source
  injection and receiver sampling.
- `dh` (`float`, optional): Scalar grid spacing.
- `dt` (`float`, optional): Time step in seconds.
- `dev` (device or context, optional): Stored device/context argument. Actual
  JAX execution placement is still driven by JAX arrays and transforms.
- `use_ckpt` (`bool`, optional): Enables chunk-based rematerialization in the
  scanned time loop.
- `ckpt_chunks` (`int`, optional): Chunk size used when `use_ckpt=True`.
- `pml_type` (`str`, optional): PML implementation passed into the equation
  setup.

## Forward Parameters

```python
forward(
    wavelet,
    sources,
    receivers,
    models=None,
    source_encoding=False,
    return_wavefield=False,
    adj=False,
    **kwargs,
)
```

- `wavelet` (array-like): Source time function. Typical shape: `(nt,)`.
- `sources` (array-like): Source coordinates. Typical shape: `(nshots, ndim)`.
- `receivers` (array-like): Receiver coordinates. Typical shape:
  `(nshots, nreceivers, ndim)`.
- `models` (list of arrays, optional): List of model arrays in the exact order
  required by `equation.models`.
- `source_encoding` (`bool`, optional): If `True`, collapses shots into a
  single encoded batch.
- `return_wavefield` (`bool`, optional): If `True`, returns an auxiliary
  wavefield output in addition to the recorded data.
- `adj` (`bool`, optional): Adjoint-style forward switch.

## Return Value

- default: `record`
- if `return_wavefield=True`: `(record, snapshots)`
