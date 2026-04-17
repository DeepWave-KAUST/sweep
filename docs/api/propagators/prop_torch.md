# PropTorch

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

Implementation:

- `src/sweep/propagator/torch.py`

Pure PyTorch propagator with optional `torch.compile` on the single-step
equation update.

!!! note

    `PropTorch` is the easiest backend to inspect and modify, because the time
    loop and source/receiver logic are implemented directly in Python.

## Parameters

- `equation` (equation instance): Equation object to run, such as `Acoustic`,
  `Acoustic3D`, `Elastic`, or `Elastic3D`.
- `shape` (`tuple[int, ...]`): Physical model shape before absorbing
  boundaries are added. Use `(nz, nx)` in 2D and `(nz, ny, nx)` in 3D.
- `source_type` (`list[str]`, optional): Wavefield names that receive source
  injection. These names must exist in `equation.wavefields`. `PropTorch` does
  not auto-fill defaults, so they should usually be set explicitly.
- `receiver_type` (`list[str]`, optional): Wavefield names sampled at receiver
  locations. These names must also exist in `equation.wavefields`.
- `abcn` (`int`, optional): Absorbing boundary width.
- `free_surface` (`bool`, optional): Whether the top boundary is treated as a
  free surface. This changes how source and receiver coordinates are offset
  internally.
- `dh` (`float`, optional): Scalar grid spacing. Tuple-valued spacing is not
  supported in `PropTorch`.
- `dt` (`float`, optional): Time step in seconds.
- `dev` (device, optional): Execution device, typically a `torch.device`.
- `use_ckpt` (`bool`, optional): Enables chunk checkpointing to reduce memory
  use during backpropagation.
- `ckpt_chunks` (`int`, optional): Number of time steps per checkpoint chunk
  when `use_ckpt=True`.
- `ckpt_mode` (`str`, optional): Checkpointing mode. For `PropTorch`, only
  `"chunk"` is currently supported.
- `pml_type` (`str`, optional): Absorbing boundary implementation passed into
  the equation setup.
- `use_compile` (`bool`, optional): If `True`, wraps the single-step equation
  update with `torch.compile`. This does not compile the full forward loop.
- `compile_backend` (`str or callable`, optional): Optional backend argument
  passed to `torch.compile`.
- `compile_mode` (`str`, optional): Compile mode passed to `torch.compile`,
  such as `"default"`.
- `compile_dynamic` (`bool`, optional): Whether to allow dynamic behavior in
  the compiled step graph.
- `compile_fullgraph` (`bool`, optional): Whether to request full-graph
  compilation for the single-step function.

## Forward Parameters

```python
forward(
    wavelet,
    sources,
    receivers,
    models=None,
    source_encoding=False,
    adj=False,
    return_wavefield=False,
    **kwargs,
)
```

- `wavelet` (array-like): Source time function. It is converted to
  `torch.float32` on `dev`. Typical shape: `(nt,)`.
- `sources` (array-like): Source coordinates. They are converted to integer
  tensors on `dev`. Typical shape: `(nshots, ndim)`.
- `receivers` (array-like): Receiver coordinates. They are converted to integer
  tensors on `dev`. Typical shape: `(nshots, nreceivers, ndim)`.
- `models` (`list[torch.Tensor]`, optional): List of model tensors, provided in
  the exact order required by `equation.models`.
- `source_encoding` (`bool`, optional): If `True`, collapses shots into a
  single encoded batch during propagation.
- `adj` (`bool`, optional): Switches source time indexing for adjoint-style
  forward usage.
- `return_wavefield` (`bool`, optional): If `True`, returns snapshots in
  addition to recorded data. This option is not supported together with
  checkpointing in the current implementation.

## Return Value

- default: `record`
- if `return_wavefield=True`: `(record, snapshots)`
