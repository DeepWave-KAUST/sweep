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
    backend="eager",
    backend_options=None,
    eager_options=None,
    cuda_options=None,
    use_ckpt=True,
    ckpt_chunks=100,
    pml_type="spml",
)
```

Implementation:

- `src/sweep/propagator/torch.py`

Torch-family propagator facade.

- `backend="eager"` uses the Python/Torch implementation
- `backend="cuda"` dispatches to the compiled CUDA implementation

In other words, the CUDA path is part of the `PropTorch` API surface now. The
lower-level CUDA implementation details sit underneath
`PropTorch(..., backend="cuda")`.

!!! note

    `PropTorch` is the main Torch-facing solver entry point. If you choose
    `backend="eager"`, the time loop and source/receiver logic live directly in
    Python. If you choose `backend="cuda"`, `PropTorch` forwards to the
    compiled CUDA backend while keeping the same top-level constructor style.

## Parameters

- `equation` (equation instance): Equation object to run, such as `Acoustic`,
  `Acoustic3D`, `Elastic`, or `Elastic3D`.
- `shape` (`tuple[int, ...]`): Physical model shape before absorbing
  boundaries are added. Use `(nz, nx)` in 2D and `(nz, ny, nx)` in 3D.
- `source_type` (`list[str]`, optional): Wavefield names that receive source
  injection. These names are resolved against the equation field metadata, so
  aliases such as `pressure` or `stress_xx` may also be accepted depending on
  the equation. If omitted, `PropTorch` uses `equation.default_source_fields`.
- `receiver_type` (`list[str]`, optional): Wavefield names sampled at receiver
  locations. These are also resolved through the equation field metadata and
  default to `equation.default_receiver_fields` when omitted.
- `abcn` (`int`, optional): Absorbing boundary width.
- `free_surface` (`bool`, optional): Whether the top boundary is treated as a
  free surface. This changes how source and receiver coordinates are offset
  internally.
- `dh` (`float or tuple[float, ...]`, optional): Grid spacing. Scalar input is
  broadcast to every spatial axis. Directional spacing is also supported:
  `(dz, dx)` in 2D and `(dz, dy, dx)` in 3D.
- `dt` (`float`, optional): Time step in seconds.
- `dev` (device, optional): Execution device, typically a `torch.device`.
- `backend` (`"eager"` or `"cuda"`, optional): Selects which Torch-family
  implementation is used.
- `backend_options` (dict or dataclass, optional): Backend-specific options
  block merged after top-level kwargs. See [backend_options](options.md#backend_options).
- `eager_options` ([`EagerOptions`](options.md#eageroptions) or dict, optional):
  eager-only options such as `use_compile`, `compile_mode`, and
  `store_last_wavefield`.
- `cuda_options` ([`CUDAOptions`](options.md#cudaoptions) or dict, optional):
  CUDA-only options. The main CUDA-specific block is
  [`CUDAOptions(memory=...)`](options.md#cudaoptions).
- `use_ckpt` (`bool`, optional): Enables chunk checkpointing on the eager
  backend. For CUDA usage, prefer
  [`CUDAOptions(memory=MemoryOptions(strategy="ckpt", ...))`](options.md#memoryoptions)
  instead of relying on this top-level flag.
- `ckpt_chunks` (`int`, optional): Number of time steps per checkpoint chunk
  when `use_ckpt=True` on the eager backend. For CUDA usage, prefer
  [`CkptOptions(chunks=...)`](options.md#ckptoptions).
- `pml_type` (`str`, optional): Absorbing boundary implementation passed into
  the equation setup.

See the full [Propagator Options](options.md) reference for field-by-field
descriptions and validation rules.

## Field Selection Helpers

To discover valid `source_type` and `receiver_type` values, prefer the equation
helpers instead of reading `equation.wavefields` directly:

- `equation.available_fields()`
- `equation.available_fields(role="source")`
- `equation.available_fields(role="receiver")`
- `equation.describe_field(name)`

These helpers return the user-facing physical fields and filter out internal
boundary variables by default.

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
