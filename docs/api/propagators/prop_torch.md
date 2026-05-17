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

Implementation:

- `src/sweep/propagator/torch.py`

Torch-family propagator facade.

- `backend="torch", impl="eager"` uses the Python/Torch implementation
- `backend="torch", impl="c"` dispatches to compiled C++/CUDA extension kernels

In other words, the `c` path is part of the `PropTorch` API surface now.
The lower-level compiled C++/CUDA implementation details sit underneath
`PropTorch(..., backend="torch", impl="c")`.

!!! note

    `PropTorch` is the main Torch-facing solver entry point. If you choose
    `backend="torch", impl="eager"`, the time loop and source/receiver logic
    live directly in Python. If you choose `backend="torch", impl="c"`,
    `PropTorch` forwards to compiled CPU or CUDA kernels while keeping the same
    top-level constructor style.

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
- `backend` (`"torch"`, optional): Selects the Torch backend family. Legacy
  values such as `"eager"` and `"cuda"` are still accepted and normalized.
- `impl` (`"eager"` or `"c"`, optional): Selects the Torch-family
  implementation. `"eager"` uses PyTorch operators; `"c"` uses compiled
  C++/CUDA kernels from the Torch extension and dispatches by tensor device.
- `backend_options` (dict or dataclass, optional): Backend-specific options
  block merged after top-level kwargs. See [backend_options](options.md#backend_options).
- `eager_options` ([`EagerOptions`](options.md#eageroptions) or dict, optional):
  eager-only options such as `use_compile`, `compile_mode`, and
  `store_last_wavefield`.
- `cuda_options` ([`CUDAOptions`](options.md#cudaoptions) or dict, optional):
  `c`-implementation options. The name is retained for compatibility; the main
  memory block is
  [`CUDAOptions(memory=...)`](options.md#cudaoptions).
- `use_ckpt` (`bool`, optional): Enables chunk checkpointing on the eager
  implementation. For `c` usage, prefer
  [`CUDAOptions(memory=MemoryOptions(strategy="ckpt", ...))`](options.md#memoryoptions)
  instead of relying on this top-level flag.
- `ckpt_chunks` (`int`, optional): Number of time steps per checkpoint chunk
  when `use_ckpt=True` on the eager implementation. For `c` usage, prefer
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
  `torch.float32` on `dev`. Common layouts are `(nt,)`, `(B, nt)`,
  `(B, nsrc, nt)`, and the source-encoding super-shot layout
  `(1, nsrc, nt)`.
- `sources` (array-like): Source coordinates. They are converted to integer
  tensors on `dev`. Common layouts are `(B, dim)`, `(B, nsrc, dim)`, and
  `(1, nsrc, dim)` for a source-encoding super-shot.
- `receivers` (array-like): Receiver coordinates. They are converted to integer
  tensors on `dev`. Common layout is `(B, nreceivers, dim)`, including
  `(1, nreceivers, dim)` for a source-encoding super-shot.
- `models` (`list[torch.Tensor]`, optional): List of model tensors, provided in
  the exact order required by `equation.models`.
- `source_encoding` (`bool`, optional): If `True`, collapses shots into a
  single encoded batch during propagation. The eager and `c` implementation Torch
  paths also auto-detect source encoding when the runtime inputs use
  `(1, nsrc, nt)`, `(1, nsrc, dim)`, and `(1, nreceivers, dim)`.
- `adj` (`bool`, optional): Switches source time indexing for adjoint-style
  forward usage.
- `return_wavefield` (`bool`, optional): If `True`, returns snapshots in
  addition to recorded data. This option is not supported together with
  checkpointing in the current implementation.

In the shape descriptions above:

- `B` is the runtime batch size
- `nsrc` is the number of sources inside one batch element
- `dim` is `2` in 2D and `3` in 3D

## Return Value

- default: `record` — shape `(B, nt, nrec, nfield)` for every backend
  (`impl="eager"` and `impl="c"`). `nfield` is the number of recorded
  components of the equation (e.g. `1` for acoustic pressure, `2` for
  elastic `vx/vz`, `5` for Zhao DAS). This matches the canonical layout
  expected by `sweep_loss`.
- if `return_wavefield=True`: `(record, snapshots)`
