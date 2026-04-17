# PropCUDA

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
    full_mode="full",
    boundary_saving_config=None,
)
```

Implementation:

- `src/sweep/propagator/cuda.py`

Compiled CUDA propagator backed by equation-specific bindings from `sweep._C`.

!!! note

    `PropCUDA` is the backend with the most runtime-specific behavior:
    anisotropic `dh`, reusable buffers, boundary saving, recursive checkpointing,
    and RTM all live here.

## Parameters

- `equation` (equation instance): Equation instance whose compiled CUDA
  binding will be used. `PropCUDA` expects the equation to expose `_C()`, and
  optionally `_C_rtm()` for RTM.
- `shape` (`tuple[int, ...]`): Physical model shape before absorbing
  boundaries are added. Use `(nz, nx)` in 2D and `(nz, ny, nx)` in 3D.
- `source_type` (`list[str]`, optional): Wavefield names used for source
  injection. If omitted, `PropCUDA` can infer defaults: acoustic-like equations
  use the first wavefield, 2D elastic uses `["sxx", "szz"]`, and 3D elastic
  uses `["sxx", "syy", "szz"]`.
- `receiver_type` (`list[str]`, optional): Wavefield names sampled at receiver
  locations. If omitted, `PropCUDA` can infer defaults: acoustic-like equations
  use the first wavefield, 2D elastic uses `["vx", "vz"]`, and 3D elastic uses
  `["vx", "vy", "vz"]`.
- `abcn` (`int`, optional): Absorbing boundary width.
- `free_surface` (`bool`, optional): Whether the top boundary is treated as a
  free surface. This affects coordinate shifts before entering the CUDA
  kernels.
- `dh` (`float or tuple[float, ...]`, optional): Grid spacing. Unlike the
  other propagators, `PropCUDA` supports anisotropic tuple-valued spacing:
  `(dz, dx)` in 2D and `(dz, dy, dx)` in 3D.
- `dt` (`float`, optional): Time step in seconds.
- `dev` (device, optional): Execution device for tensors and reusable CUDA
  buffers.
- `use_ckpt` (`bool`, optional): Enables checkpoint-based memory reduction in
  the CUDA path.
- `ckpt_chunks` (`int`, optional): Checkpoint interval used in chunk
  checkpointing.
- `ckpt_mode` (`str`, optional): Checkpoint strategy. Supported values here are
  `"chunk"` and `"recursive"`.
- `ckpt_num` (`int`, optional): Number of persistent checkpoints used by
  recursive checkpointing.
- `pml_type` (`str`, optional): PML implementation passed into equation setup.
- `nt` (`int`, optional): Stored time-step count. The actual working value is
  normally inferred from the runtime wavelet.
- `B` (`int`, optional): Initial batch capacity for reusable runtime buffers.
- `allow_growth` (`bool`, optional): If `True`, runtime buffers may grow when a
  larger batch is seen later. If `False`, larger batches than the preallocated
  capacity raise an error.
- `full_mode` (`str`, optional): Stored on the base class and not currently the
  main runtime switch for this backend.
- `boundary_saving_config` (`dict`, optional): Configuration for saving forward
  boundary values instead of storing all wavefields. Normalized form:
  `{"enabled": False, "storage": "gpu", "transfer_interval": 1, "pinned_memory": False}`.

  Supported keys are:

  - `enabled` (`bool`): Whether boundary saving is enabled.
  - `storage` (`str`): Where saved boundary values live. Supported values are
    `"gpu"` and `"cpu"`.
  - `transfer_interval` (`int`): How often boundary values are transferred when
    CPU storage is used. This must be at least `1`. When `storage="gpu"`, the
    effective interval is forced to `1`.
  - `pinned_memory` (`bool`): Whether to use pinned host memory when
    `storage="cpu"`. When `storage="gpu"`, this is effectively disabled.

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
    use_boundary_saving=None,
    boundary_saving_config=None,
    **kwargs,
)
```

- `wavelet` (array-like): Source time function. Accepted layouts are `(nt,)`,
  `(B, nt)`, and `(B, nsrc, nt)`.
- `sources` (array-like): Source coordinates. Accepted layouts are `(B, dim)`
  and `(B, nsrc, dim)`.
- `receivers` (array-like): Receiver coordinates. This path expects batched
  receiver coordinates as well.
- `models` (`list[torch.Tensor]`, optional): List of model tensors in the exact
  order required by `equation.models`. They are padded and expanded across the
  active batch before being passed into the binding.
- `source_encoding` (`bool`, optional): If `True`, runs with a single encoded
  batch instead of one batch element per shot.
- `adj` (`bool`, optional): Adjoint-style forward switch.
- `return_wavefield` (`bool`, optional): Present in the signature, but the
  current main CUDA forward path still returns only the synthetic data.
- `use_boundary_saving` (`bool`, optional): Runtime override for enabling
  boundary saving.
- `boundary_saving_config` (`dict`, optional): Runtime override for the
  boundary-saving policy.

## RTM Parameters

- `adjoint_source` (array-like): Input data for reverse-time migration.
  Accepted layouts are `(B, nt, nrec[, 1])` and `(B, nrec, nt)`.

## Return Value

- `forward(...)`: synthetic data `record`
- `rtm(...)`: `(syn, image, source_illumination, receiver_illumination)`
