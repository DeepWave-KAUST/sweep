# Acoustic FWI With Source Encoding (CUDA/Torch)

Source file:

- [examples/acoustic_fwi_encoding_torch.py](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/acoustic_fwi_encoding_torch.py)

## What This Example Does

This example runs acoustic full-waveform inversion with source encoding using a
single script that supports two propagator backends:

- `torch`: pure PyTorch propagation with `PropTorch`
- `cuda`: compiled CUDA propagation with `PropCUDA`

Compared with the standard acoustic FWI example, this script does not invert on
individual shot gathers. Instead, it builds encoded super-shots by:

- sampling a subset of shots
- applying random polarity flips
- applying random time shifts
- summing the encoded data into one source-encoded gather

## Main Components

The solver is built from:

- `equation`: `Acoustic(...)`
- `propagator`: `PropTorch(...)` or `PropCUDA(...)`
- `wave`: a Ricker wavelet
- `sources`: regularly sampled source coordinates
- `receivers`: a shared receiver line
- `models`: the velocity model `vp`

## Backend Selection

The entry point is:

```bash
python3 examples/acoustic_fwi_encoding_torch.py --backend torch
```

or:

```bash
python3 examples/acoustic_fwi_encoding_torch.py --backend cuda
```

Internally, the script keeps:

- `COMMON_CONFIG`: shared acquisition, encoding, and inversion settings
- `BACKEND_CONFIG`: backend-specific options such as
  - `use_compile` for the PyTorch path
  - `boundary_saving_config` for the CUDA path
  - display transpose rules for saved figures

## Key Configuration

Shared configuration includes:

- `nt`, `dt`: temporal sampling
- `dh`: spatial sampling
- `spatial_order`: finite-difference order
- `src_step`, `rec_step`: acquisition sampling in the x direction
- `true_model`, `init_model`: `.npy` files loaded from `examples/`
- `epochs`, `batchsize`, `lr`: inversion hyperparameters
- `max_time_shift_ratio`: maximum random encoding shift as a fraction of `nt`

Backend-specific configuration includes:

- PyTorch:
  - `use_compile`
  - `use_ckpt`
- CUDA:
  - `boundary_saving_config`
  - gather transpose for visualization

## Solver Setup

The equation side is shared across both modes:

```python
equation = Acoustic(
    spatial_order=cfg["spatial_order"],
    device=dev,
    backend="torch",
)
```

Even when the propagator is `PropCUDA`, the equation `backend` remains
`"torch"`.

Shared propagator arguments are collected first:

```python
prop_kwargs = dict(
    shape=shape,
    dev=dev,
    dh=cfg["dh"],
    dt=cfg["dt"],
    source_type=["h1"],
    receiver_type=["h1"],
    abcn=cfg["abcn"],
    free_surface=cfg["free_surface"],
    pml_type="cpmlr",
)
```

### PyTorch Mode

```python
solver = PropTorch(
    equation,
    **prop_kwargs,
    use_ckpt=cfg["use_ckpt"],
    use_compile=cfg["use_compile"],
)
```

### CUDA Mode

```python
solver = PropCUDA(
    equation,
    **prop_kwargs,
    boundary_saving_config=cfg["boundary_saving_config"],
)
```

## Source Encoding Workflow

For each inversion step, the script:

1. selects a random subset of shots
2. generates an encoded wavelet for each selected shot
3. applies the same random polarity and time shift to the corresponding
   observed gather
4. sums the encoded gathers into a super-shot target
5. runs forward modeling with `source_encoding=True`

The CUDA and PyTorch paths differ in two important ways:

### PyTorch Source Encoding

The PyTorch propagator keeps one encoded source per selected shot and collapses
them internally into a single batch when `source_encoding=True`.

The script therefore calls `PropTorch` with:

- `wavelet`: `(nsel, nt)`
- `sources`: `(nsel, 2)`
- `receivers`: `(1, nreceivers, 2)`

where `nsel` is the number of randomly selected shots in the current inversion
step.

### CUDA Source Encoding

The CUDA propagator uses a different convention. When `source_encoding=True`,
it expects a single batch that contains multiple encoded sources inside that
batch.

The script therefore calls `PropCUDA` with:

- `wavelet`: `(1, nsrc, nt)`
- `sources`: `(1, nsrc, 2)`
- `receivers`: `(1, nreceivers, 2)`

where `nsrc` is the number of encoded sources combined into the current
super-shot.

### Record Layout Difference

The recorded data layout also differs between backends:

- `torch` uses a time-major shot layout for the encoded gathers
- `cuda` returns a layout where the receiver and time axes are ordered
  differently

For that reason, the script:

- applies time shifts along the true time axis for each backend
- keeps each backend in its native layout during loss computation
- only normalizes the orientation when saving figures for display

## Geometry

The example builds a simple fixed-depth acquisition:

- sources are placed every `src_step` grid points
- receivers are placed every `rec_step` grid points
- all sources use the same source depth `srcz`
- all receivers use the same receiver depth `recz`

The final array shapes are:

- `sources`: `(nshots, 2)`
- `receivers`: `(nshots, nreceivers, 2)`
- inversion receivers: `(1, nreceivers, 2)`

## Outputs

The script creates an output directory under `examples/` and saves:

- `ricker.png`
- `observed_data.png`
- `loss.png`
- `data_epoch_XXXX.png` with
  - encoded observed data
  - encoded synthetic data
- `epoch_XXXX.png` snapshots of
  - the true model
  - the current inverted model
  - the current gradient

Each backend writes into its own output directory:

- `acoustic_fwi_encoding_torch`
- `acoustic_fwi_encoding_cuda`

## Running the Example

PyTorch mode:

```bash
python3 examples/acoustic_fwi_encoding_torch.py --backend torch
```

CUDA mode:

```bash
python3 examples/acoustic_fwi_encoding_torch.py --backend cuda
```

Notes:

- `torch` mode runs on GPU if available and otherwise falls back to CPU
- `cuda` mode requires a CUDA-capable PyTorch environment and compiled binding
- encoded data layout is backend-dependent internally, but saved figures are
  normalized for easier comparison
