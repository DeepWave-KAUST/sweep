# Acoustic FWI (CUDA/Torch)

Source file:

- [examples/acoustic_fwi_torch.py](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/acoustic_fwi_torch.py)

## What This Example Does

This example runs a simple acoustic full-waveform inversion workflow with a
single script that supports two propagator backends:

- `torch`: pure PyTorch propagation with `PropTorch`
- `cuda`: compiled CUDA propagation with `PropCUDA`

The script does four things:

1. Loads a true velocity model and an initial smooth model
2. Builds an acoustic solver for the selected backend
3. Generates observed shot gathers from the true model
4. Optimizes the initial model so the synthetic data matches the observed data

## Main Components

The solver is built from:

- `equation`: `Acoustic(...)`
- `propagator`: `PropTorch(...)` or `PropCUDA(...)`
- `wave`: a Ricker wavelet
- `sources`: regularly sampled source coordinates
- `receivers`: regularly sampled receiver coordinates
- `models`: the velocity model `vp`

## Backend Selection

The entry point is:

```bash
python3 examples/acoustic_fwi_torch.py --backend torch
```

or:

```bash
python3 examples/acoustic_fwi_torch.py --backend cuda
```

Internally, the script keeps:

- `COMMON_CONFIG`: shared acquisition and inversion settings
- `BACKEND_CONFIG`: backend-specific options such as
  - `use_compile` for the PyTorch path
  - `boundary_saving_config` for the CUDA path

## Key Configuration

Shared configuration includes:

- `nt`, `dt`: temporal sampling
- `dh`: spatial sampling
- `spatial_order`: finite-difference order
- `src_step`, `rec_step`: acquisition sampling in the x direction
- `true_model`, `init_model`: `.npy` files loaded from `examples/`
- `epochs`, `batchsize`, `lr`: inversion hyperparameters

Backend-specific configuration includes:

- PyTorch:
  - `use_compile`
  - `use_ckpt`
- CUDA:
  - `boundary_saving_config`
  - output gather transpose for visualization

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

## Geometry

The example builds a simple fixed-depth acquisition:

- sources are placed every `src_step` grid points
- receivers are placed every `rec_step` grid points
- all sources use the same source depth `srcz`
- all receivers use the same receiver depth `recz`

The final array shapes are:

- `sources`: `(nshots, 2)`
- `receivers`: `(nshots, nreceivers, 2)`

## Inversion Loop

Observed data is first generated from the true model:

```python
obs, elapsed_ms = timed_forward(solver, wave, sources, receivers, models=[true_vp])
```

Then the inversion updates the smooth initial model:

```python
inv_vp = torch.from_numpy(init_model).to(dev).requires_grad_(True)
optimizer = torch.optim.Adam([inv_vp], lr=cfg["lr"], eps=1e-22)
```

For each epoch:

- a random subset of shots is selected
- synthetic data is computed
- the L2 data-misfit loss is evaluated
- gradients are backpropagated to `inv_vp`
- the optimizer updates the model

## Outputs

The script creates an output directory under `examples/` and saves:

- `ricker.png`
- `observed_data.png`
- `loss.png`
- `epoch_XXXX.png` snapshots of
  - the true model
  - the current inverted model
  - the current gradient

Each backend writes into its own output directory:

- `acoustic_fwi_torch`
- `acoustic_fwi_cuda`

## Running the Example

PyTorch mode:

```bash
python3 examples/acoustic_fwi_torch.py --backend torch
```

CUDA mode:

```bash
python3 examples/acoustic_fwi_torch.py --backend cuda
```

Notes:

- `torch` mode runs on GPU if available and otherwise falls back to CPU
- `cuda` mode requires a CUDA-capable PyTorch environment and compiled binding

## Full Script

```python
from pathlib import Path
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from sweep.equations import Acoustic
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


torch.backends.cudnn.benchmark = True


COMMON_CONFIG = {
    "nt": 2500,
    "dt": 0.002,
    "delay": 0.256,
    "fm": 5.0,
    "dh": 25.0,
    "spatial_order": 8,
    "src_step": 2,
    "rec_step": 1,
    "srcz": 1,
    "recz": 18,
    "lr": 25.0,
    "epochs": 101,
    "batchsize": 8,
    "show_every": 10,
    "true_model": "marmousi_true.npy",
    "init_model": "marmousi_smooth.npy",
}


BACKEND_CONFIG = {
    "torch": {
        "abcn": 20,
        "free_surface": False,
        "output_dir": "acoustic_fwi_torch",
        "use_compile": True,
        "use_ckpt": False,
        "transpose_shot": False,
    },
    "cuda": {
        "abcn": 20,
        "free_surface": False,
        "output_dir": "acoustic_fwi_cuda",
        "transpose_shot": True,
        "boundary_saving_config": {
            "enabled": True,
            "storage": "gpu",
            "transfer_interval": 10,
            "pinned_memory": True,
        },
    },
}


def build_config(backend):
    if backend not in BACKEND_CONFIG:
        raise ValueError(f"Unsupported backend '{backend}'. Expected one of {sorted(BACKEND_CONFIG)}.")
    cfg = COMMON_CONFIG.copy()
    cfg.update(BACKEND_CONFIG[backend])
    cfg["backend"] = backend
    return cfg


def build_solver(shape, dev, cfg):
    equation = Acoustic(
        spatial_order=cfg["spatial_order"],
        device=dev,
        backend="torch",
    )

    if cfg["backend"] == "torch":
        return PropTorch(
            equation,
            shape=shape,
            dev=dev,
            dh=cfg["dh"],
            dt=cfg["dt"],
            source_type=["h1"],
            receiver_type=["h1"],
            abcn=cfg["abcn"],
            free_surface=cfg["free_surface"],
            use_ckpt=cfg["use_ckpt"],
            pml_type="cpmlr",
            use_compile=cfg["use_compile"],
        )

    if cfg["backend"] == "cuda":
        return PropCUDA(
            equation,
            shape=shape,
            dev=dev,
            dh=cfg["dh"],
            dt=cfg["dt"],
            source_type=["h1"],
            receiver_type=["h1"],
            abcn=cfg["abcn"],
            free_surface=cfg["free_surface"],
            pml_type="cpmlr",
            boundary_saving_config=cfg["boundary_saving_config"],
        )

    raise ValueError(f"Unsupported backend '{cfg['backend']}'.")


def build_geometry(shape, cfg):
    _, nx = shape

    src_x = np.arange(0, nx, cfg["src_step"], dtype=np.int64).reshape(-1, 1)
    src_z = np.full_like(src_x, cfg["srcz"])
    sources = np.concatenate([src_x, src_z], axis=1)

    rec_x = np.arange(0, nx, cfg["rec_step"], dtype=np.int64).reshape(-1, 1)
    rec_z = np.full_like(rec_x, cfg["recz"])
    receivers = np.concatenate([rec_x, rec_z], axis=1)
    receivers = receivers[None, ...].repeat(sources.shape[0], axis=0)

    return sources, receivers


def build_wavelet(cfg):
    t = np.arange(0, cfg["nt"] * cfg["dt"], cfg["dt"], dtype=np.float32)
    return t, ricker(t - cfg["delay"], f=cfg["fm"]).astype(np.float32)


def timed_forward(solver, wave, sources, receivers, models):
    if solver.dev.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            obs = solver(wave, sources, receivers, models=models)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            obs = solver(wave, sources, receivers, models=models)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return obs.detach().cpu().numpy(), elapsed_ms


def save_wavelet_figure(wave, output_dir):
    plt.figure(figsize=(6, 3))
    plt.plot(wave, color="black")
    plt.title("Ricker Wavelet")
    plt.tight_layout()
    plt.savefig(output_dir / "ricker.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_observed_figure(obs, output_dir, cfg):
    shot = obs[-1].squeeze()
    if cfg.get("transpose_shot", False):
        shot = shot.T
    vmin, vmax = np.percentile(shot, [2, 98])
    plt.figure(figsize=(6, 4))
    plt.imshow(shot, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
    plt.colorbar()
    plt.title("Observed Shot Gather")
    plt.tight_layout()
    plt.savefig(output_dir / "observed_data.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_progress_figure(true_model, vp, grad, losses, epoch, cfg, output_dir):
    nz, nx = true_model.shape
    extent = [0, nx * cfg["dh"], nz * cfg["dh"], 0]
    vmin_model, vmax_model = true_model.min(), true_model.max()
    vmin_grad, vmax_grad = np.percentile(grad, [2, 98])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(true_model, vmin=vmin_model, vmax=vmax_model, cmap="seismic", aspect="auto", extent=extent)
    axes[0].set_title("True Model")
    axes[1].imshow(vp, vmin=vmin_model, vmax=vmax_model, cmap="seismic", aspect="auto", extent=extent)
    axes[1].set_title("Inverted Model")
    axes[2].imshow(grad, vmin=vmin_grad, vmax=vmax_grad, cmap="seismic", aspect="auto", extent=extent)
    axes[2].set_title("Gradient")
    plt.tight_layout()
    plt.savefig(output_dir / f"epoch_{epoch:04d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(losses, color="black", label="Loss")
    ax.legend()
    ax.set_title("FWI Loss")
    plt.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_fwi(backend="torch"):
    cfg = build_config(backend)
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)

    true_model = np.load(base_dir / cfg["true_model"]).astype(np.float32)
    init_model = np.load(base_dir / cfg["init_model"]).astype(np.float32)
    shape = true_model.shape

    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The CUDA acoustic FWI example requires a CUDA-capable PyTorch environment.")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = build_solver(shape, dev, cfg)

    _, wave = build_wavelet(cfg)
    save_wavelet_figure(wave, output_dir)

    sources, receivers = build_geometry(shape, cfg)
    print("(nshots, ndim):", sources.shape)
    print("(nshots, nreceivers, ndim):", receivers.shape)

    true_vp = torch.from_numpy(true_model).to(dev)
    obs, elapsed_ms = timed_forward(solver, wave, sources, receivers, models=[true_vp])
    print(f"Forward modeling time ({backend}): {elapsed_ms:.2f} ms")
    save_observed_figure(obs, output_dir, cfg)

    inv_vp = torch.from_numpy(init_model).to(dev).requires_grad_(True)
    optimizer = torch.optim.Adam([inv_vp], lr=cfg["lr"], eps=1e-22)

    obs_torch = torch.from_numpy(obs).to(dev)
    losses = []
    nshots = sources.shape[0]
    batchsize = min(cfg["batchsize"], nshots)

    for epoch in tqdm.trange(cfg["epochs"]):
        optimizer.zero_grad()

        shot_idx = np.random.choice(nshots, size=batchsize, replace=False)
        syn = solver(wave, sources[shot_idx], receivers[shot_idx], models=[inv_vp])
        loss = (syn - obs_torch[shot_idx]).pow(2).mean()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
```
