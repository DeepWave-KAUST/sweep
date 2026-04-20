# Acoustic FWI (JAX)

Source file:

- [examples/FWI/2d/acoustic/jax/fwi_marmousi.py](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/FWI/2d/acoustic/jax/fwi_marmousi.py)

## What This Example Does

This example runs a simple acoustic full-waveform inversion workflow with the
JAX propagator.

The script does four things:

1. Loads a true velocity model and an initial smooth model
2. Builds a `PropJax(Acoustic(...))` solver
3. Generates observed shot gathers from the true model
4. Optimizes the initial model so the synthetic data matches the observed data

## Main Components

The solver is built from:

- `equation`: `Acoustic(..., backend="jax")`
- `propagator`: `PropJax(...)`
- `wave`: a Ricker wavelet
- `sources`: regularly sampled source coordinates
- `receivers`: regularly sampled receiver coordinates
- `models`: the velocity model `vp`

## Key Configuration

The example keeps its runtime settings in a single `CONFIG` dictionary.

Important entries include:

- `nt`, `dt`: temporal sampling
- `dh`: spatial sampling
- `spatial_order`: finite-difference order
- `abcn`: absorbing boundary width
- `src_step`, `rec_step`: acquisition sampling in the x direction
- `true_model`, `init_model`: `.npy` files loaded from `examples/models/`
- `epochs`, `batchsize`, `lr`: inversion hyperparameters
- `use_ckpt`: whether JAX chunk rematerialization is enabled

## Solver Setup

The equation is created with the JAX backend:

```python
equation = Acoustic(
    spatial_order=cfg["spatial_order"],
    backend="jax",
)
```

Shared propagator arguments are collected first:

```python
prop_kwargs = dict(
    shape=shape,
    dev=None,
    dh=cfg["dh"],
    dt=cfg["dt"],
    source_type=["h1"],
    receiver_type=["h1"],
    abcn=cfg["abcn"],
    free_surface=cfg["free_surface"],
    use_ckpt=cfg["use_ckpt"],
    pml_type="cpmlr",
)
```

Then the solver is created as:

```python
solver = PropJax(equation, **prop_kwargs)
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
obs = solver(wave, sources, receivers, models=[true_vp])
```

Then the inversion updates the smooth initial model with `optax.adam`:

```python
optimizer = optax.adam(cfg["lr"], eps=1e-22)
vp = jnp.array(init_model)
opt_state = optimizer.init(vp)
```

Each iteration:

- samples a subset of shots
- computes the synthetic data
- evaluates the L2 data-misfit loss
- gets gradients with `jax.value_and_grad`
- updates the model with `optax`

## Outputs

The script creates an output directory under `examples/` and saves:

- `ricker.png`
- `observed_data.png`
- `loss.png`
- `epoch_XXXX.png` snapshots of
  - the true model
  - the current inverted model
  - the current gradient

## Running the Example

From the repository root:

```bash
python3 examples/FWI/2d/acoustic/jax/fwi_marmousi.py
```

## Full Script

```python
import os
from pathlib import Path

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tqdm

from sweep.equations import Acoustic
from sweep.propagator.jax import PropJax
from sweep.signal import ricker


CONFIG = {
    "nt": 2500,
    "dt": 0.002,
    "delay": 0.256,
    "fm": 5.0,
    "dh": 25.0,
    "spatial_order": 8,
    "abcn": 20,
    "free_surface": True,
    "src_step": 2,
    "rec_step": 1,
    "srcz": 1,
    "recz": 18,
    "lr": 25.0,
    "epochs": 101,
    "batchsize": 8,
    "show_every": 10,
    "true_model": "models/marmousi/true.npy",
    "init_model": "models/marmousi/smooth.npy",
    "output_dir": "acoustic_fwi_jax",
    "use_ckpt": False,
}


def build_solver(shape, cfg):
    equation = Acoustic(
        spatial_order=cfg["spatial_order"],
        backend="jax",
    )

    prop_kwargs = dict(
        shape=shape,
        dev=None,
        dh=cfg["dh"],
        dt=cfg["dt"],
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=cfg["abcn"],
        free_surface=cfg["free_surface"],
        use_ckpt=cfg["use_ckpt"],
        pml_type="cpmlr",
    )

    return PropJax(equation, **prop_kwargs)


def build_geometry(shape, cfg):
    _, nx = shape

    src_x = np.arange(0, nx, cfg["src_step"], dtype=np.int32).reshape(-1, 1)
    src_z = np.full_like(src_x, cfg["srcz"])
    sources = np.concatenate([src_x, src_z], axis=1)

    rec_x = np.arange(0, nx, cfg["rec_step"], dtype=np.int32).reshape(-1, 1)
    rec_z = np.full_like(rec_x, cfg["recz"])
    receivers = np.concatenate([rec_x, rec_z], axis=1)
    receivers = receivers[None, ...].repeat(sources.shape[0], axis=0)

    return sources, receivers


def build_wavelet(cfg):
    t = np.arange(0, cfg["nt"] * cfg["dt"], cfg["dt"], dtype=np.float32)
    return t, jnp.array(ricker(t - cfg["delay"], f=cfg["fm"]).astype(np.float32))


def save_wavelet_figure(wave, output_dir):
    plt.figure(figsize=(6, 3))
    plt.plot(np.asarray(wave), color="black")
    plt.title("Ricker Wavelet")
    plt.tight_layout()
    plt.savefig(output_dir / "ricker.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_observed_figure(obs, output_dir):
    shot = np.asarray(obs[-1].squeeze())
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


def main():
    cfg = CONFIG.copy()
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)

    true_model = np.load(base_dir / cfg["true_model"]).astype(np.float32)
    init_model = np.load(base_dir / cfg["init_model"]).astype(np.float32)
    shape = true_model.shape

    solver = build_solver(shape, cfg)

    _, wave = build_wavelet(cfg)
    save_wavelet_figure(wave, output_dir)

    sources, receivers = build_geometry(shape, cfg)
    print("(nshots, ndim):", sources.shape)
    print("(nshots, nreceivers, ndim):", receivers.shape)

    true_vp = jnp.array(true_model)
    obs = solver(wave, sources, receivers, models=[true_vp])
    save_observed_figure(obs, output_dir)

    sources_jax = jnp.array(sources)
    receivers_jax = jnp.array(receivers)
    obs_jax = jnp.array(obs)

    optimizer = optax.adam(cfg["lr"], eps=1e-22)
    vp = jnp.array(init_model)
    opt_state = optimizer.init(vp)
    losses = []
    nshots = sources.shape[0]
    batchsize = min(cfg["batchsize"], nshots)

    @jax.jit
    def update_fn(params, grads, state):
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return params, state

    @jax.jit
    def fwi_step(params, shot_idx):
        def loss_fn(current_params, current_shots):
            syn = solver(
                wave,
                sources=sources_jax[current_shots],
                receivers=receivers_jax[current_shots],
                models=[current_params],
            )
            current_obs = obs_jax[current_shots]
            return jnp.mean((syn - current_obs) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params, shot_idx)
        return loss, grads

    for epoch in tqdm.trange(cfg["epochs"]):
        shot_idx = np.random.choice(nshots, size=batchsize, replace=False)
        loss, grads = fwi_step(vp, shot_idx)
        vp, opt_state = update_fn(vp, grads, opt_state)

        loss_value = float(loss)
        losses.append(loss_value)
        print(f"Epoch {epoch:04d} | Loss: {loss_value:.6e}")

        if epoch % cfg["show_every"] == 0:
            save_progress_figure(
                true_model,
                np.asarray(vp),
                np.asarray(grads),
                losses,
                epoch,
                cfg,
                output_dir,
            )


if __name__ == "__main__":
    main()
```
