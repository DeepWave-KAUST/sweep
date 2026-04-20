import os
from pathlib import Path
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def find_examples_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if candidate.name == "examples":
            return candidate
    raise RuntimeError("Could not locate the examples directory.")


EXAMPLES_DIR = find_examples_root()
sys.path.insert(0, str(EXAMPLES_DIR / "_shared"))
from bootstrap import configure_example_imports

IMPORT_MODE = configure_example_imports(EXAMPLES_DIR)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tqdm

import configure_marmousi as shared_config
from sweep.equations import Acoustic
from sweep.propagator.jax import PropJax
from sweep.signal import ricker

CONFIG = shared_config.get_config("fwi_2d_acoustic_jax")


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
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)

    true_model = np.load(EXAMPLES_DIR / cfg["true_model"]).astype(np.float32)
    init_model = np.load(EXAMPLES_DIR / cfg["init_model"]).astype(np.float32)
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

    wave_jax = jnp.asarray(wave)
    sources_jax = jnp.asarray(sources)
    receivers_jax = jnp.asarray(receivers)
    obs_jax = jnp.asarray(obs)

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
    def fwi_step(params, shot_idx, wavelet, sources_all, receivers_all, obs_all):
        def loss_fn(current_params, current_shots, current_wavelet, all_sources, all_receivers, all_obs):
            syn = solver(
                current_wavelet,
                sources=all_sources[current_shots],
                receivers=all_receivers[current_shots],
                models=[current_params],
            )
            current_obs = all_obs[current_shots]
            return jnp.mean((syn - current_obs) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params, shot_idx, wavelet, sources_all, receivers_all, obs_all)
        return loss, grads

    for epoch in tqdm.trange(cfg["epochs"]):
        shot_idx = np.random.choice(nshots, size=batchsize, replace=False)
        loss, grads = fwi_step(vp, shot_idx, wave_jax, sources_jax, receivers_jax, obs_jax)
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
