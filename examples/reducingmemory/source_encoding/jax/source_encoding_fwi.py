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
from plotting import gather_extent, plot_loss_curve

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import optax
import tqdm

import configure_marmousi as shared_config
from sweep.equations import Acoustic
from sweep.propagator.jax import PropJax
from sweep.signal import ricker


CONFIG = shared_config.get_config("fwi_2d_acoustic_encoding_jax")


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


def save_observed_figure(obs, receivers, cfg, output_dir):
    shot = np.asarray(obs[-1].squeeze())
    vmin, vmax = np.percentile(shot, [2, 98])
    plt.figure(figsize=(6, 4))
    plt.imshow(
        shot,
        vmin=vmin,
        vmax=vmax,
        cmap="seismic",
        aspect="auto",
        extent=gather_extent(shot.shape[0], cfg["dt"], receivers, cfg["dh"]),
    )
    plt.colorbar(label="Amplitude")
    plt.title("Observed Shot Gather")
    plt.xlabel("Distance (m)")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_dir / "observed_data.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_encoded_data_figure(encoded_obs, encoded_syn, receivers, cfg, epoch, output_dir):
    obs_np = np.asarray(encoded_obs).squeeze()
    syn_np = np.asarray(encoded_syn).squeeze()
    vmin, vmax = np.percentile(obs_np, [2, 98])

    fig, axes = plt.subplots(1, 2, figsize=(7, 5))
    extent = gather_extent(obs_np.shape[0], cfg["dt"], receivers, cfg["dh"])
    axes[0].imshow(obs_np, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto", extent=extent)
    axes[0].set_title("Encoded Observed Data")
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Time (s)")
    axes[1].imshow(syn_np, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto", extent=extent)
    axes[1].set_title("Encoded Synthetic Data")
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_dir / f"data_epoch_{epoch:04d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    plot_loss_curve(ax, losses, "Mean Squared Error")
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
    key = random.PRNGKey(0)

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
    save_observed_figure(obs, receivers, cfg, output_dir)

    sources_jax = jnp.array(sources)
    receivers_shared = jnp.array(receivers[:1])
    obs_jax = jnp.array(obs)
    wave_jax = jnp.array(wave)

    optimizer = optax.adam(cfg["lr"], eps=1e-22)
    vp = jnp.array(init_model)
    opt_state = optimizer.init(vp)
    losses = []
    nshots = sources.shape[0]
    batchsize = min(cfg["batchsize"], nshots)
    max_time_shift = int(cfg["max_time_shift_ratio"] * cfg["nt"])
    time_shifts = jnp.arange(max(1, max_time_shift), dtype=jnp.int32)

    @jax.jit
    def update_fn(params, grads, state):
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return params, state

    @jax.jit
    def fwi_step(step_key, params, shot_idx):
        def loss_fn(current_params, current_shots):
            nsel = current_shots.shape[0]
            subkeys = random.split(step_key, nsel * 2)
            keys_p = subkeys[:nsel]
            keys_tau = subkeys[nsel:]

            def process_one_shot(current_shot, key_p, key_tau):
                polarity = random.randint(key_p, shape=(), minval=0, maxval=2)
                tau = random.choice(key_tau, time_shifts)
                mask = jnp.arange(cfg["nt"]) < tau

                encoded_wave = (-1) ** polarity * jnp.roll(wave_jax, tau, axis=0)
                encoded_wave = encoded_wave * (~mask)

                encoded_obs = (-1) ** polarity * jnp.roll(obs_jax[current_shot], tau, axis=0)
                encoded_obs = encoded_obs * (~mask.reshape(-1, 1, 1))

                return encoded_wave, encoded_obs

            super_wave, super_obs = jax.vmap(process_one_shot)(current_shots, keys_p, keys_tau)
            syn = solver(
                super_wave,
                sources=sources_jax[current_shots],
                receivers=receivers_shared,
                models=[current_params],
                source_encoding=True,
            )
            encoded_obs = jnp.sum(super_obs, axis=0)
            loss = jnp.mean((syn - encoded_obs) ** 2)
            return loss, (syn, encoded_obs)

        (loss, data), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, shot_idx)
        return loss, grads, data

    for epoch in tqdm.trange(cfg["epochs"]):
        key, perm_key, step_key = random.split(key, 3)
        shot_idx = random.permutation(perm_key, sources.shape[0])[:batchsize]

        loss, grads, (encoded_syn, encoded_obs) = fwi_step(step_key, vp, shot_idx)
        vp, opt_state = update_fn(vp, grads, opt_state)

        loss_value = float(loss)
        losses.append(loss_value)
        print(f"Epoch {epoch:04d} | Loss: {loss_value:.6e}")

        if epoch % cfg["show_every"] == 0:
            save_encoded_data_figure(encoded_obs, encoded_syn, receivers_shared, cfg, epoch, output_dir)
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
