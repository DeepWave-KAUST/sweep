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
import jax.lax as lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tqdm

import configure_overthrust as shared_config
from sweep.equations import Elastic
from sweep.propagator.jax import PropJax
from sweep.signal import ricker

CONFIG = shared_config.get_config("fwi_2d_elastic_jax")


def build_solver(shape, cfg):
    equation = Elastic(
        spatial_order=cfg["spatial_order"],
        backend="jax",
    )

    return PropJax(
        equation,
        shape=shape,
        dev=None,
        abcn=cfg["abcn"],
        dh=cfg["dh"],
        dt=cfg["dt"],
        source_type=["sxx", "szz"],
        receiver_type=["vx", "vz"],
        free_surface=cfg["free_surface"],
        pml_type="cpmls",
        use_ckpt=cfg["use_ckpt"],
        ckpt_chunks=cfg["ckpt_chunks"],
    )


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
    wave = ricker(t - cfg["delay"], f=cfg["fm"]).astype(np.float32) * cfg["wavelet_scale"]
    return t, jnp.asarray(wave)


def take_shots(array, shot_idx):
    return jnp.take(array, shot_idx, axis=0)


def save_wavelet_figure(wave, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(np.asarray(wave), color="black")
    ax.set_title("Ricker Wavelet")
    ax.set_xlabel("Time Sample")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(output_dir / "ricker.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_observed_figure(obs, receivers, cfg, output_dir):
    vx = np.asarray(obs[-1][..., 0])
    vz = np.asarray(obs[-1][..., 1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for ax, data, title in zip(axes[0], [vx, vz], ["Observed Vx", "Observed Vz"]):
        vmin, vmax = np.percentile(data, [2, 98])
        im = ax.imshow(
            data,
            vmin=vmin,
            vmax=vmax,
            cmap="seismic",
            aspect="auto",
            extent=gather_extent(data.shape[0], cfg["dt"], receivers, cfg["dh"]),
        )
        fig.colorbar(im, ax=ax, shrink=0.85, label="Amplitude")
        ax.set_title(title)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "observed_data.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_progress_figure(vp_true, vs_true, vp, vs, grad_vp, grad_vs, losses, epoch, cfg, output_dir):
    nz, nx = vp_true.shape
    extent = [0, nx * cfg["dh"], nz * cfg["dh"], 0]
    vmin_vp, vmax_vp = vp_true.min(), vp_true.max()
    vmin_vs, vmax_vs = vs_true.min(), vs_true.max()

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    panels = [
        ("True Vp", vp_true, (vmin_vp, vmax_vp), "Velocity (m/s)"),
        ("True Vs", vs_true, (vmin_vs, vmax_vs), "Velocity (m/s)"),
        ("Inverted Vp", vp, (vmin_vp, vmax_vp), "Velocity (m/s)"),
        ("Inverted Vs", vs, (vmin_vs, vmax_vs), "Velocity (m/s)"),
        ("Gradient Vp", grad_vp, tuple(np.percentile(grad_vp, [2, 98])), "Gradient"),
        ("Gradient Vs", grad_vs, tuple(np.percentile(grad_vs, [2, 98])), "Gradient"),
    ]

    for ax, (title, data, (vmin, vmax), cbar_label) in zip(axes.ravel(), panels):
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto", extent=extent)
        fig.colorbar(im, ax=ax, shrink=0.82, label=cbar_label)
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")

    fig.tight_layout()
    fig.savefig(output_dir / f"epoch_{epoch:04d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plot_loss_curve(ax, losses, "Scaled Mean Squared Error")
    ax.set_title("FWI Loss")
    fig.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = CONFIG.copy()
    output_dir = Path(__file__).resolve().parent / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)

    vp_true = np.load(EXAMPLES_DIR / cfg["true_model"]).astype(np.float32)
    vs_true = (vp_true / 1.732).astype(np.float32)
    rho_true = np.ones_like(vp_true, dtype=np.float32) * 1000.0
    vp_init = np.load(EXAMPLES_DIR / cfg["init_model"]).astype(np.float32)
    shape = vp_true.shape

    solver = build_solver(shape, cfg)
    _, wave = build_wavelet(cfg)
    save_wavelet_figure(wave, output_dir)

    sources, receivers = build_geometry(shape, cfg)
    print("(nshots, ndim):", sources.shape)
    print("(nshots, nreceivers, ndim):", receivers.shape)

    obs = solver(
        wave,
        sources,
        receivers,
        models=[jnp.asarray(vp_true), jnp.asarray(vs_true), jnp.asarray(rho_true)],
    )
    save_observed_figure(obs, receivers, cfg, output_dir)

    wave_jax = jnp.asarray(wave)
    sources_jax = jnp.asarray(sources)
    receivers_jax = jnp.asarray(receivers)
    receivers_shared = receivers_jax[:1]
    obs_jax = jnp.asarray(obs)

    vp = jnp.asarray(vp_init)
    vs = jnp.asarray(vp_init / 1.732)
    rho = jnp.asarray(rho_true)

    opts = [
        optax.adam(cfg["lr_vp"], eps=1e-22),
        optax.adam(cfg["lr_vs"], eps=1e-22),
        optax.adam(cfg["lr_rho"], eps=1e-22),
    ]
    opt_states = [opt.init(param) for param, opt in zip([vp, vs, rho], opts)]
    losses = []
    nshots = sources.shape[0]
    batchsize = min(cfg["batchsize"], nshots)

    @jax.jit
    def update_vp(param, grad, opt_state):
        updates, opt_state = opts[0].update(grad, opt_state)
        param = optax.apply_updates(param, updates)
        return param, opt_state

    @jax.jit
    def update_vs(param, grad, opt_state):
        updates, opt_state = opts[1].update(grad, opt_state)
        param = optax.apply_updates(param, updates)
        return param, opt_state

    @jax.jit
    def update_rho(param, grad, opt_state):
        updates, opt_state = opts[2].update(grad, opt_state)
        param = optax.apply_updates(param, updates)
        return param, opt_state

    @jax.jit
    def fwi_step(current_vp, current_vs, current_rho, shot_idx, enc_signs, current_wave, all_sources, all_receivers, shared_receivers, all_obs):
        def loss_fn(vp_param, vs_param, rho_param, shots, signs, wavelet, sources_all, receivers_all, receivers_shared, obs_all):
            if cfg["source_encoding"]:
                encoded_wave = signs * jnp.broadcast_to(wavelet, (shots.shape[0], wavelet.shape[0]))
                encoded_obs = jnp.sum(take_shots(obs_all, shots) * signs[:, None, None, None], axis=0, keepdims=True)
                syn = solver(
                    encoded_wave,
                    sources=take_shots(sources_all, shots),
                    receivers=receivers_shared,
                    source_encoding=True,
                    models=[vp_param, vs_param, rho_param],
                )
                return jnp.mean((syn - encoded_obs) ** 2) * 1e10

            shot_sources = take_shots(sources_all, shots)
            shot_receivers = take_shots(receivers_all, shots)
            shot_obs = take_shots(obs_all, shots)
            syn = solver(
                wavelet,
                sources=shot_sources,
                receivers=shot_receivers,
                models=[vp_param, vs_param, rho_param],
            )
            return jnp.mean((syn - shot_obs) ** 2) * 1e10

        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(
            current_vp,
            current_vs,
            current_rho,
            shot_idx,
            enc_signs,
            current_wave,
            all_sources,
            all_receivers,
            shared_receivers,
            all_obs,
        )
        return loss, grads

    shot_rng = np.random.RandomState(0)
    shot_schedule = [
        shot_rng.choice(nshots, size=batchsize, replace=False).astype(np.int32)
        for _ in range(cfg["epochs"])
    ]

    for epoch in tqdm.trange(cfg["epochs"]):
        shot_idx = jnp.asarray(shot_schedule[epoch], dtype=jnp.int32)
        if cfg["source_encoding"]:
            signs = jnp.asarray(
                shot_rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(batchsize, 1)),
                dtype=jnp.float32,
            )
        else:
            signs = jnp.ones((batchsize, 1), dtype=jnp.float32)

        loss, grads = fwi_step(vp, vs, rho, shot_idx, signs, wave_jax, sources_jax, receivers_jax, receivers_shared, obs_jax)
        grad_vp, grad_vs, grad_rho = grads
        loss_value = float(loss)

        vp, opt_states[0] = update_vp(vp, grad_vp, opt_states[0])
        vs, opt_states[1] = update_vs(vs, grad_vs, opt_states[1])
        rho, opt_states[2] = update_rho(rho, grad_rho, opt_states[2])

        losses.append(loss_value)
        mode = "encoded" if cfg["source_encoding"] else "batched"
        print(f"Epoch {epoch:04d} | Loss: {loss_value:.6e} | Shots: {batchsize} | Mode: {mode}")

        if epoch % cfg["show_every"] == 0:
            save_progress_figure(
                vp_true,
                vs_true,
                np.asarray(vp),
                np.asarray(vs),
                np.asarray(grad_vp),
                np.asarray(grad_vs),
                losses,
                epoch,
                cfg,
                output_dir,
            )


if __name__ == "__main__":
    main()
