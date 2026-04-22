from pathlib import Path
import argparse
import os
import sys


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

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import optax
import tqdm

import configure_marmousi as shared_config
from sweep.equations import Acoustic, AcousticLSRTM
from sweep.propagator.jax import PropJax
from sweep.signal import ricker


def build_config():
    cfg = shared_config.get_config("fwi_2d_acoustic_torch_common")
    cfg.update(
        {
            "output_dir": "acoustic_lsrtm_jax",
            "fm": 10.0,
            "lr_ref": 0.01,
        }
    )
    return cfg


def build_geometry(shape, cfg):
    _, nx = shape

    src_x = np.arange(0, nx, cfg["src_step"], dtype=np.int64).reshape(-1, 1)
    src_z = np.full_like(src_x, cfg["srcz"])
    sources = np.concatenate([src_x, src_z], axis=1)

    rec_x = np.arange(0, nx, cfg["rec_step"], dtype=np.int64).reshape(-1, 1)
    rec_z = np.full_like(rec_x, cfg["recz"])
    receivers = np.concatenate([rec_x, rec_z], axis=1)
    receivers = receivers[None, ...].repeat(sources.shape[0], axis=0)
    return jnp.asarray(sources), jnp.asarray(receivers)


def build_wavelet(cfg):
    t = np.arange(0, cfg["nt"] * cfg["dt"], cfg["dt"], dtype=np.float32)
    wave = ricker(t - cfg["delay"], f=cfg["fm"]).astype(np.float32)
    return t, wave


def build_acoustic_solver(shape, cfg):
    return PropJax(
        Acoustic(spatial_order=cfg["spatial_order"], backend="jax"),
        shape=shape,
        dev=None,
        dh=cfg["dh"],
        dt=cfg["dt"],
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=cfg["abcn"],
        pml_type="cpmlr",
        free_surface=cfg["free_surface"],
    )


def build_lsrtm_solver(shape, cfg):
    return PropJax(
        AcousticLSRTM(spatial_order=cfg["spatial_order"], backend="jax"),
        shape=shape,
        dev=None,
        dh=cfg["dh"],
        dt=cfg["dt"],
        source_type=["h1"],
        receiver_type=["sh1"],
        abcn=cfg["abcn"],
        pml_type="cpmlr",
        free_surface=cfg["free_surface"],
    )


def save_wavelet_figure(wave, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(wave, color="black")
    ax.set_title("Ricker Wavelet")
    fig.tight_layout()
    fig.savefig(output_dir / "ricker.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_observed_figure(obs, output_dir):
    shot = np.asarray(obs[-1]).squeeze().T
    vmin, vmax = np.percentile(shot, [2, 98])
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    im = ax.imshow(shot, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.9, label="Amplitude")
    ax.set_title("Scattered Observed Shot Gather")
    ax.set_xlabel("Receiver Index")
    ax.set_ylabel("Time Sample")
    fig.tight_layout()
    fig.savefig(output_dir / "observed_data.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_progress_figure(ref, grad_ref, losses, epoch, cfg, output_dir):
    nz, nx = ref.shape
    extent = [0, nx * cfg["dh"], nz * cfg["dh"], 0]
    vmin_ref, vmax_ref = np.percentile(ref, [2, 98])
    vmin_grad, vmax_grad = np.percentile(grad_ref, [2, 98])

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im0 = axes[0].imshow(ref, vmin=vmin_ref, vmax=vmax_ref, cmap="gray", aspect="auto", extent=extent)
    axes[0].set_title("Reflectivity")
    im1 = axes[1].imshow(grad_ref, vmin=vmin_grad, vmax=vmax_grad, cmap="gray", aspect="auto", extent=extent)
    axes[1].set_title("Reflectivity Gradient")
    for ax in axes:
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="Reflectivity")
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="Gradient")
    fig.tight_layout()
    fig.savefig(output_dir / f"epoch_{epoch:04d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(losses, color="black", label="Loss")
    ax.legend()
    ax.set_title("LSRTM Loss")
    fig.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_observed_data(acoustic_solver, wave, sources, receivers, true_model, smooth_model):
    acoustic_solver.set_parameters([jnp.asarray(true_model, dtype=jnp.float32)])
    obs = acoustic_solver.forward(wave, sources, receivers)

    acoustic_solver.set_parameters([jnp.asarray(smooth_model, dtype=jnp.float32)])
    background = acoustic_solver.forward(wave, sources, receivers)
    return obs - background


def build_update_fn(lr):
    opt = optax.adam(lr, eps=1e-22)

    @jax.jit
    def update_fn(param, grads, state):
        updates, state = opt.update(grads, state)
        param = optax.apply_updates(param, updates)
        return param, state

    return opt, update_fn


def build_step_fn(lsrtm_solver):
    @jax.jit
    def lsrtm_step(vp, ref, wave_batch, source_batch, receiver_batch, obs_batch):
        def loss_fn(vp_, ref_):
            syn = lsrtm_solver(
                wave_batch,
                sources=source_batch,
                receivers=receiver_batch,
                models=[vp_, ref_],
            )
            loss = jnp.mean((syn.reshape(obs_batch.shape) - obs_batch) ** 2)
            return loss, syn

        (loss, syn), gradients = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(vp, ref)
        return loss, gradients, syn

    return lsrtm_step


def run_lsrtm(epochs=None):
    cfg = build_config()
    if epochs is not None:
        cfg["epochs"] = int(epochs)

    output_dir = Path(__file__).resolve().parent / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)
    key = random.PRNGKey(0)

    true_model = np.load(EXAMPLES_DIR / cfg["true_model"]).astype(np.float32)
    smooth_model = np.load(EXAMPLES_DIR / cfg["init_model"]).astype(np.float32)
    shape = true_model.shape

    acoustic_solver = build_acoustic_solver(shape, cfg)
    lsrtm_solver = build_lsrtm_solver(shape, cfg)

    _, wave_np = build_wavelet(cfg)
    wave = jnp.asarray(wave_np)
    save_wavelet_figure(wave_np, output_dir)

    sources, receivers = build_geometry(shape, cfg)
    print("(nshots, ndim):", sources.shape)
    print("(nshots, nreceivers, ndim):", receivers.shape)

    obs = generate_observed_data(acoustic_solver, wave, sources, receivers, true_model, smooth_model)
    save_observed_figure(np.asarray(obs), output_dir)

    vp = jnp.asarray(smooth_model, dtype=jnp.float32)
    ref = jnp.zeros_like(vp)
    opt, update_fn = build_update_fn(cfg["lr_ref"])
    opt_state = opt.init(ref)
    lsrtm_step = build_step_fn(lsrtm_solver)

    losses = []
    nshots = int(sources.shape[0])
    batchsize = min(cfg["batchsize"], nshots)

    for epoch in tqdm.trange(cfg["epochs"]):
        key, subkey = random.split(key)
        rand_shots = random.randint(subkey, (batchsize,), 0, nshots)
        shot_sources = sources[rand_shots]
        shot_receivers = receivers[rand_shots]
        shot_obs = obs[rand_shots]

        loss, grads, _ = lsrtm_step(vp, ref, wave, shot_sources, shot_receivers, shot_obs)
        grad_ref = grads[-1]
        ref, opt_state = update_fn(ref, grad_ref, opt_state)

        loss_value = float(loss.item())
        losses.append(loss_value)
        print(f"[jax] Epoch {epoch:04d} | Loss: {loss_value:.6e}")

        if epoch % cfg["show_every"] == 0:
            save_progress_figure(np.asarray(ref), np.asarray(grad_ref), losses, epoch, cfg, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Acoustic LSRTM example for JAX.")
    parser.add_argument(
        "--import-mode",
        choices=("env", "source"),
        default=IMPORT_MODE,
        help="Load sweep from the current environment or from the repository source tree.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the default number of epochs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_lsrtm(epochs=args.epochs)


if __name__ == "__main__":
    main()
