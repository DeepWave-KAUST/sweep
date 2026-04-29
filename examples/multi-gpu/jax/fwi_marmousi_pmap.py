import os
from pathlib import Path
import argparse
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


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
import matplotlib
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
import tqdm

import configure_marmousi as shared_config
from sweep.equations import Acoustic
from sweep.propagator.jax import PropJax
from sweep.signal import ricker


def build_config():
    cfg = shared_config.get_config("fwi_2d_acoustic_jax")
    cfg["output_dir"] = "multi_gpu_acoustic_fwi_jax"
    return cfg


def build_solver(shape, cfg):
    equation = Acoustic(
        spatial_order=cfg["spatial_order"],
        backend="jax",
    )
    return PropJax(
        equation,
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
    return ricker(t - cfg["delay"], f=cfg["fm"]).astype(np.float32)


def device_axis_sharding(devices):
    return NamedSharding(Mesh(np.asarray(devices), ("devices",)), P("devices"))


def replicate_array(array, ndevices, sharding):
    stacked = np.stack([np.asarray(array, dtype=np.float32)] * ndevices)
    return jax.device_put(stacked, sharding)


def shard_batch(array, sharding):
    return jax.device_put(np.asarray(array), sharding)


def save_wavelet_figure(wave, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(np.asarray(wave), color="black")
    ax.set_title("Ricker Wavelet")
    fig.tight_layout()
    fig.savefig(output_dir / "ricker.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_observed_figure(obs, receivers, cfg, output_dir):
    shot = np.asarray(obs[-1].squeeze())
    vmin, vmax = np.percentile(shot, [2, 98])
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    im = ax.imshow(
        shot,
        vmin=vmin,
        vmax=vmax,
        cmap="seismic",
        aspect="auto",
        extent=gather_extent(shot.shape[0], cfg["dt"], receivers, cfg["dh"]),
    )
    fig.colorbar(im, ax=ax, shrink=0.9, label="Amplitude")
    ax.set_title("Observed Shot Gather")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "observed_data.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_progress_figure(true_model, vp, grad, losses, epoch, cfg, output_dir):
    nz, nx = true_model.shape
    extent = [0, nx * cfg["dh"], nz * cfg["dh"], 0]
    vmin_model, vmax_model = true_model.min(), true_model.max()
    vmin_grad, vmax_grad = np.percentile(grad, [2, 98])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        ("True Model", true_model, (vmin_model, vmax_model), "Velocity (m/s)"),
        ("Inverted Model", vp, (vmin_model, vmax_model), "Velocity (m/s)"),
        ("Gradient", grad, (vmin_grad, vmax_grad), "Gradient"),
    ]
    for ax, (title, data, limits, label) in zip(axes, panels):
        im = ax.imshow(data, vmin=limits[0], vmax=limits[1], cmap="seismic", aspect="auto", extent=extent)
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        fig.colorbar(im, ax=ax, shrink=0.85, label=label)
    fig.tight_layout()
    fig.savefig(output_dir / f"epoch_{epoch:04d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plot_loss_curve(ax, losses, "Mean Squared Error")
    ax.set_title("JAX pmap FWI Loss")
    fig.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_fwi(args):
    cfg = build_config()
    cfg["epochs"] = args.epochs if args.epochs is not None else cfg["epochs"]
    cfg["batchsize"] = args.batchsize if args.batchsize is not None else cfg["batchsize"]
    cfg["show_every"] = args.show_every if args.show_every is not None else cfg["show_every"]

    devices = jax.local_devices()
    ndevices = len(devices)
    if ndevices < 1:
        raise RuntimeError("No JAX devices are available.")
    sharding = device_axis_sharding(devices)

    output_dir = Path(__file__).resolve().parent / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    true_model = np.load(EXAMPLES_DIR / cfg["true_model"]).astype(np.float32)
    init_model = np.load(EXAMPLES_DIR / cfg["init_model"]).astype(np.float32)
    shape = true_model.shape
    sources, receivers = build_geometry(shape, cfg)
    if args.max_shots is not None:
        if args.max_shots <= 0:
            raise ValueError("--max-shots must be positive when provided.")
        sources = sources[: args.max_shots]
        receivers = receivers[: args.max_shots]

    wave = build_wavelet(cfg)
    save_wavelet_figure(wave, output_dir)

    solver = build_solver(shape, cfg)

    print(f"JAX devices: {ndevices}")
    print("(nshots, ndim):", sources.shape)
    print("(nshots, nreceivers, ndim):", receivers.shape)

    true_vp = jnp.asarray(true_model)
    obs_device = solver(jnp.asarray(wave), sources, receivers, models=[true_vp])
    obs = np.asarray(jax.device_get(obs_device))
    save_observed_figure(obs, receivers, cfg, output_dir)
    del true_vp, obs_device

    nshots = sources.shape[0]
    batchsize = min(cfg["batchsize"], nshots)
    if batchsize % ndevices != 0:
        raise ValueError(
            f"Global batchsize {batchsize} must be divisible by the number of JAX devices {ndevices}. "
            "Pass --batchsize with a divisible value."
        )
    shots_per_device = batchsize // ndevices

    optimizer = optax.adam(cfg["lr"], eps=1e-22)
    params = replicate_array(init_model, ndevices, sharding)
    opt_state = jax.pmap(optimizer.init)(params)
    wave_shards = replicate_array(wave, ndevices, sharding)
    losses = []

    def device_step(params_shard, opt_state_shard, wavelet, sources_shard, receivers_shard, obs_shard):
        def local_loss_sum(current_params):
            syn = solver(
                wavelet,
                sources=sources_shard,
                receivers=receivers_shard,
                models=[current_params],
            )
            residual = syn - obs_shard
            local_numel = jnp.asarray(residual.size, dtype=jnp.float32)
            return jnp.sum(residual ** 2), local_numel

        (loss_sum, local_numel), local_grad = jax.value_and_grad(local_loss_sum, has_aux=True)(params_shard)
        global_loss_sum = jax.lax.psum(loss_sum, axis_name="devices")
        global_numel = jax.lax.psum(local_numel, axis_name="devices")
        grad = jax.lax.psum(local_grad, axis_name="devices") / global_numel
        updates, opt_state_shard = optimizer.update(grad, opt_state_shard, params_shard)
        params_shard = optax.apply_updates(params_shard, updates)
        return params_shard, opt_state_shard, global_loss_sum / global_numel, grad

    p_fwi_step = jax.pmap(
        device_step,
        axis_name="devices",
        in_axes=(0, 0, 0, 0, 0, 0),
    )

    for epoch in tqdm.trange(cfg["epochs"]):
        shot_idx = np.random.choice(nshots, size=batchsize, replace=False).astype(np.int32)
        shot_idx = shot_idx.reshape(ndevices, shots_per_device)
        sources_shards = shard_batch(sources[shot_idx], sharding)
        receivers_shards = shard_batch(receivers[shot_idx], sharding)
        obs_shards = shard_batch(obs[shot_idx], sharding)
        params, opt_state, loss, grads = p_fwi_step(
            params,
            opt_state,
            wave_shards,
            sources_shards,
            receivers_shards,
            obs_shards,
        )

        loss_value = float(jax.device_get(loss[0]))
        losses.append(loss_value)
        print(f"[jax pmap] Epoch {epoch:04d} | Loss: {loss_value:.6e}")

        if epoch % cfg["show_every"] == 0:
            save_progress_figure(
                true_model,
                np.asarray(jax.device_get(params[0])),
                np.asarray(jax.device_get(grads[0])),
                losses,
                epoch,
                cfg,
                output_dir,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="JAX pmap multi-GPU acoustic FWI on Marmousi.")
    parser.add_argument(
        "--import-mode",
        choices=("env", "source"),
        default=IMPORT_MODE,
        help="Load sweep from the current environment or from the repository source tree.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override the shared Marmousi epoch count.")
    parser.add_argument("--batchsize", type=int, default=None, help="Override the global shot batch size.")
    parser.add_argument("--show-every", type=int, default=None, help="Override progress figure interval.")
    parser.add_argument(
        "--max-shots",
        type=int,
        default=None,
        help="Use only the first N shots; intended for quick smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_fwi(args)


if __name__ == "__main__":
    main()
