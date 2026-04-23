import os
from pathlib import Path
import argparse
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ.setdefault("SWEEP_IMPORT_MODE", "source")


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
import numpy as np
import optax
import matplotlib.pyplot as plt
import tqdm

import configure_overthrust as shared_config
from fwi3d_overthrust import build_geometry, build_wavelet, load_models, save_wavelet_figure
from sweep.equations import Acoustic3D, AcousticLSRTM3D
from sweep.propagator.jax import PropJax
from sweep.signal import ricker


CONFIG = shared_config.get_config("lsrtm_3d_acoustic_jax")


def build_acoustic_solver(shape, cfg):
    equation = Acoustic3D(
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
        ckpt_chunks=cfg["ckpt_chunks"],
        pml_type="cpmlr",
    )


def build_lsrtm_solver(shape, cfg):
    equation = AcousticLSRTM3D(
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
        receiver_type=["sh1"],
        abcn=cfg["abcn"],
        free_surface=cfg["free_surface"],
        use_ckpt=cfg["use_ckpt"],
        ckpt_chunks=cfg["ckpt_chunks"],
        pml_type="cpmlr",
    )


def forward_batched(solver, wave, sources, receivers, models, shot_batchsize):
    nshots = int(sources.shape[0])
    shot_batchsize = max(1, min(int(shot_batchsize), nshots))
    outputs = []

    # Warm up the first shape once so the initial JIT/launch latency does not
    # show up as a long pause inside the progress loop.
    warm_stop = min(shot_batchsize, nshots)
    warm_batch = solver(
        wave,
        sources[:warm_stop],
        receivers[:warm_stop],
        models=models,
    )
    warm_batch = jax.block_until_ready(warm_batch)
    outputs.append(np.asarray(warm_batch))

    for start in tqdm.trange(warm_stop, nshots, shot_batchsize, desc="Forward shots", unit="batch"):
        stop = min(start + shot_batchsize, nshots)
        batch = solver(
            wave,
            sources[start:stop],
            receivers[start:stop],
            models=models,
        )
        outputs.append(np.asarray(batch))
    return np.concatenate(outputs, axis=0)


def save_observed_figure(obs, receivers, cfg, output_dir):
    arr = np.asarray(obs)
    if arr.ndim == 4:
        shot = arr[-1, :, :, 0]
    elif arr.ndim == 3:
        shot = arr[-1]
    else:
        raise ValueError(f"Expected scattered data with 3 or 4 dims, got {arr.shape}")
    vmin, vmax = np.percentile(shot, [2, 98])
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    im = ax.imshow(
        shot,
        vmin=vmin,
        vmax=vmax,
        cmap="seismic",
        aspect="auto",
        extent=gather_extent(shot.shape[0], cfg["dt"], receivers, cfg["dh"]),
    )
    fig.colorbar(im, ax=ax, shrink=0.9, label="Amplitude")
    ax.set_title("Observed Scattered Shot Gather")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "observed_data.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _middle_slices(volume):
    nz, ny, nx = volume.shape
    return [
        ("XY @ Zmid", volume[nz // 2], "X", "Y"),
        ("XZ @ Ymid", volume[:, ny // 2, :], "X", "Z"),
        ("YZ @ Xmid", volume[:, :, nx // 2], "Y", "Z"),
    ]


def _slice_extent(shape, dh, plane):
    nz, ny, nx = shape
    extents = {
        "XY @ Zmid": [0.0, nx * dh, ny * dh, 0.0],
        "XZ @ Ymid": [0.0, nx * dh, nz * dh, 0.0],
        "YZ @ Xmid": [0.0, ny * dh, nz * dh, 0.0],
    }
    return extents[plane]


def save_progress_figure(ref, grad, losses, epoch, cfg, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    dh = float(cfg["dh"])
    ref_limits = tuple(np.percentile(ref, [2, 98]))
    grad_limits = tuple(np.percentile(grad, [2, 98]))
    panels = [
        ("Reflectivity", ref, ref_limits, "Reflectivity"),
        ("Reflectivity Gradient", grad, grad_limits, "Gradient"),
    ]
    for row, (row_title, volume, limits, colorbar_label) in enumerate(panels):
        for ax, (slice_title, data, xlabel, ylabel) in zip(axes[row], _middle_slices(volume)):
            im = ax.imshow(
                data,
                vmin=limits[0],
                vmax=limits[1],
                cmap="gray",
                aspect="auto",
                extent=_slice_extent(volume.shape, dh, slice_title),
            )
            ax.set_title(f"{row_title}: {slice_title}")
            ax.set_xlabel(f"{xlabel} (m)")
            ax.set_ylabel(f"{ylabel} (m)")
            fig.colorbar(im, ax=ax, shrink=0.8, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(output_dir / f"epoch_{epoch:04d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plot_loss_curve(ax, losses, "Mean Squared Error")
    ax.set_title("3D LSRTM Loss")
    fig.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_observed_data(acoustic_solver, wave, sources, receivers, true_model, smooth_model, cfg):
    true_vp = jnp.asarray(true_model, dtype=jnp.float32)
    smooth_vp = jnp.asarray(smooth_model, dtype=jnp.float32)
    forward_batchsize = min(int(cfg["forward_batchsize"]), int(sources.shape[0]))
    if forward_batchsize >= int(sources.shape[0]):
        obs = np.asarray(acoustic_solver(wave, sources, receivers, models=[true_vp]))
        background = np.asarray(acoustic_solver(wave, sources, receivers, models=[smooth_vp]))
    else:
        obs = forward_batched(acoustic_solver, wave, sources, receivers, [true_vp], forward_batchsize)
        background = forward_batched(acoustic_solver, wave, sources, receivers, [smooth_vp], forward_batchsize)
    return jnp.asarray(obs - background, dtype=jnp.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="3D acoustic LSRTM on the Overthrust model for the JAX propagator.")
    parser.add_argument(
        "--import-mode",
        choices=("env", "source"),
        default=IMPORT_MODE,
        help="Load sweep from the current environment or from the repository source tree.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override the default number of epochs.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = CONFIG.copy()
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)

    output_dir = Path(__file__).resolve().parent / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(0)

    true_model, smooth_model = load_models(EXAMPLES_DIR, cfg)
    shape = true_model.shape
    acoustic_solver = build_acoustic_solver(shape, cfg)
    lsrtm_solver = build_lsrtm_solver(shape, cfg)

    _, wave_np = build_wavelet(cfg, ricker)
    wave = jnp.asarray(wave_np)
    save_wavelet_figure(wave_np, output_dir)

    sources_np, receivers_np = build_geometry(shape, cfg)
    sources = jnp.asarray(sources_np)
    receivers = jnp.asarray(receivers_np)
    print("prepared model shape:", shape)
    print("(nshots, ndim):", sources.shape)
    print("(nshots, nreceivers, ndim):", receivers.shape)

    obs = generate_observed_data(acoustic_solver, wave, sources, receivers, true_model, smooth_model, cfg)
    save_observed_figure(np.asarray(obs), receivers_np, cfg, output_dir)

    vp = jnp.asarray(smooth_model, dtype=jnp.float32)
    ref = jnp.zeros_like(vp)

    optimizer = optax.adam(cfg["lr_ref"], eps=1e-22)
    opt_state = optimizer.init(ref)
    losses = []
    nshots = int(sources.shape[0])
    batchsize = min(int(cfg["batchsize"]), nshots)

    @jax.jit
    def update_fn(params, grads, state):
        updates, state = optimizer.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state

    @jax.jit
    def lsrtm_step(current_ref, shot_idx, wavelet, all_sources, all_receivers, all_obs, background_vp):
        def loss_fn(reflectivity):
            syn = lsrtm_solver(
                wavelet,
                sources=all_sources[shot_idx],
                receivers=all_receivers[shot_idx],
                models=[background_vp, reflectivity],
            )
            return jnp.mean((syn - all_obs[shot_idx]) ** 2)

        loss, grad = jax.value_and_grad(loss_fn)(current_ref)
        return loss, grad

    for epoch in tqdm.trange(cfg["epochs"]):
        shot_idx = np.random.choice(nshots, size=batchsize, replace=False).astype(np.int32)
        shot_idx = jnp.asarray(shot_idx)
        loss, grad_ref = lsrtm_step(ref, shot_idx, wave, sources, receivers, obs, vp)
        ref, opt_state = update_fn(ref, grad_ref, opt_state)

        loss_value = float(loss)
        losses.append(loss_value)
        print(f"[jax] Epoch {epoch:04d} | Loss: {loss_value:.6e}")

        if epoch % cfg["show_every"] == 0:
            save_progress_figure(
                np.asarray(ref),
                np.asarray(grad_ref),
                losses,
                epoch,
                cfg,
                output_dir,
            )


if __name__ == "__main__":
    main()
