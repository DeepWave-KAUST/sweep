from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def prepare_model(model, cfg):
    stride_z = int(cfg.get("model_stride_z", 1))
    stride_y = int(cfg.get("model_stride_y", 1))
    stride_x = int(cfg.get("model_stride_x", 1))
    prepared = model[::stride_z, ::stride_y, ::stride_x]
    return np.ascontiguousarray(prepared.astype(np.float32))


def load_models(examples_dir, cfg):
    true_path = examples_dir / cfg["true_model"]
    init_path = examples_dir / cfg["init_model"]
    if not true_path.exists() or not init_path.exists():
        missing = [str(path.relative_to(examples_dir)) for path in (true_path, init_path) if not path.exists()]
        raise FileNotFoundError(
            "Missing Overthrust 3D model files: "
            + ", ".join(missing)
            + ". Generate them with the scripts in examples/models/overthrust/README.md."
        )

    true_model = prepare_model(np.load(true_path), cfg)
    init_model = prepare_model(np.load(init_path), cfg)
    return true_model, init_model


def build_geometry(shape, cfg):
    nz, ny, nx = shape

    src_x = np.arange(cfg["src_margin"], max(nx - cfg["src_margin"], cfg["src_margin"] + 1), cfg["src_step"], dtype=np.int32)
    src_y = np.arange(cfg["src_margin"], max(ny - cfg["src_margin"], cfg["src_margin"] + 1), cfg["src_step"], dtype=np.int32)
    if src_x.size == 0:
        src_x = np.array([nx // 2], dtype=np.int32)
    if src_y.size == 0:
        src_y = np.array([ny // 2], dtype=np.int32)
    src_grid_x, src_grid_y = np.meshgrid(src_x, src_y, indexing="xy")
    src_z = np.full(src_grid_x.size, cfg["srcz"], dtype=np.int32)
    sources = np.stack([src_grid_x.reshape(-1), src_grid_y.reshape(-1), src_z], axis=-1)

    rec_x = np.arange(cfg["rec_margin"], max(nx - cfg["rec_margin"], cfg["rec_margin"] + 1), cfg["rec_step"], dtype=np.int32)
    rec_y = np.arange(cfg["rec_margin"], max(ny - cfg["rec_margin"], cfg["rec_margin"] + 1), cfg["rec_step"], dtype=np.int32)
    if rec_x.size == 0:
        rec_x = np.array([nx // 2], dtype=np.int32)
    if rec_y.size == 0:
        rec_y = np.array([ny // 2], dtype=np.int32)
    rec_grid_x, rec_grid_y = np.meshgrid(rec_x, rec_y, indexing="xy")
    rec_z = np.full(rec_grid_x.size, cfg["recz"], dtype=np.int32)
    receivers = np.stack([rec_grid_x.reshape(-1), rec_grid_y.reshape(-1), rec_z], axis=-1)
    receivers = receivers[None, ...].repeat(sources.shape[0], axis=0)

    return sources.astype(np.int32), receivers.astype(np.int32)


def build_wavelet(cfg, ricker_fn):
    t = np.arange(0, cfg["nt"] * cfg["dt"], cfg["dt"], dtype=np.float32)
    wave = ricker_fn(t - cfg["delay"], f=cfg["fm"]).astype(np.float32)
    return t, wave


def save_wavelet_figure(wave, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(wave, color="black")
    ax.set_title("Ricker Wavelet")
    ax.set_xlabel("Time Sample")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(output_dir / "ricker.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_observed_figure(obs, output_dir):
    arr = np.asarray(obs)
    if arr.ndim == 4:
        shot = arr[-1, :, :, 0]
    elif arr.ndim == 3:
        # CUDA single-channel acoustic output omits the trailing channel axis.
        shot = arr[-1]
    else:
        raise ValueError(f"Expected observed data with 3 or 4 dims, got shape {arr.shape}")
    vmin, vmax = np.percentile(shot, [2, 98])
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    im = ax.imshow(shot, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.9, label="Amplitude")
    ax.set_title("Observed Shot Gather")
    ax.set_xlabel("Receiver Index")
    ax.set_ylabel("Time Sample")
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


def save_progress_figure(true_model, vp, grad, losses, epoch, cfg, output_dir):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    dh = float(cfg["dh"])
    model_limits = (float(true_model.min()), float(true_model.max()))
    grad_limits = tuple(np.percentile(grad, [2, 98]))
    panels = [
        ("True Model", true_model, model_limits, "Velocity (m/s)"),
        ("Inverted Model", vp, model_limits, "Velocity (m/s)"),
        ("Gradient", grad, grad_limits, "Gradient"),
    ]

    for row, (row_title, volume, limits, colorbar_label) in enumerate(panels):
        for ax, (slice_title, data, xlabel, ylabel) in zip(axes[row], _middle_slices(volume)):
            im = ax.imshow(
                data,
                vmin=limits[0],
                vmax=limits[1],
                cmap="seismic",
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
    ax.plot(losses, color="black")
    ax.set_title("FWI Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
