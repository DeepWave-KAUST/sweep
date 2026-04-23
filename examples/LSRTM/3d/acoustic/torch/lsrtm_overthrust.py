from pathlib import Path
import argparse
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
from plotting import gather_extent, plot_loss_curve

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

import configure_overthrust as shared_config
from fwi3d_overthrust import build_geometry, build_wavelet, load_models, save_wavelet_figure
from sweep.equations import Acoustic3D, AcousticLSRTM3D
from sweep.propagator.options import BoundaryOptions, CUDAOptions, CkptOptions, EagerOptions, MemoryOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


torch.backends.cudnn.benchmark = True


def build_config(backend):
    backend_key = f"lsrtm_3d_acoustic_torch_{backend}"
    cfg = shared_config.get_config("lsrtm_3d_acoustic_torch_common")
    cfg.update(shared_config.get_config(backend_key))
    cfg["backend"] = backend
    cfg.setdefault("cuda_memory", "bs" if backend == "cuda" else "ckpt")
    return cfg


def build_cuda_options(cfg):
    memory_mode = cfg.get("cuda_memory", "full")
    if memory_mode == "full":
        return CUDAOptions(memory=None)
    if memory_mode == "bs":
        boundary_cfg = cfg["boundary_saving_config"]
        kwargs = {"storage": boundary_cfg["storage"]}
        if boundary_cfg["storage"] == "cpu":
            kwargs["transfer_interval"] = boundary_cfg["transfer_interval"]
            kwargs["pinned_memory"] = boundary_cfg["pinned_memory"]
        return CUDAOptions(memory=MemoryOptions(strategy="boundary", boundary=BoundaryOptions(**kwargs)))
    if memory_mode == "ckpt":
        return CUDAOptions(memory=MemoryOptions(strategy="ckpt", ckpt=CkptOptions(mode="chunk", chunks=cfg["ckpt_chunks"])))
    if memory_mode == "recursive":
        return CUDAOptions(memory=MemoryOptions(strategy="ckpt", ckpt=CkptOptions(mode="recursive", count=cfg["ckpt_num"])))
    raise ValueError(f"Unsupported cuda_memory '{memory_mode}'.")


def build_acoustic_solver(shape, dev, cfg):
    equation = Acoustic3D(spatial_order=cfg["spatial_order"], device=dev, backend="torch")
    common = dict(
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
    if cfg["backend"] == "eager":
        return PropTorch(
            equation,
            backend="eager",
            eager_options=EagerOptions(use_compile=cfg.get("use_compile", True)),
            use_ckpt=cfg.get("use_ckpt", False),
            ckpt_chunks=cfg.get("ckpt_chunks", 16),
            **common,
        )
    return PropTorch(equation, backend="cuda", cuda_options=build_cuda_options(cfg), **common)


def build_lsrtm_solver(shape, dev, cfg):
    equation = AcousticLSRTM3D(spatial_order=cfg["spatial_order"], device=dev, backend="torch")
    common = dict(
        shape=shape,
        dev=dev,
        dh=cfg["dh"],
        dt=cfg["dt"],
        source_type=["h1"],
        receiver_type=["sh1"],
        abcn=cfg["abcn"],
        free_surface=cfg["free_surface"],
        pml_type="cpmlr",
    )
    if cfg["backend"] == "eager":
        return PropTorch(
            equation,
            backend="eager",
            eager_options=EagerOptions(use_compile=cfg.get("use_compile", True)),
            use_ckpt=cfg.get("use_ckpt", False),
            ckpt_chunks=cfg.get("ckpt_chunks", 16),
            **common,
        )
    return PropTorch(equation, backend="cuda", cuda_options=build_cuda_options(cfg), **common)


def save_observed_figure(obs, receivers, cfg, output_dir):
    arr = np.asarray(obs)
    if arr.ndim == 4:
        shot = arr[-1]
    elif arr.ndim == 3:
        shot = arr[-1]
    else:
        raise ValueError(f"Expected scattered data with 3 or 4 dims, got {arr.shape}")
    vmin, vmax = np.percentile(shot, [2, 98])
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    shot_time_major = shot.T
    im = ax.imshow(
        shot_time_major,
        vmin=vmin,
        vmax=vmax,
        cmap="seismic",
        aspect="auto",
        extent=gather_extent(shot_time_major.shape[0], cfg["dt"], receivers, cfg["dh"]),
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


def generate_observed_data(acoustic_solver, wave, sources, receivers, true_model, smooth_model, dev):
    true_vp = torch.from_numpy(true_model).to(dev)
    background_vp = torch.from_numpy(smooth_model).to(dev)
    kwargs = {"use_boundary_saving": False} if getattr(acoustic_solver, "backend", "eager") == "cuda" else {}
    scattered_shots = []
    with torch.no_grad():
        for shot_id in tqdm.trange(sources.shape[0], desc="Forward shots", unit="shot"):
            shot_sources = sources[shot_id : shot_id + 1]
            shot_receivers = receivers[shot_id : shot_id + 1]
            obs = acoustic_solver(wave, shot_sources, shot_receivers, models=[true_vp], **kwargs).detach().clone()
            background = acoustic_solver(
                wave,
                shot_sources,
                shot_receivers,
                models=[background_vp],
                **kwargs,
            ).detach().clone()
            scattered_shots.append((obs - background).detach().cpu())
            del obs, background
    return torch.cat(scattered_shots, dim=0)


def run_lsrtm(backend="eager", cuda_memory=None, epochs=None):
    cfg = build_config(backend)
    if cuda_memory is None:
        cuda_memory = cfg["cuda_memory"]
    cfg["cuda_memory"] = cuda_memory
    if epochs is not None:
        cfg["epochs"] = int(epochs)
    cfg.setdefault("ckpt_num", 6)

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / f"{cfg['output_dir']}_{backend}_{cuda_memory}"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)

    true_model, smooth_model = load_models(EXAMPLES_DIR, cfg)
    shape = true_model.shape

    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("The CUDA 3D LSRTM example requires a CUDA-capable PyTorch environment.")
        dev = torch.device("cuda")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acoustic_solver = build_acoustic_solver(shape, dev, cfg)
    lsrtm_solver = build_lsrtm_solver(shape, dev, cfg)

    _, wave = build_wavelet(cfg, ricker)
    save_wavelet_figure(wave, output_dir)

    sources, receivers = build_geometry(shape, cfg)
    print("(nshots, ndim):", sources.shape)
    print("(nshots, nreceivers, ndim):", receivers.shape)
    print(f"Running backend={backend}, cuda_memory={cuda_memory} on {dev} ...")

    obs = generate_observed_data(acoustic_solver, wave, sources, receivers, true_model, smooth_model, dev)
    save_observed_figure(obs.numpy(), receivers, cfg, output_dir)

    vp = torch.from_numpy(smooth_model).to(dev)
    ref = torch.zeros_like(vp, requires_grad=True)
    optimizer = torch.optim.Adam([ref], lr=cfg["lr_ref"], eps=1e-22)

    losses = []
    nshots = sources.shape[0]
    batchsize = min(cfg["batchsize"], nshots)

    for epoch in tqdm.trange(cfg["epochs"]):
        optimizer.zero_grad()
        shot_idx = np.random.choice(nshots, size=batchsize, replace=False)
        epoch_loss = 0.0
        for shot_id in shot_idx:
            solver_kwargs = {}
            if getattr(lsrtm_solver, "backend", "eager") == "cuda" and cuda_memory == "bs":
                solver_kwargs["use_boundary_saving"] = True
            syn = lsrtm_solver(
                wave,
                sources[shot_id : shot_id + 1],
                receivers[shot_id : shot_id + 1],
                models=[vp, ref],
                **solver_kwargs,
            )
            obs_shot = obs[shot_id : shot_id + 1].to(dev, non_blocking=True)
            shot_loss = (syn - obs_shot).pow(2).mean()
            (shot_loss / batchsize).backward()
            epoch_loss += shot_loss.item()
            del syn, obs_shot, shot_loss
        optimizer.step()

        epoch_loss /= batchsize
        losses.append(epoch_loss)
        print(f"[{backend}/{cuda_memory}] Epoch {epoch:04d} | Loss: {epoch_loss:.6e}")

        if epoch % cfg["show_every"] == 0:
            ref_np = ref.detach().cpu().numpy()
            grad_np = ref.grad.detach().cpu().numpy()
            save_progress_figure(ref_np, grad_np, losses, epoch, cfg, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="3D acoustic LSRTM on the Overthrust model for PyTorch eager and CUDA backends.")
    parser.add_argument(
        "--import-mode",
        choices=("env", "source"),
        default=IMPORT_MODE,
        help="Load sweep from the current environment or from the repository source tree.",
    )
    parser.add_argument("--backend", choices=("eager", "cuda"), default="eager")
    parser.add_argument("--cuda-memory", choices=("full", "bs", "ckpt", "recursive"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_lsrtm(backend=args.backend, cuda_memory=args.cuda_memory, epochs=args.epochs)


if __name__ == "__main__":
    main()
