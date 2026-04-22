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

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

import configure_marmousi as shared_config
from sweep.equations import Acoustic, AcousticLSRTM
from sweep.propagator.options import BoundaryOptions, CUDAOptions, CkptOptions, EagerOptions, MemoryOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


torch.backends.cudnn.benchmark = True


def build_config():
    cfg = shared_config.get_config("fwi_2d_acoustic_torch_common")
    cfg.update(
        {
            "output_dir": "acoustic_lsrtm_torch",
            "backend": "eager",
            "cuda_memory": "full",
            "use_compile": False,
            "use_ckpt": False,
            "boundary_saving_config": {
                "enabled": True,
                "storage": "gpu",
                "transfer_interval": 10,
                "pinned_memory": True,
            },
            "ckpt_chunks": 64,
            "ckpt_num": 6,
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
    return sources, receivers


def build_wavelet(cfg):
    t = np.arange(0, cfg["nt"] * cfg["dt"], cfg["dt"], dtype=np.float32)
    wave = ricker(t - cfg["delay"], f=cfg["fm"]).astype(np.float32)
    return t, wave


def build_acoustic_solver(shape, dev, cfg):
    equation = Acoustic(
        spatial_order=cfg["spatial_order"],
        device=dev,
        backend="torch",
    )

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

    if cfg["backend"] == "eager":
        return PropTorch(
            equation,
            **prop_kwargs,
            backend="eager",
            eager_options=EagerOptions(use_compile=cfg["use_compile"]),
        )

    return PropTorch(
        equation,
        **prop_kwargs,
        backend="cuda",
        cuda_options=build_cuda_options(cfg),
    )


def build_lsrtm_solver(shape, dev, cfg):
    equation = AcousticLSRTM(
        spatial_order=cfg["spatial_order"],
        device=dev,
        backend="torch",
    )

    prop_kwargs = dict(
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
            **prop_kwargs,
            backend="eager",
            eager_options=EagerOptions(use_compile=cfg["use_compile"]),
        )

    return PropTorch(
        equation,
        **prop_kwargs,
        backend="cuda",
        cuda_options=build_cuda_options(cfg),
    )


def build_cuda_options(cfg):
    memory_mode = cfg["cuda_memory"]
    if memory_mode == "full":
        return CUDAOptions(memory=None)
    if memory_mode == "bs":
        return CUDAOptions(
            memory=MemoryOptions(
                strategy="boundary",
                boundary=BoundaryOptions(
                    storage=cfg["boundary_saving_config"]["storage"],
                    transfer_interval=cfg["boundary_saving_config"]["transfer_interval"],
                    pinned_memory=cfg["boundary_saving_config"]["pinned_memory"],
                ),
            )
        )
    if memory_mode == "ckpt":
        return CUDAOptions(
            memory=MemoryOptions(
                strategy="ckpt",
                ckpt=CkptOptions(mode="chunk", chunks=cfg["ckpt_chunks"]),
            )
        )
    if memory_mode == "recursive":
        return CUDAOptions(
            memory=MemoryOptions(
                strategy="ckpt",
                ckpt=CkptOptions(mode="recursive", count=cfg["ckpt_num"]),
            )
        )
    raise ValueError(f"Unsupported cuda_memory '{memory_mode}'.")


def save_wavelet_figure(wave, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(wave, color="black")
    ax.set_title("Ricker Wavelet")
    fig.tight_layout()
    fig.savefig(output_dir / "ricker.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_observed_figure(obs, output_dir):
    shot = obs[-1].squeeze().T
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


def save_progress_figure(true_model, ref, grad_ref, losses, epoch, cfg, output_dir):
    nz, nx = true_model.shape
    extent = [0, nx * cfg["dh"], nz * cfg["dh"], 0]
    vmin_true, vmax_true = true_model.min(), true_model.max()
    vmin_ref, vmax_ref = np.percentile(ref, [2, 98])
    vmin_grad, vmax_grad = np.percentile(grad_ref, [2, 98])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(true_model, vmin=vmin_true, vmax=vmax_true, cmap="seismic", aspect="auto", extent=extent)
    axes[0].set_title("True Velocity")
    im1 = axes[1].imshow(ref, vmin=vmin_ref, vmax=vmax_ref, cmap="gray", aspect="auto", extent=extent)
    axes[1].set_title("Reflectivity")
    im2 = axes[2].imshow(grad_ref, vmin=vmin_grad, vmax=vmax_grad, cmap="gray", aspect="auto", extent=extent)
    axes[2].set_title("Reflectivity Gradient")
    for ax in axes:
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="Velocity (m/s)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="Reflectivity")
    fig.colorbar(im2, ax=axes[2], shrink=0.85, label="Gradient")
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


def generate_observed_data(acoustic_solver, wave, sources, receivers, true_model, smooth_model, dev):
    true_vp = torch.from_numpy(true_model).to(dev)
    background_vp = torch.from_numpy(smooth_model).to(dev)
    with torch.no_grad():
        obs = acoustic_solver(wave, sources, receivers, models=[true_vp]).detach().clone()
        background = acoustic_solver(wave, sources, receivers, models=[background_vp]).detach().clone()
    return (obs - background).detach()


def run_lsrtm(backend="eager", cuda_memory="full", epochs=None):
    cfg = build_config()
    cfg["backend"] = backend
    cfg["cuda_memory"] = cuda_memory
    if epochs is not None:
        cfg["epochs"] = int(epochs)

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / f"{cfg['output_dir']}_{backend}_{cuda_memory}"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)

    true_model = np.load(EXAMPLES_DIR / cfg["true_model"]).astype(np.float32)
    smooth_model = np.load(EXAMPLES_DIR / cfg["init_model"]).astype(np.float32)
    shape = true_model.shape

    if backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("The CUDA LSRTM example requires a CUDA-capable PyTorch environment.")
        dev = torch.device("cuda")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acoustic_solver = build_acoustic_solver(shape, dev, cfg)
    lsrtm_solver = build_lsrtm_solver(shape, dev, cfg)

    _, wave = build_wavelet(cfg)
    save_wavelet_figure(wave, output_dir)

    sources, receivers = build_geometry(shape, cfg)
    print("(nshots, ndim):", sources.shape)
    print("(nshots, nreceivers, ndim):", receivers.shape)
    print(f"Running backend={backend}, cuda_memory={cuda_memory} on {dev} ...")

    obs = generate_observed_data(acoustic_solver, wave, sources, receivers, true_model, smooth_model, dev)
    obs_np = obs.detach().cpu().numpy()
    save_observed_figure(obs_np, output_dir)

    vp = torch.from_numpy(smooth_model).to(dev)
    ref = torch.zeros_like(vp, requires_grad=True)
    optimizer = torch.optim.Adam([ref], lr=cfg["lr_ref"], eps=1e-22)

    losses = []
    nshots = sources.shape[0]
    batchsize = min(cfg["batchsize"], nshots)

    for epoch in tqdm.trange(cfg["epochs"]):
        optimizer.zero_grad()

        shot_idx = np.random.choice(nshots, size=batchsize, replace=False)
        syn = lsrtm_solver(wave, sources[shot_idx], receivers[shot_idx], models=[vp, ref])
        loss = (syn - obs[shot_idx]).pow(2).mean()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"[{backend}/{cuda_memory}] Epoch {epoch:04d} | Loss: {loss.item():.6e}")

        if epoch % cfg["show_every"] == 0:
            ref_np = ref.detach().cpu().numpy()
            grad_np = ref.grad.detach().cpu().numpy()
            save_progress_figure(true_model, ref_np, grad_np, losses, epoch, cfg, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Acoustic LSRTM example for PyTorch eager and CUDA backends.")
    parser.add_argument(
        "--import-mode",
        choices=("env", "source"),
        default=IMPORT_MODE,
        help="Load sweep from the current environment or from the repository source tree.",
    )
    parser.add_argument(
        "--backend",
        choices=("eager", "cuda"),
        default="eager",
        help="Select which PropTorch backend to use.",
    )
    parser.add_argument(
        "--cuda-memory",
        choices=("full", "bs", "ckpt", "recursive"),
        default="full",
        help="CUDA memory strategy. Ignored when --backend eager.",
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
    run_lsrtm(backend=args.backend, cuda_memory=args.cuda_memory, epochs=args.epochs)


if __name__ == "__main__":
    main()
