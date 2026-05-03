from pathlib import Path
import argparse
import os
import time
import sys

_ALLOC_CONF = (
    os.environ.get("PYTORCH_ALLOC_CONF")
    or os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    or "expandable_segments:True"
)
os.environ.setdefault("PYTORCH_ALLOC_CONF", _ALLOC_CONF)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", _ALLOC_CONF)


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

import configure_marmousi as shared_config
from sweep.equations import Acoustic
from sweep.propagator.options import BoundaryOptions, CUDAOptions, EagerOptions, MemoryOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


torch.backends.cudnn.benchmark = True

def build_config(backend):
    backend_key = f"fwi_2d_acoustic_torch_{backend}"
    if backend not in ("eager", "cuda"):
        raise ValueError(f"Unsupported backend '{backend}'. Expected one of ['cuda', 'eager'].")
    cfg = shared_config.get_config("fwi_2d_acoustic_torch_common")
    cfg.update(shared_config.get_config(backend_key))
    cfg["backend"] = backend
    return cfg


def build_solver(shape, dev, cfg):
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
            eager_options=EagerOptions(
                use_compile=cfg["use_compile"],
            ),
            use_ckpt=cfg["use_ckpt"],
            ckpt_chunks=cfg.get("ckpt_chunks", 100),
        )

    if cfg["backend"] == "cuda":
        return PropTorch(
            equation,
            **prop_kwargs,
            backend="cuda",
            cuda_options=CUDAOptions(
                memory=MemoryOptions(
                    strategy="boundary",
                    boundary=build_boundary_options(cfg["boundary_saving_config"]),
                ),
            ),
        )

    raise ValueError(f"Unsupported backend '{cfg['backend']}'.")


def build_boundary_options(boundary_cfg):
    kwargs = {"storage": boundary_cfg["storage"]}
    if boundary_cfg["storage"] == "cpu":
        kwargs["transfer_interval"] = boundary_cfg["transfer_interval"]
        kwargs["pinned_memory"] = boundary_cfg["pinned_memory"]
    return BoundaryOptions(**kwargs)


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
    return t, ricker(t - cfg["delay"], f=cfg["fm"]).astype(np.float32)


def timed_forward(solver, wave, sources, receivers, models):
    kwargs = {}
    if getattr(solver, "backend", "eager") == "cuda":
        kwargs["use_boundary_saving"] = False

    if solver.dev.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            obs = solver(wave, sources, receivers, models=models, **kwargs)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            obs = solver(wave, sources, receivers, models=models, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return obs.detach().cpu().numpy(), elapsed_ms


def timed_forward_batched(solver, wave, sources, receivers, models, shot_batchsize):
    nshots = sources.shape[0]
    shot_batchsize = max(1, min(int(shot_batchsize), nshots))
    obs_batches = []
    batch_starts = list(range(0, nshots, shot_batchsize))
    progress = tqdm.tqdm(batch_starts, desc="Forward shots", unit="batch")

    if solver.dev.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            for start in progress:
                stop = min(start + shot_batchsize, nshots)
                solver_kwargs = {}
                if getattr(solver, "backend", "eager") == "cuda":
                    solver_kwargs["use_boundary_saving"] = False
                batch_obs = solver(
                    wave,
                    sources[start:stop],
                    receivers[start:stop],
                    models=models,
                    **solver_kwargs,
                )
                obs_batches.append(batch_obs.detach().cpu())
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for start in progress:
                stop = min(start + shot_batchsize, nshots)
                batch_obs = solver(
                    wave,
                    sources[start:stop],
                    receivers[start:stop],
                    models=models,
                )
                obs_batches.append(batch_obs.detach().cpu())
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    progress.close()
    obs = torch.cat(obs_batches, dim=0).numpy()
    return obs, elapsed_ms


def save_wavelet_figure(wave, output_dir):
    plt.figure(figsize=(6, 3))
    plt.plot(wave, color="black")
    plt.title("Ricker Wavelet")
    plt.tight_layout()
    plt.savefig(output_dir / "ricker.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_observed_figure(obs, receivers, output_dir, cfg):
    shot = obs[-1].squeeze()
    if cfg.get("transpose_shot", False):
        shot = shot.T
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


def robust_positive_limits(data):
    vmax = float(np.percentile(data, 99))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(data)) if data.size else 1.0
    return 0.0, max(vmax, 1e-12)


def save_progress_figure(
    true_model,
    vp,
    grad,
    losses,
    epoch,
    cfg,
    output_dir,
    source_illumination=None,
    receiver_illumination=None,
):
    nz, nx = true_model.shape
    extent = [0, nx * cfg["dh"], nz * cfg["dh"], 0]
    vmin_model, vmax_model = true_model.min(), true_model.max()
    vmin_grad, vmax_grad = np.percentile(grad, [2, 98])
    show_illumination = source_illumination is not None and receiver_illumination is not None

    ncols = 5 if show_illumination else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    im0 = axes[0].imshow(true_model, vmin=vmin_model, vmax=vmax_model, cmap="seismic", aspect="auto", extent=extent)
    axes[0].set_title("True Model")
    im1 = axes[1].imshow(vp, vmin=vmin_model, vmax=vmax_model, cmap="seismic", aspect="auto", extent=extent)
    axes[1].set_title("Inverted Model")
    im2 = axes[2].imshow(grad, vmin=vmin_grad, vmax=vmax_grad, cmap="seismic", aspect="auto", extent=extent)
    axes[2].set_title("Gradient")
    images = [im0, im1, im2]
    colorbar_labels = ["Velocity (m/s)", "Velocity (m/s)", "Gradient"]

    if show_illumination:
        vmin_src, vmax_src = robust_positive_limits(source_illumination)
        im3 = axes[3].imshow(
            source_illumination,
            vmin=vmin_src,
            vmax=vmax_src,
            cmap="magma",
            aspect="auto",
            extent=extent,
        )
        axes[3].set_title("Source Illumination")
        vmin_rec, vmax_rec = robust_positive_limits(receiver_illumination)
        im4 = axes[4].imshow(
            receiver_illumination,
            vmin=vmin_rec,
            vmax=vmax_rec,
            cmap="magma",
            aspect="auto",
            extent=extent,
        )
        axes[4].set_title("Receiver Illumination")
        images.extend([im3, im4])
        colorbar_labels.extend(["Illumination", "Illumination"])

    for ax in axes:
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
    for ax, image, label in zip(axes, images, colorbar_labels):
        fig.colorbar(image, ax=ax, shrink=0.85, label=label)
    plt.tight_layout()
    plt.savefig(output_dir / f"epoch_{epoch:04d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plot_loss_curve(ax, losses, "Mean Squared Error")
    ax.set_title("FWI Loss")
    plt.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def inversion_step_batched(
    solver,
    wave,
    sources,
    receivers,
    obs_torch,
    inv_vp,
    dev,
    shot_idx,
    train_shot_batchsize,
):
    train_shot_batchsize = max(1, min(int(train_shot_batchsize), shot_idx.shape[0]))
    total_elements = int(obs_torch[shot_idx].numel())
    total_loss_sum = 0.0

    for start in range(0, shot_idx.shape[0], train_shot_batchsize):
        stop = min(start + train_shot_batchsize, shot_idx.shape[0])
        batch_idx = shot_idx[start:stop]
        solver_kwargs = {}
        if getattr(solver, "backend", "eager") == "cuda":
            solver_kwargs["use_boundary_saving"] = True

        syn = solver(wave, sources[batch_idx], receivers[batch_idx], models=[inv_vp], **solver_kwargs)
        obs_batch = obs_torch[batch_idx].to(dev)
        loss_sum = (syn - obs_batch).pow(2).sum()
        (loss_sum / total_elements).backward()
        total_loss_sum += float(loss_sum.detach().cpu())

    return total_loss_sum / total_elements


def run_fwi(
    backend="eager",
    batchsize_override=None,
    train_shot_batchsize_override=None,
    forward_batchsize_override=None,
    epochs_override=None,
    use_compile_override=None,
    use_ckpt_override=None,
    ckpt_chunks_override=None,
):
    cfg = build_config(backend)
    if batchsize_override is not None:
        cfg["batchsize"] = int(batchsize_override)
    if train_shot_batchsize_override is not None:
        cfg["train_shot_batchsize"] = int(train_shot_batchsize_override)
    if forward_batchsize_override is not None:
        cfg["forward_batchsize"] = int(forward_batchsize_override)
    if epochs_override is not None:
        cfg["epochs"] = int(epochs_override)
    if use_compile_override is not None:
        cfg["use_compile"] = bool(use_compile_override)
    if use_ckpt_override is not None:
        cfg["use_ckpt"] = bool(use_ckpt_override)
    if ckpt_chunks_override is not None:
        cfg["ckpt_chunks"] = int(ckpt_chunks_override)

    if int(cfg["batchsize"]) < 1:
        raise ValueError(f"batchsize must be >= 1, got {cfg['batchsize']}.")
    if int(cfg.get("forward_batchsize", 1)) < 1:
        raise ValueError(f"forward_batchsize must be >= 1, got {cfg['forward_batchsize']}.")
    if int(cfg.get("train_shot_batchsize", cfg["batchsize"])) < 1:
        raise ValueError(
            "train_shot_batchsize must be >= 1, "
            f"got {cfg.get('train_shot_batchsize')}."
        )

    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)

    true_model = np.load(EXAMPLES_DIR / cfg["true_model"]).astype(np.float32)
    init_model = np.load(EXAMPLES_DIR / cfg["init_model"]).astype(np.float32)
    shape = true_model.shape

    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The CUDA acoustic FWI example requires a CUDA-capable PyTorch environment.")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = build_solver(shape, dev, cfg)

    _, wave = build_wavelet(cfg)
    save_wavelet_figure(wave, output_dir)

    sources, receivers = build_geometry(shape, cfg)
    print("(nshots, ndim):", sources.shape)
    print("(nshots, nreceivers, ndim):", receivers.shape)

    true_vp = torch.from_numpy(true_model).to(dev)
    forward_batchsize = min(int(cfg.get("forward_batchsize", sources.shape[0])), sources.shape[0])
    if forward_batchsize >= sources.shape[0]:
        obs, elapsed_ms = timed_forward(solver, wave, sources, receivers, models=[true_vp])
        print(f"Forward modeling time ({backend}): {elapsed_ms:.2f} ms")
    else:
        obs, elapsed_ms = timed_forward_batched(
            solver,
            wave,
            sources,
            receivers,
            models=[true_vp],
            shot_batchsize=forward_batchsize,
        )
        print(
            f"Forward modeling time ({backend}, batched {forward_batchsize} shots/step): "
            f"{elapsed_ms:.2f} ms"
        )
    del true_vp
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    save_observed_figure(obs, receivers, output_dir, cfg)

    inv_vp = torch.from_numpy(init_model).to(dev).requires_grad_(True)
    optimizer = torch.optim.Adam([inv_vp], lr=cfg["lr"], eps=1e-22)

    obs_torch = torch.from_numpy(obs)
    losses = []
    nshots = sources.shape[0]
    batchsize = min(cfg["batchsize"], nshots)
    train_shot_batchsize = min(int(cfg.get("train_shot_batchsize", batchsize)), batchsize)
    print(
        "Training shot selection:",
        f"batchsize={batchsize},",
        f"train_shot_batchsize={train_shot_batchsize},",
        f"use_ckpt={cfg.get('use_ckpt', False)},",
        f"ckpt_chunks={cfg.get('ckpt_chunks', None)},",
        f"use_compile={cfg.get('use_compile', False)}",
    )

    for epoch in tqdm.trange(cfg["epochs"]):
        optimizer.zero_grad()

        shot_idx = np.random.choice(nshots, size=batchsize, replace=False)
        loss_value = inversion_step_batched(
            solver,
            wave,
            sources,
            receivers,
            obs_torch,
            inv_vp,
            dev,
            shot_idx,
            train_shot_batchsize,
        )
        optimizer.step()

        losses.append(loss_value)
        print(f"[{backend}] Epoch {epoch:04d} | Loss: {loss_value:.6e}")

        if epoch % cfg["show_every"] == 0:
            vp_np = inv_vp.detach().cpu().numpy()
            grad_np = inv_vp.grad.detach().cpu().numpy()
            source_illumination = getattr(solver, "source_illumination", None)
            receiver_illumination = getattr(solver, "receiver_illumination", None)
            source_illumination_np = (
                source_illumination.detach().cpu().numpy()
                if isinstance(source_illumination, torch.Tensor)
                else None
            )
            receiver_illumination_np = (
                receiver_illumination.detach().cpu().numpy()
                if isinstance(receiver_illumination, torch.Tensor)
                else None
            )
            save_progress_figure(
                true_model,
                vp_np,
                grad_np,
                losses,
                epoch,
                cfg,
                output_dir,
                source_illumination=source_illumination_np,
                receiver_illumination=receiver_illumination_np,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Acoustic FWI example for PyTorch and CUDA propagators.")
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
        "--batchsize",
        type=int,
        default=None,
        help="Override the number of randomly selected shots used in each optimization step.",
    )
    parser.add_argument(
        "--train-shot-batchsize",
        type=int,
        default=None,
        help="Process selected training shots in smaller chunks while accumulating gradients.",
    )
    parser.add_argument(
        "--forward-batchsize",
        type=int,
        default=None,
        help="Generate observed data in shot batches to limit memory use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of FWI optimization epochs.",
    )
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument(
        "--compile",
        dest="use_compile",
        action="store_true",
        default=None,
        help="Enable torch.compile for the eager backend step function.",
    )
    compile_group.add_argument(
        "--no-compile",
        dest="use_compile",
        action="store_false",
        help="Disable torch.compile for the eager backend step function.",
    )
    ckpt_group = parser.add_mutually_exclusive_group()
    ckpt_group.add_argument(
        "--ckpt",
        dest="use_ckpt",
        action="store_true",
        default=None,
        help="Enable eager checkpointing to reduce activation memory.",
    )
    ckpt_group.add_argument(
        "--no-ckpt",
        dest="use_ckpt",
        action="store_false",
        help="Disable eager checkpointing. This can require much more GPU memory.",
    )
    parser.add_argument(
        "--ckpt-chunks",
        type=int,
        default=None,
        help="Override eager checkpoint chunk length in time steps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_fwi(
        backend=args.backend,
        batchsize_override=args.batchsize,
        train_shot_batchsize_override=args.train_shot_batchsize,
        forward_batchsize_override=args.forward_batchsize,
        epochs_override=args.epochs,
        use_compile_override=args.use_compile,
        use_ckpt_override=args.use_ckpt,
        ckpt_chunks_override=args.ckpt_chunks,
    )


if __name__ == "__main__":
    main()
