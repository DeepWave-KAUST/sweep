from pathlib import Path
import argparse
import os
import sys
import time


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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import tqdm

import configure_marmousi as shared_config
from sweep.equations import Acoustic
from sweep.propagator.options import BoundaryOptions, CUDAOptions, EagerOptions, MemoryOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


torch.backends.cudnn.benchmark = True


def setup_distributed():
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Run this script with torchrun so RANK and WORLD_SIZE are defined.")
    if not torch.cuda.is_available():
        raise RuntimeError("The distributed Torch FWI example requires CUDA.")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible."
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), world_size, torch.device(f"cuda:{local_rank}")


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_config(backend):
    backend_key = f"fwi_2d_acoustic_torch_{backend}"
    if backend not in ("eager", "cuda"):
        raise ValueError(f"Unsupported backend '{backend}'. Expected one of ['cuda', 'eager'].")
    cfg = shared_config.get_config("fwi_2d_acoustic_torch_common")
    cfg.update(shared_config.get_config(backend_key))
    cfg["backend"] = backend
    cfg["output_dir"] = f"multi_gpu_acoustic_fwi_{backend}"
    return cfg


def build_boundary_options(boundary_cfg):
    kwargs = {"storage": boundary_cfg["storage"]}
    if boundary_cfg["storage"] == "cpu":
        kwargs["transfer_interval"] = boundary_cfg["transfer_interval"]
        kwargs["pinned_memory"] = boundary_cfg["pinned_memory"]
    return BoundaryOptions(**kwargs)


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
            eager_options=EagerOptions(use_compile=cfg["use_compile"]),
            use_ckpt=cfg["use_ckpt"],
        )

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
    return ricker(t - cfg["delay"], f=cfg["fm"]).astype(np.float32)


def chunked_observed_data(solver, wave, sources, receivers, model, forward_batchsize):
    records = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for start in range(0, sources.shape[0], forward_batchsize):
            end = min(start + forward_batchsize, sources.shape[0])
            record = solver(wave, sources[start:end], receivers[start:end], models=[model])
            records.append(record.detach())
    if solver.dev.type == "cuda":
        torch.cuda.synchronize(solver.dev)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return torch.cat(records, dim=0), elapsed_ms


def broadcast_observed_data(rank, device, obs):
    if rank == 0:
        ndim = torch.tensor([obs.dim()], dtype=torch.int64, device=device)
    else:
        ndim = torch.empty(1, dtype=torch.int64, device=device)
    dist.broadcast(ndim, src=0)

    if rank == 0:
        shape = torch.tensor(obs.shape, dtype=torch.int64, device=device)
    else:
        shape = torch.empty(int(ndim.item()), dtype=torch.int64, device=device)
    dist.broadcast(shape, src=0)

    if rank != 0:
        obs = torch.empty(tuple(shape.tolist()), dtype=torch.float32, device=device)
    dist.broadcast(obs, src=0)
    return obs


def split_indices_for_rank(indices, rank, world_size):
    chunks = torch.tensor_split(indices, world_size)
    return chunks[rank].contiguous()


def all_reduce_optional_tensor(tensor, like):
    if isinstance(tensor, torch.Tensor):
        reduced = tensor.detach().clone()
    else:
        reduced = torch.zeros_like(like)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced


def robust_positive_limits(data):
    vmax = float(np.percentile(data, 99))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.max(data)) if data.size else 1.0
    return 0.0, max(vmax, 1e-12)


def save_wavelet_figure(wave, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(wave, color="black")
    ax.set_title("Ricker Wavelet")
    fig.tight_layout()
    fig.savefig(output_dir / "ricker.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


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


def save_progress_figure(true_model, vp, grad, losses, epoch, cfg, output_dir, src_illum, rec_illum):
    nz, nx = true_model.shape
    extent = [0, nx * cfg["dh"], nz * cfg["dh"], 0]
    vmin_model, vmax_model = true_model.min(), true_model.max()
    vmin_grad, vmax_grad = np.percentile(grad, [2, 98])

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    panels = [
        ("True Model", true_model, "seismic", (vmin_model, vmax_model), "Velocity (m/s)"),
        ("Inverted Model", vp, "seismic", (vmin_model, vmax_model), "Velocity (m/s)"),
        ("Gradient", grad, "seismic", (vmin_grad, vmax_grad), "Gradient"),
        ("Source Illumination", src_illum, "magma", robust_positive_limits(src_illum), "Illumination"),
        ("Receiver Illumination", rec_illum, "magma", robust_positive_limits(rec_illum), "Illumination"),
    ]
    for ax, (title, data, cmap, limits, label) in zip(axes, panels):
        im = ax.imshow(data, vmin=limits[0], vmax=limits[1], cmap=cmap, aspect="auto", extent=extent)
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        fig.colorbar(im, ax=ax, shrink=0.85, label=label)
    fig.tight_layout()
    fig.savefig(output_dir / f"epoch_{epoch:04d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plot_loss_curve(ax, losses, "Mean Squared Error")
    ax.set_title("Distributed FWI Loss")
    fig.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_fwi(args):
    rank, world_size, device = setup_distributed()
    try:
        cfg = build_config(args.backend)
        cfg["epochs"] = args.epochs if args.epochs is not None else cfg["epochs"]
        cfg["batchsize"] = args.batchsize if args.batchsize is not None else cfg["batchsize"]
        cfg["show_every"] = args.show_every if args.show_every is not None else cfg["show_every"]
        cfg["forward_batchsize"] = args.forward_batchsize

        output_dir = Path(__file__).resolve().parent / cfg["output_dir"]
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(args.seed)
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

        solver = build_solver(shape, device, cfg)
        true_vp = torch.from_numpy(true_model).to(device)

        if rank == 0:
            save_wavelet_figure(wave, output_dir)
            obs, elapsed_ms = chunked_observed_data(
                solver,
                wave,
                sources,
                receivers,
                true_vp,
                forward_batchsize=cfg["forward_batchsize"],
            )
            print(f"[rank 0] Forward modeling time ({args.backend}): {elapsed_ms:.2f} ms")
        else:
            obs = None
        obs_torch = broadcast_observed_data(rank, device, obs)

        if rank == 0:
            save_observed_figure(obs_torch.detach().cpu().numpy(), receivers, output_dir, cfg)
            print("(nshots, ndim):", sources.shape)
            print("(nshots, nreceivers, ndim):", receivers.shape)
            print(f"world_size: {world_size}")

        inv_vp = torch.from_numpy(init_model).to(device).requires_grad_(True)
        optimizer = torch.optim.Adam([inv_vp], lr=cfg["lr"], eps=1e-22)

        nshots = sources.shape[0]
        batchsize = min(cfg["batchsize"], nshots)
        losses = []
        iterator = tqdm.trange(cfg["epochs"], disable=rank != 0)

        for epoch in iterator:
            optimizer.zero_grad(set_to_none=True)

            if rank == 0:
                shot_idx = torch.from_numpy(
                    np.random.choice(nshots, size=batchsize, replace=False).astype(np.int64)
                ).to(device)
            else:
                shot_idx = torch.empty(batchsize, dtype=torch.int64, device=device)
            dist.broadcast(shot_idx, src=0)
            local_idx = split_indices_for_rank(shot_idx, rank, world_size)

            local_loss_sum = torch.zeros((), device=device)
            local_src_illum = None
            local_rec_illum = None
            global_numel = int(obs_torch[shot_idx].numel())
            if local_idx.numel() > 0:
                local_sources = sources[local_idx.detach().cpu().numpy()]
                local_receivers = receivers[local_idx.detach().cpu().numpy()]
                syn = solver(wave, local_sources, local_receivers, models=[inv_vp])
                residual = syn - obs_torch[local_idx]
                local_loss_sum = residual.pow(2).sum()
                (local_loss_sum / float(global_numel)).backward()
                local_src_illum = getattr(solver, "source_illumination", None)
                local_rec_illum = getattr(solver, "receiver_illumination", None)

            if inv_vp.grad is None:
                inv_vp.grad = torch.zeros_like(inv_vp)
            dist.all_reduce(inv_vp.grad, op=dist.ReduceOp.SUM)

            reduced_loss_sum = local_loss_sum.detach().clone()
            dist.all_reduce(reduced_loss_sum, op=dist.ReduceOp.SUM)
            loss_value = float((reduced_loss_sum / float(global_numel)).detach().cpu())

            src_illum = all_reduce_optional_tensor(local_src_illum, inv_vp)
            rec_illum = all_reduce_optional_tensor(local_rec_illum, inv_vp)

            optimizer.step()
            dist.barrier(device_ids=[device.index])

            if rank == 0:
                losses.append(loss_value)
                iterator.set_description(f"loss={loss_value:.6e}")
                print(f"[dist {args.backend}] Epoch {epoch:04d} | Loss: {loss_value:.6e}")
                if epoch % cfg["show_every"] == 0:
                    save_progress_figure(
                        true_model,
                        inv_vp.detach().cpu().numpy(),
                        inv_vp.grad.detach().cpu().numpy(),
                        losses,
                        epoch,
                        cfg,
                        output_dir,
                        src_illum.detach().cpu().numpy(),
                        rec_illum.detach().cpu().numpy(),
                    )
    finally:
        cleanup_distributed()


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed multi-GPU Torch acoustic FWI on Marmousi.")
    parser.add_argument(
        "--import-mode",
        choices=("env", "source"),
        default=IMPORT_MODE,
        help="Load sweep from the current environment or from the repository source tree.",
    )
    parser.add_argument(
        "--backend",
        choices=("cuda", "eager"),
        default="cuda",
        help="PropTorch backend used independently on each rank.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override the shared Marmousi epoch count.")
    parser.add_argument("--batchsize", type=int, default=None, help="Override the global shot batch size.")
    parser.add_argument("--show-every", type=int, default=None, help="Override progress figure interval.")
    parser.add_argument(
        "--forward-batchsize",
        type=int,
        default=1,
        help="Rank-0 shot batch size used when generating observed data.",
    )
    parser.add_argument(
        "--max-shots",
        type=int,
        default=None,
        help="Use only the first N shots; intended for quick smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed shared by all ranks.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_fwi(args)


if __name__ == "__main__":
    main()
