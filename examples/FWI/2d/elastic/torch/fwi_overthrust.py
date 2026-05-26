from pathlib import Path
import argparse
import time
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
from backend_cli import DEVICES, add_backend_impl_device_args, resolve_backend_impl_device
from mpi_utils import (
    allgather_shot_records,
    allreduce_gradients,
    allreduce_tensors,
    barrier,
    init_mpi,
    rank_zero_print,
    reduce_scalar,
    split_indices,
)

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

import configure_overthrust as shared_config
from sweep.equations import Elastic
from sweep.propagator.options import BoundaryOptions, CUDAOptions, EagerOptions, MemoryOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


torch.backends.cudnn.benchmark = True


def build_config(backend, impl=None, device="auto"):
    backend, impl, device = resolve_backend_impl_device(backend, impl, device)
    backend_key = f"fwi_2d_elastic_torch_{'eager' if impl == 'eager' else device}"
    cfg = shared_config.get_config("fwi_2d_elastic_torch_common")
    cfg.update(shared_config.get_config(backend_key))
    cfg["backend"] = backend
    cfg["impl"] = impl
    cfg["device"] = device
    if impl == "eager" and device == "cpu":
        cfg["output_dir"] = f"{cfg['output_dir']}_cpu"
    return cfg


def select_device(cfg):
    return torch.device(cfg["device"])


def build_solver(shape, dev, cfg):
    equation = Elastic(
        spatial_order=cfg["spatial_order"],
        device=dev,
        backend="torch",
    )

    prop_kwargs = dict(
        shape=shape,
        dev=dev,
        dh=cfg["dh"],
        dt=cfg["dt"],
        source_type=["sxx", "szz"],
        receiver_type=["vx", "vz"],
        abcn=cfg["abcn"],
        free_surface=cfg["free_surface"],
        pml_type="cpmls",
    )

    if cfg["impl"] == "eager":
        return PropTorch(
            equation,
            **prop_kwargs,
            backend="torch",
            impl="eager",
            eager_options=EagerOptions(
                use_compile=cfg["use_compile"],
            ),
            use_ckpt=cfg["use_ckpt"],
            ckpt_chunks=cfg["ckpt_chunks"],
        )

    if cfg["impl"] == "c":
        return PropTorch(
            equation,
            **prop_kwargs,
            backend="torch",
            impl="c",
            cuda_options=CUDAOptions(
                memory=MemoryOptions(
                    strategy="boundary",
                    boundary=build_boundary_options(cfg["boundary_saving_config"]),
                ),
            ),
        )

    raise ValueError(f"Unsupported backend/impl '{cfg['backend']}'.")


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
    return t, ricker(t - cfg["delay"], f=cfg["fm"]).astype(np.float32) * cfg["wavelet_scale"]


def to_standard_layout(record, cfg):
    # All sweep backends (eager + CUDA) now return records in the canonical
    # (nshots, nt, nrec, nfield) layout, so no per-backend permutation is
    # needed. Kept as a thin wrapper for symmetry with older config files
    # that still set ``record_layout``.
    return np.asarray(record)


def to_backend_layout(record, cfg):
    # Inverse of :func:`to_standard_layout`. After backend alignment this is
    # also a no-op — solvers consume the canonical layout directly.
    return np.asarray(record)


def timed_forward(solver, wave, sources, receivers, models, cfg):
    kwargs = {}
    if cfg["impl"] == "c":
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


def observed_shot_axis(cfg):
    # All backends produce records in canonical (nshots, nt, nrec, nfield)
    # layout with the shot index at axis 0.
    return 0


def mpi_forward_batchsize(cfg, local_count):
    batchsize = int(cfg.get("mpi_forward_batchsize", 1))
    return max(1, min(batchsize, max(1, int(local_count))))


def generate_observed_data(solver, wave, sources, receivers, models, cfg, mpi):
    nshots = sources.shape[0]
    if not mpi.enabled:
        return timed_forward(solver, wave, sources, receivers, models=models, cfg=cfg)

    local_idx = split_indices(np.arange(nshots, dtype=np.int64), mpi)
    local_batchsize = mpi_forward_batchsize(cfg, local_idx.size)
    local_rounds = (local_idx.size + local_batchsize - 1) // local_batchsize
    rounds = int(reduce_scalar(local_rounds, mpi, op="max"))
    shot_axis = observed_shot_axis(cfg)
    progress = tqdm.tqdm(
        total=nshots,
        desc="MPI forward shots",
        unit="shot",
        disable=not mpi.is_root,
    )

    local_obs_batches = []
    elapsed_ms = 0.0
    for round_idx in range(rounds):
        start = round_idx * local_batchsize
        stop = min(start + local_batchsize, local_idx.size)
        completed = 0
        if start < local_idx.size:
            batch_idx = local_idx[start:stop]
            batch_obs, batch_elapsed_ms = timed_forward(
                solver,
                wave,
                sources[batch_idx],
                receivers[batch_idx],
                models=models,
                cfg=cfg,
            )
            local_obs_batches.append(batch_obs)
            elapsed_ms += batch_elapsed_ms
            completed = batch_idx.size
        completed = int(reduce_scalar(completed, mpi, op="sum"))
        if mpi.is_root:
            progress.update(completed)
    progress.close()

    local_obs = np.concatenate(local_obs_batches, axis=shot_axis) if local_obs_batches else None
    elapsed_ms = reduce_scalar(elapsed_ms, mpi, op="max")
    obs = allgather_shot_records(
        local_idx,
        local_obs,
        nshots,
        shot_axis=shot_axis,
        mpi=mpi,
    )
    return obs, elapsed_ms


def save_wavelet_figure(wave, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(wave, color="black")
    ax.set_title("Ricker Wavelet")
    ax.set_xlabel("Time Sample")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    fig.savefig(output_dir / "ricker.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_observed_figure(obs, receivers, output_dir, cfg):
    standard_obs = to_standard_layout(obs, cfg)
    vx = standard_obs[-1][..., 0]
    vz = standard_obs[-1][..., 1]

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

    for ax, (title, data, (vmin, vmax), label) in zip(axes.ravel(), panels):
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto", extent=extent)
        fig.colorbar(im, ax=ax, shrink=0.82, label=label)
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")

    fig.tight_layout()
    fig.savefig(output_dir / f"epoch_{epoch:04d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    plot_loss_curve(ax, losses, "Mean Squared Error")
    ax.set_title("FWI Loss")
    fig.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_encoded_batch(wave, obs, shot_idx, cfg):
    standard_obs = to_standard_layout(obs, cfg)
    shot_idx = np.asarray(shot_idx, dtype=np.int64).reshape(-1)
    selected_obs = standard_obs[shot_idx]
    if selected_obs.ndim == 3:
        selected_obs = selected_obs[None, ...]
    signs = np.random.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(shot_idx.size,))
    encoded_wave = signs[:, None] * wave[None, :]
    encoded_obs = np.sum(selected_obs * signs[:, None, None, None], axis=0, keepdims=True)
    if cfg["impl"] == "c":
        encoded_wave = encoded_wave[None, ...]
    return encoded_wave.astype(np.float32), to_backend_layout(encoded_obs, cfg).astype(np.float32)


def select_observed_batch(obs_torch, shot_idx, cfg):
    # Canonical layout puts shot index at axis 0 for every backend.
    return obs_torch[shot_idx]


def run_fwi(backend="torch", impl=None, device="auto", use_mpi=False, mpi_forward_batchsize_override=None):
    cfg = build_config(backend, impl, device)
    mpi = init_mpi(use_mpi, cfg["backend"], cfg["device"], cfg["impl"])
    if mpi_forward_batchsize_override is not None:
        cfg["mpi_forward_batchsize"] = int(mpi_forward_batchsize_override)
    if int(cfg.get("mpi_forward_batchsize", 1)) < 1:
        raise ValueError(f"mpi_forward_batchsize must be >= 1, got {cfg['mpi_forward_batchsize']}.")
    output_dir = Path(__file__).resolve().parent / cfg["output_dir"]
    if mpi.is_root:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(mpi)

    torch.manual_seed(0)
    np.random.seed(0)
    if mpi.enabled:
        rank_zero_print(
            mpi,
            f"MPI enabled: ranks={mpi.size}, backend={cfg['backend']}, impl={cfg['impl']}, device={cfg['device']}, "
            f"mpi_forward_batchsize={cfg.get('mpi_forward_batchsize', 1)}",
        )

    vp_true = np.load(EXAMPLES_DIR / cfg["true_model"]).astype(np.float32)
    vs_true = (vp_true / 1.732).astype(np.float32)
    rho_true = np.ones_like(vp_true, dtype=np.float32) * 1000.0
    vp_init = np.load(EXAMPLES_DIR / cfg["init_model"]).astype(np.float32)
    shape = vp_true.shape

    dev = select_device(cfg)
    solver = build_solver(shape, dev, cfg)
    run_label = f"{cfg['backend']}/{cfg['impl']}/{cfg['device']}"

    _, wave = build_wavelet(cfg)
    if mpi.is_root:
        save_wavelet_figure(wave, output_dir)

    sources, receivers = build_geometry(shape, cfg)
    rank_zero_print(mpi, "(nshots, ndim):", sources.shape)
    rank_zero_print(mpi, "(nshots, nreceivers, ndim):", receivers.shape)
    if mpi.enabled:
        rank_zero_print(mpi, f"MPI ranks: {mpi.size} (CPU backend, shot-parallel)")

    obs, elapsed_ms = generate_observed_data(
        solver,
        wave,
        sources,
        receivers,
        models=[
            torch.from_numpy(vp_true).to(dev),
            torch.from_numpy(vs_true).to(dev),
            torch.from_numpy(rho_true).to(dev),
        ],
        cfg=cfg,
        mpi=mpi,
    )
    rank_zero_print(mpi, f"Forward modeling time ({run_label}): {elapsed_ms:.2f} ms")
    if mpi.is_root:
        save_observed_figure(obs, receivers, output_dir, cfg)

    vp = torch.from_numpy(vp_init).float().to(dev)
    vs = torch.from_numpy(vp_init / 1.732).float().to(dev)
    rho = (torch.ones_like(vp) * 1000.0).float().to(dev)

    vp.requires_grad_()
    vs.requires_grad_()
    rho.requires_grad_()

    optimizer = torch.optim.Adam(
        [
            {"params": [vp], "lr": cfg["lr_vp"]},
            {"params": [vs], "lr": cfg["lr_vs"]},
            {"params": [rho], "lr": cfg["lr_rho"]},
        ],
        eps=1e-22,
    )

    obs_torch = torch.from_numpy(obs).to(dev)
    losses = []
    nshots = sources.shape[0]
    batchsize = min(cfg["batchsize"], nshots)
    shot_rng = np.random.RandomState(0)
    shot_schedule = [
        shot_rng.choice(nshots, size=batchsize, replace=False)
        for _ in range(cfg["epochs"])
    ]

    progress = tqdm.trange(cfg["epochs"], disable=mpi.enabled and not mpi.is_root)
    for epoch in progress:
        with torch.no_grad():
            top = cfg["fix_top_layers"]
            if top > 0:
                vp[:top] = torch.from_numpy(vp_true[:top]).to(dev)
                vs[:top] = torch.from_numpy(vs_true[:top]).to(dev)

        optimizer.zero_grad()
        global_shot_idx = shot_schedule[epoch]
        shot_idx = split_indices(global_shot_idx, mpi)

        forward_kwargs = {}
        if cfg["impl"] == "c":
            forward_kwargs["use_boundary_saving"] = True

        local_loss_sum = 0.0
        local_elements = 0
        if shot_idx.size > 0 and cfg["source_encoding"]:
            encoded_wave, encoded_obs = build_encoded_batch(wave, obs, shot_idx, cfg)
            # Source-encoding mode: sources/receivers must be 3D with leading
            # batch dim of 1 ((1, nsrc, ndim) / (1, nrec, ndim)). The mode is
            # auto-detected from these shapes; no kwarg needed.
            encoded_sources = sources[shot_idx][None, ...]
            encoded_receivers = receivers[:1]
            syn = solver(
                encoded_wave,
                encoded_sources,
                encoded_receivers,
                models=[vp, vs, rho],
                **forward_kwargs,
            )
            obs_batch = torch.from_numpy(encoded_obs).to(dev)
            local_elements = int(obs_batch.numel())
        elif shot_idx.size > 0:
            syn = solver(
                wave,
                sources[shot_idx],
                receivers[shot_idx],
                models=[vp, vs, rho],
                **forward_kwargs,
            )
            obs_batch = select_observed_batch(obs_torch, shot_idx, cfg)
            local_elements = int(obs_batch.numel())

        global_elements = reduce_scalar(local_elements, mpi, op="sum")
        if shot_idx.size > 0:
            loss_sum = (syn - obs_batch).pow(2).sum()
            (loss_sum / global_elements).backward()
            local_loss_sum = float(loss_sum.detach().cpu())
        elif mpi.enabled:
            solver.source_illumination = torch.zeros_like(vp)
            solver.receiver_illumination = torch.zeros_like(vp)
        allreduce_gradients([vp, vs, rho], mpi)
        allreduce_tensors(
            [
                getattr(solver, "source_illumination", None),
                getattr(solver, "receiver_illumination", None),
            ],
            mpi,
        )
        optimizer.step()

        loss_value = reduce_scalar(local_loss_sum, mpi, op="sum") / global_elements
        losses.append(loss_value)
        mode = "encoded" if cfg["source_encoding"] else "standard"
        rank_zero_print(mpi, f"[{run_label}] Epoch {epoch:04d} | Loss: {loss_value:.6e} | Mode: {mode}")

        if mpi.is_root and epoch % cfg["show_every"] == 0:
            save_progress_figure(
                vp_true,
                vs_true,
                vp.detach().cpu().numpy(),
                vs.detach().cpu().numpy(),
                vp.grad.detach().cpu().numpy(),
                vs.grad.detach().cpu().numpy(),
                losses,
                epoch,
                cfg,
                output_dir,
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Elastic Overthrust FWI example. Choose backend with --backend, implementation with --impl, "
            "and device with --device. Use --backend torch --impl c --device cpu for the C++ CPU binding."
        )
    )
    parser.add_argument(
        "--import-mode",
        choices=("env", "source"),
        default=IMPORT_MODE,
        help="Load sweep from the current environment or from the repository source tree.",
    )
    add_backend_impl_device_args(parser)
    parser.add_argument(
        "--mpi",
        action="store_true",
        help="Enable MPI shot parallelism for --backend torch --impl c --device cpu. Launch with mpirun or mpiexec.",
    )
    parser.add_argument(
        "--mpi-forward-batchsize",
        type=int,
        default=None,
        help="Number of local shots per MPI forward batch. Defaults to 1 for visible progress.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_fwi(
        backend=args.backend,
        impl=args.impl,
        device=args.device,
        use_mpi=args.mpi,
        mpi_forward_batchsize_override=args.mpi_forward_batchsize,
    )


if __name__ == "__main__":
    main()
