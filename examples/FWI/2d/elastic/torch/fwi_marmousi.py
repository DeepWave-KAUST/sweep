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

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

import configure_marmousi as shared_config
from sweep.equations import Elastic
from sweep.propagator.options import BoundaryOptions, CUDAOptions, EagerOptions, MemoryOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


torch.backends.cudnn.benchmark = True


def build_config(backend):
    backend_key = f"fwi_2d_elastic_torch_{backend}"
    if backend not in ("eager", "cuda"):
        raise ValueError(f"Unsupported backend '{backend}'. Expected one of ['cuda', 'eager'].")
    cfg = shared_config.get_config("fwi_2d_elastic_torch_common")
    cfg.update(shared_config.get_config(backend_key))
    cfg["backend"] = backend
    return cfg


def build_boundary_options(boundary_cfg):
    kwargs = {"storage": boundary_cfg["storage"]}
    if boundary_cfg["storage"] == "cpu":
        kwargs["transfer_interval"] = boundary_cfg["transfer_interval"]
        kwargs["pinned_memory"] = boundary_cfg["pinned_memory"]
    return BoundaryOptions(**kwargs)


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

    if cfg["backend"] == "eager":
        return PropTorch(
            equation,
            **prop_kwargs,
            backend="eager",
            eager_options=EagerOptions(
                use_compile=cfg["use_compile"],
            ),
            use_ckpt=cfg["use_ckpt"],
            ckpt_chunks=cfg["ckpt_chunks"],
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
    if cfg["record_layout"] == "cuda":
        return np.transpose(record, (1, 3, 2, 0))
    return np.asarray(record)


def to_backend_layout(record, cfg):
    if cfg["record_layout"] == "cuda":
        return np.transpose(record, (3, 0, 2, 1))
    return np.asarray(record)


def timed_forward(solver, wave, sources, receivers, models, cfg):
    kwargs = {}
    if cfg["backend"] == "cuda":
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
    if cfg["backend"] == "cuda":
        encoded_wave = encoded_wave[None, ...]
    return encoded_wave.astype(np.float32), to_backend_layout(encoded_obs, cfg).astype(np.float32)


def select_observed_batch(obs_torch, shot_idx, cfg):
    if cfg["record_layout"] == "cuda":
        return obs_torch[:, shot_idx]
    return obs_torch[shot_idx]


def run_fwi(backend="eager"):
    cfg = build_config(backend)
    output_dir = Path(__file__).resolve().parent / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)

    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The CUDA elastic FWI example requires a CUDA-capable PyTorch environment.")

    vp_true = np.load(EXAMPLES_DIR / cfg["true_model"]).astype(np.float32)
    vs_true = (vp_true / 1.732).astype(np.float32)
    rho_true = np.ones_like(vp_true, dtype=np.float32) * 1000.0
    vp_init = np.load(EXAMPLES_DIR / cfg["init_model"]).astype(np.float32)
    shape = vp_true.shape

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = build_solver(shape, dev, cfg)

    _, wave = build_wavelet(cfg)
    save_wavelet_figure(wave, output_dir)

    sources, receivers = build_geometry(shape, cfg)
    print("(nshots, ndim):", sources.shape)
    print("(nshots, nreceivers, ndim):", receivers.shape)

    obs, elapsed_ms = timed_forward(
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
    )
    print(f"Forward modeling time ({backend}): {elapsed_ms:.2f} ms")
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

    for epoch in tqdm.trange(cfg["epochs"]):
        with torch.no_grad():
            top = cfg["fix_top_layers"]
            if top > 0:
                vp[:top] = torch.from_numpy(vp_true[:top]).to(dev)
                vs[:top] = torch.from_numpy(vs_true[:top]).to(dev)

        optimizer.zero_grad()
        shot_idx = shot_schedule[epoch]

        forward_kwargs = {}
        if cfg["backend"] == "cuda":
            forward_kwargs["use_boundary_saving"] = True

        if cfg["source_encoding"]:
            encoded_wave, encoded_obs = build_encoded_batch(wave, obs, shot_idx, cfg)
            encoded_sources = sources[shot_idx]
            encoded_receivers = receivers[:1]
            if cfg["backend"] == "cuda":
                encoded_sources = encoded_sources[None, ...]
            syn = solver(
                encoded_wave,
                encoded_sources,
                encoded_receivers,
                models=[vp, vs, rho],
                source_encoding=True,
                **forward_kwargs,
            )
            obs_batch = torch.from_numpy(encoded_obs).to(dev)
        else:
            syn = solver(
                wave,
                sources[shot_idx],
                receivers[shot_idx],
                models=[vp, vs, rho],
                **forward_kwargs,
            )
            obs_batch = select_observed_batch(obs_torch, shot_idx, cfg)

        loss = (syn - obs_batch).pow(2).mean()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)
        mode = "encoded" if cfg["source_encoding"] else "standard"
        print(f"[{backend}] Epoch {epoch:04d} | Loss: {loss_value:.6e} | Mode: {mode}")

        if epoch % cfg["show_every"] == 0:
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
    parser = argparse.ArgumentParser(description="Elastic Marmousi FWI example for PyTorch and CUDA propagators.")
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
    return parser.parse_args()


def main():
    args = parse_args()
    run_fwi(backend=args.backend)


if __name__ == "__main__":
    main()
