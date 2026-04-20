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

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

from sweep.equations import Acoustic
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


torch.backends.cudnn.benchmark = True


COMMON_CONFIG = {
    "nt": 2500,
    "dt": 0.002,
    "delay": 0.256,
    "fm": 5.0,
    "dh": 25.0,
    "spatial_order": 8,
    "abcn": 20,
    "free_surface": False,
    "src_step": 2,
    "rec_step": 1,
    "srcz": 1,
    "recz": 18,
    "lr": 25.0,
    "epochs": 101,
    "batchsize": 8,
    "show_every": 10,
    "true_model": "models/marmousi/true.npy",
    "init_model": "models/marmousi/smooth.npy",
    "max_time_shift_ratio": 0.2,
}


BACKEND_CONFIG = {
    "torch": {
        "abcn": 20,
        "free_surface": False,
        "output_dir": "acoustic_fwi_encoding_torch",
        "use_ckpt": False,
        "use_compile": True,
        "transpose_shot": False,
    },
    "cuda": {
        "abcn": 20,
        "free_surface": False,
        "output_dir": "acoustic_fwi_encoding_cuda",
        "transpose_shot": True,
        "boundary_saving_config": {
            "enabled": True,
            "storage": "gpu",
            "transfer_interval": 10,
            "pinned_memory": True,
        },
    },
}


def build_config(backend):
    if backend not in BACKEND_CONFIG:
        raise ValueError(f"Unsupported backend '{backend}'. Expected one of {sorted(BACKEND_CONFIG)}.")
    cfg = COMMON_CONFIG.copy()
    cfg.update(BACKEND_CONFIG[backend])
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

    if cfg["backend"] == "torch":
        return PropTorch(
            equation,
            **prop_kwargs,
            use_ckpt=cfg["use_ckpt"],
            use_compile=cfg["use_compile"],
        )

    if cfg["backend"] == "cuda":
        return PropCUDA(
            equation,
            **prop_kwargs,
            boundary_saving_config=cfg["boundary_saving_config"],
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
    return t, ricker(t - cfg["delay"], f=cfg["fm"]).astype(np.float32)


def timed_forward(solver, wave, sources, receivers, models):
    if solver.dev.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            obs = solver(wave, sources, receivers, models=models)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            obs = solver(wave, sources, receivers, models=models)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return obs.detach().cpu().numpy(), elapsed_ms


def save_wavelet_figure(wave, output_dir):
    plt.figure(figsize=(6, 3))
    plt.plot(wave, color="black")
    plt.title("Ricker Wavelet")
    plt.tight_layout()
    plt.savefig(output_dir / "ricker.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_observed_figure(obs, output_dir, cfg):
    shot = obs[-1].squeeze()
    if cfg.get("transpose_shot", False):
        shot = shot.T
    vmin, vmax = np.percentile(shot, [2, 98])
    plt.figure(figsize=(6, 4))
    plt.imshow(shot, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
    plt.colorbar()
    plt.title("Observed Shot Gather")
    plt.tight_layout()
    plt.savefig(output_dir / "observed_data.png", dpi=300, bbox_inches="tight")
    plt.close()


def format_shot_for_display(shot, cfg):
    shot = np.asarray(shot).squeeze()
    if cfg.get("transpose_shot", False):
        shot = shot.T
    return shot


def save_encoded_data_figure(encoded_obs, encoded_syn, epoch, output_dir, cfg):
    obs_np = format_shot_for_display(encoded_obs, cfg)
    syn_np = format_shot_for_display(encoded_syn, cfg)
    vmin, vmax = np.percentile(obs_np, [2, 98])

    fig, axes = plt.subplots(1, 2, figsize=(7, 5))
    axes[0].imshow(obs_np, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
    axes[0].set_title("Encoded Observed Data")
    axes[1].imshow(syn_np, vmin=vmin, vmax=vmax, cmap="seismic", aspect="auto")
    axes[1].set_title("Encoded Synthetic Data")
    plt.tight_layout()
    plt.savefig(output_dir / f"data_epoch_{epoch:04d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_progress_figure(true_model, vp, grad, losses, epoch, cfg, output_dir):
    nz, nx = true_model.shape
    extent = [0, nx * cfg["dh"], nz * cfg["dh"], 0]
    vmin_model, vmax_model = true_model.min(), true_model.max()
    vmin_grad, vmax_grad = np.percentile(grad, [2, 98])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(true_model, vmin=vmin_model, vmax=vmax_model, cmap="seismic", aspect="auto", extent=extent)
    axes[0].set_title("True Model")
    axes[1].imshow(vp, vmin=vmin_model, vmax=vmax_model, cmap="seismic", aspect="auto", extent=extent)
    axes[1].set_title("Inverted Model")
    axes[2].imshow(grad, vmin=vmin_grad, vmax=vmax_grad, cmap="seismic", aspect="auto", extent=extent)
    axes[2].set_title("Gradient")
    plt.tight_layout()
    plt.savefig(output_dir / f"epoch_{epoch:04d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(losses, color="black", label="Loss")
    ax.legend()
    ax.set_title("FWI Loss")
    plt.tight_layout()
    fig.savefig(output_dir / "loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def record_time_axis(record, cfg):
    if cfg.get("transpose_shot", False) and np.asarray(record).ndim >= 2:
        return 1
    return 0


def build_encoded_batch(wave, obs, shot_idx, cfg, dev, backend):
    nsel = len(shot_idx)
    nt = cfg["nt"]
    max_shift = max(1, int(cfg["max_time_shift_ratio"] * nt))

    encoded_wave = np.zeros((nsel, nt), dtype=np.float32)
    encoded_obs = np.zeros_like(obs[shot_idx[0]], dtype=np.float32)
    for i, shot in enumerate(shot_idx):
        polarity = -1.0 if np.random.randint(0, 2) else 1.0
        tau = int(np.random.randint(0, max_shift))
        shot_wave = polarity * np.roll(wave, shift=tau, axis=0)
        shot_wave[:tau] = 0.0

        shot_obs = polarity * np.roll(obs[shot], shift=tau, axis=record_time_axis(obs[shot], cfg))
        if record_time_axis(obs[shot], cfg) == 0:
            shot_obs[:tau] = 0.0
        else:
            shot_obs[:, :tau] = 0.0

        encoded_wave[i] = shot_wave
        encoded_obs += shot_obs

    if backend == "cuda":
        encoded_wave = encoded_wave[None, ...]
        encoded_obs = torch.as_tensor(encoded_obs[None, ...], dtype=torch.float32, device=dev)
        return encoded_wave, encoded_obs

    encoded_wave = torch.as_tensor(encoded_wave, dtype=torch.float32, device=dev)
    encoded_obs = torch.as_tensor(encoded_obs, dtype=torch.float32, device=dev)
    return encoded_wave, encoded_obs


def prepare_encoded_inputs(wave, obs, sources, receivers_shared, shot_idx, cfg, dev):
    encoded_wave, encoded_obs = build_encoded_batch(wave, obs, shot_idx, cfg, dev, cfg["backend"])
    sources_sel = sources[shot_idx]

    if cfg["backend"] == "cuda":
        return (
            encoded_wave,
            sources_sel[None, ...],
            receivers_shared,
            encoded_obs,
        )

    return (
        encoded_wave,
        sources_sel,
        receivers_shared,
        encoded_obs,
    )


def run_fwi(backend="torch"):
    cfg = build_config(backend)
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)

    true_model = np.load(EXAMPLES_DIR / cfg["true_model"]).astype(np.float32)
    init_model = np.load(EXAMPLES_DIR / cfg["init_model"]).astype(np.float32)
    shape = true_model.shape

    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The CUDA acoustic FWI encoding example requires a CUDA-capable PyTorch environment.")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = build_solver(shape, dev, cfg)

    _, wave = build_wavelet(cfg)
    save_wavelet_figure(wave, output_dir)

    sources, receivers = build_geometry(shape, cfg)
    print("(nshots, ndim):", sources.shape)
    print("(nshots, nreceivers, ndim):", receivers.shape)

    true_vp = torch.from_numpy(true_model).to(dev)
    obs, elapsed_ms = timed_forward(solver, wave, sources, receivers, models=[true_vp])
    print(f"Forward modeling time ({backend}): {elapsed_ms:.2f} ms")
    save_observed_figure(obs, output_dir, cfg)

    receivers_shared = receivers[:1]
    print("Source shape for inversion:", sources.shape)
    print("Receiver shape for inversion:", receivers_shared.shape)

    inv_vp = torch.from_numpy(init_model).to(dev).requires_grad_(True)
    optimizer = torch.optim.Adam([inv_vp], lr=cfg["lr"], eps=1e-22)
    losses = []
    nshots = sources.shape[0]
    batchsize = min(cfg["batchsize"], nshots)

    for epoch in tqdm.trange(cfg["epochs"]):
        optimizer.zero_grad()

        shot_idx = np.random.choice(nshots, size=batchsize, replace=False)
        encoded_wave, encoded_sources, encoded_receivers, encoded_obs = prepare_encoded_inputs(
            wave,
            obs,
            sources,
            receivers_shared,
            shot_idx,
            cfg,
            dev,
        )
        encoded_syn = solver(
            encoded_wave,
            encoded_sources,
            encoded_receivers,
            models=[inv_vp],
            source_encoding=True,
        )
        
        loss = (encoded_syn - encoded_obs).pow(2).mean()
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        losses.append(loss_value)
        print(f"[{backend}] Epoch {epoch:04d} | Loss: {loss_value:.6e}")

        if epoch % cfg["show_every"] == 0:
            vp_np = inv_vp.detach().cpu().numpy()
            grad_np = inv_vp.grad.detach().cpu().numpy()
            save_encoded_data_figure(
                encoded_obs.detach().cpu().numpy(),
                encoded_syn.detach().cpu().numpy(),
                epoch,
                output_dir,
                cfg,
            )
            save_progress_figure(true_model, vp_np, grad_np, losses, epoch, cfg, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Acoustic source-encoding FWI example for PyTorch and CUDA propagators.")
    parser.add_argument(
        "--import-mode",
        choices=("env", "source"),
        default=IMPORT_MODE,
        help="Load sweep from the current environment or from the repository source tree.",
    )
    parser.add_argument(
        "--backend",
        choices=("torch", "cuda"),
        default="torch",
        help="Select which propagator backend to use.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_fwi(backend=args.backend)


if __name__ == "__main__":
    main()
