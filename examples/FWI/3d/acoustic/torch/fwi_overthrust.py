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

import numpy as np
import torch
import tqdm

import configure_overthrust as shared_config
from fwi3d_overthrust import (
    build_geometry,
    build_wavelet,
    load_models,
    save_observed_figure,
    save_progress_figure,
    save_wavelet_figure,
)
from sweep.equations import Acoustic3D
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


torch.backends.cudnn.benchmark = True


def build_config(backend):
    backend_key = f"fwi_3d_acoustic_torch_{backend}"
    if backend not in ("torch", "cuda"):
        raise ValueError(f"Unsupported backend '{backend}'. Expected one of ['cuda', 'torch'].")
    cfg = shared_config.get_config("fwi_3d_acoustic_torch_common")
    cfg.update(shared_config.get_config(backend_key))
    cfg["backend"] = backend
    return cfg


def build_solver(shape, dev, cfg):
    equation = Acoustic3D(
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
            ckpt_chunks=cfg["ckpt_chunks"],
            use_compile=cfg["use_compile"],
        )

    if cfg["backend"] == "cuda":
        return PropCUDA(
            equation,
            **prop_kwargs,
            boundary_saving_config=cfg["boundary_saving_config"],
        )

    raise ValueError(f"Unsupported backend '{cfg['backend']}'.")


def timed_forward(solver, wave, sources, receivers, models):
    kwargs = {}
    if isinstance(solver, PropCUDA):
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
                if isinstance(solver, PropCUDA):
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


def run_fwi(backend="torch"):
    cfg = build_config(backend)
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    np.random.seed(0)

    true_model, init_model = load_models(EXAMPLES_DIR, cfg)
    shape = true_model.shape

    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The CUDA acoustic 3D FWI example requires a CUDA-capable PyTorch environment.")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = build_solver(shape, dev, cfg)

    _, wave = build_wavelet(cfg, ricker)
    save_wavelet_figure(wave, output_dir)

    sources, receivers = build_geometry(shape, cfg)
    print("prepared model shape:", shape)
    print("(nshots, ndim):", sources.shape)
    print("(nshots, nreceivers, ndim):", receivers.shape)

    true_vp = torch.from_numpy(true_model).to(dev)
    forward_batchsize = min(cfg["forward_batchsize"], sources.shape[0])
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
    save_observed_figure(obs, output_dir)

    inv_vp = torch.from_numpy(init_model).to(dev).requires_grad_(True)
    optimizer = torch.optim.Adam([inv_vp], lr=cfg["lr"], eps=1e-22)

    obs_torch = torch.from_numpy(obs)
    losses = []
    nshots = sources.shape[0]
    batchsize = min(cfg["batchsize"], nshots)

    for epoch in tqdm.trange(cfg["epochs"]):
        optimizer.zero_grad()

        shot_idx = np.random.choice(nshots, size=batchsize, replace=False)
        solver_kwargs = {}
        if isinstance(solver, PropCUDA):
            solver_kwargs["use_boundary_saving"] = True

        syn = solver(wave, sources[shot_idx], receivers[shot_idx], models=[inv_vp], **solver_kwargs)
        obs_batch = obs_torch[shot_idx].to(dev)
        loss = (syn - obs_batch).pow(2).sum()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"[{backend}] Epoch {epoch:04d} | Loss: {loss.item():.6e}")

        if epoch % cfg["show_every"] == 0:
            vp_np = inv_vp.detach().cpu().numpy()
            grad_np = inv_vp.grad.detach().cpu().numpy()
            save_progress_figure(true_model, vp_np, grad_np, losses, epoch, cfg, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="3D acoustic FWI on the Overthrust model for PyTorch and CUDA propagators.")
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
