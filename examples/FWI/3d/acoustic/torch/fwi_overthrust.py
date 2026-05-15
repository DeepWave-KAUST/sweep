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
from backend_cli import add_backend_impl_device_args, resolve_backend_impl_device

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
from sweep.propagator.options import BoundaryOptions, CUDAOptions, EagerOptions, MemoryOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


torch.backends.cudnn.benchmark = True


def build_config(backend, impl=None, device="auto"):
    backend, impl, device = resolve_backend_impl_device(backend, impl, device)
    if impl == "c" and device != "cuda":
        raise ValueError("The 3D acoustic FWI example currently supports the c implementation only on CUDA.")
    mode = "eager" if impl == "eager" else "cuda"
    backend_key = f"fwi_3d_acoustic_torch_{mode}"
    cfg = shared_config.get_config("fwi_3d_acoustic_torch_common")
    cfg.update(shared_config.get_config(backend_key))
    cfg["backend"] = backend
    cfg["impl"] = impl
    cfg["device"] = device
    cfg["mode"] = mode
    return cfg


def select_device(cfg):
    return torch.device(cfg["device"])


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

    raise ValueError(f"Unsupported backend/impl '{cfg['backend']}/{cfg['impl']}'.")


def build_boundary_options(boundary_cfg):
    kwargs = {"storage": boundary_cfg["storage"]}
    if boundary_cfg["storage"] == "cpu":
        kwargs["transfer_interval"] = boundary_cfg["transfer_interval"]
        kwargs["pinned_memory"] = boundary_cfg["pinned_memory"]
    return BoundaryOptions(**kwargs)


def timed_forward(solver, wave, sources, receivers, models):
    kwargs = {}
    if getattr(solver, "impl", "eager") == "c":
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
                if getattr(solver, "impl", "eager") == "c":
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
    total_loss = 0.0

    for start in range(0, shot_idx.shape[0], train_shot_batchsize):
        stop = min(start + train_shot_batchsize, shot_idx.shape[0])
        batch_idx = shot_idx[start:stop]
        solver_kwargs = {}
        if getattr(solver, "impl", "eager") == "c":
            solver_kwargs["use_boundary_saving"] = True

        syn = solver(wave, sources[batch_idx], receivers[batch_idx], models=[inv_vp], **solver_kwargs)
        obs_batch = obs_torch[batch_idx].to(dev)
        loss = (syn - obs_batch).pow(2).sum()
        loss.backward()
        total_loss += loss.item()

    return total_loss


def build_encoded_batch_3d(wave, obs, shot_idx, cfg, dev):
    nsel = len(shot_idx)
    nt = cfg["nt"]
    max_shift = max(1, int(cfg["max_time_shift_ratio"] * nt))

    encoded_wave = np.zeros((nsel, nt), dtype=np.float32)
    encoded_obs = np.zeros_like(obs[shot_idx[0]], dtype=np.float32)
    # obs[shot] has shape (nrec, nt) for c path or (nrec, nt) post-squeeze — time is the last axis.
    time_axis = obs[shot_idx[0]].ndim - 1
    head_zero = (slice(None),) * time_axis + (slice(0, None),)  # filled below per tau
    for i, shot in enumerate(shot_idx):
        polarity = -1.0 if np.random.randint(0, 2) else 1.0
        tau = int(np.random.randint(0, max_shift))
        shot_wave = polarity * np.roll(wave, shift=tau, axis=0)
        shot_wave[:tau] = 0.0
        shot_obs = polarity * np.roll(obs[shot], shift=tau, axis=time_axis)
        zero_idx = (slice(None),) * time_axis + (slice(0, tau),)
        shot_obs[zero_idx] = 0.0
        encoded_wave[i] = shot_wave
        encoded_obs += shot_obs

    # Source-encoding requires 3D (1, nsel, ...) shapes for both eager and c paths
    # so that _auto_detect_source_encoding agrees with the explicit flag.
    encoded_wave = encoded_wave[None, ...]
    if cfg["impl"] == "c":
        encoded_obs = torch.as_tensor(encoded_obs[None, ...], dtype=torch.float32, device=dev)
    else:
        encoded_obs = torch.as_tensor(encoded_obs, dtype=torch.float32, device=dev)
    return encoded_wave, encoded_obs


def prepare_encoded_inputs_3d(wave, obs, sources, receivers_shared, shot_idx, cfg, dev):
    encoded_wave, encoded_obs = build_encoded_batch_3d(wave, obs, shot_idx, cfg, dev)
    sources_sel = sources[shot_idx][None, ...]                              # (1, nsel, 3)
    return encoded_wave, sources_sel, receivers_shared, encoded_obs


def run_fwi(backend="torch", impl=None, device="auto", batchsize_override=None, train_shot_batchsize_override=None, use_source_encoding=False):
    cfg = build_config(backend, impl, device)
    if batchsize_override is not None:
        cfg["batchsize"] = int(batchsize_override)
    if train_shot_batchsize_override is not None:
        cfg["train_shot_batchsize"] = int(train_shot_batchsize_override)
    if int(cfg["batchsize"]) < 1:
        raise ValueError(f"batchsize must be >= 1, got {cfg['batchsize']}.")
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

    true_model, init_model = load_models(EXAMPLES_DIR, cfg)
    shape = true_model.shape
    
    dev = select_device(cfg)
    solver = build_solver(shape, dev, cfg)
    run_label = f"{cfg['backend']}/{cfg['impl']}/{cfg['device']}"

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
        print(f"Forward modeling time ({run_label}): {elapsed_ms:.2f} ms")
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
            f"Forward modeling time ({run_label}, batched {forward_batchsize} shots/step): "
            f"{elapsed_ms:.2f} ms"
        )
    save_observed_figure(obs, receivers, cfg, output_dir)

    inv_vp = torch.from_numpy(init_model).to(dev).requires_grad_(True)
    nshots = sources.shape[0]
    batchsize = min(cfg["batchsize"], nshots)

    if use_source_encoding:
        lr = float(cfg.get("lr_encoding", cfg["lr"]))
        optimizer = torch.optim.Adam([inv_vp], lr=lr, eps=1e-22)
        receivers_shared = receivers[:1]
        print(
            f"[{run_label}] source encoding ON,",
            f"lr={lr}, batchsize={batchsize}, nshots={nshots}",
        )
        losses = []
        for epoch in tqdm.trange(cfg["epochs"]):
            optimizer.zero_grad()
            shot_idx = np.random.choice(nshots, size=batchsize, replace=False)
            encoded_wave, encoded_sources, encoded_receivers, encoded_obs = prepare_encoded_inputs_3d(
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
            loss = (encoded_syn - encoded_obs).pow(2).sum()
            loss.backward()
            optimizer.step()

            loss_value = float(loss.item())
            losses.append(loss_value)
            print(f"[{run_label}] Epoch {epoch:04d} | Loss: {loss_value:.6e}")

            if epoch % cfg["show_every"] == 0:
                vp_np = inv_vp.detach().cpu().numpy()
                grad_np = inv_vp.grad.detach().cpu().numpy()
                save_progress_figure(
                    true_model,
                    vp_np,
                    grad_np,
                    losses,
                    epoch,
                    cfg,
                    output_dir,
                    loss_ylabel="Sum of Squared Error",
                )
        np.save(output_dir / "vp_inverted.npy", inv_vp.detach().cpu().numpy())
        np.save(output_dir / "losses.npy", np.asarray(losses, dtype=np.float32))
        return

    optimizer = torch.optim.Adam([inv_vp], lr=cfg["lr"], eps=1e-22)

    obs_torch = torch.from_numpy(obs)
    losses = []
    train_shot_batchsize = min(int(cfg.get("train_shot_batchsize", batchsize)), batchsize)
    print(
        "Training shot selection:",
        f"batchsize={batchsize},",
        f"train_shot_batchsize={train_shot_batchsize}",
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
        print(f"[{run_label}] Epoch {epoch:04d} | Loss: {loss_value:.6e}")

        if epoch % cfg["show_every"] == 0:
            vp_np = inv_vp.detach().cpu().numpy()
            grad_np = inv_vp.grad.detach().cpu().numpy()
            save_progress_figure(
                true_model,
                vp_np,
                grad_np,
                losses,
                epoch,
                cfg,
                output_dir,
                loss_ylabel="Sum of Squared Error",
            )
    np.save(output_dir / "vp_inverted.npy", inv_vp.detach().cpu().numpy())
    np.save(output_dir / "losses.npy", np.asarray(losses, dtype=np.float32))


def parse_args():
    parser = argparse.ArgumentParser(description="3D acoustic FWI on the Overthrust model for Torch eager and c propagators.")
    parser.add_argument(
        "--import-mode",
        choices=("env", "source"),
        default=IMPORT_MODE,
        help="Load sweep from the current environment or from the repository source tree.",
    )
    add_backend_impl_device_args(parser)
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
        help="Process the selected training shots in smaller chunks during each optimizer step. Use 1 to run one shot at a time while accumulating gradients.",
    )
    parser.add_argument(
        "--use-source-encoding",
        action="store_true",
        help="Use source encoding (combine selected shots into a super-shot per iteration) instead of mini-batch stochastic FWI.",
    )
    args = parser.parse_args()
    if args.use_source_encoding and args.train_shot_batchsize is not None:
        parser.error("--use-source-encoding ignores --train-shot-batchsize; do not pass both.")
    return args


def main():
    args = parse_args()
    run_fwi(
        backend=args.backend,
        impl=args.impl,
        device=args.device,
        batchsize_override=args.batchsize,
        train_shot_batchsize_override=args.train_shot_batchsize,
        use_source_encoding=args.use_source_encoding,
    )


if __name__ == "__main__":
    main()
