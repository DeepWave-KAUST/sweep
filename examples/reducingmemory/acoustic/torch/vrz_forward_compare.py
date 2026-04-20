import os
from pathlib import Path
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

from sweep.equations import AcousticVRZ
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


torch.backends.cudnn.benchmark = True


SAVE_DIR = Path(__file__).resolve().parent / "acoustic_vrz_forward_compare"


def smooth2d(model, radius=8):
    kernel_size = 2 * radius + 1
    padded = np.pad(model, ((radius, radius), (radius, radius)), mode="edge")
    out = np.zeros_like(model)

    for iz in range(model.shape[0]):
        for ix in range(model.shape[1]):
            window = padded[iz : iz + kernel_size, ix : ix + kernel_size]
            out[iz, ix] = window.mean()

    return out.astype(np.float32)


def build_true_models(shape):
    nz, nx = shape

    vp = np.full(shape, 2000.0, dtype=np.float32)
    rho = np.full(shape, 1200.0, dtype=np.float32)

    vp[nz // 2 :, :] = 2600.0
    rho[nz // 2 :, :] = 1700.0

    anomaly_z0 = nz // 3
    anomaly_z1 = anomaly_z0 + nz // 8
    anomaly_x0 = nx // 3
    anomaly_x1 = anomaly_x0 + nx // 6
    vp[anomaly_z0:anomaly_z1, anomaly_x0:anomaly_x1] += 250.0
    rho[anomaly_z0:anomaly_z1, anomaly_x0:anomaly_x1] += 120.0

    z = vp * rho
    return vp, z


def build_initial_models(vp_true, z_true):
    vp_init = smooth2d(vp_true, radius=10)
    z_init = smooth2d(z_true, radius=10)
    return vp_init, z_init


def build_geometry(nx, src_z, rec_z, rec_step):
    src_x = np.array([[nx // 2]], dtype=np.int32)
    src_z_arr = np.full_like(src_x, src_z)
    sources = np.concatenate([src_x, src_z_arr], axis=1)

    rec_x = np.arange(0, nx, rec_step, dtype=np.int32).reshape(-1, 1)
    rec_z_arr = np.full_like(rec_x, rec_z)
    receivers = np.concatenate([rec_x, rec_z_arr], axis=1)
    receivers = receivers[None, ...].repeat(sources.shape[0], axis=0)

    return sources, receivers


def normalize_syn_shape(syn):
    syn = np.asarray(syn)
    if syn.ndim == 3:
        return syn
    if syn.ndim == 4 and syn.shape[-1] == 1:
        return np.transpose(syn[..., 0], (0, 2, 1))
    raise ValueError(f"Unsupported synthetic shape: {syn.shape}")


def run_forward(solver, wavelet, sources, receivers, vp, z, device, use_boundary_saving=False):
    with torch.no_grad():
        kwargs = {}
        if use_boundary_saving:
            kwargs["use_boundary_saving"] = True
        syn = solver.forward(
            wavelet,
            sources,
            receivers,
            models=[
                torch.from_numpy(vp).to(device),
                torch.from_numpy(z).to(device),
            ],
            **kwargs,
        )
    return syn.detach().cpu().numpy()


def run_loss_and_grads(solver, wavelet, sources, receivers, obs, vp, z, device):
    vp_t = torch.from_numpy(vp).to(device=device, dtype=torch.float32).requires_grad_()
    z_t = torch.from_numpy(z).to(device=device, dtype=torch.float32).requires_grad_()
    obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)

    syn = solver(
        wavelet,
        sources,
        receivers,
        models=[vp_t, z_t],
    )
    loss = (syn - obs_t).pow(2).mean()
    loss.backward()

    return {
        "syn": syn.detach().cpu().numpy(),
        "loss": float(loss.detach().cpu()),
        "grad_vp": vp_t.grad.detach().cpu().numpy(),
        "grad_z": z_t.grad.detach().cpu().numpy(),
    }


def build_cuda_solver(shape, dh, dt, spatial_order, abcn, free_surface, device, mode):
    kwargs = dict(
        shape=shape,
        dev=device,
        dh=dh,
        dt=dt,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=abcn,
        free_surface=free_surface,
        pml_type="cpmlr",
    )

    if mode == "full":
        kwargs["use_ckpt"] = False
    elif mode == "ckpt":
        kwargs["use_ckpt"] = True
        kwargs["ckpt_mode"] = "chunk"
        kwargs["ckpt_chunks"] = 100
    elif mode == "recursive_ckpt":
        kwargs["use_ckpt"] = True
        kwargs["ckpt_mode"] = "recursive"
        kwargs["ckpt_num"] = 6
    elif mode == "boundary_saving":
        kwargs["use_ckpt"] = False
    else:
        raise ValueError(f"Unsupported CUDA mode: {mode}")

    return PropCUDA(
        AcousticVRZ(spatial_order=spatial_order, device=device),
        **kwargs,
    )


def run_mode_forward(mode, solver, wavelet, sources, receivers, vp, z, device):
    return run_forward(
        solver,
        wavelet,
        sources,
        receivers,
        vp,
        z,
        device,
        use_boundary_saving=(mode == "boundary_saving"),
    )


def summarize_difference(name, cuda_arr, torch_arr):
    diff = cuda_arr - torch_arr
    abs_diff = np.abs(diff)
    rel_l2 = np.linalg.norm(diff) / (np.linalg.norm(torch_arr) + 1e-12)

    print(f"{name} max abs diff: {abs_diff.max():.6e}")
    print(f"{name} mean abs diff: {abs_diff.mean():.6e}")
    print(f"{name} relative L2 diff: {rel_l2:.6e}")

    return abs_diff, rel_l2


def summarize_scalar_difference(name, value, ref):
    abs_diff = abs(value - ref)
    rel_diff = abs_diff / (abs(ref) + 1e-12)
    print(f"{name} abs diff: {abs_diff:.6e}")
    print(f"{name} relative diff: {rel_diff:.6e}")


def summarize_scale_factor(name, mode_arr, ref_arr, dt=None, dh=None):
    mode_flat = np.asarray(mode_arr, dtype=np.float64).ravel()
    ref_flat = np.asarray(ref_arr, dtype=np.float64).ravel()

    denom = np.dot(ref_flat, ref_flat)
    lsq_scale = np.dot(mode_flat, ref_flat) / (denom + 1e-24)

    mode_norm = np.linalg.norm(mode_flat)
    ref_norm = np.linalg.norm(ref_flat)
    corr = np.dot(mode_flat, ref_flat) / (mode_norm * ref_norm + 1e-24)

    mask = np.abs(ref_flat) > (0.05 * np.max(np.abs(ref_flat)) + 1e-24)
    median_scale = np.median(mode_flat[mask] / (ref_flat[mask] + 1e-24)) if np.any(mask) else np.nan

    print(f"{name} least-squares scale (mode ~= a * ref): {lsq_scale:.6e}")
    print(f"{name} median pointwise scale: {median_scale:.6e}")
    print(f"{name} cosine similarity: {corr:.6e}")

    if dt is not None and dh is not None:
        candidates = {
            "dt^2": dt * dt,
            "1/dt^2": 1.0 / (dt * dt),
            "dh^2": dh * dh,
            "1/dh^2": 1.0 / (dh * dh),
            "dt^2/dh^2": (dt * dt) / (dh * dh),
            "dh^2/dt^2": (dh * dh) / (dt * dt),
        }
        print(f"{name} candidate factors:")
        for label, value in candidates.items():
            ratio = lsq_scale / (value + 1e-24)
            print(f"  {label:>10s}: {value:.6e}  lsq_scale/{label}={ratio:.6e}")


def plot_model_compare(title_ref, title_mode, ref_arr, mode_arr, save_path):
    abs_diff = np.abs(mode_arr - ref_arr)
    ref_vmin, ref_vmax = np.percentile(ref_arr, [2, 98])
    mode_vmin, mode_vmax = np.percentile(mode_arr, [2, 98])
    diff_clip = max(np.percentile(abs_diff, 98), 1e-12)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    im0 = ax[0].imshow(ref_arr, cmap="seismic", aspect="auto", vmin=ref_vmin, vmax=ref_vmax)
    ax[0].set_title(title_ref)
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    im1 = ax[1].imshow(mode_arr, cmap="seismic", aspect="auto", vmin=mode_vmin, vmax=mode_vmax)
    ax[1].set_title(title_mode)
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    im2 = ax[2].imshow(abs_diff, cmap="magma", aspect="auto", vmin=0.0, vmax=diff_clip)
    ax[2].set_title("abs diff")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_setup(vp_true, vp_init, sources, receivers, dh, save_path):
    nz, nx = vp_true.shape
    extent = [0.0, nx * dh, nz * dh, 0.0]

    src_x = sources[:, 0] * dh
    src_z = sources[:, 1] * dh
    rec_x = receivers[0, :, 0] * dh
    rec_z = receivers[0, :, 1] * dh

    vmin = min(float(vp_true.min()), float(vp_init.min()))
    vmax = max(float(vp_true.max()), float(vp_init.max()))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    im0 = ax[0].imshow(vp_true, cmap="seismic", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
    ax[0].scatter(rec_x, rec_z, s=10, c="white", marker="v", edgecolors="black", linewidths=0.3)
    ax[0].scatter(src_x, src_z, s=80, c="gold", marker="*", edgecolors="black", linewidths=0.5)
    ax[0].set_title("True vp + geometry")
    ax[0].set_xlabel("x (m)")
    ax[0].set_ylabel("z (m)")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(vp_init, cmap="seismic", aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
    ax[1].scatter(rec_x, rec_z, s=10, c="white", marker="v", edgecolors="black", linewidths=0.3)
    ax[1].scatter(src_x, src_z, s=80, c="gold", marker="*", edgecolors="black", linewidths=0.5)
    ax[1].set_title("Initial vp + geometry")
    ax[1].set_xlabel("x (m)")
    ax[1].set_ylabel("z (m)")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_synthetic_compare(mode_name, syn_ref, syn_mode, save_path):
    shot_id = min(1, syn_ref.shape[0] - 1)
    abs_diff = np.abs(syn_mode - syn_ref)
    vmin, vmax = np.percentile(syn_ref[shot_id], [2, 98])
    diff_clip = np.percentile(abs_diff[shot_id], 98)
    diff_clip = max(diff_clip, 1e-8)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    im0 = ax[0].imshow(syn_ref[shot_id], cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
    ax[0].set_title("Torch synthetic")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    im1 = ax[1].imshow(syn_mode[shot_id], cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
    ax[1].set_title(f"{mode_name} synthetic")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    im2 = ax[2].imshow(abs_diff[shot_id], cmap="magma", aspect="auto", vmin=0.0, vmax=diff_clip)
    ax[2].set_title(f"{mode_name} abs diff")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires CUDA.")

    np.random.seed(0)
    torch.manual_seed(0)

    shape = (96, 160)
    dh = 10.0
    dt = 0.001
    nt = 1000
    fm = 10.0
    delay = 0.15
    spatial_order = 8
    abcn = 20
    free_surface = False

    src_z = spatial_order // 2
    rec_z = spatial_order // 2
    rec_step = 2

    time_axis = np.arange(nt, dtype=np.float32) * dt
    wavelet = ricker(time_axis - delay, f=fm)

    vp_true, z_true = build_true_models(shape)
    vp_init, z_init = build_initial_models(vp_true, z_true)

    sources, receivers = build_geometry(shape[1], src_z, rec_z, rec_step)

    plot_setup(
        vp_true,
        vp_init,
        sources,
        receivers,
        dh,
        os.path.join(SAVE_DIR, "model_geometry_setup.png"),
    )

    cuda_dev = torch.device("cuda")
    cpu_dev = torch.device("cpu")

    torch_solver = PropTorch(
        AcousticVRZ(spatial_order=spatial_order, device=cpu_dev),
        shape=shape,
        dev=cpu_dev,
        dh=dh,
        dt=dt,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=abcn,
        free_surface=free_surface,
        pml_type="cpmlr",
        use_ckpt=False,
    )

    syn_torch_raw = run_forward(torch_solver, wavelet, sources, receivers, vp_init, z_init, cpu_dev)
    syn_torch = normalize_syn_shape(syn_torch_raw)
    obs_torch_raw = run_forward(torch_solver, wavelet, sources, receivers, vp_true, z_true, cpu_dev)

    mode_specs = [
        ("full", "cuda_full"),
        ("boundary_saving", "cuda_bs"),
        ("ckpt", "cuda_ckpt"),
        ("recursive_ckpt", "cuda_recursive_ckpt"),
    ]

    for mode, tag in mode_specs:
        solver = build_cuda_solver(
            shape, dh, dt, spatial_order, abcn, free_surface, cuda_dev, mode=mode
        )
        syn_mode = normalize_syn_shape(
            run_mode_forward(
                mode,
                solver,
                wavelet,
                sources,
                receivers,
                vp_init,
                z_init,
                cuda_dev,
            )
        )

        print("")
        summarize_difference(f"{tag} synthetic", syn_mode, syn_torch)

        plot_synthetic_compare(
            tag,
            syn_torch,
            syn_mode,
            os.path.join(SAVE_DIR, f"{tag}_synthetic_compare.png"),
        )

    cuda_full_solver = build_cuda_solver(
        shape, dh, dt, spatial_order, abcn, free_surface, cuda_dev, mode="full"
    )
    cuda_bs_solver = build_cuda_solver(
        shape, dh, dt, spatial_order, abcn, free_surface, cuda_dev, mode="boundary_saving"
    )
    obs_cuda_full_raw = run_forward(
        cuda_full_solver,
        wavelet,
        sources,
        receivers,
        vp_true,
        z_true,
        cuda_dev,
    )
    obs_cuda_bs_raw = run_forward(
        cuda_bs_solver,
        wavelet,
        sources,
        receivers,
        vp_true,
        z_true,
        cuda_dev,
        use_boundary_saving=True,
    )

    print("")
    print("Gradient comparison: torch vs cuda_full/cuda_bs")

    grad_torch = run_loss_and_grads(
        torch_solver,
        wavelet,
        sources,
        receivers,
        obs_torch_raw,
        vp_init,
        z_init,
        cpu_dev,
    )

    grad_cuda_full = run_loss_and_grads(
        cuda_full_solver,
        wavelet,
        sources,
        receivers,
        obs_cuda_full_raw,
        vp_init,
        z_init,
        cuda_dev,
    )
    grad_cuda_bs = run_loss_and_grads(
        cuda_bs_solver,
        wavelet,
        sources,
        receivers,
        obs_cuda_bs_raw,
        vp_init,
        z_init,
        cuda_dev,
    )

    summarize_scalar_difference("full loss", grad_cuda_full["loss"], grad_torch["loss"])
    summarize_difference("full grad_vp", grad_cuda_full["grad_vp"], grad_torch["grad_vp"])
    summarize_difference("full grad_z", grad_cuda_full["grad_z"], grad_torch["grad_z"])
    summarize_scale_factor("full grad_vp", grad_cuda_full["grad_vp"], grad_torch["grad_vp"], dt=dt, dh=dh)
    summarize_scale_factor("full grad_z", grad_cuda_full["grad_z"], grad_torch["grad_z"], dt=dt, dh=dh)

    print("")
    summarize_scalar_difference("bs loss", grad_cuda_bs["loss"], grad_torch["loss"])
    summarize_difference("bs grad_vp", grad_cuda_bs["grad_vp"], grad_torch["grad_vp"])
    summarize_difference("bs grad_z", grad_cuda_bs["grad_z"], grad_torch["grad_z"])
    summarize_scale_factor("bs grad_vp", grad_cuda_bs["grad_vp"], grad_torch["grad_vp"], dt=dt, dh=dh)
    summarize_scale_factor("bs grad_z", grad_cuda_bs["grad_z"], grad_torch["grad_z"], dt=dt, dh=dh)

    plot_model_compare(
        "Torch grad_vp",
        "CUDA full grad_vp",
        grad_torch["grad_vp"],
        grad_cuda_full["grad_vp"],
        os.path.join(SAVE_DIR, "cuda_full_grad_vp_compare.png"),
    )
    plot_model_compare(
        "Torch grad_z",
        "CUDA full grad_z",
        grad_torch["grad_z"],
        grad_cuda_full["grad_z"],
        os.path.join(SAVE_DIR, "cuda_full_grad_z_compare.png"),
    )
    plot_model_compare(
        "Torch grad_vp",
        "CUDA bs grad_vp",
        grad_torch["grad_vp"],
        grad_cuda_bs["grad_vp"],
        os.path.join(SAVE_DIR, "cuda_bs_grad_vp_compare.png"),
    )
    plot_model_compare(
        "Torch grad_z",
        "CUDA bs grad_z",
        grad_torch["grad_z"],
        grad_cuda_bs["grad_z"],
        os.path.join(SAVE_DIR, "cuda_bs_grad_z_compare.png"),
    )


if __name__ == "__main__":
    main()
