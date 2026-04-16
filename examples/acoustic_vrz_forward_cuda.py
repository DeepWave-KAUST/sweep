import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.equations import AcousticVRZ
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


torch.backends.cudnn.benchmark = True


SAVE_DIR = "acoustic_vrz_grad_compare"


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


def build_geometry(nx, src_z, rec_z, src_step, rec_step):
    src_x = np.arange(0, nx, src_step, dtype=np.int32).reshape(-1, 1)
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


def run_observed_data(solver, wavelet, sources, receivers, vp, z, device):
    with torch.no_grad():
        syn = solver.forward(
            wavelet,
            sources,
            receivers,
            models=[
                torch.from_numpy(vp).to(device),
                torch.from_numpy(z).to(device),
            ],
        )
    return syn.detach().cpu().numpy()


def run_single_gradient_step(solver, wavelet, sources, receivers, obs, vp0, z0, device):
    vp = torch.tensor(vp0, device=device, dtype=torch.float32, requires_grad=True)
    z = torch.tensor(z0, device=device, dtype=torch.float32, requires_grad=True)

    obs_t = torch.tensor(obs, device=device, dtype=torch.float32)
    syn = solver(wavelet, sources, receivers, models=[vp, z])

    if syn.ndim == 3:
        target = obs_t
    elif syn.ndim == 4:
        target = obs_t.permute(0, 2, 1).unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported synthetic tensor shape: {tuple(syn.shape)}")

    loss = (syn - target).pow(2).mean()
    loss.backward()

    return {
        "loss": float(loss.detach().cpu()),
        "syn": syn.detach().cpu().numpy(),
        "vp_grad": vp.grad.detach().cpu().numpy(),
        "z_grad": z.grad.detach().cpu().numpy(),
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


def run_mode_gradient_step(mode, solver, wavelet, sources, receivers, obs, vp0, z0, device):
    if mode == "boundary_saving":
        vp = torch.tensor(vp0, device=device, dtype=torch.float32, requires_grad=True)
        z = torch.tensor(z0, device=device, dtype=torch.float32, requires_grad=True)
        obs_t = torch.tensor(obs, device=device, dtype=torch.float32)
        syn = solver(
            wavelet,
            sources,
            receivers,
            models=[vp, z],
            use_boundary_saving=True,
        )
        target = obs_t if syn.ndim == 3 else obs_t.permute(0, 2, 1).unsqueeze(-1)
        loss = (syn - target).pow(2).mean()
        loss.backward()
        return {
            "loss": float(loss.detach().cpu()),
            "syn": syn.detach().cpu().numpy(),
            "vp_grad": vp.grad.detach().cpu().numpy(),
            "z_grad": z.grad.detach().cpu().numpy(),
        }

    return run_single_gradient_step(
        solver,
        wavelet,
        sources,
        receivers,
        obs,
        vp0,
        z0,
        device,
    )


def summarize_difference(name, cuda_arr, torch_arr):
    diff = cuda_arr - torch_arr
    abs_diff = np.abs(diff)
    rel_l2 = np.linalg.norm(diff) / (np.linalg.norm(torch_arr) + 1e-12)

    print(f"{name} max abs diff: {abs_diff.max():.6e}")
    print(f"{name} mean abs diff: {abs_diff.mean():.6e}")
    print(f"{name} relative L2 diff: {rel_l2:.6e}")

    return abs_diff, rel_l2


def plot_gradient_compare(name, torch_grad, cuda_grad, save_path, display_gain=1.0):
    torch_show = torch_grad * display_gain
    cuda_show = cuda_grad * display_gain
    abs_diff = np.abs(cuda_grad - torch_grad) * display_gain

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    clip = np.percentile(np.abs(torch_show), 98)
    clip = max(clip, 1e-8)
    im0 = ax[0].imshow(torch_show, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
    ax[0].set_title(f"Torch {name} grad x{display_gain:g}")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(cuda_show, cmap="seismic", aspect="auto", vmin=-clip, vmax=clip)
    ax[1].set_title(f"CUDA {name} grad x{display_gain:g}")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    diff_clip = np.percentile(abs_diff, 98)
    diff_clip = max(diff_clip, 1e-8)
    im2 = ax[2].imshow(abs_diff, cmap="magma", aspect="auto", vmin=0.0, vmax=diff_clip)
    ax[2].set_title(f"{name} abs diff x{display_gain:g}")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

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

    src_z = 2
    rec_z = 2
    src_step = 32
    rec_step = 2

    time_axis = np.arange(nt, dtype=np.float32) * dt
    wavelet = ricker(time_axis - delay, f=fm)

    vp_true, z_true = build_true_models(shape)
    vp_init, z_init = build_initial_models(vp_true, z_true)

    sources, receivers = build_geometry(shape[1], src_z, rec_z, src_step, rec_step)

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

    cuda_full_solver = build_cuda_solver(
        shape, dh, dt, spatial_order, abcn, free_surface, cuda_dev, mode="full"
    )

    obs = normalize_syn_shape(
        run_observed_data(cuda_full_solver, wavelet, sources, receivers, vp_true, z_true, cuda_dev)
    )

    torch_result = run_single_gradient_step(
        torch_solver, wavelet, sources, receivers, obs, vp_init, z_init, cpu_dev
    )
    syn_torch = normalize_syn_shape(torch_result["syn"])

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
        result = run_mode_gradient_step(
            mode, solver, wavelet, sources, receivers, obs, vp_init, z_init, cuda_dev
        )
        syn_mode = normalize_syn_shape(result["syn"])

        print("")
        print(f"[{tag}] loss:      {result['loss']:.6e}")
        print(f"[{tag}] torch loss: {torch_result['loss']:.6e}")
        print(f"[{tag}] loss abs diff: {abs(result['loss'] - torch_result['loss']):.6e}")

        summarize_difference(f"{tag} synthetic", syn_mode, syn_torch)
        summarize_difference(f"{tag} vp grad", result["vp_grad"], torch_result["vp_grad"])
        summarize_difference(f"{tag} z grad", result["z_grad"], torch_result["z_grad"])

        plot_synthetic_compare(
            tag,
            syn_torch,
            syn_mode,
            os.path.join(SAVE_DIR, f"{tag}_synthetic_compare.png"),
        )
        plot_gradient_compare(
            f"vp ({tag})",
            torch_result["vp_grad"],
            result["vp_grad"],
            os.path.join(SAVE_DIR, f"{tag}_vp_grad_compare.png"),
            display_gain=1.0,
        )
        plot_gradient_compare(
            f"z ({tag})",
            torch_result["z_grad"],
            result["z_grad"],
            os.path.join(SAVE_DIR, f"{tag}_z_grad_compare.png"),
            display_gain=200.0,
        )


if __name__ == "__main__":
    main()
