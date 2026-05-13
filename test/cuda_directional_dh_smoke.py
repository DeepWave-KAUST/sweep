import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep import is_torch_binding_available
from sweep.equations import Acoustic, Elastic
from sweep.propagator._c import _CompiledPropagator
from sweep.signal import ricker

OUTPUT_DIR = Path(__file__).resolve().parent / "test_outputs" / "cuda_directional_dh"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not is_torch_binding_available():
        raise RuntimeError("sweep._C is not available or the CUDA binding cannot be imported.")
    return torch.device("cuda")


def make_wavelet(nt, dt, freq):
    t = np.arange(nt, dtype=np.float32) * dt
    delay = 1.2 / freq
    return ricker(t - delay, f=freq).astype(np.float32)


def make_geometry_2d(nx, src_z, rec_z, nshots=2, nrec=8):
    src_x = np.linspace(4, nx - 5, nshots, dtype=np.int32)
    rec_x = np.linspace(2, nx - 3, nrec, dtype=np.int32)

    sources = np.stack([src_x, np.full_like(src_x, src_z)], axis=1)
    receivers_single = np.stack([rec_x, np.full_like(rec_x, rec_z)], axis=1)
    receivers = np.repeat(receivers_single[None, :, :], nshots, axis=0)
    return sources, receivers


def physical_to_grid(coords_m, dh):
    coords_m = np.asarray(coords_m, dtype=np.float32)
    dh = np.asarray(dh, dtype=np.float32)
    return np.rint(coords_m / dh).astype(np.int32)


def make_acoustic_model(shape):
    nz, nx = shape
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)[None, :]
    true_vp = 1800.0 + 400.0 * z + 120.0 * np.sin(2 * np.pi * x) * np.exp(-2.0 * z)
    init_vp = 1750.0 + 320.0 * z + np.zeros_like(x)
    return true_vp.astype(np.float32), init_vp.astype(np.float32)


def make_acoustic_model_from_extent(shape, dh, physical_size):
    nz, nx = shape
    dz, dx = dh
    lz, lx = physical_size

    z = ((np.arange(nz, dtype=np.float32) + 0.5) * dz / lz)[:, None]
    x = ((np.arange(nx, dtype=np.float32) + 0.5) * dx / lx)[None, :]

    vp = (
        1800.0
        + 500.0 * z
        + 120.0 * np.sin(2.0 * np.pi * x) * np.exp(-1.8 * z)
        + 80.0 * np.cos(3.0 * np.pi * z) * np.exp(-2.5 * (x - 0.5) ** 2)
    )
    return vp.astype(np.float32)


def make_elastic_model(shape):
    nz, nx = shape
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)[None, :]
    true_vp = 2200.0 + 500.0 * z + 100.0 * np.cos(np.pi * x) * np.exp(-1.5 * z)
    true_vs = 0.58 * true_vp
    true_rho = 1000.0 + 120.0 * z + np.zeros_like(x)

    init_vp = 2100.0 + 420.0 * z + np.zeros_like(x)
    init_vs = 0.58 * init_vp
    init_rho = np.full_like(true_rho, 1000.0)

    return (
        true_vp.astype(np.float32),
        true_vs.astype(np.float32),
        true_rho.astype(np.float32),
        init_vp.astype(np.float32),
        init_vs.astype(np.float32),
        init_rho.astype(np.float32),
    )


def check_tensor(name, tensor):
    if tensor is None:
        raise AssertionError(f"{name} is None.")
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains non-finite values.")


def save_record_comparison(prefix, coarse, fine):
    coarse = np.asarray(coarse, dtype=np.float32)
    fine = np.asarray(fine, dtype=np.float32)
    diff = coarse - fine

    nshots = coarse.shape[0]
    fig, axes = plt.subplots(nshots, 3, figsize=(12, 3.5 * nshots), squeeze=False)

    for ishot in range(nshots):
        coarse_panel = coarse[ishot].T
        fine_panel = fine[ishot].T
        diff_panel = diff[ishot].T

        amp = max(
            float(np.percentile(np.abs(coarse_panel), 99)),
            float(np.percentile(np.abs(fine_panel), 99)),
            1e-6,
        )
        diff_amp = max(float(np.percentile(np.abs(diff_panel), 99)), 1e-6)

        axes[ishot, 0].imshow(coarse_panel, cmap="seismic", aspect="auto", vmin=-amp, vmax=amp)
        axes[ishot, 0].set_title(f"Coarse Shot {ishot}")
        axes[ishot, 0].set_xlabel("Receiver")
        axes[ishot, 0].set_ylabel("Time")

        axes[ishot, 1].imshow(fine_panel, cmap="seismic", aspect="auto", vmin=-amp, vmax=amp)
        axes[ishot, 1].set_title(f"Fine Shot {ishot}")
        axes[ishot, 1].set_xlabel("Receiver")
        axes[ishot, 1].set_ylabel("Time")

        axes[ishot, 2].imshow(diff_panel, cmap="seismic", aspect="auto", vmin=-diff_amp, vmax=diff_amp)
        axes[ishot, 2].set_title(f"Difference Shot {ishot}")
        axes[ishot, 2].set_xlabel("Receiver")
        axes[ishot, 2].set_ylabel("Time")

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{prefix}_records.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_normalized_record_comparison(prefix, coarse, fine):
    coarse = np.asarray(coarse, dtype=np.float32)
    fine = np.asarray(fine, dtype=np.float32)

    def normalize_per_trace(data):
        scale = np.max(np.abs(data), axis=-1, keepdims=True)
        scale = np.maximum(scale, 1e-6)
        return data / scale

    coarse_n = normalize_per_trace(coarse)
    fine_n = normalize_per_trace(fine)
    diff_n = coarse_n - fine_n

    nshots = coarse.shape[0]
    fig, axes = plt.subplots(nshots, 3, figsize=(12, 3.5 * nshots), squeeze=False)

    for ishot in range(nshots):
        axes[ishot, 0].imshow(coarse_n[ishot].T, cmap="seismic", aspect="auto", vmin=-1.0, vmax=1.0)
        axes[ishot, 0].set_title(f"Coarse Norm Shot {ishot}")
        axes[ishot, 0].set_xlabel("Receiver")
        axes[ishot, 0].set_ylabel("Time")

        axes[ishot, 1].imshow(fine_n[ishot].T, cmap="seismic", aspect="auto", vmin=-1.0, vmax=1.0)
        axes[ishot, 1].set_title(f"Fine Norm Shot {ishot}")
        axes[ishot, 1].set_xlabel("Receiver")
        axes[ishot, 1].set_ylabel("Time")

        axes[ishot, 2].imshow(diff_n[ishot].T, cmap="seismic", aspect="auto", vmin=-1.0, vmax=1.0)
        axes[ishot, 2].set_title(f"Norm Difference Shot {ishot}")
        axes[ishot, 2].set_xlabel("Receiver")
        axes[ishot, 2].set_ylabel("Time")

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{prefix}_normalized_records.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_acoustic(dev):
    shape = (40, 48)
    dh = (8.0, 12.0)  # (dz, dx)
    dt = 0.001
    nt = 500
    wave = make_wavelet(nt, dt, freq=10.0)
    sources, receivers = make_geometry_2d(shape[1], src_z=3, rec_z=4, nshots=2, nrec=10)
    true_vp, init_vp = make_acoustic_model(shape)

    solver = _CompiledPropagator(
        Acoustic(spatial_order=4, device=dev),
        shape=shape,
        dev=dev,
        dh=dh,
        dt=dt,
        nt=nt,
        abcn=12,
        source_type=["h1"],
        receiver_type=["h1"],
        free_surface=False,
        pml_type="cpmlr",
        use_ckpt=False,
        boundary_saving_config={"enabled": False},
    )

    with torch.no_grad():
        obs = solver.forward(
            wave,
            sources,
            receivers,
            models=[torch.from_numpy(true_vp).to(dev)],
        )
    check_tensor("acoustic obs", obs)

    vp = torch.from_numpy(init_vp).to(dev).requires_grad_()
    syn = solver(wave, sources, receivers, models=[vp])
    loss = torch.mean((syn - obs) ** 2)
    check_tensor("acoustic syn", syn)
    check_tensor("acoustic loss", loss)
    loss.backward()
    check_tensor("acoustic vp.grad", vp.grad)
    if float(vp.grad.abs().max()) == 0.0:
        raise AssertionError("acoustic vp.grad is identically zero.")

    return {
        "forward_shape": tuple(obs.shape),
        "loss": float(loss.detach().cpu()),
        "grad_max": float(vp.grad.abs().max().detach().cpu()),
    }


def run_elastic(dev):
    shape = (36, 44)
    dh = (9.0, 13.0)  # (dz, dx)
    dt = 0.0008
    nt = 50
    wave = make_wavelet(nt, dt, freq=12.0)
    sources, receivers = make_geometry_2d(shape[1], src_z=3, rec_z=4, nshots=2, nrec=8)
    true_vp, true_vs, true_rho, init_vp, init_vs, init_rho = make_elastic_model(shape)

    solver = _CompiledPropagator(
        Elastic(spatial_order=4, device=dev),
        shape=shape,
        dev=dev,
        dh=dh,
        dt=dt,
        nt=nt,
        abcn=12,
        source_type=["sxx", "szz"],
        receiver_type=["vx", "vz"],
        free_surface=False,
        pml_type="cpmls",
        use_ckpt=False,
        boundary_saving_config={"enabled": False},
    )

    with torch.no_grad():
        obs = solver.forward(
            wave,
            sources,
            receivers,
            models=[
                torch.from_numpy(true_vp).to(dev),
                torch.from_numpy(true_vs).to(dev),
                torch.from_numpy(true_rho).to(dev),
            ],
        )
    check_tensor("elastic obs", obs)

    vp = torch.from_numpy(init_vp).to(dev).requires_grad_()
    vs = torch.from_numpy(init_vs).to(dev).requires_grad_()
    rho = torch.from_numpy(init_rho).to(dev)
    syn = solver(wave, sources, receivers, models=[vp, vs, rho])
    loss = torch.mean((syn - obs) ** 2)
    check_tensor("elastic syn", syn)
    check_tensor("elastic loss", loss)
    loss.backward()
    check_tensor("elastic vp.grad", vp.grad)
    check_tensor("elastic vs.grad", vs.grad)
    if float(vp.grad.abs().max()) == 0.0:
        raise AssertionError("elastic vp.grad is identically zero.")
    if float(vs.grad.abs().max()) == 0.0:
        raise AssertionError("elastic vs.grad is identically zero.")

    return {
        "forward_shape": tuple(obs.shape),
        "loss": float(loss.detach().cpu()),
        "vp_grad_max": float(vp.grad.abs().max().detach().cpu()),
        "vs_grad_max": float(vs.grad.abs().max().detach().cpu()),
    }


def run_acoustic_physical_consistency(dev):
    coarse_shape = (40, 48)
    fine_shape = (80, 96)
    coarse_dh = (8.0, 12.0)   # (dz, dx)
    fine_dh = (4.0, 6.0)      # same physical size, refined grid
    coarse_abcn = 12
    fine_abcn = 24
    physical_size = (
        coarse_shape[0] * coarse_dh[0],
        coarse_shape[1] * coarse_dh[1],
    )

    dt = 0.001
    nt = 1000
    base_wave = make_wavelet(nt, dt, freq=8.0)
    coarse_wave = base_wave / float(np.prod(coarse_dh))
    fine_wave = base_wave / float(np.prod(fine_dh))

    # Use physical coordinates that fall exactly on both grids.
    source_coords_m = np.array(
        [
            [96.0, 24.0],
            [240.0, 24.0],
            [384.0, 24.0],
        ],
        dtype=np.float32,
    )
    receiver_x_m = np.arange(24.0, physical_size[1] - 24.0 + 1e-6, 24.0, dtype=np.float32)
    receiver_coords_m = np.stack([receiver_x_m, np.full_like(receiver_x_m, 32.0)], axis=1)

    coarse_sources = physical_to_grid(source_coords_m, np.array([coarse_dh[1], coarse_dh[0]], dtype=np.float32))
    fine_sources = physical_to_grid(source_coords_m, np.array([fine_dh[1], fine_dh[0]], dtype=np.float32))
    coarse_receivers_single = physical_to_grid(receiver_coords_m, np.array([coarse_dh[1], coarse_dh[0]], dtype=np.float32))
    fine_receivers_single = physical_to_grid(receiver_coords_m, np.array([fine_dh[1], fine_dh[0]], dtype=np.float32))

    coarse_receivers = np.repeat(coarse_receivers_single[None, :, :], coarse_sources.shape[0], axis=0)
    fine_receivers = np.repeat(fine_receivers_single[None, :, :], fine_sources.shape[0], axis=0)

    coarse_model = make_acoustic_model_from_extent(coarse_shape, coarse_dh, physical_size)
    fine_model = make_acoustic_model_from_extent(fine_shape, fine_dh, physical_size)

    coarse_solver = _CompiledPropagator(
        Acoustic(spatial_order=4, device=dev),
        shape=coarse_shape,
        dev=dev,
        dh=coarse_dh,
        dt=dt,
        nt=nt,
        abcn=coarse_abcn,
        source_type=["h1"],
        receiver_type=["h1"],
        free_surface=False,
        pml_type="cpmlr",
        use_ckpt=False,
        boundary_saving_config={"enabled": False},
    )
    fine_solver = _CompiledPropagator(
        Acoustic(spatial_order=4, device=dev),
        shape=fine_shape,
        dev=dev,
        dh=fine_dh,
        dt=dt,
        nt=nt,
        abcn=fine_abcn,
        source_type=["h1"],
        receiver_type=["h1"],
        free_surface=False,
        pml_type="cpmlr",
        use_ckpt=False,
        boundary_saving_config={"enabled": False},
    )

    with torch.no_grad():
        coarse_obs = coarse_solver.forward(
            coarse_wave,
            coarse_sources,
            coarse_receivers,
            models=[torch.from_numpy(coarse_model).to(dev)],
        )
        fine_obs = fine_solver.forward(
            fine_wave,
            fine_sources,
            fine_receivers,
            models=[torch.from_numpy(fine_model).to(dev)],
        )

    check_tensor("coarse acoustic obs", coarse_obs)
    check_tensor("fine acoustic obs", fine_obs)

    coarse_np = coarse_obs.detach().cpu().numpy()
    fine_np = fine_obs.detach().cpu().numpy()
    figure_path = save_record_comparison("acoustic_physical_consistency", coarse_np, fine_np)
    normalized_figure_path = save_normalized_record_comparison(
        "acoustic_physical_consistency",
        coarse_np,
        fine_np,
    )

    diff = coarse_np - fine_np

    flat_coarse = coarse_np.reshape(-1)
    flat_fine = fine_np.reshape(-1)
    global_corr = np.corrcoef(flat_coarse, flat_fine)[0, 1]
    
    trace_corrs = []
    for shot in range(coarse_np.shape[0]):
        for rec in range(coarse_np.shape[1]):
            a = coarse_np[shot, rec]
            b = fine_np[shot, rec]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom == 0.0:
                trace_corrs.append(1.0)
            else:
                trace_corrs.append(float(np.dot(a, b) / denom))
    mean_trace_corr = float(np.mean(trace_corrs))

    if not np.isfinite(global_corr):
        raise AssertionError("physical-consistency comparison produced non-finite metrics.")

    return {
        "coarse_shape": tuple(coarse_obs.shape),
        "fine_shape": tuple(fine_obs.shape),
        "coarse_abcn": coarse_abcn,
        "fine_abcn": fine_abcn,
        "coarse_source_scale": float(1.0 / np.prod(coarse_dh)),
        "fine_source_scale": float(1.0 / np.prod(fine_dh)),
        "global_corr": float(global_corr),
        "mean_trace_corr": mean_trace_corr,
        "figure_path": str(figure_path),
        "normalized_figure_path": str(normalized_figure_path),
    }


def main():
    dev = require_cuda()
    torch.manual_seed(0)
    np.random.seed(0)

    acoustic = run_acoustic(dev)
    elastic = run_elastic(dev)
    consistency = run_acoustic_physical_consistency(dev)

    print("Directional dh CUDA smoke test passed.")
    print(f"Acoustic forward shape: {acoustic['forward_shape']}")
    print(f"Acoustic inversion loss: {acoustic['loss']:.6e}")
    print(f"Acoustic grad max: {acoustic['grad_max']:.6e}")
    print(f"Elastic forward shape: {elastic['forward_shape']}")
    print(f"Elastic inversion loss: {elastic['loss']:.6e}")
    print(f"Elastic vp grad max: {elastic['vp_grad_max']:.6e}")
    print(f"Elastic vs grad max: {elastic['vs_grad_max']:.6e}")
    print(f"Physical-size coarse record shape: {consistency['coarse_shape']}")
    print(f"Physical-size fine record shape: {consistency['fine_shape']}")
    print(f"Physical-size coarse/fine abcn: {consistency['coarse_abcn']} / {consistency['fine_abcn']}")
    print(f"Physical-size coarse/fine source scale: {consistency['coarse_source_scale']:.6e} / {consistency['fine_source_scale']:.6e}")
    print(f"Physical-size record global corr: {consistency['global_corr']:.6f}")
    print(f"Physical-size record mean trace corr: {consistency['mean_trace_corr']:.6f}")
    print(f"Physical-size record figure: {consistency['figure_path']}")
    print(f"Physical-size normalized record figure: {consistency['normalized_figure_path']}")


if __name__ == "__main__":
    main()
