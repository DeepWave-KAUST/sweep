import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from sweep import is_torch_binding_available
from sweep.equations import Acoustic
from sweep.propagator.cuda import PropCUDA
from sweep.signal import ricker

OUTPUT_DIR = Path(__file__).resolve().parent / "test_outputs" / "acoustic2d_rtm_layered"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if not is_torch_binding_available():
        raise RuntimeError("sweep._C is not available or the CUDA binding cannot be imported.")
    return torch.device("cuda")


def make_wavelet(nt, dt, freq_hz, delay_s=None):
    t = np.arange(nt, dtype=np.float32) * dt
    if delay_s is None:
        delay_s = 1.2 / float(freq_hz)
    return ricker(t - delay_s, f=float(freq_hz)).astype(np.float32)


def make_layered_model(shape):
    nz, nx = shape
    vp = np.full((nz, nx), 1800.0, dtype=np.float32)
    z1 = int(round(nz * 0.28))
    z2 = int(round(nz * 0.58))
    vp[z1:, :] = 2300.0
    vp[z2:, :] = 2900.0

    # Add a gentle lateral perturbation so the image is not perfectly flat.
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)[None, :]
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None]
    vp += 90.0 * np.exp(-((x / 0.35) ** 2 + ((z - 0.62) / 0.10) ** 2))
    return vp.astype(np.float32)


def smooth_along_z(model, half_width=8):
    model = np.asarray(model, dtype=np.float32)
    padded = np.pad(model, ((half_width, half_width), (0, 0)), mode="edge")
    kernel = np.ones(2 * half_width + 1, dtype=np.float32) / float(2 * half_width + 1)
    out = np.empty_like(model)
    for ix in range(model.shape[1]):
        out[:, ix] = np.convolve(padded[:, ix], kernel, mode="valid")
    return out


def build_background_from_top_layer(model):
    top = np.asarray(model[0:1, :], dtype=np.float32)
    return np.repeat(top, model.shape[0], axis=0).astype(np.float32)


def make_geometry(nx, src_z, rec_z, nshots, nrec):
    src_x = np.linspace(8, nx - 9, nshots, dtype=np.int32)
    rec_x = np.linspace(4, nx - 5, nrec, dtype=np.int32)
    sources = np.stack([src_x, np.full_like(src_x, src_z)], axis=1)
    receivers_single = np.stack([rec_x, np.full_like(rec_x, rec_z)], axis=1)
    receivers = np.repeat(receivers_single[None, :, :], nshots, axis=0)
    return sources, receivers


def crop_cuda_volume_2d(volume, abcn, spatial_order, free_surface):
    m = spatial_order // 2
    if free_surface:
        z_slice = slice(m, -(abcn + m))
    else:
        z_slice = slice(abcn + m, -(abcn + m))
    x_slice = slice(abcn + m, -(abcn + m))
    return volume[z_slice, x_slice]


def plot_models(true_vp, mig_vp, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    vmin, vmax = np.percentile(true_vp, [1, 99])
    for ax, data, title in zip(axes, [true_vp, mig_vp], ["True Layered Model", "Migration Model"]):
        im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("X Grid")
        ax.set_ylabel("Z Grid")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_records(obs, syn, residual, out_path):
    obs = np.asarray(obs, dtype=np.float32).T
    syn = np.asarray(syn, dtype=np.float32).T
    residual = np.asarray(residual, dtype=np.float32).T
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    panels = [obs, syn, residual]
    titles = ["Observed", "Synthetic", "Residual"]
    for ax, panel, title in zip(axes, panels, titles):
        amp = max(float(np.percentile(np.abs(panel), 99)), 1e-6)
        ax.imshow(panel, cmap="seismic", aspect="auto", vmin=-amp, vmax=amp)
        ax.set_title(title)
        ax.set_xlabel("Receiver")
        ax.set_ylabel("Time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rtm_panels(true_vp, image, src_illum, rec_illum, normalized, out_path):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    model_vmin, model_vmax = np.percentile(true_vp, [1, 99])
    image_amp = max(float(np.percentile(np.abs(image), 99)), 1e-6)
    illum_vmax = max(float(np.percentile(src_illum + rec_illum, 99)), 1e-6)
    norm_amp = max(float(np.percentile(np.abs(normalized), 99)), 1e-6)
    panels = [
        (true_vp, "True Model", "viridis", model_vmin, model_vmax),
        (image, "RTM Image", "seismic", -image_amp, image_amp),
        (src_illum, "Src Illum", "magma", 0.0, illum_vmax),
        (rec_illum, "Rec Illum", "magma", 0.0, illum_vmax),
        (normalized, "Illum-Norm", "seismic", -norm_amp, norm_amp),
    ]
    for ax, (panel, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("X Grid")
        ax.set_ylabel("Z Grid")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_solver(args, shape, device):
    return PropCUDA(
        Acoustic(spatial_order=args.spatial_order, device=device),
        shape=shape,
        dev=device,
        dh=(args.dz, args.dx),
        dt=args.dt,
        nt=args.nt,
        abcn=args.abcn,
        source_type=["h1"],
        receiver_type=["h1"],
        free_surface=args.free_surface,
        pml_type="cpmlr",
        use_ckpt=False,
        allow_growth=True,
        boundary_saving_config={"enabled": False},
    )


def build_parser():
    parser = argparse.ArgumentParser(description="2D acoustic CUDA RTM layered-model smoke test.")
    parser.add_argument("--nz", type=int, default=120)
    parser.add_argument("--nx", type=int, default=220)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--nt", type=int, default=1500)
    parser.add_argument("--fm", type=float, default=20.0)
    parser.add_argument("--abcn", type=int, default=20)
    parser.add_argument("--spatial-order", type=int, default=4)
    parser.add_argument("--nshots", type=int, default=30)
    parser.add_argument("--nrec", type=int, default=96)
    parser.add_argument("--src-z", type=int, default=2)
    parser.add_argument("--rec-z", type=int, default=2)
    parser.add_argument("--free-surface", action="store_true", default=True)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--output-prefix", default="layered_2d_rtm")
    return parser


def main():
    args = build_parser().parse_args()
    device = require_cuda()

    shape = (args.nz, args.nx)
    true_vp = make_layered_model(shape)
    mig_vp = smooth_along_z(true_vp, half_width=8)
    bg_true = build_background_from_top_layer(true_vp)
    bg_mig = build_background_from_top_layer(mig_vp)

    sources, receivers = make_geometry(args.nx, args.src_z, args.rec_z, args.nshots, args.nrec)
    wave = make_wavelet(args.nt, args.dt, args.fm)
    solver = build_solver(args, shape, device)

    true_vp_t = torch.from_numpy(true_vp).to(device)
    mig_vp_t = torch.from_numpy(mig_vp).to(device)
    bg_true_t = torch.from_numpy(bg_true).to(device)
    bg_mig_t = torch.from_numpy(bg_mig).to(device)

    image = np.zeros_like(true_vp, dtype=np.float32)
    src_illum = np.zeros_like(true_vp, dtype=np.float32)
    rec_illum = np.zeros_like(true_vp, dtype=np.float32)
    preview_obs = None
    preview_syn = None
    preview_residual = None

    prefix = OUTPUT_DIR / args.output_prefix
    plot_models(true_vp, mig_vp, Path(f"{prefix}_models.png"))

    print(f"Device: {device}")
    print(f"Output prefix: {prefix}")
    print(f"Shots: {sources.shape[0]}, receivers per shot: {receivers.shape[1]}")

    for ishot in range(sources.shape[0]):
        shot_slice = slice(ishot, ishot + 1)
        shot_src = sources[shot_slice]
        shot_rec = receivers[shot_slice]

        with torch.no_grad():
            obs = solver(wave, shot_src, shot_rec, models=[true_vp_t])
            obs_direct = solver(wave, shot_src, shot_rec, models=[bg_true_t])
            syn = solver(wave, shot_src, shot_rec, models=[mig_vp_t])
            syn_direct = solver(wave, shot_src, shot_rec, models=[bg_mig_t])

        residual = (obs - obs_direct) - (syn - syn_direct)
        _, shot_image, shot_src_illum, shot_rec_illum = solver.rtm(
            wave,
            shot_src,
            shot_rec,
            residual.detach().cpu().numpy(),
            models=[mig_vp_t],
            use_boundary_saving=False,
        )
        torch.cuda.synchronize(device)

        image += crop_cuda_volume_2d(
            shot_image.detach().cpu().numpy()[0, 0],
            args.abcn,
            args.spatial_order,
            free_surface=args.free_surface,
        )
        src_illum += crop_cuda_volume_2d(
            shot_src_illum.detach().cpu().numpy()[0, 0],
            args.abcn,
            args.spatial_order,
            free_surface=args.free_surface,
        )
        rec_illum += crop_cuda_volume_2d(
            shot_rec_illum.detach().cpu().numpy()[0, 0],
            args.abcn,
            args.spatial_order,
            free_surface=args.free_surface,
        )

        if preview_obs is None:
            preview_obs = obs[0].detach().cpu().numpy()
            preview_syn = syn[0].detach().cpu().numpy()
            preview_residual = residual[0].detach().cpu().numpy()

    normalized = image / np.sqrt(src_illum * rec_illum + float(args.eps))

    np.save(f"{prefix}_image.npy", image)
    np.save(f"{prefix}_source_illumination.npy", src_illum)
    np.save(f"{prefix}_receiver_illumination.npy", rec_illum)
    np.save(f"{prefix}_normalized.npy", normalized)

    if preview_obs is not None:
        plot_records(preview_obs, preview_syn, preview_residual, Path(f"{prefix}_records.png"))
    plot_rtm_panels(true_vp, image, src_illum, rec_illum, normalized, Path(f"{prefix}_panels.png"))

    print(f"Saved RTM image to {prefix}_image.npy")
    print(f"Saved source illumination to {prefix}_source_illumination.npy")
    print(f"Saved receiver illumination to {prefix}_receiver_illumination.npy")
    print(f"Saved normalized RTM to {prefix}_normalized.npy")
    print(f"Saved figures to {prefix}_models.png, {prefix}_records.png, {prefix}_panels.png")


if __name__ == "__main__":
    main()
