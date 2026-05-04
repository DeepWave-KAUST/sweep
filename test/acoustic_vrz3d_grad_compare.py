import argparse
import sys
from pathlib import Path

import numpy as np
import torch


from sweep.equations import AcousticVRZ3D
from sweep.propagator.options import BoundaryOptions, CkptOptions, CUDAOptions, MemoryOptions
from sweep.propagator.torch import PropTorch


OUTPUT_DIR = Path(__file__).resolve().parent / "test_outputs" / "acoustic_vrz3d_grad_compare"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare AcousticVRZ3D eager gradients with CUDA boundary-saving or checkpointing modes."
        )
    )
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.0015)
    parser.add_argument("--dh", type=float, default=10.0)
    parser.add_argument("--freq", type=float, default=10.0)
    parser.add_argument("--delay", type=float, default=0.06)
    parser.add_argument("--spatial-order", type=int, default=4)
    parser.add_argument("--abcn", type=int, default=10)
    parser.add_argument("--receiver-spacing", type=int, default=4)
    parser.add_argument("--receiver-margin", type=int, default=3)
    parser.add_argument("--receiver-depth", type=int, default=0)
    parser.add_argument("--vrz3d-mode", choices=("full", "bs", "ckpt"), default="full")
    parser.add_argument("--ckpt-mode", choices=("chunk", "recursive"), default="chunk")
    parser.add_argument("--ckpt-chunks", type=int, default=100)
    parser.add_argument("--ckpt-count", type=int, default=4)
    parser.add_argument("--boundary-storage", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--transfer-interval", type=int, default=1)
    parser.add_argument("--pinned-memory", action="store_true")
    parser.add_argument("--rel-l2-threshold", type=float, default=1.5)
    parser.add_argument("--z-rel-l2-threshold", type=float, default=5.0)
    parser.add_argument("--cosine-threshold", type=float, default=0.8)
    parser.add_argument("--no-fail", action="store_true", help="Print metrics without failing on threshold mismatch.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-plot", action="store_true", help="Do not save gradient slice figures.")
    parser.add_argument("--show", action="store_true", help="Display gradient slice figures with matplotlib.")
    parser.add_argument("--plot-path", type=Path, default=None, help="Optional output path for the gradient slice figure.")
    parser.add_argument("--record-plot-path", type=Path, default=None, help="Optional output path for the record comparison figure.")
    return parser


def require_cuda_binding():
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-capable PyTorch environment.")

    try:
        import sweep._C as sweep_c
    except Exception as exc:
        raise RuntimeError("Could not import sweep._C. Rebuild the CUDA extension first.") from exc

    required = (
        "acoustic_vrz3d_forward",
        "acoustic_vrz3d_backward",
        "acoustic_vrz3d_backward_bs",
        "acoustic_vrz3d_backward_ckpt",
        "acoustic_vrz3d_backward_recursive_ckpt",
    )
    missing = [name for name in required if not hasattr(sweep_c, name)]
    if missing:
        raise RuntimeError(
            "The loaded sweep._C does not expose AcousticVRZ3D CUDA bindings "
            f"{missing}. Rebuild with `{sys.executable} setup_cuda.py build_ext --inplace`."
        )
    return sweep_c


def ricker(nt, dt, freq, delay):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    x = np.pi * freq * t
    return ((1.0 - 2.0 * x * x) * np.exp(-(x * x))).astype(np.float32)


def make_geometry(shape, receiver_spacing=4, receiver_margin=3, receiver_depth=0):
    nz, ny, nx = shape
    sources = np.array([[nx // 2, ny // 2, max(2, nz // 6)]], dtype=np.int32)

    margin = max(0, int(receiver_margin))
    spacing = max(1, int(receiver_spacing))
    depth = int(np.clip(receiver_depth, 0, nz - 1))

    rec_x = np.arange(margin, nx - margin, spacing, dtype=np.int32)
    rec_y = np.arange(margin, ny - margin, spacing, dtype=np.int32)
    if rec_x.size == 0:
        rec_x = np.array([nx // 2], dtype=np.int32)
    if rec_y.size == 0:
        rec_y = np.array([ny // 2], dtype=np.int32)
    grid_y, grid_x = np.meshgrid(rec_y, rec_x, indexing="ij")
    rec_z = np.full(grid_x.size, depth, dtype=np.int32)
    receivers = np.stack([grid_x.ravel(), grid_y.ravel(), rec_z], axis=-1)[None, ...]
    return sources, receivers


def gaussian_kernel1d(sigma):
    radius = max(1, int(3.0 * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma)).astype(np.float32)
    kernel /= kernel.sum()
    return kernel


def smooth3d(array, sigma_z=4.0, sigma_y=4.0, sigma_x=4.0):
    kz = gaussian_kernel1d(sigma_z)
    ky = gaussian_kernel1d(sigma_y)
    kx = gaussian_kernel1d(sigma_x)

    pad_z = kz.size // 2
    padded = np.pad(array, ((pad_z, pad_z), (0, 0), (0, 0)), mode="edge")
    smoothed_z = np.empty_like(array, dtype=np.float32)
    for iy in range(array.shape[1]):
        for ix in range(array.shape[2]):
            smoothed_z[:, iy, ix] = np.convolve(padded[:, iy, ix], kz, mode="valid")

    pad_y = ky.size // 2
    padded = np.pad(smoothed_z, ((0, 0), (pad_y, pad_y), (0, 0)), mode="edge")
    smoothed_y = np.empty_like(array, dtype=np.float32)
    for iz in range(array.shape[0]):
        for ix in range(array.shape[2]):
            smoothed_y[iz, :, ix] = np.convolve(padded[iz, :, ix], ky, mode="valid")

    pad_x = kx.size // 2
    padded = np.pad(smoothed_y, ((0, 0), (0, 0), (pad_x, pad_x)), mode="edge")
    out = np.empty_like(array, dtype=np.float32)
    for iz in range(array.shape[0]):
        for iy in range(array.shape[1]):
            out[iz, iy, :] = np.convolve(padded[iz, iy, :], kx, mode="valid")
    return out


def make_models(shape):
    nz, ny, nx = shape
    depth = np.linspace(0.0, 1.0, nz, dtype=np.float32)

    vp_true_1d = np.select(
        [depth < 0.25, depth < 0.50, depth < 0.75],
        [1800.0, 2200.0, 2600.0],
        default=3100.0,
    ).astype(np.float32)
    z_true_1d = np.select(
        [depth < 0.25, depth < 0.50, depth < 0.75],
        [1.05, 1.18, 1.34],
        default=1.55,
    ).astype(np.float32)

    vp_true = np.broadcast_to(vp_true_1d[:, None, None], (nz, ny, nx)).copy()
    z_true = np.broadcast_to(z_true_1d[:, None, None], (nz, ny, nx)).copy()

    sigma_z = max(2.0, 0.06 * nz)
    sigma_y = max(2.0, 0.04 * ny)
    sigma_x = max(2.0, 0.04 * nx)
    vp_init = smooth3d(vp_true, sigma_z=sigma_z, sigma_y=sigma_y, sigma_x=sigma_x)
    z_init = smooth3d(z_true, sigma_z=sigma_z, sigma_y=sigma_y, sigma_x=sigma_x)
    return (vp_true, z_true), (vp_init, z_init)


def build_cuda_variant_label(args):
    if args.vrz3d_mode == "full":
        return "vrz3d"
    if args.vrz3d_mode == "bs":
        return "vrz3d_bs"
    if args.vrz3d_mode == "ckpt":
        if args.ckpt_mode == "chunk":
            return f"vrz3d_ckpt_chunk_{args.ckpt_chunks}"
        return f"vrz3d_ckpt_recursive_{args.ckpt_count}"
    raise ValueError(f"Unsupported vrz3d mode {args.vrz3d_mode}.")


def build_solver(backend, shape, device, args):
    base_kwargs = dict(
        shape=shape,
        dev=device,
        dh=(args.dh, args.dh, args.dh),
        dt=args.dt,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=args.abcn,
        pml_type="cpmlr",
        free_surface=False,
    )

    if backend == "eager":
        return PropTorch(
            AcousticVRZ3D(spatial_order=args.spatial_order, device=device, backend="torch"),
            backend="eager",
            **base_kwargs,
        )

    if args.vrz3d_mode == "full":
        return PropTorch(
            AcousticVRZ3D(spatial_order=args.spatial_order, device=device, backend="torch"),
            backend="cuda",
            **base_kwargs,
            use_ckpt=False,
            boundary_saving_config={"enabled": False},
        )

    if args.vrz3d_mode == "bs":
        return PropTorch(
            AcousticVRZ3D(spatial_order=args.spatial_order, device=device, backend="torch"),
            backend="cuda",
            **base_kwargs,
            cuda_options=CUDAOptions(
                memory=MemoryOptions(
                    strategy="boundary",
                    boundary=BoundaryOptions(
                        storage=args.boundary_storage,
                        transfer_interval=args.transfer_interval,
                        pinned_memory=args.pinned_memory,
                    ),
                )
            ),
        )

    if args.vrz3d_mode == "ckpt":
        ckpt_options = CkptOptions(
            mode=args.ckpt_mode,
            chunks=args.ckpt_chunks if args.ckpt_mode == "chunk" else 100,
            count=args.ckpt_count if args.ckpt_mode == "recursive" else 0,
        )
        return PropTorch(
            AcousticVRZ3D(spatial_order=args.spatial_order, device=device, backend="torch"),
            backend="cuda",
            **base_kwargs,
            cuda_options=CUDAOptions(
                memory=MemoryOptions(
                    strategy="ckpt",
                    ckpt=ckpt_options,
                )
            ),
        )

    raise ValueError(f"Unsupported vrz3d mode {args.vrz3d_mode}.")


def normalize_record(record, nshots, nreceivers, nt):
    if record.ndim == 4:
        if record.shape[-1] != 1:
            raise ValueError(f"Unsupported 4-D record shape {tuple(record.shape)}")
        record = record[..., 0]
    if tuple(record.shape) == (nshots, nreceivers, nt):
        return record
    if tuple(record.shape) == (nshots, nt, nreceivers):
        return record.transpose(1, 2)
    raise ValueError(
        f"Unsupported record shape {tuple(record.shape)}; expected "
        f"{(nshots, nreceivers, nt)} or {(nshots, nt, nreceivers)}."
    )


def run_forward(backend, variant_label, shape, wavelet, sources, receivers, vp_np, z_np, device, args):
    solver = build_solver(backend, shape, device, args)
    vp = torch.tensor(vp_np, device=device, dtype=torch.float32)
    z_model = torch.tensor(z_np, device=device, dtype=torch.float32)

    with torch.no_grad():
        record = solver(wavelet, sources, receivers, models=[vp, z_model])
        record = normalize_record(record, sources.shape[0], receivers.shape[1], wavelet.shape[-1])
        torch.cuda.synchronize(device)
    if not torch.isfinite(record).all():
        raise RuntimeError(f"{backend}:{variant_label} record contains NaN/Inf.")
    return record.detach()


def run_gradient(backend, variant_label, shape, wavelet, sources, receivers, observed, vp_np, z_np, device, args):
    solver = build_solver(backend, shape, device, args)
    vp = torch.tensor(vp_np, device=device, dtype=torch.float32, requires_grad=True)
    z_model = torch.tensor(z_np, device=device, dtype=torch.float32, requires_grad=True)

    record = solver(wavelet, sources, receivers, models=[vp, z_model])
    record = normalize_record(record, sources.shape[0], receivers.shape[1], wavelet.shape[-1])
    residual = record - observed
    loss = residual.square().mean()
    loss.backward()
    torch.cuda.synchronize(device)

    grads = {"vp": vp.grad, "z": z_model.grad}
    for name, grad in grads.items():
        if grad is None:
            raise RuntimeError(f"{backend}:{variant_label} {name} gradient is missing.")
        if not torch.isfinite(grad).all():
            raise RuntimeError(f"{backend}:{variant_label} {name} gradient contains NaN/Inf.")
        if float(grad.detach().abs().max().cpu()) == 0.0:
            raise RuntimeError(f"{backend}:{variant_label} {name} gradient is identically zero.")

    return {
        "loss": float(loss.detach().cpu()),
        "record": record.detach().cpu(),
        "residual": residual.detach().cpu(),
        "grads": {name: grad.detach().cpu() for name, grad in grads.items()},
    }


def metric_pair(reference, candidate):
    ref = reference.reshape(-1)
    cand = candidate.reshape(-1)
    diff = cand - ref
    denom = torch.linalg.norm(ref).clamp_min(1e-12)
    return {
        "rel_l2": float(torch.linalg.norm(diff) / denom),
        "cosine": float(torch.nn.functional.cosine_similarity(cand[None], ref[None])),
        "diff_l2": float(torch.linalg.norm(diff)),
        "diff_linf": float(diff.abs().max()),
        "ref_l2": float(torch.linalg.norm(ref)),
        "cand_l2": float(torch.linalg.norm(cand)),
    }


def save_outputs(output_dir, results, candidate_key, metrics, observed, true_models, init_models):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"acoustic_vrz3d_grad_compare_{candidate_key}.npz"
    payload = {
        "observed_record": observed.detach().cpu().numpy(),
        "eager_record": results["eager"]["record"].numpy(),
        f"{candidate_key}_record": results[candidate_key]["record"].numpy(),
        "eager_residual": results["eager"]["residual"].numpy(),
        f"{candidate_key}_residual": results[candidate_key]["residual"].numpy(),
        "eager_vp_grad": results["eager"]["grads"]["vp"].numpy(),
        f"{candidate_key}_vp_grad": results[candidate_key]["grads"]["vp"].numpy(),
        "eager_z_grad": results["eager"]["grads"]["z"].numpy(),
        f"{candidate_key}_z_grad": results[candidate_key]["grads"]["z"].numpy(),
        "vp_true": true_models[0],
        "z_true": true_models[1],
        "vp_init": init_models[0],
        "z_init": init_models[1],
        "metrics": np.array([str(metrics)], dtype=object),
    }
    np.savez(path, **payload)
    return path


def center_slices(volume):
    nz, ny, nx = volume.shape
    return (
        (f"z={nz // 2} (y-x)", volume[nz // 2, :, :]),
        (f"y={ny // 2} (z-x)", volume[:, ny // 2, :]),
        (f"x={nx // 2} (z-y)", volume[:, :, nx // 2]),
    )


def symmetric_limit(*arrays):
    max_abs = max(float(np.nanmax(np.abs(array))) for array in arrays if array.size)
    return (-max_abs, max_abs) if max_abs > 0 else (-1.0, 1.0)


def percentile_limits(array):
    abs_arr = np.abs(array)
    vmax = float(np.percentile(abs_arr, 99.5)) if abs_arr.size else 1.0
    if vmax <= 0:
        vmax = float(abs_arr.max()) if abs_arr.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def load_pyplot(show):
    if not show:
        import matplotlib

        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_record_comparison(results, candidate_key, output_dir, plot_path=None, show=False):
    plt = load_pyplot(show)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = plot_path or (output_dir / f"acoustic_vrz3d_record_comparison_{candidate_key}.png")

    eager = results["eager"]["residual"].numpy()
    candidate = results[candidate_key]["residual"].numpy()
    residual = candidate - eager
    limit = symmetric_limit(eager, candidate)

    panels = (
        ("eager data residual", eager.reshape(-1, eager.shape[-1])),
        (f"{candidate_key} data residual", candidate.reshape(-1, candidate.shape[-1])),
        (f"{candidate_key}-eager residual diff", residual.reshape(-1, residual.shape[-1])),
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.2), constrained_layout=True)
    for ax, (title, data) in zip(axes, panels):
        vmin, vmax = percentile_limits(data) if "residual" in title else limit
        im = ax.imshow(data, cmap="seismic", origin="upper", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("time step")
        ax.set_ylabel("shot/receiver")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_gradient_slices(results, candidate_key, output_dir, plot_path=None, show=False):
    plt = load_pyplot(show)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = plot_path or (output_dir / f"acoustic_vrz3d_gradient_slices_{candidate_key}.png")

    row_specs = []
    for model_name in ("vp", "z"):
        eager = results["eager"]["grads"][model_name].numpy()
        candidate = results[candidate_key]["grads"][model_name].numpy()
        diff = candidate - eager
        model_limits = percentile_limits(np.concatenate([eager.ravel(), candidate.ravel()]))
        row_specs.extend(
            (
                (f"{model_name} eager", eager, model_limits),
                (f"{model_name} {candidate_key}", candidate, model_limits),
                (f"{model_name} {candidate_key}-eager", diff, percentile_limits(diff)),
            )
        )

    fig, axes = plt.subplots(len(row_specs), 3, figsize=(13.0, 2.5 * len(row_specs)), constrained_layout=True)
    if len(row_specs) == 1:
        axes = axes[None, :]

    for row, (row_label, volume, (vmin, vmax)) in enumerate(row_specs):
        for col, (slice_label, slice_data) in enumerate(center_slices(volume)):
            ax = axes[row, col]
            im = ax.imshow(slice_data, cmap="seismic", origin="upper", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(slice_label)
            if col == 0:
                ax.set_ylabel(row_label)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)
    return path


def main():
    args = build_parser().parse_args()
    sweep_c = require_cuda_binding()
    torch.manual_seed(0)
    np.random.seed(0)

    shape = (args.nz, args.ny, args.nx)
    device = torch.device("cuda")
    wavelet_np = ricker(args.nt, args.dt, args.freq, args.delay)[None, None, :]
    wavelet = torch.tensor(wavelet_np, device=device)
    sources, receivers = make_geometry(
        shape,
        receiver_spacing=args.receiver_spacing,
        receiver_margin=args.receiver_margin,
        receiver_depth=args.receiver_depth,
    )
    true_models, init_models = make_models(shape)
    vp_true_np, z_true_np = true_models
    vp_init_np, z_init_np = init_models

    candidate_key = build_cuda_variant_label(args)

    print(f"python={sys.executable}")
    print(f"sweep._C={sweep_c.__file__}")
    print(f"shape={shape}, nt={args.nt}, abcn={args.abcn}, spatial_order={args.spatial_order}, vrz3d_mode={args.vrz3d_mode}")
    print(f"candidate={candidate_key}")
    print(f"sources={sources.shape}, receivers={receivers.shape}")
    print(
        "models: true=layered, init=smoothed true; "
        f"vp_true=[{vp_true_np.min():.1f}, {vp_true_np.max():.1f}], "
        f"vp_init=[{vp_init_np.min():.1f}, {vp_init_np.max():.1f}], "
        f"z_true=[{z_true_np.min():.3f}, {z_true_np.max():.3f}], "
        f"z_init=[{z_init_np.min():.3f}, {z_init_np.max():.3f}]"
    )

    observed = run_forward(
        "eager",
        "observed_true",
        shape,
        wavelet,
        sources,
        receivers,
        vp_true_np,
        z_true_np,
        device,
        args,
    )

    results = {
        "eager": run_gradient(
            "eager", "eager", shape, wavelet, sources, receivers, observed,
            vp_init_np, z_init_np, device, args
        ),
        candidate_key: run_gradient(
            "cuda", candidate_key, shape, wavelet, sources, receivers, observed,
            vp_init_np, z_init_np, device, args
        ),
    }

    metrics = {
        "record": metric_pair(results["eager"]["record"], results[candidate_key]["record"]),
        "vp_grad": metric_pair(results["eager"]["grads"]["vp"], results[candidate_key]["grads"]["vp"]),
        "z_grad": metric_pair(results["eager"]["grads"]["z"], results[candidate_key]["grads"]["z"]),
    }
    output_path = save_outputs(args.output_dir, results, candidate_key, metrics, observed, true_models, init_models)
    gradient_plot_path = None
    record_plot_path = None
    if not args.no_plot:
        record_plot_path = plot_record_comparison(results, candidate_key, args.output_dir, args.record_plot_path, args.show)
        gradient_plot_path = plot_gradient_slices(results, candidate_key, args.output_dir, args.plot_path, args.show)

    print(f"eager loss: {results['eager']['loss']:.6e}")
    print(f"{candidate_key} loss:  {results[candidate_key]['loss']:.6e}")
    for name, item in metrics.items():
        print(
            f"{name}: rel_l2={item['rel_l2']:.6e}, cosine={item['cosine']:.6f}, "
            f"diff_l2={item['diff_l2']:.6e}, diff_linf={item['diff_linf']:.6e}, "
            f"eager_l2={item['ref_l2']:.6e}, {candidate_key}_l2={item['cand_l2']:.6e}"
        )
    print(f"saved: {output_path}")
    if record_plot_path is not None:
        print(f"record comparison: {record_plot_path}")
    if gradient_plot_path is not None:
        print(f"gradient slices: {gradient_plot_path}")

    if args.no_fail:
        return

    failures = []
    for name in ("vp_grad", "z_grad"):
        item = metrics[name]
        rel_l2_threshold = args.z_rel_l2_threshold if name == "z_grad" else args.rel_l2_threshold
        if item["rel_l2"] > rel_l2_threshold:
            failures.append(f"{name} rel_l2 {item['rel_l2']:.6e} > {rel_l2_threshold:.6e}")
        if item["cosine"] < args.cosine_threshold:
            failures.append(f"{name} cosine {item['cosine']:.6f} < {args.cosine_threshold:.6f}")
    if failures:
        raise AssertionError(
            f"[candidate={candidate_key}] " + "; ".join(failures)
        )


if __name__ == "__main__":
    main()
