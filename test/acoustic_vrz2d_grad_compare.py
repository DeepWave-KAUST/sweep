import argparse
import sys
from pathlib import Path

import numpy as np
import torch


from sweep.equations import AcousticVRZ
from sweep.propagator.options import BoundaryOptions, CUDAOptions, MemoryOptions
from sweep.propagator.torch import PropTorch


OUTPUT_DIR = Path(__file__).resolve().parent / "test_outputs" / "acoustic_vrz2d_grad_compare"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare AcousticVRZ 2D eager and CUDA model gradients."
    )
    parser.add_argument("--nz", type=int, default=96)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--nt", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.0015)
    parser.add_argument("--dh", type=float, default=10.0)
    parser.add_argument("--freq", type=float, default=10.0)
    parser.add_argument("--delay", type=float, default=0.06)
    parser.add_argument("--spatial-order", type=int, default=8)
    parser.add_argument("--abcn", type=int, default=10)
    parser.add_argument("--receiver-spacing", type=int, default=4)
    parser.add_argument("--receiver-margin", type=int, default=3)
    parser.add_argument("--receiver-depth", type=int, default=0)
    parser.add_argument("--vrz2d-mode", choices=("full", "bs"), default="bs")
    parser.add_argument("--boundary-storage", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--transfer-interval", type=int, default=1)
    parser.add_argument("--pinned-memory", action="store_true")
    parser.add_argument("--rel-l2-threshold", type=float, default=1.5)
    parser.add_argument("--z-rel-l2-threshold", type=float, default=5.0)
    parser.add_argument("--cosine-threshold", type=float, default=0.8)
    parser.add_argument("--no-fail", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--plot-path", type=Path, default=None)
    parser.add_argument("--record-plot-path", type=Path, default=None)
    return parser


def require_cuda_binding():
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-capable PyTorch environment.")

    try:
        import sweep._C as sweep_c
    except Exception as exc:
        raise RuntimeError("Could not import sweep._C. Rebuild the CUDA extension first.") from exc

    required = (
        "acoustic_vrz2d_forward",
        "acoustic_vrz2d_backward",
        "acoustic_vrz2d_backward_bs",
    )
    missing = [name for name in required if not hasattr(sweep_c, name)]
    if missing:
        raise RuntimeError(f"The loaded sweep._C is missing AcousticVRZ 2D bindings {missing}.")
    return sweep_c


def ricker(nt, dt, freq, delay):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    x = np.pi * freq * t
    return ((1.0 - 2.0 * x * x) * np.exp(-(x * x))).astype(np.float32)


def make_geometry(shape, receiver_spacing=4, receiver_margin=3, receiver_depth=0):
    nz, nx = shape
    sources = np.array([[nx // 2, max(2, nz // 6)]], dtype=np.int32)

    margin = max(0, int(receiver_margin))
    spacing = max(1, int(receiver_spacing))
    depth = int(np.clip(receiver_depth, 0, nz - 1))

    rec_x = np.arange(margin, nx - margin, spacing, dtype=np.int32)
    if rec_x.size == 0:
        rec_x = np.array([nx // 2], dtype=np.int32)
    rec_z = np.full(rec_x.size, depth, dtype=np.int32)
    receivers = np.stack([rec_x, rec_z], axis=-1)[None, ...]
    return sources, receivers


def gaussian_kernel1d(sigma):
    radius = max(1, int(3.0 * sigma + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma)).astype(np.float32)
    kernel /= kernel.sum()
    return kernel


def smooth2d(array, sigma_z=5.0, sigma_x=5.0):
    kz = gaussian_kernel1d(sigma_z)
    kx = gaussian_kernel1d(sigma_x)

    pad_z = kz.size // 2
    padded = np.pad(array, ((pad_z, pad_z), (0, 0)), mode="edge")
    smoothed = np.empty_like(array, dtype=np.float32)
    for ix in range(array.shape[1]):
        smoothed[:, ix] = np.convolve(padded[:, ix], kz, mode="valid")

    pad_x = kx.size // 2
    padded = np.pad(smoothed, ((0, 0), (pad_x, pad_x)), mode="edge")
    out = np.empty_like(array, dtype=np.float32)
    for iz in range(array.shape[0]):
        out[iz, :] = np.convolve(padded[iz, :], kx, mode="valid")
    return out


def make_models(shape):
    nz, nx = shape
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

    vp_true = np.repeat(vp_true_1d[:, None], nx, axis=1).astype(np.float32)
    z_true = np.repeat(z_true_1d[:, None], nx, axis=1).astype(np.float32)

    sigma_z = max(2.0, 0.06 * nz)
    sigma_x = max(2.0, 0.04 * nx)
    vp_init = smooth2d(vp_true, sigma_z=sigma_z, sigma_x=sigma_x)
    z_init = smooth2d(z_true, sigma_z=sigma_z, sigma_x=sigma_x)
    return (vp_true, z_true), (vp_init, z_init)


def candidate_label(args):
    return "vrz2d" if args.vrz2d_mode == "full" else "vrz2d_bs"


def build_solver(backend, shape, device, args):
    base_kwargs = dict(
        shape=shape,
        dev=device,
        dh=(args.dh, args.dh),
        dt=args.dt,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=args.abcn,
        pml_type="cpmlr",
        free_surface=False,
    )

    if backend == "eager":
        return PropTorch(
            AcousticVRZ(spatial_order=args.spatial_order, device=device, backend="torch"),
            backend="eager",
            **base_kwargs,
        )

    if args.vrz2d_mode == "full":
        return PropTorch(
            AcousticVRZ(spatial_order=args.spatial_order, device=device, backend="torch"),
            backend="cuda",
            **base_kwargs,
            use_ckpt=False,
            boundary_saving_config={"enabled": False},
        )

    return PropTorch(
        AcousticVRZ(spatial_order=args.spatial_order, device=device, backend="torch"),
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


def normalize_record(record, nshots, nreceivers, nt):
    if record.ndim == 4:
        if record.shape[-1] != 1:
            raise ValueError(f"Unsupported 4-D record shape {tuple(record.shape)}")
        record = record[..., 0]
    if tuple(record.shape) == (nshots, nreceivers, nt):
        return record
    if tuple(record.shape) == (nshots, nt, nreceivers):
        return record.transpose(1, 2)
    raise ValueError(f"Unsupported record shape {tuple(record.shape)}.")


def run_forward(backend, label, shape, wavelet, sources, receivers, vp_np, z_np, device, args):
    solver = build_solver(backend, shape, device, args)
    vp = torch.tensor(vp_np, device=device, dtype=torch.float32)
    z_model = torch.tensor(z_np, device=device, dtype=torch.float32)

    with torch.no_grad():
        record = solver(wavelet, sources, receivers, models=[vp, z_model])
        record = normalize_record(record, sources.shape[0], receivers.shape[1], wavelet.shape[-1])
        torch.cuda.synchronize(device)
    if not torch.isfinite(record).all():
        raise RuntimeError(f"{backend}:{label} record contains NaN/Inf.")
    return record.detach()


def run_gradient(backend, label, shape, wavelet, sources, receivers, observed, vp_np, z_np, device, args):
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
            raise RuntimeError(f"{backend}:{label} {name} gradient is missing.")
        if not torch.isfinite(grad).all():
            raise RuntimeError(f"{backend}:{label} {name} gradient contains NaN/Inf.")
        if float(grad.detach().abs().max().cpu()) == 0.0:
            raise RuntimeError(f"{backend}:{label} {name} gradient is identically zero.")

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
    ref_norm = torch.linalg.vector_norm(ref).item()
    cand_norm = torch.linalg.vector_norm(cand).item()
    diff_norm = torch.linalg.vector_norm(diff).item()
    denom = max(ref_norm, 1e-30)
    cosine = torch.dot(ref, cand).item() / max(ref_norm * cand_norm, 1e-30)
    return {
        "rel_l2": diff_norm / denom,
        "cosine": cosine,
        "diff_l2": diff_norm,
        "diff_linf": diff.abs().max().item(),
        "ref_l2": ref_norm,
        "cand_l2": cand_norm,
    }


def save_outputs(output_dir, results, key, metrics, observed, true_models, init_models):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"acoustic_vrz2d_grad_compare_{key}.npz"
    np.savez(
        path,
        observed_record=observed.detach().cpu().numpy(),
        eager_record=results["eager"]["record"].numpy(),
        **{f"{key}_record": results[key]["record"].numpy()},
        eager_residual=results["eager"]["residual"].numpy(),
        **{f"{key}_residual": results[key]["residual"].numpy()},
        eager_vp_grad=results["eager"]["grads"]["vp"].numpy(),
        **{f"{key}_vp_grad": results[key]["grads"]["vp"].numpy()},
        eager_z_grad=results["eager"]["grads"]["z"].numpy(),
        **{f"{key}_z_grad": results[key]["grads"]["z"].numpy()},
        vp_true=true_models[0],
        z_true=true_models[1],
        vp_init=init_models[0],
        z_init=init_models[1],
        metrics=np.array([str(metrics)], dtype=object),
    )
    return path


def load_pyplot(show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def symmetric_limit(*arrays):
    max_abs = max(float(np.nanmax(np.abs(arr))) for arr in arrays if arr.size)
    return (-max_abs, max_abs) if max_abs > 0 else (-1.0, 1.0)


def percentile_limits(array):
    abs_arr = np.abs(array)
    vmax = float(np.percentile(abs_arr, 99.5)) if abs_arr.size else 1.0
    if vmax <= 0:
        vmax = float(abs_arr.max()) if abs_arr.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def plot_record_comparison(results, key, output_dir, plot_path=None, show=False):
    plt = load_pyplot(show)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = plot_path or (output_dir / f"acoustic_vrz2d_record_comparison_{key}.png")
    eager = results["eager"]["residual"].numpy()
    cand = results[key]["residual"].numpy()
    residual = cand - eager
    limit = symmetric_limit(eager, cand)
    panels = (
        ("eager data residual", eager.reshape(-1, eager.shape[-1])),
        (f"{key} data residual", cand.reshape(-1, cand.shape[-1])),
        (f"{key}-eager residual diff", residual.reshape(-1, residual.shape[-1])),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.2), constrained_layout=True)
    for ax, (title, data) in zip(axes, panels):
        vmin, vmax = percentile_limits(data) if "residual" in title else limit
        im = ax.imshow(data, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("trace")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_gradient_comparison(results, key, output_dir, plot_path=None, show=False):
    plt = load_pyplot(show)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = plot_path or (output_dir / f"acoustic_vrz2d_gradient_compare_{key}.png")
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), constrained_layout=True)

    for row, model_name in enumerate(("vp", "z")):
        eager = results["eager"]["grads"][model_name].numpy()
        cand = results[key]["grads"][model_name].numpy()
        diff = cand - eager
        vmin, vmax = percentile_limits(np.concatenate([eager.ravel(), cand.ravel()]))
        dmin, dmax = percentile_limits(diff)
        panels = (
            (f"{model_name} eager", eager, vmin, vmax),
            (f"{model_name} {key}", cand, vmin, vmax),
            (f"{model_name} {key}-eager", diff, dmin, dmax),
        )
        for col, (title, data, lo, hi) in enumerate(panels):
            im = axes[row, col].imshow(data, cmap="seismic", origin="upper", aspect="auto", vmin=lo, vmax=hi)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel("x")
            axes[row, col].set_ylabel("z")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.02)

    fig.savefig(path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return path


def main():
    args = build_parser().parse_args()
    sweep_c = require_cuda_binding()
    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)

    shape = (args.nz, args.nx)
    sources, receivers = make_geometry(
        shape,
        receiver_spacing=args.receiver_spacing,
        receiver_margin=args.receiver_margin,
        receiver_depth=args.receiver_depth,
    )
    wavelet_np = ricker(args.nt, args.dt, args.freq, args.delay)[None, None, :]
    wavelet = torch.tensor(wavelet_np, device=device)
    true_models, init_models = make_models(shape)
    vp_true_np, z_true_np = true_models
    vp_init_np, z_init_np = init_models
    key = candidate_label(args)

    print(f"python={sys.executable}")
    print(f"sweep._C={sweep_c.__file__}")
    print(f"shape={shape}, nt={args.nt}, abcn={args.abcn}, spatial_order={args.spatial_order}, vrz2d_mode={args.vrz2d_mode}")
    print(f"candidate={key}")
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
        key: run_gradient(
            "cuda", key, shape, wavelet, sources, receivers, observed,
            vp_init_np, z_init_np, device, args
        ),
    }
    metrics = {
        "record": metric_pair(results["eager"]["record"], results[key]["record"]),
        "vp_grad": metric_pair(results["eager"]["grads"]["vp"], results[key]["grads"]["vp"]),
        "z_grad": metric_pair(results["eager"]["grads"]["z"], results[key]["grads"]["z"]),
    }

    output_path = save_outputs(args.output_dir, results, key, metrics, observed, true_models, init_models)
    record_plot_path = None
    gradient_plot_path = None
    if not args.no_plot:
        record_plot_path = plot_record_comparison(results, key, args.output_dir, args.record_plot_path, args.show)
        gradient_plot_path = plot_gradient_comparison(results, key, args.output_dir, args.plot_path, args.show)

    print(f"eager loss: {results['eager']['loss']:.6e}")
    print(f"{key} loss:  {results[key]['loss']:.6e}")
    for name, item in metrics.items():
        print(
            f"{name}: rel_l2={item['rel_l2']:.6e}, cosine={item['cosine']:.6f}, "
            f"diff_l2={item['diff_l2']:.6e}, diff_linf={item['diff_linf']:.6e}, "
            f"eager_l2={item['ref_l2']:.6e}, {key}_l2={item['cand_l2']:.6e}"
        )
    print(f"saved: {output_path}")
    if record_plot_path is not None:
        print(f"record comparison: {record_plot_path}")
    if gradient_plot_path is not None:
        print(f"gradient comparison: {gradient_plot_path}")

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
        raise AssertionError(f"[candidate={key}] " + "; ".join(failures))


if __name__ == "__main__":
    main()
