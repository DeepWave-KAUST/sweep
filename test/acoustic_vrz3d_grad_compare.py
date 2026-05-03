import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def find_repo_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "src").exists() and (candidate / "test").exists():
            return candidate
    raise RuntimeError("Could not locate the repository root.")


REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

from sweep.equations import AcousticVRZ3D
from sweep.propagator.torch import PropTorch


OUTPUT_DIR = REPO_ROOT / "test" / "test_outputs" / "acoustic_vrz3d_grad_compare"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare AcousticVRZ3D eager and CUDA model gradients on a small deterministic problem."
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


def make_models(shape):
    nz, ny, nx = shape
    zz = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None, None]
    yy = np.linspace(0.0, 1.0, ny, dtype=np.float32)[None, :, None]
    xx = np.linspace(0.0, 1.0, nx, dtype=np.float32)[None, None, :]

    vp = (
        1800.0
        + 350.0 * zz
        + 80.0 * np.sin(2.0 * np.pi * xx)
        + 50.0 * np.cos(2.0 * np.pi * yy)
    ).astype(np.float32)
    z_model = (
        1.05
        + 0.18 * zz
        + 0.04 * np.sin(np.pi * yy)
        + 0.03 * np.cos(np.pi * xx)
    ).astype(np.float32)
    return vp, z_model


def build_solver(backend, shape, device, args):
    return PropTorch(
        AcousticVRZ3D(spatial_order=args.spatial_order, device=device, backend="torch"),
        backend=backend,
        shape=shape,
        dev=device,
        dh=(args.dh, args.dh, args.dh),
        dt=args.dt,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=args.abcn,
        pml_type="cpmlr",
        free_surface=False,
        use_ckpt=False,
        boundary_saving_config={"enabled": False},
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
    raise ValueError(
        f"Unsupported record shape {tuple(record.shape)}; expected "
        f"{(nshots, nreceivers, nt)} or {(nshots, nt, nreceivers)}."
    )


def run_gradient(backend, shape, wavelet, sources, receivers, vp_np, z_np, device, args):
    solver = build_solver(backend, shape, device, args)
    vp = torch.tensor(vp_np, device=device, dtype=torch.float32, requires_grad=True)
    z_model = torch.tensor(z_np, device=device, dtype=torch.float32, requires_grad=True)

    record = solver(wavelet, sources, receivers, models=[vp, z_model])
    record = normalize_record(record, sources.shape[0], receivers.shape[1], wavelet.shape[-1])
    loss = record.square().mean()
    loss.backward()
    torch.cuda.synchronize(device)

    grads = {"vp": vp.grad, "z": z_model.grad}
    for name, grad in grads.items():
        if grad is None:
            raise RuntimeError(f"{backend} {name} gradient is missing.")
        if not torch.isfinite(grad).all():
            raise RuntimeError(f"{backend} {name} gradient contains NaN/Inf.")
        if float(grad.detach().abs().max().cpu()) == 0.0:
            raise RuntimeError(f"{backend} {name} gradient is identically zero.")

    return {
        "loss": float(loss.detach().cpu()),
        "record": record.detach().cpu(),
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


def save_outputs(output_dir, results, metrics):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "acoustic_vrz3d_grad_compare.npz"
    np.savez(
        path,
        eager_record=results["eager"]["record"].numpy(),
        cuda_record=results["cuda"]["record"].numpy(),
        eager_vp_grad=results["eager"]["grads"]["vp"].numpy(),
        cuda_vp_grad=results["cuda"]["grads"]["vp"].numpy(),
        eager_z_grad=results["eager"]["grads"]["z"].numpy(),
        cuda_z_grad=results["cuda"]["grads"]["z"].numpy(),
        metrics=np.array([str(metrics)], dtype=object),
    )
    return path


def center_slices(volume):
    nz, ny, nx = volume.shape
    return (
        (f"z={nz // 2} (y-x)", volume[nz // 2, :, :]),
        (f"y={ny // 2} (z-x)", volume[:, ny // 2, :]),
        (f"x={nx // 2} (z-y)", volume[:, :, nx // 2]),
    )


def symmetric_limit(*arrays):
    max_abs = max(float(np.max(np.abs(array))) for array in arrays)
    return max(max_abs, 1e-20)


def percentile_limits(array):
    return tuple(float(value) for value in np.percentile(array, [2, 98]))


def load_pyplot(show):
    if not show:
        import matplotlib

        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_record_comparison(results, output_dir, plot_path=None, show=False):
    plt = load_pyplot(show)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = plot_path or (output_dir / "acoustic_vrz3d_record_comparison.png")

    eager = results["eager"]["record"].numpy()
    cuda = results["cuda"]["record"].numpy()
    residual = cuda - eager
    limit = symmetric_limit(eager, cuda)

    panels = (
        ("eager record", eager.reshape(-1, eager.shape[-1])),
        ("cuda record", cuda.reshape(-1, cuda.shape[-1])),
        ("cuda-eager residual", residual.reshape(-1, residual.shape[-1])),
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.2), constrained_layout=True)
    for ax, (title, data) in zip(axes, panels):
        im = ax.imshow(data, cmap="seismic", origin="upper", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("time step")
        ax.set_ylabel("shot/receiver")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle("AcousticVRZ3D Record Comparison", fontsize=14)
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_gradient_slices(results, output_dir, plot_path=None, show=False):
    plt = load_pyplot(show)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = plot_path or (output_dir / "acoustic_vrz3d_gradient_slices.png")

    row_specs = []
    for model_name in ("vp", "z"):
        eager = results["eager"]["grads"][model_name].numpy()
        cuda = results["cuda"]["grads"][model_name].numpy()
        diff = cuda - eager
        row_specs.extend(
            (
                (f"{model_name} eager", eager, percentile_limits(eager)),
                (f"{model_name} cuda", cuda, percentile_limits(cuda)),
                (f"{model_name} cuda-eager", diff, percentile_limits(diff)),
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

    fig.suptitle("AcousticVRZ3D Gradient Slices", fontsize=14)
    fig.savefig(path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)
    return path


def main():
    args = build_parser().parse_args()
    sweep_c = require_cuda_binding()

    shape = (args.nz, args.ny, args.nx)
    device = torch.device("cuda")
    wavelet = ricker(args.nt, args.dt, args.freq, args.delay)
    sources, receivers = make_geometry(
        shape,
        receiver_spacing=args.receiver_spacing,
        receiver_margin=args.receiver_margin,
        receiver_depth=args.receiver_depth,
    )
    vp_np, z_np = make_models(shape)

    print(f"python={sys.executable}")
    print(f"sweep._C={sweep_c.__file__}")
    print(f"shape={shape}, nt={args.nt}, abcn={args.abcn}, spatial_order={args.spatial_order}")
    print(f"sources={sources.shape}, receivers={receivers.shape}")

    results = {
        "eager": run_gradient("eager", shape, wavelet, sources, receivers, vp_np, z_np, device, args),
        "cuda": run_gradient("cuda", shape, wavelet, sources, receivers, vp_np, z_np, device, args),
    }

    metrics = {
        "record": metric_pair(results["eager"]["record"], results["cuda"]["record"]),
        "vp_grad": metric_pair(results["eager"]["grads"]["vp"], results["cuda"]["grads"]["vp"]),
        "z_grad": metric_pair(results["eager"]["grads"]["z"], results["cuda"]["grads"]["z"]),
    }
    output_path = save_outputs(args.output_dir, results, metrics)
    gradient_plot_path = None
    record_plot_path = None
    if not args.no_plot:
        record_plot_path = plot_record_comparison(results, args.output_dir, args.record_plot_path, args.show)
        gradient_plot_path = plot_gradient_slices(results, args.output_dir, args.plot_path, args.show)

    print(f"eager loss: {results['eager']['loss']:.6e}")
    print(f"cuda loss:  {results['cuda']['loss']:.6e}")
    for name, item in metrics.items():
        print(
            f"{name}: rel_l2={item['rel_l2']:.6e}, cosine={item['cosine']:.6f}, "
            f"diff_l2={item['diff_l2']:.6e}, diff_linf={item['diff_linf']:.6e}, "
            f"eager_l2={item['ref_l2']:.6e}, cuda_l2={item['cand_l2']:.6e}"
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
        raise AssertionError("; ".join(failures))


if __name__ == "__main__":
    main()
