import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.equations import Acoustic, Acoustic3D
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch


def ricker(t, fm):
    pi2 = np.pi * 2
    return (1 - 0.5 * (pi2 * fm * t) ** 2) * np.exp(-0.25 * (pi2 * fm * t) ** 2)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare Acoustic 2D/3D gradients from CUDA full, checkpointing, or boundary saving against PyTorch."
    )
    parser.add_argument("--dim", choices=("2d", "3d"), default="2d")
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--nt", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--dh", type=float, default=10.0)
    parser.add_argument("--fm", type=float, default=5.0)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--spatial-order", type=int, default=2)
    parser.add_argument("--abcn", type=int, default=None)
    parser.add_argument("--mode", choices=("full", "ckpt", "boundary"), default="full")
    parser.add_argument("--ckpt-mode", choices=("chunk", "recursive"), default="chunk")
    parser.add_argument("--checkpoint-chunks", type=int, default=None)
    parser.add_argument("--checkpoint-count", type=int, default=4)
    parser.add_argument("--boundary-storage", choices=("gpu", "cpu", "disk"), default="gpu")
    parser.add_argument("--transfer-interval", type=int, default=1)
    parser.add_argument("--pinned-memory", action="store_true")
    parser.add_argument("--source-x", type=int, default=None)
    parser.add_argument("--source-y", type=int, default=None)
    parser.add_argument("--src-z", type=int, default=1)
    parser.add_argument("--rec-z", type=int, default=1)
    parser.add_argument("--receiver-stride", type=int, default=4)
    parser.add_argument("--constant-model", action="store_true")
    parser.add_argument("--output", default=None)
    return parser


def apply_dim_defaults(args):
    if args.dim == "2d":
        if args.nz is None:
            args.nz = 100
        if args.nx is None:
            args.nx = 512
        if args.nt is None:
            args.nt = 1200
        if args.abcn is None:
            args.abcn = 20
        if args.checkpoint_chunks is None:
            args.checkpoint_chunks = 100
        args.ny = None
    else:
        if args.nz is None:
            args.nz = 64
        if args.ny is None:
            args.ny = 64
        if args.nx is None:
            args.nx = 64
        if args.nt is None:
            args.nt = 1000
        if args.abcn is None:
            args.abcn = 30
        if args.checkpoint_chunks is None:
            args.checkpoint_chunks = 50


def build_case(args):
    if args.dim == "2d":
        vp = np.full((args.nz, args.nx), 2000.0, dtype=np.float32)
        if not args.constant_model:
            vp[args.nz // 2 :, :] = 2600.0
            vp[args.nz // 3 : (2 * args.nz) // 3, args.nx // 4 : (3 * args.nx) // 4] += 150.0

        source_x = args.source_x if args.source_x is not None else args.nx // 2
        sources = np.array([[source_x, args.src_z]], dtype=np.int32)
        receiver_x = np.arange(0, args.nx, args.receiver_stride, dtype=np.int32)
        receivers = np.stack(
            [receiver_x, np.full(receiver_x.shape[0], args.rec_z, dtype=np.int32)],
            axis=1,
        )[None, ...]
    else:
        vp = np.full((args.nz, args.ny, args.nx), 1800.0, dtype=np.float32)
        if not args.constant_model:
            vp[args.nz // 2 :, :, :] = 2400.0
            vp[
                args.nz // 3 : (2 * args.nz) // 3,
                args.ny // 4 : (3 * args.ny) // 4,
                args.nx // 4 : (3 * args.nx) // 4,
            ] += 100.0

        source_x = args.source_x if args.source_x is not None else args.nx // 2
        source_y = args.source_y if args.source_y is not None else args.ny // 2
        sources = np.array([[source_x, source_y, args.src_z]], dtype=np.int32)
        rec_x, rec_y = np.meshgrid(
            np.arange(0, args.nx, args.receiver_stride, dtype=np.int32),
            np.arange(0, args.ny, args.receiver_stride, dtype=np.int32),
            indexing="xy",
        )
        rec_z = np.full(rec_x.size, args.rec_z, dtype=np.int32)
        receivers = np.stack((rec_x.reshape(-1), rec_y.reshape(-1), rec_z), axis=1)[None, ...]

    t = np.arange(args.nt, dtype=np.float32) * args.dt - args.delay
    wave = ricker(t, fm=args.fm).astype(np.float32)
    return vp, wave, sources, receivers


def build_solvers(args, device):
    if args.dim == "2d":
        equation = Acoustic(spatial_order=args.spatial_order, device=device)
        shape = (args.nz, args.nx)
    else:
        equation = Acoustic3D(spatial_order=args.spatial_order, device=device)
        shape = (args.nz, args.ny, args.nx)

    common_kwargs = dict(
        shape=shape,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=args.abcn,
        dh=args.dh,
        dt=args.dt,
        pml_type="cpmlr",
        dev=device,
        free_surface=False,
        B=1,
        allow_growth=True,
        nt=args.nt,
    )

    if args.mode == "full":
        solver_kwargs = {
            "boundary_saving_config": {
                "enabled": False,
                "storage": "gpu",
                "transfer_interval": 1,
                "pinned_memory": False,
            },
            "use_ckpt": False,
        }
        cuda_label = "cuda_full"
    elif args.mode == "ckpt":
        solver_kwargs = {
            "boundary_saving_config": {
                "enabled": False,
                "storage": "gpu",
                "transfer_interval": 1,
                "pinned_memory": False,
            },
            "use_ckpt": True,
            "ckpt_mode": args.ckpt_mode,
        }
        if args.ckpt_mode == "chunk":
            solver_kwargs["ckpt_chunks"] = args.checkpoint_chunks
            cuda_label = f"cuda_ckpt_chunk_{args.checkpoint_chunks}"
        else:
            solver_kwargs["ckpt_num"] = args.checkpoint_count
            cuda_label = f"cuda_ckpt_recursive_{args.checkpoint_count}"
    else:
        solver_kwargs = {
            "boundary_saving_config": {
                "enabled": True,
                "storage": args.boundary_storage,
                "transfer_interval": args.transfer_interval,
                "pinned_memory": args.pinned_memory,
            },
            "use_ckpt": False,
        }
        if args.dim == "2d":
            cuda_label = f"cuda_boundary_{args.boundary_storage}"
        else:
            cuda_label = f"cuda_boundary_{args.boundary_storage}_interval_{args.transfer_interval}"

    cuda_solver = PropCUDA(
        equation,
        **common_kwargs,
        **solver_kwargs,
    )
    torch_solver = PropTorch(
        equation,
        use_ckpt=False,
        **common_kwargs,
    )
    return {
        cuda_label: cuda_solver,
        "torch": torch_solver,
    }


def run_case(name, solver, vp_np, wave, sources, receivers, device):
    vp = torch.from_numpy(vp_np.copy()).to(device).requires_grad_(True)
    record = solver(
        wavelet=wave,
        sources=sources,
        receivers=receivers,
        models=[vp],
    )
    loss = record.pow(2).sum()
    loss.backward()
    torch.cuda.synchronize(device)
    return {
        "name": name,
        "loss": float(loss.detach().cpu().item()),
        "grads": {"vp": vp.grad.detach().cpu().numpy().copy()},
    }


def grad_slice(grad, dim):
    if dim == "2d":
        return grad
    return (
        grad[grad.shape[0] // 2],
        grad[:, grad.shape[1] // 2, :],
        grad[:, :, grad.shape[2] // 2],
    )


def summarize_array(name, arr):
    finite = np.isfinite(arr)
    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())
    finite_count = int(finite.sum())
    total = int(arr.size)
    if finite_count == 0:
        return f"{name}: finite 0/{total}, nan {nan_count}, inf {inf_count}"
    finite_vals = arr[finite]
    return (
        f"{name}: finite {finite_count}/{total}, nan {nan_count}, inf {inf_count}, "
        f"min {finite_vals.min():.3e}, max {finite_vals.max():.3e}"
    )


def summarize_scale(cuda_grad, torch_grad):
    cuda_flat = np.asarray(cuda_grad, dtype=np.float64).reshape(-1)
    torch_flat = np.asarray(torch_grad, dtype=np.float64).reshape(-1)
    finite = np.isfinite(cuda_flat) & np.isfinite(torch_flat)
    if not np.any(finite):
        return "scale: no finite overlap"

    cuda_flat = cuda_flat[finite]
    torch_flat = torch_flat[finite]

    torch_peak = float(np.max(np.abs(torch_flat)))
    cuda_peak = float(np.max(np.abs(cuda_flat)))
    peak_ratio = np.nan if torch_peak == 0.0 else cuda_peak / torch_peak

    torch_energy = float(np.dot(torch_flat, torch_flat))
    ls_ratio = np.nan if torch_energy == 0.0 else float(np.dot(cuda_flat, torch_flat) / torch_energy)

    return f"scale: peak(cuda/torch) {peak_ratio:.6f}, ls(cuda≈s*torch) {ls_ratio:.6f}"


def plot_results(results, output_path, dim):
    cuda_name = next(name for name in results if name != "torch")
    row_names = (cuda_name, "cuda-torch")
    ncols = 1 if dim == "2d" else 3
    fig, axes = plt.subplots(len(row_names), ncols, figsize=(6 * ncols, 4 * len(row_names)), squeeze=False)

    cuda_grad = grad_slice(results[cuda_name]["grads"]["vp"], dim)
    torch_grad = grad_slice(results["torch"]["grads"]["vp"], dim)
    row_images = {
        cuda_name: cuda_grad,
        "cuda-torch": cuda_grad - torch_grad if dim == "2d" else tuple(a - b for a, b in zip(cuda_grad, torch_grad)),
    }
    slice_labels = ("z", "y", "x") if dim == "3d" else ("",)

    for row_idx, row_name in enumerate(row_names):
        images = (row_images[row_name],) if dim == "2d" else row_images[row_name]
        for col_idx, image in enumerate(images):
            ax = axes[row_idx, col_idx]
            finite = image[np.isfinite(image)]
            if finite.size == 0:
                vmin, vmax = -1.0, 1.0
            elif row_name == "cuda-torch":
                vmax = np.nanpercentile(np.abs(finite), 99.0)
                if not np.isfinite(vmax) or vmax == 0.0:
                    vmax = 1.0
                vmin = -vmax
            else:
                vmin, vmax = np.nanpercentile(finite, [1.0, 99.0])
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
                    if vmax == 0.0:
                        vmax = 1.0
                    vmin = -vmax
            im = ax.imshow(image, cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
            title = f"{row_name} vp"
            if dim == "3d":
                title += f" ({slice_labels[col_idx]})"
            if col_idx == 0 and row_name in results:
                title += f"\nloss {results[row_name]['loss']:.6e}"
            ax.set_title(title)
            fig.colorbar(im, ax=ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def default_output_name(args):
    prefix = f"acoustic{args.dim}"
    if args.mode == "full":
        return f"{prefix}_full_gradient_compare.png"
    if args.mode == "ckpt":
        if args.ckpt_mode == "chunk":
            return f"{prefix}_ckpt_chunk_{args.checkpoint_chunks}_gradient_compare.png"
        return f"{prefix}_ckpt_recursive_{args.checkpoint_count}_gradient_compare.png"
    if args.dim == "2d":
        return f"{prefix}_boundary_{args.boundary_storage}_gradient_compare.png"
    return f"{prefix}_boundary_{args.boundary_storage}_interval_{args.transfer_interval}_gradient_compare.png"


def print_mode_summary(args):
    if args.mode == "full":
        print("Mode:", "full wavefield")
    elif args.mode == "ckpt":
        if args.ckpt_mode == "chunk":
            print("Mode:", f"checkpoint chunk ({args.checkpoint_chunks})")
        else:
            print("Mode:", f"checkpoint recursive ({args.checkpoint_count})")
    elif args.dim == "2d":
        print("Mode:", f"boundary saving ({args.boundary_storage})")
    else:
        print("Mode:", f"boundary saving ({args.boundary_storage}, interval={args.transfer_interval})")


def main(argv=None):
    args = build_parser().parse_args(argv)
    apply_dim_defaults(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this comparison.")

    device = torch.device("cuda:0")
    vp_np, wave, sources, receivers = build_case(args)

    print("Device:", device)
    print("Dimension:", args.dim)
    print("Shape:", vp_np.shape, "nt:", args.nt)
    print("Model:", "constant" if args.constant_model else "layered")
    print_mode_summary(args)

    output = args.output or default_output_name(args)
    solvers = build_solvers(args, device)
    results = {}
    for name, solver in solvers.items():
        try:
            results[name] = run_case(name, solver, vp_np, wave, sources, receivers, device)
        except Exception:
            print(f"{name:>24} | failed")
            raise
        print(f"{name:>24} | loss {results[name]['loss']:.6e}")
        print(" " * 26 + summarize_array("vp", results[name]["grads"]["vp"]))

    cuda_name = next(name for name in results if name != "torch")
    print(" " * 26 + summarize_scale(results[cuda_name]["grads"]["vp"], results["torch"]["grads"]["vp"]))

    output_path = Path(output)
    plot_results(results, output_path, args.dim)
    print(f"\nSaved figure to {output_path}")


if __name__ == "__main__":
    main()
