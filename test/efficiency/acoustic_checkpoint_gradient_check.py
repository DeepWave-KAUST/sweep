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
        description="Check final gradient consistency across full, boundary, chunk checkpoint, recursive checkpoint, and torch."
    )
    parser.add_argument("--dim", choices=("2d", "3d"), default="2d")
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--nt", type=int, default=1200)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--dh", type=float, default=10.0)
    parser.add_argument("--fm", type=float, default=5.0)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--spatial-order", type=int, default=2)
    parser.add_argument("--abcn", type=int, default=20)
    parser.add_argument("--checkpoint-chunks", type=int, default=100)
    parser.add_argument("--checkpoint-count", type=int, default=4)
    parser.add_argument("--transfer-interval", type=int, default=1)
    parser.add_argument("--source-x", type=int, default=None)
    parser.add_argument("--source-y", type=int, default=None)
    parser.add_argument("--src-z", type=int, default=1)
    parser.add_argument("--rec-z", type=int, default=1)
    parser.add_argument("--receiver-stride", type=int, default=4)
    parser.add_argument("--output", default=None)
    return parser


def run_case(name, solver, vp, wave, sources, receivers, call_kwargs=None):
    if call_kwargs is None:
        call_kwargs = {}

    if vp.grad is not None:
        vp.grad = None

    record = solver(
        wavelet=wave,
        sources=sources,
        receivers=receivers,
        models=[vp],
        **call_kwargs,
    )
    loss = record.pow(2).sum()
    loss.backward()

    grad = vp.grad.detach().cpu().numpy().copy()
    vp.grad = None
    return {"name": name, "loss": float(loss.detach().cpu().item()), "grad": grad}


def normalize_grad(grad):
    max_abs = 1#float(np.max(np.abs(grad)))
    if max_abs == 0.0:
        return grad.copy(), max_abs
    return grad / max_abs, max_abs


def build_2d_case(args):
    nz = 100 if args.nz is None else args.nz
    nx = 512 if args.nx is None else args.nx
    vp = np.ones((nz, nx), dtype=np.float32) * 2000.0
    t = np.arange(args.nt, dtype=np.float32) * args.dt - args.delay
    wave = ricker(t, fm=args.fm).astype(np.float32)
    source_x = args.source_x if args.source_x is not None else nx // 2
    sources = np.array([[source_x, args.src_z]], dtype=np.int32)
    receiver_x = np.arange(0, nx, args.receiver_stride, dtype=np.int32)
    receivers = np.stack(
        [receiver_x, np.full(receiver_x.shape[0], args.rec_z, dtype=np.int32)],
        axis=1,
    )[None, ...]
    return vp, wave, sources, receivers


def build_3d_case(args):
    nz = 64 if args.nz is None else args.nz
    ny = 64 if args.ny is None else args.ny
    nx = 64 if args.nx is None else args.nx
    vp = np.full((nz, ny, nx), 1500.0, dtype=np.float32)
    vp[nz // 2 :, :, :] = 2000.0
    t = np.arange(args.nt, dtype=np.float32) * args.dt - args.delay
    wave = ricker(t, fm=args.fm).astype(np.float32)
    source_x = args.source_x if args.source_x is not None else nx // 2
    source_y = args.source_y if args.source_y is not None else ny // 2
    sources = np.array([[source_x, source_y, args.src_z]], dtype=np.int32)
    rec_x, rec_y = np.meshgrid(
        np.arange(0, nx, args.receiver_stride, dtype=np.int32),
        np.arange(0, ny, args.receiver_stride, dtype=np.int32),
        indexing="xy",
    )
    rec_z = np.full(rec_x.size, args.rec_z, dtype=np.int32)
    receivers = np.stack((rec_x.reshape(-1), rec_y.reshape(-1), rec_z), axis=1)[None, ...]
    return vp, wave, sources, receivers


def build_solvers(args, device, shape):
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

    equation = Acoustic if args.dim == "2d" else Acoustic3D
    eq = equation(spatial_order=args.spatial_order, device=device)

    full = PropCUDA(
        eq,
        boundary_saving_config={"enabled": False, "storage": "gpu", "transfer_interval": 1, "pinned_memory": False},
        use_ckpt=False,
        **common_kwargs,
    )
    boundary = PropCUDA(
        equation(spatial_order=args.spatial_order, device=device),
        boundary_saving_config={"enabled": True, "storage": "gpu", "transfer_interval": 1, "pinned_memory": False},
        use_ckpt=False,
        **common_kwargs,
    )
    chunk = PropCUDA(
        equation(spatial_order=args.spatial_order, device=device),
        boundary_saving_config={"enabled": False, "storage": "gpu", "transfer_interval": 1, "pinned_memory": False},
        use_ckpt=True,
        ckpt_mode="chunk",
        ckpt_chunks=args.checkpoint_chunks,
        **common_kwargs,
    )
    recursive = PropCUDA(
        equation(spatial_order=args.spatial_order, device=device),
        boundary_saving_config={"enabled": False, "storage": "gpu", "transfer_interval": 1, "pinned_memory": False},
        use_ckpt=True,
        ckpt_mode="recursive",
        ckpt_num=args.checkpoint_count,
        **common_kwargs,
    )
    torch_solver = PropTorch(
        equation(spatial_order=args.spatial_order, device=device),
        use_ckpt=False,
        **common_kwargs,
    )

    return {
        "cuda_full": (full, {}),
        "cuda_boundary": (
            boundary,
            {
                "use_boundary_saving": True,
                "boundary_saving_config": {
                    "enabled": True,
                    "storage": "gpu",
                    "transfer_interval": args.transfer_interval,
                    "pinned_memory": False,
                },
            },
        ),
        f"cuda_ckpt_chunk_{args.checkpoint_chunks}": (chunk, {}),
        f"cuda_ckpt_recursive_{args.checkpoint_count}": (recursive, {}),
        "torch": (torch_solver, {}),
    }


def plot_2d(results, output_path):
    row_names = list(results.keys())
    fig, axes = plt.subplots(len(row_names), 1, figsize=(7, 4 * len(row_names)))
    if len(row_names) == 1:
        axes = [axes]

    for row_idx, name in enumerate(row_names):
        ax = axes[row_idx]
        grad, max_abs = normalize_grad(results[name]["grad"])
        vmin, vmax = np.percentile(grad, [1.0, 99.0])
        im = ax.imshow(grad, cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"{name} | loss {results[name]['loss']:.6e} | max {max_abs:.6e}")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_3d(results, output_path):
    row_names = list(results.keys())
    fig, axes = plt.subplots(len(row_names), 3, figsize=(15, 4 * len(row_names)))
    if len(row_names) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_idx, name in enumerate(row_names):
        grad, max_abs = normalize_grad(results[name]["grad"])
        mid_z = grad.shape[0] // 2
        mid_y = grad.shape[1] // 2
        mid_x = grad.shape[2] // 2
        vmin, vmax = np.percentile(grad, [1.0, 99.0])

        im = axes[row_idx, 0].imshow(grad[mid_z], cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
        axes[row_idx, 0].set_title(f"{name} z | loss {results[name]['loss']:.6e} | max {max_abs:.6e}")
        fig.colorbar(im, ax=axes[row_idx, 0])

        im = axes[row_idx, 1].imshow(grad[:, mid_y], cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
        axes[row_idx, 1].set_title(f"{name} y")
        fig.colorbar(im, ax=axes[row_idx, 1])

        im = axes[row_idx, 2].imshow(grad[:, :, mid_x], cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
        axes[row_idx, 2].set_title(f"{name} x")
        fig.colorbar(im, ax=axes[row_idx, 2])

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this gradient check.")

    device = torch.device("cuda:0")
    if args.dim == "2d":
        vp_np, wave, sources, receivers = build_2d_case(args)
        output = args.output or "acoustic2d_checkpoint_gradient_check.png"
    else:
        vp_np, wave, sources, receivers = build_3d_case(args)
        output = args.output or "acoustic3d_checkpoint_gradient_check.png"

    print("Device:", device)
    print("Dimension:", args.dim)
    print("Shape:", vp_np.shape, "nt:", args.nt)

    solvers = build_solvers(args, device, vp_np.shape)
    results = {}
    for name, (solver, call_kwargs) in solvers.items():
        vp = torch.from_numpy(vp_np.copy()).to(device).requires_grad_(True)
        results[name] = run_case(name, solver, vp, wave, sources, receivers, call_kwargs)
        print(f"{name:>24} | loss {results[name]['loss']:.6e}")

    output_path = Path(output)
    if args.dim == "2d":
        plot_2d(results, output_path)
    else:
        plot_3d(results, output_path)
    print(f"\nSaved gradient figure to {output_path}")


if __name__ == "__main__":
    main()
