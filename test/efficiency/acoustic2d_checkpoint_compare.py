import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.equations import Acoustic
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch


def ricker(t, fm):
    pi2 = np.pi * 2
    return (1 - 0.5 * (pi2 * fm * t) ** 2) * np.exp(-0.25 * (pi2 * fm * t) ** 2)


def build_parser():
    def parse_int_list(text):
        values = []
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            value = int(item)
            if value < 1:
                raise argparse.ArgumentTypeError("checkpoint chunks must be >= 1")
            values.append(value)
        if not values:
            raise argparse.ArgumentTypeError("at least one checkpoint chunk value is required")
        return tuple(dict.fromkeys(values))

    parser = argparse.ArgumentParser(
        description=(
            "Compare 2D acoustic gradients and benchmark full-wavefield storage, "
            "boundary saving, chunk checkpointing, recursive checkpointing, and PyTorch autograd."
        )
    )
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--nt", type=int, default=1200)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--dh", type=float, default=10.0)
    parser.add_argument("--fm", type=float, default=5.0)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--spatial-order", type=int, default=2)
    parser.add_argument("--abcn", type=int, default=20)
    parser.add_argument("--checkpoint-chunks", type=parse_int_list, default=(200,))
    parser.add_argument("--checkpoint-counts", type=parse_int_list, default=(4,))
    parser.add_argument("--transfer-interval", type=int, default=50)
    parser.add_argument("--source-x", type=int, default=None)
    parser.add_argument("--src-z", type=int, default=1)
    parser.add_argument("--rec-z", type=int, default=1)
    parser.add_argument("--receiver-stride", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", default="acoustic2d_checkpoint_compare.png")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def run_case(name, solver, vp, wave, sources, receivers, call_kwargs=None, warmup=0, repeats=1):
    if call_kwargs is None:
        call_kwargs = {}

    def single_run():
        if vp.grad is not None:
            vp.grad = None

        if vp.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(vp.device)
        torch.cuda.synchronize(vp.device)
        start = time.perf_counter()
        record = solver(
            wavelet=wave,
            sources=sources,
            receivers=receivers,
            models=[vp],
            **call_kwargs,
        )
        loss = record.pow(2).sum()
        loss.backward()
        torch.cuda.synchronize(vp.device)
        elapsed_s = time.perf_counter() - start
        peak_memory_mib = 0.0
        if vp.device.type == "cuda":
            peak_memory_mib = torch.cuda.max_memory_allocated(vp.device) / (1024.0 ** 2)
        return loss, elapsed_s, peak_memory_mib

    for _ in range(warmup):
        loss, _, _ = single_run()

    timings = []
    peaks = []
    loss = None
    for _ in range(repeats):
        loss, elapsed_s, peak_memory_mib = single_run()
        timings.append(elapsed_s)
        peaks.append(peak_memory_mib)

    result = {
        "name": name,
        "loss": float(loss.detach().cpu().item()),
        "grad": vp.grad.detach().cpu().numpy().copy(),
        "elapsed_s": float(np.mean(timings)),
        "elapsed_std_s": float(np.std(timings, ddof=0)),
        "peak_memory_mib": float(np.mean(peaks)),
        "peak_memory_max_mib": float(np.max(peaks)),
    }
    vp.grad = None
    return result


def maybe_plot(results, output_path):
    row_names = list(results.keys())
    fig, axes = plt.subplots(len(row_names), 1, figsize=(6, 4 * len(row_names)))
    if len(row_names) == 1:
        axes = [axes]

    all_grads = np.stack([results[name]["grad"] for name in row_names], axis=0)
    vmin, vmax = np.percentile(all_grads, [1.0, 99.0])

    for row_idx, name in enumerate(row_names):
        grad = results[name]["grad"]

        ax = axes[row_idx]
        im = ax.imshow(grad, cmap="seismic", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(
            f"{name} grad | mean {results[name]['elapsed_s'] * 1000.0:.2f} ms | "
            f"peak {results[name]['peak_memory_mib']:.2f} MiB"
        )
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def maybe_plot_checkpoint_sweeps(chunk_results, recursive_results, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    if chunk_results:
        x = [item["checkpoint_chunks"] for item in chunk_results]
        timings = [item["elapsed_s"] * 1000.0 for item in chunk_results]
        peaks = [item["peak_memory_mib"] for item in chunk_results]
        axes[0, 0].plot(x, timings, marker="o")
        axes[1, 0].plot(x, peaks, marker="o")
    axes[0, 0].set_xlabel("checkpoint_chunks")
    axes[0, 0].set_ylabel("time (ms)")
    axes[0, 0].set_title("Chunk Checkpoint Timing")
    axes[0, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("checkpoint_chunks")
    axes[1, 0].set_ylabel("peak memory (MiB)")
    axes[1, 0].set_title("Chunk Checkpoint Peak Memory")
    axes[1, 0].grid(True, alpha=0.3)

    if recursive_results:
        x = [item["checkpoint_count"] for item in recursive_results]
        timings = [item["elapsed_s"] * 1000.0 for item in recursive_results]
        peaks = [item["peak_memory_mib"] for item in recursive_results]
        axes[0, 1].plot(x, timings, marker="o")
        axes[1, 1].plot(x, peaks, marker="o")
    axes[0, 1].set_xlabel("checkpoint_count")
    axes[0, 1].set_ylabel("time (ms)")
    axes[0, 1].set_title("Recursive Checkpoint Timing")
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlabel("checkpoint_count")
    axes[1, 1].set_ylabel("peak memory (MiB)")
    axes[1, 1].set_title("Recursive Checkpoint Peak Memory")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = build_parser().parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this comparison.")

    device = torch.device("cuda:0")

    true_vp = np.ones((args.nz, args.nx), dtype=np.float32) * 2000.0
    t = np.arange(args.nt, dtype=np.float32) * args.dt - args.delay
    wave = ricker(t, fm=args.fm).astype(np.float32)

    source_x = args.source_x if args.source_x is not None else args.nx // 2
    sources = np.array([[source_x, args.src_z]], dtype=np.int32)
    receiver_x = np.arange(0, args.nx, args.receiver_stride, dtype=np.int32)
    receivers = np.stack(
        [receiver_x, np.full(receiver_x.shape[0], args.rec_z, dtype=np.int32)],
        axis=1,
    )[None, ...]

    print("Device:", device)
    print("Shape:", true_vp.shape, "nt:", args.nt)
    print("Sources:\n", sources)
    print("Receivers:\n", receivers[:, 0, :])

    common_kwargs = dict(
        shape=true_vp.shape,
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

    cuda_full = PropCUDA(
        Acoustic(spatial_order=args.spatial_order, device=device),
        boundary_saving_config={
            "enabled": False,
            "storage": "gpu",
            "transfer_interval": 1,
            "pinned_memory": False,
        },
        use_ckpt=False,
        **common_kwargs,
    )
    cuda_boundary = PropCUDA(
        Acoustic(spatial_order=args.spatial_order, device=device),
        boundary_saving_config={
            "enabled": True,
            "storage": "gpu",
            "transfer_interval": 1,
            "pinned_memory": False,
        },
        use_ckpt=False,
        **common_kwargs,
    )
    torch_solver = PropTorch(
        Acoustic(spatial_order=args.spatial_order, device=device),
        use_ckpt=False,
        **common_kwargs,
    )

    results = {}
    cases = [
        ("cuda_full", cuda_full, {}),
        (
            "cuda_boundary",
            cuda_boundary,
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
        ("torch", torch_solver, {}),
    ]

    for name, solver, call_kwargs in cases:
        vp = torch.from_numpy(true_vp.copy()).to(device).requires_grad_(True)
        results[name] = run_case(
            name,
            solver,
            vp,
            wave,
            sources,
            receivers,
            call_kwargs,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        print(f"{name:>16} | loss {results[name]['loss']:.6e}")

    chunk_checkpoint_results = []
    for checkpoint_chunks in args.checkpoint_chunks:
        checkpoint_solver = PropCUDA(
            Acoustic(spatial_order=args.spatial_order, device=device),
            boundary_saving_config={
                "enabled": False,
                "storage": "gpu",
                "transfer_interval": 1,
                "pinned_memory": False,
            },
            use_ckpt=True,
            ckpt_mode="chunk",
            ckpt_chunks=checkpoint_chunks,
            **common_kwargs,
        )
        name = f"cuda_ckpt_chunk_{checkpoint_chunks}"
        vp = torch.from_numpy(true_vp.copy()).to(device).requires_grad_(True)
        result = run_case(
            name,
            checkpoint_solver,
            vp,
            wave,
            sources,
            receivers,
            {},
            warmup=args.warmup,
            repeats=args.repeats,
        )
        result["checkpoint_chunks"] = checkpoint_chunks
        chunk_checkpoint_results.append(result)
        results[name] = result
        print(f"{name:>24} | loss {result['loss']:.6e}")

    recursive_checkpoint_results = []
    for checkpoint_count in args.checkpoint_counts:
        checkpoint_solver = PropCUDA(
            Acoustic(spatial_order=args.spatial_order, device=device),
            boundary_saving_config={
                "enabled": False,
                "storage": "gpu",
                "transfer_interval": 1,
                "pinned_memory": False,
            },
            use_ckpt=True,
            ckpt_mode="recursive",
            ckpt_num=checkpoint_count,
            **common_kwargs,
        )
        name = f"cuda_ckpt_recursive_{checkpoint_count}"
        vp = torch.from_numpy(true_vp.copy()).to(device).requires_grad_(True)
        result = run_case(
            name,
            checkpoint_solver,
            vp,
            wave,
            sources,
            receivers,
            {},
            warmup=args.warmup,
            repeats=args.repeats,
        )
        result["checkpoint_count"] = checkpoint_count
        recursive_checkpoint_results.append(result)
        results[name] = result
        print(f"{name:>24} | loss {result['loss']:.6e}")

    print("\nTiming summary")
    for name in ("cuda_full", "cuda_boundary", "torch"):
        print(
            f"{name:>16} | "
            f"mean {results[name]['elapsed_s'] * 1000.0:.2f} ms | "
            f"std {results[name]['elapsed_std_s'] * 1000.0:.2f} ms | "
            f"peak_mean {results[name]['peak_memory_mib']:.2f} MiB | "
            f"peak_max {results[name]['peak_memory_max_mib']:.2f} MiB | "
            f"loss {results[name]['loss']:.6e}"
        )
    for item in chunk_checkpoint_results:
        print(
            f"{item['name']:>24} | "
            f"mean {item['elapsed_s'] * 1000.0:.2f} ms | "
            f"std {item['elapsed_std_s'] * 1000.0:.2f} ms | "
            f"peak_mean {item['peak_memory_mib']:.2f} MiB | "
            f"peak_max {item['peak_memory_max_mib']:.2f} MiB | "
            f"loss {item['loss']:.6e}"
        )
    for item in recursive_checkpoint_results:
        print(
            f"{item['name']:>24} | "
            f"mean {item['elapsed_s'] * 1000.0:.2f} ms | "
            f"std {item['elapsed_std_s'] * 1000.0:.2f} ms | "
            f"peak_mean {item['peak_memory_mib']:.2f} MiB | "
            f"peak_max {item['peak_memory_max_mib']:.2f} MiB | "
            f"loss {item['loss']:.6e}"
        )

    if not args.no_plot:
        output_path = Path(args.output)
        maybe_plot(results, output_path)
        sweep_path = output_path.with_name(output_path.stem + "_checkpoint_sweep" + output_path.suffix)
        maybe_plot_checkpoint_sweeps(chunk_checkpoint_results, recursive_checkpoint_results, sweep_path)
        print(f"\nSaved gradient figure to {output_path}")
        print(f"Saved checkpoint sweep figure to {sweep_path}")


if __name__ == "__main__":
    main()
