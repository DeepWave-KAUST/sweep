import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import torch

from sweep.equations import Acoustic, Acoustic3D
from sweep.propagator.cuda import PropCUDA
from sweep.propagator.torch import PropTorch


STRATEGIES = (
    "eager",
    "full",
    "boundary_gpu",
    "boundary_cpu",
    "boundary_disk",
    "ckpt_chunk",
)


def ricker(t, fm):
    x = np.pi * fm * t
    return ((1.0 - 2.0 * x * x) * np.exp(-(x * x))).astype(np.float32)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark Acoustic 2D/3D CUDA memory strategies for time, GPU memory, CPU RSS, and disk cache size."
    )
    parser.add_argument("--dim", choices=("2d", "3d"), default="2d")
    parser.add_argument("--strategy", choices=STRATEGIES, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--nt", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--dh", type=float, default=10.0)
    parser.add_argument("--fm", type=float, default=8.0)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--spatial-order", type=int, default=2)
    parser.add_argument("--abcn", type=int, default=20)
    parser.add_argument("--transfer-interval", type=int, default=8)
    parser.add_argument("--ring-buffers", type=int, default=1)
    parser.add_argument("--checkpoint-chunks", type=int, default=100)
    parser.add_argument("--receiver-stride", type=int, default=8)
    parser.add_argument("--source-x", type=int, default=None)
    parser.add_argument("--source-y", type=int, default=None)
    parser.add_argument("--src-z", type=int, default=1)
    parser.add_argument("--rec-z", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--disk-dir", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--save-grads", type=Path, default=None)
    return parser


def apply_dim_defaults(args):
    if args.dim == "2d":
        if args.nz is None:
            args.nz = 128
        if args.nx is None:
            args.nx = 512
        if args.nt is None:
            args.nt = 800
        args.ny = None
        return

    if args.nz is None:
        args.nz = 64
    if args.ny is None:
        args.ny = 64
    if args.nx is None:
        args.nx = 64
    if args.nt is None:
        args.nt = 400


def build_case(args):
    t = np.arange(args.nt, dtype=np.float32) * args.dt - args.delay
    wave = ricker(t, args.fm)

    source_x = args.source_x if args.source_x is not None else args.nx // 2
    if args.dim == "2d":
        vp = np.full((args.nz, args.nx), 2000.0, dtype=np.float32)
        vp[args.nz // 2 :, :] = 2600.0
        vp[args.nz // 3 : (2 * args.nz) // 3, args.nx // 4 : (3 * args.nx) // 4] += 150.0

        sources = np.array([[source_x, args.src_z]], dtype=np.int32)
        rx = np.arange(0, args.nx, args.receiver_stride, dtype=np.int32)
        receivers = np.stack(
            [rx, np.full(rx.size, args.rec_z, dtype=np.int32)],
            axis=1,
        )[None, ...]
        return vp, wave, sources, receivers

    vp = np.full((args.nz, args.ny, args.nx), 1800.0, dtype=np.float32)
    vp[args.nz // 2 :, :, :] = 2400.0
    vp[
        args.nz // 3 : (2 * args.nz) // 3,
        args.ny // 4 : (3 * args.ny) // 4,
        args.nx // 4 : (3 * args.nx) // 4,
    ] += 100.0

    source_y = args.source_y if args.source_y is not None else args.ny // 2
    sources = np.array([[source_x, source_y, args.src_z]], dtype=np.int32)
    rx, ry = np.meshgrid(
        np.arange(0, args.nx, args.receiver_stride, dtype=np.int32),
        np.arange(0, args.ny, args.receiver_stride, dtype=np.int32),
        indexing="xy",
    )
    receivers = np.stack(
        [
            rx.reshape(-1),
            ry.reshape(-1),
            np.full(rx.size, args.rec_z, dtype=np.int32),
        ],
        axis=1,
    )[None, ...]
    return vp, wave, sources, receivers


def solver_kwargs_for_strategy(strategy, args):
    if strategy == "full":
        return {
            "use_ckpt": False,
            "boundary_saving_config": {"enabled": False},
        }
    if strategy == "ckpt_chunk":
        return {
            "use_ckpt": True,
            "ckpt_mode": "chunk",
            "ckpt_chunks": args.checkpoint_chunks,
            "boundary_saving_config": {"enabled": False},
        }

    storage = strategy.removeprefix("boundary_")
    cfg = {
        "enabled": True,
        "storage": storage,
        "transfer_interval": args.transfer_interval,
        "ring_buffers": args.ring_buffers,
        "pinned_memory": False,
    }
    if storage == "disk" and args.disk_dir is not None:
        cfg["disk_dir"] = str(args.disk_dir)
    return {
        "use_ckpt": False,
        "boundary_saving_config": cfg,
    }


def build_solver(strategy, args, device):
    if args.dim == "2d":
        equation = Acoustic(spatial_order=args.spatial_order, device=device)
        shape = (args.nz, args.nx)
    else:
        equation = Acoustic3D(spatial_order=args.spatial_order, device=device)
        shape = (args.nz, args.ny, args.nx)

    common_kwargs = {
        "shape": shape,
        "source_type": ["h1"],
        "receiver_type": ["h1"],
        "abcn": args.abcn,
        "dh": args.dh,
        "dt": args.dt,
        "pml_type": "cpmlr",
        "dev": device,
        "free_surface": False,
        "B": 1,
        "allow_growth": True,
        "nt": args.nt,
    }

    if strategy == "eager":
        return PropTorch(
            equation,
            backend="eager",
            **common_kwargs,
        )

    return PropCUDA(
        equation,
        **common_kwargs,
        **solver_kwargs_for_strategy(strategy, args),
    )


def read_rss_bytes():
    with open("/proc/self/status", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


class RssMonitor:
    def __init__(self, interval=0.01):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, read_rss_bytes())
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = read_rss_bytes()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, read_rss_bytes())


def tensor_bytes(tensors):
    total = 0
    for tensor in tensors or ():
        if isinstance(tensor, torch.Tensor):
            total += tensor.numel() * tensor.element_size()
    return total


def disk_cache_bytes(solver):
    return sum(
        os.path.getsize(path)
        for path in getattr(solver, "_boundary_disk_files", ())
        if os.path.exists(path)
    )


def run_once(solver, vp_np, wave, sources, receivers, device):
    vp = torch.from_numpy(vp_np.copy()).to(device).requires_grad_(True)
    record = solver(wavelet=wave, sources=sources, receivers=receivers, models=[vp])
    loss = record.square().sum()
    loss.backward()
    torch.cuda.synchronize(device)
    return float(loss.detach().cpu()), vp.grad.detach().cpu().numpy().copy()


def run_worker(args):
    if args.strategy is None:
        raise ValueError("--strategy is required in --worker mode.")

    device = torch.device("cuda:0")
    vp_np, wave, sources, receivers = build_case(args)
    solver = build_solver(args.strategy, args, device)

    for _ in range(args.warmup):
        run_once(solver, vp_np, wave, sources, receivers, device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    losses = []
    grad = None
    with RssMonitor() as rss:
        for _ in range(args.repeats):
            start = time.perf_counter()
            loss, grad = run_once(solver, vp_np, wave, sources, receivers, device)
            losses.append(loss)
            times.append(time.perf_counter() - start)

    result = {
        "strategy": args.strategy,
        "dim": args.dim,
        "shape": [args.nz, args.nx] if args.dim == "2d" else [args.nz, args.ny, args.nx],
        "nt": args.nt,
        "repeats": args.repeats,
        "time_mean_s": float(np.mean(times)),
        "time_min_s": float(np.min(times)),
        "time_max_s": float(np.max(times)),
        "loss_last": losses[-1],
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        "rss_peak_mb": rss.peak / 1024**2,
        "boundary_cpu_mb": tensor_bytes(getattr(solver, "boundary_cpu", ())) / 1024**2,
        "boundary_gpu_mb": (
            tensor_bytes(getattr(solver, "boundary_gpu", ()))
            + tensor_bytes(getattr(solver, "boundary_gpu_full", ()))
        ) / 1024**2,
        "boundary_disk_mb": disk_cache_bytes(solver) / 1024**2,
    }
    if args.save_grads is not None:
        args.save_grads.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_grads, grad)
        result["grad_path"] = str(args.save_grads)
    print(json.dumps(result), flush=True)


def run_all(args):
    strategies = STRATEGIES if args.strategy is None else (args.strategy,)
    results = []
    grad_paths = {}
    grad_dir = None
    if args.plot is not None or args.save_grads is not None:
        grad_dir = args.save_grads or Path(tempfile.mkdtemp(prefix="sweep_grad_compare_"))
        grad_dir.mkdir(parents=True, exist_ok=True)

    for strategy in strategies:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--strategy",
            strategy,
            "--dim",
            args.dim,
            "--nz",
            str(args.nz),
            "--nx",
            str(args.nx),
            "--nt",
            str(args.nt),
            "--dt",
            str(args.dt),
            "--dh",
            str(args.dh),
            "--fm",
            str(args.fm),
            "--delay",
            str(args.delay),
            "--spatial-order",
            str(args.spatial_order),
            "--abcn",
            str(args.abcn),
            "--transfer-interval",
            str(args.transfer_interval),
            "--ring-buffers",
            str(args.ring_buffers),
            "--checkpoint-chunks",
            str(args.checkpoint_chunks),
            "--receiver-stride",
            str(args.receiver_stride),
            "--src-z",
            str(args.src_z),
            "--rec-z",
            str(args.rec_z),
            "--warmup",
            str(args.warmup),
            "--repeats",
            str(args.repeats),
        ]
        if args.dim == "3d":
            cmd.extend(["--ny", str(args.ny)])
        if args.source_x is not None:
            cmd.extend(["--source-x", str(args.source_x)])
        if args.source_y is not None:
            cmd.extend(["--source-y", str(args.source_y)])
        if args.disk_dir is not None:
            cmd.extend(["--disk-dir", str(args.disk_dir)])
        if grad_dir is not None:
            grad_path = grad_dir / f"{strategy}_vp_grad.npy"
            cmd.extend(["--save-grads", str(grad_path)])
            grad_paths[strategy] = grad_path
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src") + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)
        line = proc.stdout.strip().splitlines()[-1]
        result = json.loads(line)
        results.append(result)
        print_result(result)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"saved json: {args.json}")
    if args.plot is not None:
        plot_gradients(grad_paths, args.plot)
        print(f"saved gradient plot: {args.plot}")


def print_result(result):
    print(
        f"{result['strategy']:>14} | "
        f"time {result['time_mean_s']:.4f}s "
        f"cuda_alloc {result['cuda_peak_allocated_mb']:.1f} MB "
        f"cuda_reserved {result['cuda_peak_reserved_mb']:.1f} MB "
        f"rss {result['rss_peak_mb']:.1f} MB "
        f"b_cpu {result['boundary_cpu_mb']:.1f} MB "
        f"b_gpu {result['boundary_gpu_mb']:.1f} MB "
        f"b_disk {result['boundary_disk_mb']:.1f} MB"
    )


def percentile_limit(arrays):
    values = np.concatenate([np.asarray(arr).reshape(-1) for arr in arrays])
    vmax = float(np.percentile(np.abs(values), 99.5))
    return vmax if vmax > 0 else 1.0


def plot_gradients(grad_paths, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grads = {name: np.load(grad_path) for name, grad_path in grad_paths.items()}
    reference_name = "eager" if "eager" in grads else ("full" if "full" in grads else next(iter(grads)))
    reference = grads[reference_name]
    vmax = percentile_limit(grads.values())

    nrows = len(grads)
    if reference.ndim == 2:
        fig, axes = plt.subplots(nrows, 2, figsize=(10.0, 3.0 * nrows), squeeze=False)
        slice_names = ("",)
    elif reference.ndim == 3:
        fig, axes = plt.subplots(nrows, 3, figsize=(12.0, 3.0 * nrows), squeeze=False)
        slice_names = ("z", "y", "x")
    else:
        raise ValueError(f"Expected 2D or 3D gradients, got shape {reference.shape}")

    for row, (name, grad) in enumerate(grads.items()):
        if grad.ndim == 2:
            diff = grad - reference
            grad_slices = (grad,)
            diff_slices = (diff,)
            for idx, (slice_name, grad_slice, diff_slice) in enumerate(zip(slice_names, grad_slices, diff_slices)):
                col = 2 * idx
                title_suffix = f" {slice_name}-slice" if slice_name else ""
                im0 = axes[row, col].imshow(grad_slice, cmap="seismic", aspect="auto", vmin=-vmax, vmax=vmax)
                axes[row, col].set_title(f"{name} vp grad{title_suffix}")
                fig.colorbar(im0, ax=axes[row, col], fraction=0.046, pad=0.02)

                im1 = axes[row, col + 1].imshow(diff_slice, cmap="seismic", aspect="auto", vmin=-vmax, vmax=vmax)
                axes[row, col + 1].set_title(f"{name} - {reference_name}{title_suffix}")
                fig.colorbar(im1, ax=axes[row, col + 1], fraction=0.046, pad=0.02)
            continue

        grad_slices = (
            grad[grad.shape[0] // 2],
            grad[:, grad.shape[1] // 2, :],
            grad[:, :, grad.shape[2] // 2],
        )
        for idx, (slice_name, grad_slice) in enumerate(zip(slice_names, grad_slices)):
            im0 = axes[row, idx].imshow(grad_slice, cmap="seismic", aspect="auto", vmin=-vmax, vmax=vmax)
            axes[row, idx].set_title(f"{name} vp grad {slice_name}-slice")
            fig.colorbar(im0, ax=axes[row, idx], fraction=0.046, pad=0.02)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    apply_dim_defaults(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.worker:
        run_worker(args)
    else:
        if not args.all and args.strategy is None:
            args.all = True
        run_all(args)


if __name__ == "__main__":
    main()
