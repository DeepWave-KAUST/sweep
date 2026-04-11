import argparse
import gc
import importlib.util
import json
import time
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


def parse_int_list(text):
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_str_list(text):
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare acoustic CUDA full, checkpoint, and boundary-saving efficiency "
            "and memory usage against a PyTorch baseline."
        )
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
    parser.add_argument("--nshots", type=int, default=1)
    parser.add_argument("--source-x", type=int, default=None)
    parser.add_argument("--source-y", type=int, default=None)
    parser.add_argument("--src-z", type=int, default=1)
    parser.add_argument("--rec-z", type=int, default=1)
    parser.add_argument("--receiver-stride", type=int, default=4)
    parser.add_argument("--constant-model", action="store_true")
    parser.add_argument("--checkpoint-chunks", default=None)
    parser.add_argument("--checkpoint-counts", default="2,4,8")
    parser.add_argument("--boundary-storages", default="gpu,cpu")
    parser.add_argument("--transfer-intervals", default="1,2,4,8")
    parser.add_argument("--pinned-memory", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--include-torch", action="store_true")
    parser.add_argument("--include-deepwave", action="store_true")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--json", default=None)
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
            args.checkpoint_chunks = "50,100,200"
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
            args.checkpoint_chunks = "25,50,100"


def build_case(args):
    if args.dim == "2d":
        vp = np.full((args.nz, args.nx), 2000.0, dtype=np.float32)
        if not args.constant_model:
            vp[args.nz // 2 :, :] = 2600.0
            vp[args.nz // 3 : (2 * args.nz) // 3, args.nx // 4 : (3 * args.nx) // 4] += 150.0

        if args.nshots == 1:
            source_x = args.source_x if args.source_x is not None else args.nx // 2
            shot_x = np.array([source_x], dtype=np.int32)
        else:
            left = max(args.abcn, args.nx // 10)
            right = min(args.nx - 1 - args.abcn, args.nx - 1 - args.nx // 10)
            shot_x = np.linspace(left, right, args.nshots, dtype=np.int32)
        sources = np.stack(
            [shot_x, np.full(args.nshots, args.src_z, dtype=np.int32)],
            axis=1,
        )
        receiver_x = np.arange(0, args.nx, args.receiver_stride, dtype=np.int32)
        receiver_template = np.stack(
            [receiver_x, np.full(receiver_x.shape[0], args.rec_z, dtype=np.int32)],
            axis=1,
        )
        receivers = np.repeat(receiver_template[None, ...], args.nshots, axis=0)
    else:
        vp = np.full((args.nz, args.ny, args.nx), 1800.0, dtype=np.float32)
        if not args.constant_model:
            vp[args.nz // 2 :, :, :] = 2400.0
            vp[
                args.nz // 3 : (2 * args.nz) // 3,
                args.ny // 4 : (3 * args.ny) // 4,
                args.nx // 4 : (3 * args.nx) // 4,
            ] += 100.0

        if args.nshots == 1:
            source_x = args.source_x if args.source_x is not None else args.nx // 2
            source_y = args.source_y if args.source_y is not None else args.ny // 2
            shot_x = np.array([source_x], dtype=np.int32)
            shot_y = np.array([source_y], dtype=np.int32)
        else:
            left_x = max(args.abcn, args.nx // 10)
            right_x = min(args.nx - 1 - args.abcn, args.nx - 1 - args.nx // 10)
            left_y = max(args.abcn, args.ny // 10)
            right_y = min(args.ny - 1 - args.abcn, args.ny - 1 - args.ny // 10)
            side = int(np.ceil(np.sqrt(args.nshots)))
            grid_x = np.linspace(left_x, right_x, side, dtype=np.int32)
            grid_y = np.linspace(left_y, right_y, side, dtype=np.int32)
            mesh_x, mesh_y = np.meshgrid(grid_x, grid_y, indexing="xy")
            shot_x = mesh_x.reshape(-1)[: args.nshots]
            shot_y = mesh_y.reshape(-1)[: args.nshots]
        sources = np.stack(
            [shot_x, shot_y, np.full(args.nshots, args.src_z, dtype=np.int32)],
            axis=1,
        )
        rec_x, rec_y = np.meshgrid(
            np.arange(0, args.nx, args.receiver_stride, dtype=np.int32),
            np.arange(0, args.ny, args.receiver_stride, dtype=np.int32),
            indexing="xy",
        )
        rec_z = np.full(rec_x.size, args.rec_z, dtype=np.int32)
        receiver_template = np.stack((rec_x.reshape(-1), rec_y.reshape(-1), rec_z), axis=1)
        receivers = np.repeat(receiver_template[None, ...], args.nshots, axis=0)

    t = np.arange(args.nt, dtype=np.float32) * args.dt - args.delay
    wave = np.repeat(ricker(t, fm=args.fm).astype(np.float32)[None, :], args.nshots, axis=0)
    return vp, wave, sources, receivers


def make_equation(args, device):
    if args.dim == "2d":
        return Acoustic(spatial_order=args.spatial_order, device=device), (args.nz, args.nx)
    return Acoustic3D(spatial_order=args.spatial_order, device=device), (args.nz, args.ny, args.nx)


def common_solver_kwargs(args, shape, device):
    return dict(
        shape=shape,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=args.abcn,
        dh=args.dh,
        dt=args.dt,
        pml_type="cpmlr",
        dev=device,
        free_surface=False,
        B=args.nshots,
        allow_growth=True,
        nt=args.nt,
    )


def require_deepwave():
    if importlib.util.find_spec("deepwave") is None:
        raise RuntimeError(
            "Deepwave is not installed, but --include-deepwave was requested."
        )
    import deepwave

    return deepwave


def reorder_locations_for_deepwave(args, locations):
    if args.dim == "2d":
        return locations[..., [1, 0]]
    return locations[..., [2, 1, 0]]


def build_cases(args, device):
    equation, shape = make_equation(args, device)
    common_kwargs = common_solver_kwargs(args, shape, device)
    cases = []

    def make_cuda_factory(**solver_kwargs):
        return lambda: PropCUDA(equation, **common_kwargs, **solver_kwargs)

    def make_torch_factory():
        return lambda: PropTorch(equation, use_ckpt=False, **common_kwargs)

    cases.append(
        {
            "name": "cuda_full",
            "group": "full",
            "x_value": 0,
            "solver_factory": make_cuda_factory(
                boundary_saving_config={"enabled": False, "storage": "gpu", "transfer_interval": 1, "pinned_memory": False},
                use_ckpt=False,
            ),
        }
    )

    for chunk in parse_int_list(args.checkpoint_chunks):
        cases.append(
            {
                "name": f"cuda_ckpt_chunk_{chunk}",
                "group": "chunk",
                "x_value": chunk,
                "solver_factory": make_cuda_factory(
                    boundary_saving_config={"enabled": False, "storage": "gpu", "transfer_interval": 1, "pinned_memory": False},
                    use_ckpt=True,
                    ckpt_mode="chunk",
                    ckpt_chunks=chunk,
                ),
            }
        )

    for count in parse_int_list(args.checkpoint_counts):
        cases.append(
            {
                "name": f"cuda_ckpt_recursive_{count}",
                "group": "recursive",
                "x_value": count,
                "solver_factory": make_cuda_factory(
                    boundary_saving_config={"enabled": False, "storage": "gpu", "transfer_interval": 1, "pinned_memory": False},
                    use_ckpt=True,
                    ckpt_mode="recursive",
                    ckpt_num=count,
                ),
            }
        )

    boundary_storages = parse_str_list(args.boundary_storages)
    transfer_intervals = parse_int_list(args.transfer_intervals)
    for storage in boundary_storages:
        if storage == "gpu":
            cases.append(
                {
                    "name": "cuda_boundary_gpu",
                    "group": "boundary",
                    "x_value": "gpu",
                    "solver_factory": make_cuda_factory(
                        boundary_saving_config={"enabled": True, "storage": "gpu", "transfer_interval": 1, "pinned_memory": False},
                        use_ckpt=False,
                    ),
                }
            )
            continue

        for interval in transfer_intervals:
            cases.append(
                {
                    "name": f"cuda_boundary_cpu_interval_{interval}",
                    "group": "boundary",
                    "x_value": f"cpu@{interval}",
                    "solver_factory": make_cuda_factory(
                        boundary_saving_config={
                            "enabled": True,
                            "storage": "cpu",
                            "transfer_interval": interval,
                            "pinned_memory": args.pinned_memory,
                        },
                        use_ckpt=False,
                    ),
                }
            )

    if args.include_torch:
        cases.append(
            {
                "name": "torch",
                "group": "baseline",
                "x_value": 0,
                "solver_factory": make_torch_factory(),
                "backend": "propagator",
            }
        )

    if args.include_deepwave:
        require_deepwave()
        cases.append(
            {
                "name": "deepwave",
                "group": "baseline",
                "x_value": 0,
                "solver_factory": lambda: None,
                "backend": "deepwave",
            }
        )

    for case in cases:
        case.setdefault("backend", "propagator")

    return cases


def run_deepwave_pass(args, vp_np, wave, sources, receivers, device):
    deepwave = require_deepwave()
    vp = torch.from_numpy(vp_np.copy()).to(device).requires_grad_(True)
    source_amplitudes = torch.from_numpy(wave.copy()).to(device=device, dtype=torch.float32)[:, None, :]
    source_locations = torch.from_numpy(
        reorder_locations_for_deepwave(
            args,
            sources[:, None, :] if sources.ndim == 2 else sources,
        )
    ).to(device=device, dtype=torch.long)
    receiver_locations = torch.from_numpy(
        reorder_locations_for_deepwave(args, receivers)
    ).to(device=device, dtype=torch.long)
    out = deepwave.scalar(
        vp,
        grid_spacing=args.dh,
        dt=args.dt,
        source_amplitudes=source_amplitudes,
        source_locations=source_locations,
        receiver_locations=receiver_locations,
        accuracy=args.spatial_order,
        pml_width=args.abcn,
        pml_freq=args.fm,
    )
    record = out[-1]
    loss = record.pow(2).sum()
    loss.backward()
    return float(loss.detach().cpu().item())


def run_single_pass(case, solver, args, vp_np, wave, sources, receivers, device):
    if case.get("backend") == "deepwave":
        return run_deepwave_pass(args, vp_np, wave, sources, receivers, device)

    vp = torch.from_numpy(vp_np.copy()).to(device).requires_grad_(True)
    record = solver(
        wavelet=wave,
        sources=sources,
        receivers=receivers,
        models=[vp],
    )
    loss = record.pow(2).sum()
    loss.backward()
    return float(loss.detach().cpu().item())


def benchmark_case(case, args, vp_np, wave, sources, receivers, device, warmup, repeats):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    solver = case["solver_factory"]()
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        for _ in range(warmup):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            run_single_pass(case, solver, args, vp_np, wave, sources, receivers, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        timings = []
        peaks = []
        loss_value = None
        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            loss_value = run_single_pass(case, solver, args, vp_np, wave, sources, receivers, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peaks.append(torch.cuda.max_memory_allocated(device))
            timings.append(time.perf_counter() - start)
    finally:
        del solver
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

    timings_ms = np.asarray(timings, dtype=np.float64) * 1e3
    peaks_mib = np.asarray(peaks, dtype=np.float64) / (1024.0 ** 2) if peaks else np.zeros(0, dtype=np.float64)
    return {
        "name": case["name"],
        "group": case["group"],
        "x_value": case["x_value"],
        "loss": loss_value,
        "mean_ms": float(timings_ms.mean()),
        "std_ms": float(timings_ms.std(ddof=0)),
        "min_ms": float(timings_ms.min()),
        "max_ms": float(timings_ms.max()),
        "peak_mean_mib": float(peaks_mib.mean()) if peaks_mib.size else None,
        "peak_max_mib": float(peaks_mib.max()) if peaks_mib.size else None,
    }


def print_result(result):
    mem_text = "n/a"
    if result["peak_mean_mib"] is not None:
        mem_text = f"peak_mean {result['peak_mean_mib']:.2f} MiB | peak_max {result['peak_max_mib']:.2f} MiB"
    print(
        f"{result['name']:>28} | mean {result['mean_ms']:8.2f} ms | "
        f"std {result['std_ms']:7.2f} ms | {mem_text}"
    )


def default_output_prefix(args):
    return f"acoustic{args.dim}_checkpoint_compare"


def plot_sweeps(results, output_prefix):
    full_result = next((row for row in results if row["group"] == "full"), None)
    torch_result = next((row for row in results if row["name"] == "torch"), None)
    deepwave_result = next((row for row in results if row["name"] == "deepwave"), None)
    chunk_rows = sorted((row for row in results if row["group"] == "chunk"), key=lambda row: row["x_value"])
    recursive_rows = sorted((row for row in results if row["group"] == "recursive"), key=lambda row: row["x_value"])
    boundary_rows = [row for row in results if row["group"] == "boundary"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    def add_reference_lines(ax, key):
        if full_result is not None and full_result.get(key) is not None:
            ax.axhline(full_result[key], color="tab:gray", linestyle="--", linewidth=1.0, label="cuda_full")
        if torch_result is not None and torch_result.get(key) is not None:
            ax.axhline(torch_result[key], color="tab:purple", linestyle=":", linewidth=1.0, label="torch")
        if deepwave_result is not None and deepwave_result.get(key) is not None:
            ax.axhline(deepwave_result[key], color="tab:red", linestyle="-.", linewidth=1.0, label="deepwave")

    def plot_numeric(ax_time, ax_mem, rows, title, xlabel):
        if rows:
            x = [row["x_value"] for row in rows]
            ax_time.plot(x, [row["mean_ms"] for row in rows], marker="o", label=title)
            ax_mem.plot(x, [row["peak_mean_mib"] for row in rows], marker="o", label=title)
        ax_time.set_title(f"{title} time")
        ax_time.set_xlabel(xlabel)
        ax_time.set_ylabel("ms")
        ax_mem.set_title(f"{title} peak memory")
        ax_mem.set_xlabel(xlabel)
        ax_mem.set_ylabel("MiB")
        add_reference_lines(ax_time, "mean_ms")
        add_reference_lines(ax_mem, "peak_mean_mib")
        ax_time.legend()
        ax_mem.legend()

    plot_numeric(axes[0, 0], axes[0, 1], chunk_rows, "chunk checkpoint", "checkpoint_chunks")
    plot_numeric(axes[1, 0], axes[1, 1], recursive_rows, "recursive checkpoint", "checkpoint_count")

    if boundary_rows:
        labels = [row["x_value"] for row in boundary_rows]
        x = np.arange(len(boundary_rows))
        axes[2, 0].bar(x, [row["mean_ms"] for row in boundary_rows], color="tab:green")
        axes[2, 1].bar(x, [row["peak_mean_mib"] for row in boundary_rows], color="tab:orange")
        axes[2, 0].set_xticks(x, labels, rotation=20)
        axes[2, 1].set_xticks(x, labels, rotation=20)
    axes[2, 0].set_title("boundary saving time")
    axes[2, 0].set_ylabel("ms")
    axes[2, 1].set_title("boundary saving peak memory")
    axes[2, 1].set_ylabel("MiB")
    add_reference_lines(axes[2, 0], "mean_ms")
    add_reference_lines(axes[2, 1], "peak_mean_mib")
    axes[2, 0].legend()
    axes[2, 1].legend()

    plt.tight_layout()
    figure_path = Path(f"{output_prefix}_sweeps.png")
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def main(argv=None):
    args = build_parser().parse_args(argv)
    apply_dim_defaults(args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this comparison.")

    device = torch.device("cuda:0")
    vp_np, wave, sources, receivers = build_case(args)
    cases = build_cases(args, device)

    print("Device:", device)
    print("Dimension:", args.dim)
    print("Shape:", vp_np.shape, "nt:", args.nt)
    print("Shots:", args.nshots)
    print("Model:", "constant" if args.constant_model else "layered")
    print("Warmup/Reps:", args.warmup, args.repeats)

    results = []
    for case in cases:
        result = benchmark_case(
            case,
            args,
            vp_np,
            wave,
            sources,
            receivers,
            device,
            args.warmup,
            args.repeats,
        )
        results.append(result)
        print_result(result)

    output_prefix = args.output_prefix or default_output_prefix(args)
    figure_path = plot_sweeps(results, output_prefix)

    json_path = Path(args.json) if args.json else Path(f"{output_prefix}.json")
    payload = {
        "params": vars(args),
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    print(f"\nSaved sweep figure to {figure_path}")
    print(f"Saved raw results to {json_path}")


if __name__ == "__main__":
    main()
