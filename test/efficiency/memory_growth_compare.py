import argparse
import csv
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.equations import Acoustic, Acoustic3D, Elastic, Elastic3D
from sweep.propagator._c import _CompiledPropagator
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
            "Track whether GPU memory grows with iteration count for acoustic or elastic "
            "CUDA/torch runs across full, checkpoint, and boundary-saving modes."
        )
    )
    parser.add_argument("--equation", choices=("acoustic", "elastic"), required=True)
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
    parser.add_argument("--source-x", type=int, default=None)
    parser.add_argument("--source-y", type=int, default=None)
    parser.add_argument("--src-z", type=int, default=1)
    parser.add_argument("--rec-z", type=int, default=1)
    parser.add_argument("--receiver-stride", type=int, default=4)
    parser.add_argument("--constant-model", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--checkpoint-chunks", default=None)
    parser.add_argument("--checkpoint-counts", default="2,4,8")
    parser.add_argument("--boundary-storages", default="gpu,cpu")
    parser.add_argument("--transfer-intervals", default="1,2,4,8")
    parser.add_argument("--pinned-memory", action="store_true")
    parser.add_argument("--include-torch", action="store_true")
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--csv", default=None)
    return parser


def apply_dim_defaults(args):
    if args.equation == "acoustic":
        if args.dim == "2d":
            args.nz = 100 if args.nz is None else args.nz
            args.nx = 512 if args.nx is None else args.nx
            args.nt = 1200 if args.nt is None else args.nt
            args.abcn = 20 if args.abcn is None else args.abcn
            args.checkpoint_chunks = "50,100,200" if args.checkpoint_chunks is None else args.checkpoint_chunks
            args.ny = None
        else:
            args.nz = 64 if args.nz is None else args.nz
            args.ny = 64 if args.ny is None else args.ny
            args.nx = 64 if args.nx is None else args.nx
            args.nt = 1000 if args.nt is None else args.nt
            args.abcn = 30 if args.abcn is None else args.abcn
            args.checkpoint_chunks = "25,50,100" if args.checkpoint_chunks is None else args.checkpoint_chunks
    else:
        if args.dim == "2d":
            args.nz = 100 if args.nz is None else args.nz
            args.nx = 512 if args.nx is None else args.nx
            args.nt = 1200 if args.nt is None else args.nt
            args.abcn = 20 if args.abcn is None else args.abcn
            args.checkpoint_chunks = "50,100,200" if args.checkpoint_chunks is None else args.checkpoint_chunks
            args.ny = None
        else:
            args.nz = 64 if args.nz is None else args.nz
            args.ny = 64 if args.ny is None else args.ny
            args.nx = 64 if args.nx is None else args.nx
            args.nt = 400 if args.nt is None else args.nt
            args.abcn = 10 if args.abcn is None else args.abcn
            args.checkpoint_chunks = "25,50,100" if args.checkpoint_chunks is None else args.checkpoint_chunks


def build_case(args):
    if args.equation == "acoustic":
        if args.dim == "2d":
            vp = np.full((args.nz, args.nx), 2000.0, dtype=np.float32)
            if not args.constant_model:
                vp[args.nz // 2 :, :] = 2600.0
                vp[args.nz // 3 : (2 * args.nz) // 3, args.nx // 4 : (3 * args.nx) // 4] += 150.0
            models = (vp,)
        else:
            vp = np.full((args.nz, args.ny, args.nx), 1800.0, dtype=np.float32)
            if not args.constant_model:
                vp[args.nz // 2 :, :, :] = 2400.0
                vp[
                    args.nz // 3 : (2 * args.nz) // 3,
                    args.ny // 4 : (3 * args.ny) // 4,
                    args.nx // 4 : (3 * args.nx) // 4,
                ] += 100.0
            models = (vp,)
    else:
        if args.dim == "2d":
            vp = np.full((args.nz, args.nx), 2000.0, dtype=np.float32)
            if not args.constant_model:
                vp[args.nz // 2 :, :] = 2600.0
                vp[args.nz // 3 : (2 * args.nz) // 3, args.nx // 4 : (3 * args.nx) // 4] += 150.0
            vs = vp / 1.73
            rho = np.full((args.nz, args.nx), 1000.0, dtype=np.float32)
            if not args.constant_model:
                rho[args.nz // 2 :, :] = 1200.0
            models = (vp, vs, rho)
        else:
            vp = np.full((args.nz, args.ny, args.nx), 1800.0, dtype=np.float32)
            if not args.constant_model:
                vp[args.nz // 2 :, :, :] = 2400.0
                vp[
                    args.nz // 3 : (2 * args.nz) // 3,
                    args.ny // 4 : (3 * args.ny) // 4,
                    args.nx // 4 : (3 * args.nx) // 4,
                ] += 100.0
            vs = vp / 1.73
            rho = np.full((args.nz, args.ny, args.nx), 1000.0, dtype=np.float32)
            if not args.constant_model:
                rho[args.nz // 2 :, :, :] = 1200.0
            models = (vp, vs, rho)

    if args.dim == "2d":
        source_x = args.source_x if args.source_x is not None else args.nx // 2
        sources = np.array([[source_x, args.src_z]], dtype=np.int32)
        receiver_x = np.arange(0, args.nx, args.receiver_stride, dtype=np.int32)
        receivers = np.stack(
            [receiver_x, np.full(receiver_x.shape[0], args.rec_z, dtype=np.int32)],
            axis=1,
        )[None, ...]
    else:
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
    return models, wave, sources, receivers


def make_equation_bundle(args, device):
    if args.equation == "acoustic":
        if args.dim == "2d":
            return Acoustic(spatial_order=args.spatial_order, device=device), (args.nz, args.nx), ["h1"], ["h1"], "cpmlr"
        return Acoustic3D(spatial_order=args.spatial_order, device=device), (args.nz, args.ny, args.nx), ["h1"], ["h1"], "cpmlr"

    if args.dim == "2d":
        return Elastic(spatial_order=args.spatial_order, device=device), (args.nz, args.nx), ["sxx", "szz"], ["vx", "vz"], "cpmls"
    return Elastic3D(spatial_order=args.spatial_order, device=device), (args.nz, args.ny, args.nx), ["sxx", "syy", "szz"], ["vx", "vy", "vz"], "cpmls"


def build_case_factories(args, device):
    equation, shape, source_type, receiver_type, pml_type = make_equation_bundle(args, device)
    common_kwargs = dict(
        shape=shape,
        source_type=source_type,
        receiver_type=receiver_type,
        abcn=args.abcn,
        dh=args.dh,
        dt=args.dt,
        pml_type=pml_type,
        dev=device,
        free_surface=False,
        B=1,
        allow_growth=True,
        nt=args.nt,
    )

    def make_cuda_factory(**solver_kwargs):
        return lambda: _CompiledPropagator(equation, **common_kwargs, **solver_kwargs)

    def make_torch_factory():
        return lambda: PropTorch(equation, use_ckpt=False, **common_kwargs)

    cases = [
        {
            "name": "cuda_full",
            "group": "full",
            "solver_factory": make_cuda_factory(
                boundary_saving_config={"enabled": False, "storage": "gpu", "transfer_interval": 1, "pinned_memory": False},
                use_ckpt=False,
            ),
        }
    ]

    for chunk in parse_int_list(args.checkpoint_chunks):
        cases.append(
            {
                "name": f"cuda_ckpt_chunk_{chunk}",
                "group": "chunk",
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
                "solver_factory": make_cuda_factory(
                    boundary_saving_config={"enabled": False, "storage": "gpu", "transfer_interval": 1, "pinned_memory": False},
                    use_ckpt=True,
                    ckpt_mode="recursive",
                    ckpt_num=count,
                ),
            }
        )

    for storage in parse_str_list(args.boundary_storages):
        if storage == "gpu":
            cases.append(
                {
                    "name": "cuda_boundary_gpu",
                    "group": "boundary",
                    "solver_factory": make_cuda_factory(
                        boundary_saving_config={"enabled": True, "storage": "gpu", "transfer_interval": 1, "pinned_memory": False},
                        use_ckpt=False,
                    ),
                }
            )
        else:
            for interval in parse_int_list(args.transfer_intervals):
                cases.append(
                    {
                        "name": f"cuda_boundary_cpu_interval_{interval}",
                        "group": "boundary",
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
        cases.append({"name": "torch", "group": "baseline", "solver_factory": make_torch_factory()})
    return cases


def run_single_pass(solver, models_np, wave, sources, receivers, device):
    models = [torch.from_numpy(arr.copy()).to(device).requires_grad_(True) for arr in models_np]
    record = solver(wavelet=wave, sources=sources, receivers=receivers, models=models)
    loss = record.pow(2).sum()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return float(loss.detach().cpu().item())


def collect_case_memory(case, models_np, wave, sources, receivers, device, iterations):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    solver = case["solver_factory"]()
    rows = []
    try:
        for iteration in range(1, iterations + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            loss = run_single_pass(solver, models_np, wave, sources, receivers, device)
            allocated = torch.cuda.memory_allocated(device) / (1024.0 ** 2)
            reserved = torch.cuda.memory_reserved(device) / (1024.0 ** 2)
            peak = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            rows.append(
                {
                    "case": case["name"],
                    "group": case["group"],
                    "iteration": iteration,
                    "loss": loss,
                    "allocated_mib": allocated,
                    "reserved_mib": reserved,
                    "peak_mib": peak,
                }
            )
    finally:
        del solver
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
    return rows


def save_csv(rows, csv_path, args):
    csv_path = Path(csv_path)
    fieldnames = [
        "equation",
        "dim",
        "case",
        "group",
        "iteration",
        "loss",
        "allocated_mib",
        "reserved_mib",
        "peak_mib",
        "nz",
        "ny",
        "nx",
        "nt",
        "spatial_order",
        "abcn",
        "receiver_stride",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            out.update(
                {
                    "equation": args.equation,
                    "dim": args.dim,
                    "nz": args.nz,
                    "ny": args.ny if args.dim == "3d" else "",
                    "nx": args.nx,
                    "nt": args.nt,
                    "spatial_order": args.spatial_order,
                    "abcn": args.abcn,
                    "receiver_stride": args.receiver_stride,
                }
            )
            writer.writerow(out)


def plot_rows(rows, output_prefix):
    case_names = []
    for row in rows:
        if row["case"] not in case_names:
            case_names.append(row["case"])

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    metrics = [("allocated_mib", "Allocated"), ("reserved_mib", "Reserved"), ("peak_mib", "Peak")]

    for metric, title in metrics:
        ax = axes[metrics.index((metric, title))]
        for case_name in case_names:
            case_rows = [row for row in rows if row["case"] == case_name]
            ax.plot(
                [row["iteration"] for row in case_rows],
                [row[metric] for row in case_rows],
                marker="o",
                label=case_name,
            )
        ax.set_ylabel("MiB")
        ax.set_title(f"{title} memory vs iteration")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Iteration")
    plt.tight_layout()
    figure_path = Path(f"{output_prefix}_memory_growth.png")
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def default_output_prefix(args):
    return f"{args.equation}{args.dim}_memory_growth"


def main(argv=None):
    args = build_parser().parse_args(argv)
    apply_dim_defaults(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this memory-growth benchmark.")

    device = torch.device("cuda:0")
    models_np, wave, sources, receivers = build_case(args)
    cases = build_case_factories(args, device)

    print("Device:", device)
    print("Equation:", args.equation)
    print("Dimension:", args.dim)
    print("Shape:", models_np[0].shape, "nt:", args.nt)
    print("Iterations:", args.iterations)

    rows = []
    for case in cases:
        case_rows = collect_case_memory(case, models_np, wave, sources, receivers, device, args.iterations)
        rows.extend(case_rows)
        first = case_rows[0]
        last = case_rows[-1]
        print(
            f"{case['name']:>28} | allocated {first['allocated_mib']:.2f}->{last['allocated_mib']:.2f} MiB | "
            f"reserved {first['reserved_mib']:.2f}->{last['reserved_mib']:.2f} MiB | "
            f"peak {first['peak_mib']:.2f}->{last['peak_mib']:.2f} MiB"
        )

    output_prefix = args.output_prefix or default_output_prefix(args)
    csv_path = Path(args.csv) if args.csv else Path(f"{output_prefix}.csv")
    save_csv(rows, csv_path, args)
    figure_path = plot_rows(rows, output_prefix)

    print(f"\nSaved CSV to {csv_path}")
    print(f"Saved figure to {figure_path}")


if __name__ == "__main__":
    main()
