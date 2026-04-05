import argparse
import csv
from pathlib import Path


DTYPE_SIZES = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
}


def parse_shape(text):
    parts = text.lower().split("x")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError(
            f"Invalid shape '{text}'. Expected NZxNX or NZxNYxNX."
        )

    try:
        dims = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid shape '{text}'. Shape entries must be integers."
        ) from exc

    if any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("All shape entries must be positive.")
    return dims


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate the theoretical GPU memory saved by boundary saving, "
            "assuming only the outer N layers are stored at each time step, "
            "with N = spatial_order / 2."
        )
    )
    parser.add_argument(
        "--shape",
        type=parse_shape,
        required=True,
        help="Model shape in NZxNX or NZxNYxNX format.",
    )
    parser.add_argument("--nt", type=int, required=True, help="Number of time steps.")
    parser.add_argument(
        "--spatial-order",
        type=int,
        required=True,
        help="Finite-difference spatial order. N is taken as spatial_order / 2.",
    )
    parser.add_argument(
        "--nvar",
        type=int,
        default=1,
        help="Number of wavefield variables stored per time step. Default: 1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size / number of shots stored together. Default: 1.",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_SIZES),
        default="float32",
        help="Element dtype used for storage. Default: float32.",
    )
    parser.add_argument(
        "--include-last-two",
        action="store_true",
        help="Also add the memory for two extra full wavefields, often kept during time stepping.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional CSV output path. Defaults to "
            "theoretical_boundary_saving_memory.csv in this directory."
        ),
    )
    return parser


def numel(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def shell_cells(shape, width):
    interior = 1
    for dim in shape:
        inner_dim = max(dim - 2 * width, 0)
        interior *= inner_dim
    return numel(shape) - interior


def bytes_to_mib(value):
    return value / (1024.0 ** 2)


def format_bytes(value):
    mib = bytes_to_mib(value)
    gib = value / (1024.0 ** 3)
    if gib >= 1.0:
        return f"{gib:.3f} GiB"
    return f"{mib:.3f} MiB"


def main():
    args = build_parser().parse_args()

    if args.spatial_order <= 0 or args.spatial_order % 2 != 0:
        raise ValueError("--spatial-order must be a positive even integer.")
    if args.nt <= 0:
        raise ValueError("--nt must be positive.")
    if args.nvar <= 0:
        raise ValueError("--nvar must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    shape = args.shape
    dim = len(shape)
    shell_width = args.spatial_order // 2
    dtype_size = DTYPE_SIZES[args.dtype]

    full_cells = numel(shape)
    boundary_cells = shell_cells(shape, shell_width)
    base_multiplier = args.nt * args.nvar * args.batch_size

    full_bytes = base_multiplier * full_cells * dtype_size
    boundary_bytes = base_multiplier * boundary_cells * dtype_size

    extra_last_two_bytes = 0
    if args.include_last_two:
        extra_last_two_bytes = 2 * args.nvar * args.batch_size * full_cells * dtype_size
        boundary_bytes += extra_last_two_bytes

    saved_bytes = full_bytes - boundary_bytes
    saved_ratio = saved_bytes / full_bytes if full_bytes > 0 else 0.0

    print("Theoretical Boundary Saving Memory Estimate")
    print(f"Dimension: {dim}D")
    print(f"Shape: {shape}")
    print(f"nt: {args.nt}")
    print(f"spatial_order: {args.spatial_order}")
    print(f"N (shell width): {shell_width}")
    print(f"nvar: {args.nvar}")
    print(f"batch_size: {args.batch_size}")
    print(f"dtype: {args.dtype} ({dtype_size} bytes)")
    print()
    print(f"Full-domain cells per step: {full_cells}")
    print(f"Boundary-shell cells per step: {boundary_cells}")
    print(f"Boundary-shell fraction: {boundary_cells / full_cells:.6f}")
    print()
    print(f"Full-wavefield storage: {format_bytes(full_bytes)}")
    print(f"Boundary-saving storage: {format_bytes(boundary_bytes)}")
    if args.include_last_two:
        print(f"Included extra last-two full wavefields: {format_bytes(extra_last_two_bytes)}")
    print(f"Saved memory: {format_bytes(saved_bytes)}")
    print(f"Saved ratio: {saved_ratio:.2%}")

    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent / "theoretical_boundary_saving_memory.csv"
    )
    row = {
        "dim": dim,
        "shape": "x".join(str(v) for v in shape),
        "nt": args.nt,
        "spatial_order": args.spatial_order,
        "shell_width": shell_width,
        "nvar": args.nvar,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "dtype_bytes": dtype_size,
        "include_last_two": args.include_last_two,
        "full_cells": full_cells,
        "boundary_cells": boundary_cells,
        "boundary_fraction": boundary_cells / full_cells,
        "full_bytes": full_bytes,
        "boundary_bytes": boundary_bytes,
        "extra_last_two_bytes": extra_last_two_bytes,
        "saved_bytes": saved_bytes,
        "full_mib": bytes_to_mib(full_bytes),
        "boundary_mib": bytes_to_mib(boundary_bytes),
        "saved_mib": bytes_to_mib(saved_bytes),
        "saved_ratio": saved_ratio,
    }
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dim",
                "shape",
                "nt",
                "spatial_order",
                "shell_width",
                "nvar",
                "batch_size",
                "dtype",
                "dtype_bytes",
                "include_last_two",
                "full_cells",
                "boundary_cells",
                "boundary_fraction",
                "full_bytes",
                "boundary_bytes",
                "extra_last_two_bytes",
                "saved_bytes",
                "full_mib",
                "boundary_mib",
                "saved_mib",
                "saved_ratio",
            ],
        )
        writer.writeheader()
        writer.writerow(row)

    print(f"\nSaved CSV to {output_path}")


if __name__ == "__main__":
    main()
