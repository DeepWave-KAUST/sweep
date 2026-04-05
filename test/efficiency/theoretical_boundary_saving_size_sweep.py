import argparse
import csv
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute the theoretical boundary-saving memory ratio versus model size, "
            "assuming nt and spatial order are fixed."
        )
    )
    parser.add_argument(
        "--dim",
        choices=("2d", "3d"),
        default="3d",
        help="Whether to compute 2D squares or 3D cubes. Default: 3d.",
    )
    parser.add_argument(
        "--size-start",
        type=int,
        default=64,
        help="Starting size. Default: 64.",
    )
    parser.add_argument(
        "--size-stop",
        type=int,
        default=512,
        help="Ending size (inclusive). Default: 512.",
    )
    parser.add_argument(
        "--size-step",
        type=int,
        default=64,
        help="Size increment. Default: 64.",
    )
    parser.add_argument(
        "--nt",
        type=int,
        default=1200,
        help="Number of time steps. Fixed during the sweep. Default: 1200.",
    )
    parser.add_argument(
        "--spatial-order",
        type=int,
        default=8,
        help="Finite-difference spatial order. Default: 8.",
    )
    parser.add_argument(
        "--nvar",
        type=int,
        default=1,
        help="Number of stored variables per time step. Default: 1.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size. Default: 1.",
    )
    parser.add_argument(
        "--dtype-bytes",
        type=int,
        default=4,
        help="Bytes per element. float32 corresponds to 4. Default: 4.",
    )
    parser.add_argument(
        "--include-last-two",
        action="store_true",
        help="Also include two extra full wavefields in the boundary-saving estimate.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional CSV output path. Defaults to "
            "theoretical_boundary_saving_size_sweep_<dim>.csv in this directory."
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
        interior *= max(dim - 2 * width, 0)
    return numel(shape) - interior


def bytes_to_mib(value):
    return value / (1024.0 ** 2)


def main():
    args = build_parser().parse_args()

    if args.size_start <= 0 or args.size_stop <= 0 or args.size_step <= 0:
        raise ValueError("Sizes and size step must be positive.")
    if args.size_start > args.size_stop:
        raise ValueError("--size-start must be <= --size-stop.")
    if args.spatial_order <= 0 or args.spatial_order % 2 != 0:
        raise ValueError("--spatial-order must be a positive even integer.")
    if args.nt <= 0 or args.nvar <= 0 or args.batch_size <= 0 or args.dtype_bytes <= 0:
        raise ValueError("nt, nvar, batch-size, and dtype-bytes must be positive.")

    shell_width = args.spatial_order // 2
    sizes = list(range(args.size_start, args.size_stop + 1, args.size_step))
    rows = []

    for size in sizes:
        shape = (size, size, size) if args.dim == "3d" else (size, size)
        full_cells = numel(shape)
        boundary_cells = shell_cells(shape, shell_width)

        multiplier = args.nt * args.nvar * args.batch_size * args.dtype_bytes
        full_bytes = multiplier * full_cells
        boundary_bytes = multiplier * boundary_cells

        extra_last_two_bytes = 0
        if args.include_last_two:
            extra_last_two_bytes = 2 * args.nvar * args.batch_size * args.dtype_bytes * full_cells
            boundary_bytes += extra_last_two_bytes

        saved_bytes = full_bytes - boundary_bytes
        saved_ratio = saved_bytes / full_bytes if full_bytes > 0 else 0.0

        rows.append(
            {
                "dim": args.dim,
                "size": size,
                "shape": "x".join(str(v) for v in shape),
                "nt": args.nt,
                "spatial_order": args.spatial_order,
                "shell_width": shell_width,
                "nvar": args.nvar,
                "batch_size": args.batch_size,
                "dtype_bytes": args.dtype_bytes,
                "include_last_two": args.include_last_two,
                "full_cells": full_cells,
                "boundary_cells": boundary_cells,
                "boundary_fraction": boundary_cells / full_cells,
                "full_mib": bytes_to_mib(full_bytes),
                "boundary_mib": bytes_to_mib(boundary_bytes),
                "saved_mib": bytes_to_mib(saved_bytes),
                "saved_ratio": saved_ratio,
            }
        )

    print("Theoretical Boundary Saving Size Sweep")
    print(f"Dimension: {args.dim}")
    print(f"Sizes: {args.size_start} to {args.size_stop} step {args.size_step}")
    print(f"nt: {args.nt}")
    print(f"spatial_order: {args.spatial_order}")
    print(f"shell width N: {shell_width}")
    print()
    print(
        f"{'size':>6} | {'saved_ratio':>12} | {'saved_mib':>12} | "
        f"{'boundary_frac':>14}"
    )
    for row in rows:
        print(
            f"{row['size']:6d} | "
            f"{row['saved_ratio'] * 100:11.2f}% | "
            f"{row['saved_mib']:11.2f} | "
            f"{row['boundary_fraction'] * 100:13.2f}%"
        )

    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parent / f"theoretical_boundary_saving_size_sweep_{args.dim}.csv"
    )
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dim",
                "size",
                "shape",
                "nt",
                "spatial_order",
                "shell_width",
                "nvar",
                "batch_size",
                "dtype_bytes",
                "include_last_two",
                "full_cells",
                "boundary_cells",
                "boundary_fraction",
                "full_mib",
                "boundary_mib",
                "saved_mib",
                "saved_ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV to {output_path}")


if __name__ == "__main__":
    main()
