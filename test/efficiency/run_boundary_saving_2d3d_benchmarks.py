import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


SUMMARY_RE = re.compile(
    r"^\s*(?P<name>.+?)\s+\|\s+mean\s+(?P<mean>[0-9.]+)\s+ms\s+\|\s+std\s+(?P<std>[0-9.]+)\s+ms",
    re.MULTILINE,
)
PARAMS_RE = re.compile(r"^PARAMS\s+(?P<json>\{.*\})$", re.MULTILINE)

DEFAULT_2D_SHAPE = (100, 512)
DEFAULT_3D_SHAPE = (64, 64, 64)


def parse_2d_shape(text):
    try:
        nz_str, nx_str = text.lower().split("x", maxsplit=1)
        nz = int(nz_str)
        nx = int(nx_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid 2D shape '{text}'. Expected format NZxNX, for example 100x512."
        ) from exc

    if nz <= 0 or nx <= 0:
        raise argparse.ArgumentTypeError("NZ and NX must both be positive integers.")
    return (nz, nx)


def parse_3d_shape(text):
    try:
        nz_str, ny_str, nx_str = text.lower().split("x", maxsplit=2)
        nz = int(nz_str)
        ny = int(ny_str)
        nx = int(nx_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid 3D shape '{text}'. Expected format NZxNYxNX, for example 64x64x64."
        ) from exc

    if nz <= 0 or ny <= 0 or nx <= 0:
        raise argparse.ArgumentTypeError("NZ, NY, and NX must all be positive integers.")
    return (nz, ny, nx)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark 2D/3D acoustic CUDA inversion with and without boundary saving."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the child benchmark scripts.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional CSV output path. Defaults to "
            "benchmark_boundary_saving_2d3d_summary.csv in this directory."
        ),
    )
    parser.add_argument(
        "--shape-2d",
        type=parse_2d_shape,
        default=DEFAULT_2D_SHAPE,
        metavar="NZxNX",
        help=f"2D model shape. Default: {DEFAULT_2D_SHAPE[0]}x{DEFAULT_2D_SHAPE[1]}.",
    )
    parser.add_argument(
        "--shape-3d",
        type=parse_3d_shape,
        default=DEFAULT_3D_SHAPE,
        metavar="NZxNYxNX",
        help=(
            "3D model shape. Default: "
            f"{DEFAULT_3D_SHAPE[0]}x{DEFAULT_3D_SHAPE[1]}x{DEFAULT_3D_SHAPE[2]}."
        ),
    )
    parser.add_argument(
        "--boundary-storage",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Boundary-saving storage mode for the enabled cases.",
    )
    parser.add_argument(
        "--transfer-interval",
        type=int,
        default=1,
        help="Boundary transfer interval forwarded to boundary-saving runs.",
    )
    parser.add_argument(
        "--pinned-memory",
        action="store_true",
        help="Use pinned CPU memory when boundary storage is cpu.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to the child benchmark scripts. Prefix with '--'.",
    )
    return parser


def parse_summary(output):
    match = SUMMARY_RE.search(output)
    if match is None:
        raise ValueError("Could not find benchmark summary line in output.")
    return {
        "label": match.group("name").strip(),
        "mean_ms": float(match.group("mean")),
        "std_ms": float(match.group("std")),
    }


def parse_params(output):
    match = PARAMS_RE.search(output)
    if match is None:
        raise ValueError("Could not find PARAMS line in output.")
    return json.loads(match.group("json"))


def format_row(row):
    return (
        f"{row['dimension']:>2} | "
        f"bs {'on' if row['use_boundary_saving'] else 'off':<3} | "
        f"shape {row['shape_label']:<14} | "
        f"storage {row['boundary_storage']:<3} | "
        f"mean {row['mean_ms']:8.2f} ms | "
        f"std {row['std_ms']:7.2f} ms"
    )


def append_rows(csv_path, rows):
    fieldnames = [
        "dimension",
        "backend",
        "variant",
        "label",
        "mean_ms",
        "std_ms",
        "use_boundary_saving",
        "boundary_storage",
        "transfer_interval",
        "pinned_memory",
        "shape_label",
        "parameters_json",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parent
    forwarded = args.script_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    shape_2d = args.shape_2d
    shape_3d = args.shape_3d

    bs_extra = [
        "--use-boundary-saving",
        "--boundary-storage",
        args.boundary_storage,
        "--transfer-interval",
        str(args.transfer_interval),
    ]
    if args.pinned_memory:
        bs_extra.append("--pinned-memory")

    jobs = [
        {
            "dimension": "2d",
            "backend": "cuda",
            "variant": "boundary_saving_off",
            "script": base_dir / "acoustic_inversion_cuda.py",
            "shape_label": f"{shape_2d[0]}x{shape_2d[1]}",
            "extra_args": ["--nz", str(shape_2d[0]), "--nx", str(shape_2d[1])],
            "use_boundary_saving": False,
        },
        {
            "dimension": "2d",
            "backend": "cuda",
            "variant": "boundary_saving_on",
            "script": base_dir / "acoustic_inversion_cuda.py",
            "shape_label": f"{shape_2d[0]}x{shape_2d[1]}",
            "extra_args": ["--nz", str(shape_2d[0]), "--nx", str(shape_2d[1]), *bs_extra],
            "use_boundary_saving": True,
        },
        {
            "dimension": "3d",
            "backend": "cuda3d",
            "variant": "boundary_saving_off",
            "script": base_dir / "acoustic_inversion_3d_cuda.py",
            "shape_label": f"{shape_3d[0]}x{shape_3d[1]}x{shape_3d[2]}",
            "extra_args": [
                "--nz",
                str(shape_3d[0]),
                "--ny",
                str(shape_3d[1]),
                "--nx",
                str(shape_3d[2]),
            ],
            "use_boundary_saving": False,
        },
        {
            "dimension": "3d",
            "backend": "cuda3d",
            "variant": "boundary_saving_on",
            "script": base_dir / "acoustic_inversion_3d_cuda.py",
            "shape_label": f"{shape_3d[0]}x{shape_3d[1]}x{shape_3d[2]}",
            "extra_args": [
                "--nz",
                str(shape_3d[0]),
                "--ny",
                str(shape_3d[1]),
                "--nx",
                str(shape_3d[2]),
                *bs_extra,
            ],
            "use_boundary_saving": True,
        },
    ]

    rows = []
    for job in jobs:
        cmd = [args.python, job["script"].name, *job["extra_args"], *forwarded]
        print(
            f"\nRunning {job['dimension']} "
            f"(boundary_saving={'on' if job['use_boundary_saving'] else 'off'}): {' '.join(cmd)}"
        )
        result = subprocess.run(
            cmd,
            cwd=base_dir,
            text=True,
            capture_output=True,
            check=False,
        )

        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"Benchmark command failed with exit code {result.returncode}: {' '.join(cmd)}"
            )

        summary = parse_summary(result.stdout)
        params = parse_params(result.stdout)
        rows.append(
            {
                "dimension": job["dimension"],
                "backend": job["backend"],
                "variant": job["variant"],
                "label": summary["label"],
                "mean_ms": summary["mean_ms"],
                "std_ms": summary["std_ms"],
                "use_boundary_saving": job["use_boundary_saving"],
                "boundary_storage": (
                    args.boundary_storage if job["use_boundary_saving"] else "off"
                ),
                "transfer_interval": args.transfer_interval if job["use_boundary_saving"] else None,
                "pinned_memory": bool(args.pinned_memory if job["use_boundary_saving"] else False),
                "shape_label": job["shape_label"],
                "parameters_json": json.dumps(params, sort_keys=True),
            }
        )

    print("\nSummary")
    for row in rows:
        print(format_row(row))

    if args.output is None:
        csv_path = base_dir / "benchmark_boundary_saving_2d3d_summary.csv"
    else:
        csv_path = Path(args.output)
    append_rows(csv_path, rows)
    print(f"\nSaved CSV summary to {csv_path}")


if __name__ == "__main__":
    main()
