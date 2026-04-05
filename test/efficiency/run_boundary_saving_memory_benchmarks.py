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
MEMORY_RE = re.compile(
    r"^\s*(?P<name>.+?)\s+\|\s+peak_mean\s+(?P<peak_mean>[0-9.]+)\s+MiB\s+\|\s+peak_max\s+(?P<peak_max>[0-9.]+)\s+MiB",
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
        description="Compare CUDA inversion performance and peak GPU memory with/without boundary saving in 2D and 3D."
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the benchmark scripts.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional CSV output path. Defaults to "
            "benchmark_boundary_saving_memory_summary.csv in this directory."
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
        "script_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to each benchmark script. Prefix with '--'.",
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


def parse_memory(output):
    match = MEMORY_RE.search(output)
    if match is None:
        raise ValueError("Could not find GPU memory summary line in output.")
    return {
        "peak_mean_mib": float(match.group("peak_mean")),
        "peak_max_mib": float(match.group("peak_max")),
    }


def parse_params(output):
    match = PARAMS_RE.search(output)
    if match is None:
        raise ValueError("Could not find PARAMS line in output.")
    return json.loads(match.group("json"))


def parse_forwarded_args(forwarded):
    params = {}
    key = None

    for token in forwarded:
        if token.startswith("--"):
            if key is not None:
                params[key] = True
            key = token[2:].replace("-", "_")
        else:
            if key is None:
                continue
            params[key] = token
            key = None

    if key is not None:
        params[key] = True

    return params


def format_row(row):
    shape_text = row["shape_label"]
    bs_text = "on" if row["use_boundary_saving"] else "off"
    return (
        f"{row['dimension']:>2} | "
        f"bs {bs_text:<3} | "
        f"shape {shape_text:<14} | "
        f"mean {row['mean_ms']:8.2f} ms | "
        f"peak {row['peak_max_mib']:8.2f} MiB"
    )


def main():
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parent
    forwarded = args.script_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    run_params = {
        "mode": "inversion",
        "python": args.python,
        "script_args": forwarded,
        "script_args_map": parse_forwarded_args(forwarded),
        "shape_2d": list(args.shape_2d),
        "shape_3d": list(args.shape_3d),
    }

    shape_2d = args.shape_2d
    shape_3d = args.shape_3d
    jobs = [
        {
            "dimension": "2d",
            "backend": "cuda",
            "variant": "boundary_saving_off",
            "script": base_dir / "acoustic_inversion_cuda.py",
            "shape_label": f"{shape_2d[0]}x{shape_2d[1]}",
            "extra_args": [
                "--measure-memory",
                "--nz",
                str(shape_2d[0]),
                "--nx",
                str(shape_2d[1]),
            ],
            "use_boundary_saving": False,
        },
        {
            "dimension": "2d",
            "backend": "cuda",
            "variant": "boundary_saving_on",
            "script": base_dir / "acoustic_inversion_cuda.py",
            "shape_label": f"{shape_2d[0]}x{shape_2d[1]}",
            "extra_args": [
                "--measure-memory",
                "--use-boundary-saving",
                "--nz",
                str(shape_2d[0]),
                "--nx",
                str(shape_2d[1]),
            ],
            "use_boundary_saving": True,
        },
        {
            "dimension": "3d",
            "backend": "cuda3d",
            "variant": "boundary_saving_off",
            "script": base_dir / "acoustic_inversion_3d_cuda.py",
            "shape_label": f"{shape_3d[0]}x{shape_3d[1]}x{shape_3d[2]}",
            "extra_args": [
                "--measure-memory",
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
                "--measure-memory",
                "--use-boundary-saving",
                "--nz",
                str(shape_3d[0]),
                "--ny",
                str(shape_3d[1]),
                "--nx",
                str(shape_3d[2]),
            ],
            "use_boundary_saving": True,
        },
    ]

    rows = []
    for job in jobs:
        cmd = [args.python, job["script"].name, *job["extra_args"], *forwarded]
        print(
            f"\nRunning {job['dimension']} CUDA "
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
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
            raise RuntimeError(
                f"{job['dimension']} CUDA benchmark with boundary_saving="
                f"{job['use_boundary_saving']} failed with exit code {result.returncode}."
            )

        summary = parse_summary(result.stdout)
        memory = parse_memory(result.stdout)
        script_params = parse_params(result.stdout)
        combined_params = {
            "runner": run_params,
            "script": script_params,
        }
        rows.append(
            {
                "dimension": job["dimension"],
                "backend": job["backend"],
                "variant": job["variant"],
                "label": summary["label"],
                "mean_ms": summary["mean_ms"],
                "std_ms": summary["std_ms"],
                "peak_mean_mib": memory["peak_mean_mib"],
                "peak_max_mib": memory["peak_max_mib"],
                "mode": "inversion",
                "script": job["script"].name,
                "python": args.python,
                "script_args": " ".join([*job["extra_args"], *forwarded]),
                "shape_label": job["shape_label"],
                "use_boundary_saving": job["use_boundary_saving"],
                "nz": script_params.get("nz"),
                "ny": script_params.get("ny"),
                "nx": script_params.get("nx"),
                "nt": script_params.get("nt"),
                "nshots": script_params.get("nshots"),
                "dh": script_params.get("dh"),
                "dt": script_params.get("dt"),
                "delay": script_params.get("delay"),
                "fm": script_params.get("fm"),
                "spatial_order": script_params.get("spatial_order"),
                "abcn": script_params.get("abcn"),
                "warmup": script_params.get("warmup"),
                "repeats": script_params.get("repeats"),
                "receiver_stride": script_params.get("receiver_stride"),
                "shape": json.dumps(script_params.get("shape"), ensure_ascii=True),
                "nreceivers": script_params.get("nreceivers"),
                "device": script_params.get("device"),
                "parameters_json": json.dumps(combined_params, ensure_ascii=True, sort_keys=True),
            }
        )

    print("\nSummary")
    for row in rows:
        print(format_row(row))

    output_path = (
        Path(args.output)
        if args.output
        else base_dir / "benchmark_boundary_saving_memory_summary.csv"
    )
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dimension",
                "backend",
                "variant",
                "label",
                "mean_ms",
                "std_ms",
                "peak_mean_mib",
                "peak_max_mib",
                "mode",
                "script",
                "python",
                "script_args",
                "shape_label",
                "use_boundary_saving",
                "nz",
                "ny",
                "nx",
                "nt",
                "nshots",
                "dh",
                "dt",
                "delay",
                "fm",
                "spatial_order",
                "abcn",
                "warmup",
                "repeats",
                "receiver_stride",
                "shape",
                "nreceivers",
                "device",
                "parameters_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV to {output_path}")


if __name__ == "__main__":
    main()
