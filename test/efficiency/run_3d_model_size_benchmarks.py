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

BACKENDS = ("cuda3d", "deepwave3d")
DEFAULT_SHAPES = (
    (48, 48, 48),
    (64, 64, 64),
    (80, 80, 80),
    (96, 96, 96),
    (112, 112, 112),
    (128, 128, 128),
    (144, 144, 144),
)


def parse_shape(text):
    try:
        nz_str, ny_str, nx_str = text.lower().split("x", maxsplit=2)
        nz = int(nz_str)
        ny = int(ny_str)
        nx = int(nx_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid shape '{text}'. Expected format NZxNYxNX, for example 64x64x64."
        ) from exc

    if nz <= 0 or ny <= 0 or nx <= 0:
        raise argparse.ArgumentTypeError("NZ, NY, and NX must all be positive integers.")
    return (nz, ny, nx)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run 3D CUDA/Deepwave acoustic benchmarks across model sizes and collect mean/std."
    )
    parser.add_argument(
        "--mode",
        choices=("forward", "inversion"),
        default="forward",
        help="Benchmark mode to run.",
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
            "benchmark_3d_model_size_<mode>_summary.csv in this directory."
        ),
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=parse_shape,
        default=DEFAULT_SHAPES,
        metavar="NZxNYxNX",
        help=(
            "3D model shapes to benchmark. Defaults to: "
            + ", ".join(f"{nz}x{ny}x{nx}" for nz, ny, nx in DEFAULT_SHAPES)
        ),
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to each backend script. Prefix with '--'.",
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
    return (
        f"{row['backend']:>10} | "
        f"shape {row['nz']}x{row['ny']}x{row['nx']:<4} | "
        f"mean {row['mean_ms']:8.2f} ms | "
        f"std {row['std_ms']:7.2f} ms"
    )


def main():
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parent
    forwarded = args.script_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    run_params = {
        "mode": args.mode,
        "python": args.python,
        "script_args": forwarded,
        "script_args_map": parse_forwarded_args(forwarded),
        "shapes": [list(shape) for shape in args.shapes],
    }

    jobs = []
    for backend in BACKENDS:
        script_name = {
            "cuda3d": f"acoustic_{args.mode}_3d_cuda.py",
            "deepwave3d": f"acoustic_{args.mode}_3d_deepwave.py",
        }[backend]
        script = base_dir / script_name
        for nz, ny, nx in args.shapes:
            jobs.append(
                {
                    "backend": backend,
                    "variant": f"shape_{nz}x{ny}x{nx}",
                    "script": script,
                    "extra_args": ["--nz", str(nz), "--ny", str(ny), "--nx", str(nx)],
                    "shape": (nz, ny, nx),
                }
            )

    rows = []
    for job in jobs:
        backend = job["backend"]
        script = job["script"]
        nz, ny, nx = job["shape"]
        cmd = [args.python, script.name, *job["extra_args"], *forwarded]
        print(f"\nRunning {backend} (shape={nz}x{ny}x{nx}): {' '.join(cmd)}")
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
                f"{backend} {args.mode} benchmark for shape {nz}x{ny}x{nx} failed with exit code {result.returncode}."
            )

        summary = parse_summary(result.stdout)
        script_params = parse_params(result.stdout)
        combined_params = {
            "runner": run_params,
            "script": script_params,
        }
        rows.append(
            {
                "backend": backend,
                "variant": job["variant"],
                "label": summary["label"],
                "mean_ms": summary["mean_ms"],
                "std_ms": summary["std_ms"],
                "mode": args.mode,
                "script": script.name,
                "python": args.python,
                "script_args": " ".join([*job["extra_args"], *forwarded]),
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
        else base_dir / f"benchmark_3d_model_size_{args.mode}_summary.csv"
    )
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "variant",
                "label",
                "mean_ms",
                "std_ms",
                "mode",
                "script",
                "python",
                "script_args",
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
