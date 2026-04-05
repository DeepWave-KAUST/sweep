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

BACKENDS = ("cuda", "deepwave")
DEFAULT_SHAPES = (
    (100, 512),
    (150, 768),
    (200, 1024),
    (250, 1280),
    (300, 1536),
    (350, 1792),
    (400, 2048),
)


def parse_shape(text):
    try:
        nz_str, nx_str = text.lower().split("x", maxsplit=1)
        nz = int(nz_str)
        nx = int(nx_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid shape '{text}'. Expected format NZxNX, for example 100x512."
        ) from exc

    if nz <= 0 or nx <= 0:
        raise argparse.ArgumentTypeError("NZ and NX must both be positive integers.")
    return (nz, nx)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run CUDA/Deepwave inversion benchmarks across model sizes and collect mean/std."
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
            "benchmark_inversion_model_size_summary.csv in this directory."
        ),
    )
    parser.add_argument(
        "--shapes",
        nargs="+",
        type=parse_shape,
        default=DEFAULT_SHAPES,
        metavar="NZxNX",
        help=(
            "Model shapes to benchmark. Defaults to: "
            + ", ".join(f"{nz}x{nx}" for nz, nx in DEFAULT_SHAPES)
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
        f"shape {row['nz']}x{row['nx']:<8} | "
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
        "mode": "inversion",
        "python": args.python,
        "script_args": forwarded,
        "script_args_map": parse_forwarded_args(forwarded),
        "shapes": [list(shape) for shape in args.shapes],
    }

    jobs = []
    for backend in BACKENDS:
        script = base_dir / f"acoustic_inversion_{backend}.py"
        for nz, nx in args.shapes:
            jobs.append(
                {
                    "backend": backend,
                    "variant": f"shape_{nz}x{nx}",
                    "script": script,
                    "extra_args": ["--nz", str(nz), "--nx", str(nx)],
                    "shape": (nz, nx),
                }
            )

    rows = []
    for job in jobs:
        backend = job["backend"]
        script = job["script"]
        nz, nx = job["shape"]
        cmd = [args.python, script.name, *job["extra_args"], *forwarded]
        print(f"\nRunning {backend} (shape={nz}x{nx}): {' '.join(cmd)}")
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
                f"{backend} inversion benchmark for shape {nz}x{nx} failed with exit code {result.returncode}."
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
                "mode": "inversion",
                "script": script.name,
                "python": args.python,
                "script_args": " ".join([*job["extra_args"], *forwarded]),
                "nz": script_params.get("nz"),
                "nx": script_params.get("nx"),
                "nt": script_params.get("nt"),
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
        else base_dir / "benchmark_inversion_model_size_summary.csv"
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
                "nx",
                "nt",
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
