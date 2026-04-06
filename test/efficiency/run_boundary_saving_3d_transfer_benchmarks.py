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

DEFAULT_TRANSFER_INTERVALS = (1, 2, 4, 8, 16, 32)


def parse_transfer_intervals(text):
    values = []
    for chunk in text.split(","):
        item = chunk.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid transfer interval '{item}'. Expected a comma-separated list of integers."
            ) from exc
        if value < 1:
            raise argparse.ArgumentTypeError("Transfer intervals must be >= 1.")
        values.append(value)

    if not values:
        raise argparse.ArgumentTypeError("At least one transfer interval is required.")

    return tuple(dict.fromkeys(values))


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
        return None
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


def format_variant(job):
    if not job["use_boundary_saving"]:
        return "boundary_saving_off"
    if job["storage"] == "gpu":
        return "boundary_saving_gpu"
    pin_text = "pinned" if job["pinned_memory"] else "unpinned"
    return f"boundary_saving_cpu_{pin_text}_ti{job['transfer_interval']}"


def format_row(row):
    pin_text = "-" if row["storage"] == "gpu" else ("on" if row["pinned_memory"] else "off")
    interval_text = "-" if row["storage"] == "gpu" else str(row["transfer_interval"])
    memory_text = (
        f"{row['peak_max_mib']:8.2f} MiB"
        if row["peak_max_mib"] is not None
        else "   n/a   "
    )
    return (
        f"{row['variant']:<34} | "
        f"storage {row['storage']:<3} | "
        f"pinned {pin_text:<3} | "
        f"interval {interval_text:>3} | "
        f"mean {row['mean_ms']:8.2f} ms | "
        f"peak {memory_text}"
    )


def build_jobs(transfer_intervals, include_no_boundary_saving):
    jobs = []
    if include_no_boundary_saving:
        jobs.append(
            {
                "use_boundary_saving": False,
                "storage": "off",
                "pinned_memory": False,
                "transfer_interval": None,
                "extra_args": [],
            }
        )

    jobs.append(
        {
            "use_boundary_saving": True,
            "storage": "gpu",
            "pinned_memory": False,
            "transfer_interval": 1,
            "extra_args": [
                "--use-boundary-saving",
                "--boundary-storage",
                "gpu",
                "--transfer-interval",
                "1",
            ],
        }
    )

    for transfer_interval in transfer_intervals:
        for pinned_memory in (False, True):
            extra_args = [
                "--use-boundary-saving",
                "--boundary-storage",
                "cpu",
                "--transfer-interval",
                str(transfer_interval),
            ]
            if pinned_memory:
                extra_args.append("--pinned-memory")
            jobs.append(
                {
                    "use_boundary_saving": True,
                    "storage": "cpu",
                    "pinned_memory": pinned_memory,
                    "transfer_interval": transfer_interval,
                    "extra_args": extra_args,
                }
            )

    return jobs


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark 3D acoustic boundary saving with GPU storage and CPU staged "
            "storage using pinned/unpinned memory across transfer intervals."
        )
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the benchmark script.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional CSV output path. Defaults to "
            "benchmark_boundary_saving_3d_transfer_summary.csv in this directory."
        ),
    )
    parser.add_argument(
        "--transfer-intervals",
        type=parse_transfer_intervals,
        default=DEFAULT_TRANSFER_INTERVALS,
        help=(
            "Comma-separated CPU transfer intervals to test. "
            f"Default: {','.join(str(v) for v in DEFAULT_TRANSFER_INTERVALS)}."
        ),
    )
    parser.add_argument(
        "--include-no-boundary-saving",
        action="store_true",
        help="Also run a no-boundary-saving baseline.",
    )
    parser.add_argument(
        "--measure-memory",
        action="store_true",
        help="Also collect peak GPU memory from the child benchmark output.",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments forwarded to acoustic_inversion_3d_cuda.py. Prefix with '--'.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    base_dir = Path(__file__).resolve().parent
    script = base_dir / "acoustic_inversion_3d_cuda.py"
    forwarded = args.script_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]

    run_params = {
        "python": args.python,
        "measure_memory": args.measure_memory,
        "transfer_intervals": list(args.transfer_intervals),
        "include_no_boundary_saving": args.include_no_boundary_saving,
        "script_args": forwarded,
        "script_args_map": parse_forwarded_args(forwarded),
    }

    jobs = build_jobs(args.transfer_intervals, args.include_no_boundary_saving)
    rows = []

    for job in jobs:
        cmd = [args.python, script.name]
        if args.measure_memory:
            cmd.append("--measure-memory")
        cmd.extend(job["extra_args"])
        cmd.extend(forwarded)

        variant = format_variant(job)
        print(f"\nRunning {variant}: {' '.join(cmd)}")
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
            raise RuntimeError(f"{variant} failed with exit code {result.returncode}.")

        summary = parse_summary(result.stdout)
        memory = parse_memory(result.stdout)
        script_params = parse_params(result.stdout)
        combined_params = {
            "runner": run_params,
            "job": {
                "variant": variant,
                "storage": job["storage"],
                "pinned_memory": job["pinned_memory"],
                "transfer_interval": job["transfer_interval"],
                "use_boundary_saving": job["use_boundary_saving"],
            },
            "script": script_params,
        }

        rows.append(
            {
                "variant": variant,
                "label": summary["label"],
                "mean_ms": summary["mean_ms"],
                "std_ms": summary["std_ms"],
                "peak_mean_mib": None if memory is None else memory["peak_mean_mib"],
                "peak_max_mib": None if memory is None else memory["peak_max_mib"],
                "storage": job["storage"],
                "pinned_memory": job["pinned_memory"],
                "transfer_interval": job["transfer_interval"],
                "use_boundary_saving": job["use_boundary_saving"],
                "script": script.name,
                "python": args.python,
                "script_args": " ".join([*(["--measure-memory"] if args.measure_memory else []), *job["extra_args"], *forwarded]),
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
        else base_dir / "benchmark_boundary_saving_3d_transfer_summary.csv"
    )
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "label",
                "mean_ms",
                "std_ms",
                "peak_mean_mib",
                "peak_max_mib",
                "storage",
                "pinned_memory",
                "transfer_interval",
                "use_boundary_saving",
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
