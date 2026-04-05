import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from sweep.signal import ricker


def add_benchmark_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--nt", type=int, default=1200)
    parser.add_argument("--nshots", type=int, default=1)
    parser.add_argument("--dh", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--fm", type=float, default=5.0)
    parser.add_argument("--spatial-order", type=int, default=2)
    parser.add_argument("--abcn", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--receiver-stride", type=int, default=4)
    return parser


def make_acoustic_2d_case(args):
    shape = (args.nz, args.nx)

    vp = np.full(shape, 1500.0, dtype=np.float32)
    vp[shape[0] // 2 :, :] = 2000.0

    t = np.arange(args.nt, dtype=np.float32) * args.dt - args.delay
    wave = ricker(t, f=args.fm).astype(np.float32)

    if args.nshots == 1:
        src_x = np.array([args.nx // 2], dtype=np.int32)
    else:
        src_x = np.linspace(0, args.nx - 1, num=args.nshots)
        src_x = np.rint(src_x).astype(np.int32)
    src_z = np.ones_like(src_x, dtype=np.int32)
    sources = np.stack((src_x, src_z), axis=1)

    rec_x = np.arange(0, args.nx, args.receiver_stride, dtype=np.int32)
    rec_z = np.ones_like(rec_x, dtype=np.int32)
    receiver_line = np.stack((rec_x, rec_z), axis=1)
    receivers = np.repeat(receiver_line[None, ...], args.nshots, axis=0)

    return {
        "vp": vp,
        "wave": wave,
        "sources": sources,
        "receivers": receivers,
        "shape": shape,
    }


def make_acoustic_3d_case(args):
    shape = (args.nz, args.ny, args.nx)

    vp = np.full(shape, 1500.0, dtype=np.float32)
    vp[shape[0] // 2 :, :, :] = 2000.0

    t = np.arange(args.nt, dtype=np.float32) * args.dt - args.delay
    wave = ricker(t, f=args.fm).astype(np.float32)

    src_x = np.array([args.nx // 2], dtype=np.int32)
    src_y = np.array([args.ny // 2], dtype=np.int32)
    src_z = np.ones(1, dtype=np.int32)
    sources = np.stack((src_x, src_y, src_z), axis=1)

    rec_x, rec_y = np.meshgrid(
        np.arange(0, args.nx, args.receiver_stride, dtype=np.int32),
        np.arange(0, args.ny, args.receiver_stride, dtype=np.int32),
        indexing="xy",
    )
    rec_z = np.ones_like(rec_x, dtype=np.int32)
    receiver_grid = np.stack(
        (rec_x.reshape(-1), rec_y.reshape(-1), rec_z.reshape(-1)),
        axis=1,
    )
    receivers = receiver_grid[None, ...]

    return {
        "vp": vp,
        "wave": wave,
        "sources": sources,
        "receivers": receivers,
        "shape": shape,
    }


def summary_line(name, timings):
    timings_ms = np.asarray(timings, dtype=np.float64) * 1e3
    return (
        f"{name:>14} | mean {timings_ms.mean():8.2f} ms | "
        f"std {timings_ms.std(ddof=0):7.2f} ms | "
        f"min {timings_ms.min():8.2f} ms | max {timings_ms.max():8.2f} ms"
    )


def summary_stats(timings):
    timings_ms = np.asarray(timings, dtype=np.float64) * 1e3
    return {
        "mean_ms": float(timings_ms.mean()),
        "std_ms": float(timings_ms.std(ddof=0)),
        "min_ms": float(timings_ms.min()),
        "max_ms": float(timings_ms.max()),
    }


def memory_summary_line(name, peaks_bytes):
    peaks_mib = np.asarray(peaks_bytes, dtype=np.float64) / (1024.0 ** 2)
    return (
        f"{name:>14} | peak_mean {peaks_mib.mean():8.2f} MiB | "
        f"peak_max {peaks_mib.max():8.2f} MiB"
    )


def memory_summary_stats(peaks_bytes):
    peaks_mib = np.asarray(peaks_bytes, dtype=np.float64) / (1024.0 ** 2)
    return {
        "peak_mean_mib": float(peaks_mib.mean()),
        "peak_max_mib": float(peaks_mib.max()),
    }


def benchmark_params(args, **extra):
    params = vars(args).copy()
    params.update(extra)
    return params


def print_params(args, **extra):
    print("PARAMS " + json.dumps(benchmark_params(args, **extra), sort_keys=True))


def append_summary_csv(csv_path, row):
    csv_path = Path(csv_path)
    fieldnames = [
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
        "nshots",
        "spatial_order",
        "abcn",
        "warmup",
        "repeats",
        "receiver_stride",
        "shape",
        "nreceivers",
        "device",
        "parameters_json",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_benchmark(name, build_step, synchronize, warmup, repeats):
    step = build_step()
    synchronize()

    for _ in range(warmup):
        step()
        synchronize()

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        step()
        synchronize()
        timings.append(time.perf_counter() - start)

    print(summary_line(name, timings))
    return timings


def run_benchmark_with_memory(name, build_step, synchronize, warmup, repeats, device):
    step = build_step()
    synchronize()

    if device.type != "cuda":
        raise RuntimeError("Memory benchmarking currently requires a CUDA device.")

    for _ in range(warmup):
        torch.cuda.reset_peak_memory_stats(device)
        step()
        synchronize()

    timings = []
    peaks_bytes = []
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        step()
        synchronize()
        timings.append(time.perf_counter() - start)
        peaks_bytes.append(torch.cuda.max_memory_allocated(device))

    print(summary_line(name, timings))
    print(memory_summary_line(name, peaks_bytes))
    return timings, peaks_bytes
