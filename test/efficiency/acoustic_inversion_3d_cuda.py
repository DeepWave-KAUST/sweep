import argparse
import json
import time

import torch
import sweep._C as _C  # noqa: F401

from sweep.equations import Acoustic3D
from sweep.propagator.cuda import PropCUDA
from _common import (
    add_benchmark_args,
    make_acoustic_3d_case,
    print_params,
    run_benchmark,
    run_benchmark_with_memory,
    segment_summary_stats,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark 3D acoustic inversion/gradient computation for the CUDA binding backend."
    )
    parser = add_benchmark_args(parser)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--use-boundary-saving", action="store_true")
    parser.add_argument("--boundary-storage", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--transfer-interval", type=int, default=1)
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-mode", choices=("chunk", "recursive"), default="chunk")
    parser.add_argument("--checkpoint-chunks", type=int, default=100)
    parser.add_argument("--checkpoint-count", type=int, default=4)
    parser.add_argument("--pinned-memory", action="store_true")
    parser.add_argument("--measure-memory", action="store_true")
    parser.add_argument("--profile-segments", action="store_true")
    return parser


def run_segment_profile(name, build_step, synchronize, warmup, repeats):
    step = build_step()
    synchronize()

    for _ in range(warmup):
        step(profile=False)
        synchronize()

    total_timings = []
    segment_timings = {
        "zero_grad": [],
        "forward": [],
        "loss": [],
        "backward": [],
        "residual": [],
    }

    for _ in range(repeats):
        result = step(profile=True)
        total_timings.append(result["total_s"])
        for key in segment_timings:
            segment_timings[key].append(result[f"{key}_s"])

    total_stats = segment_summary_stats({"total": total_timings})["total"]
    segment_stats = segment_summary_stats(segment_timings)

    print(
        f"{name:>14} | mean {total_stats['mean_ms']:8.2f} ms | "
        f"std {total_stats['std_ms']:7.2f} ms | "
        f"min {total_stats['min_ms']:8.2f} ms | max {total_stats['max_ms']:8.2f} ms"
    )

    payload = {
        "total_ms": total_stats,
        "segments_ms": segment_stats,
    }
    print("SEGMENTS " + json.dumps(payload, sort_keys=True))

    print("\nSegment breakdown")
    total_mean = total_stats["mean_ms"]
    ordered_keys = ["zero_grad", "forward", "loss", "backward", "residual"]
    for key in ordered_keys:
        stats = segment_stats[key]
        pct = 0.0 if total_mean == 0.0 else 100.0 * stats["mean_ms"] / total_mean
        print(
            f"{key:>10} | mean {stats['mean_ms']:8.2f} ms | "
            f"std {stats['std_ms']:7.2f} ms | share {pct:6.2f}%"
        )

    sum_ms = sum(segment_stats[key]["mean_ms"] for key in ordered_keys)
    print(f"{'sum':>10} | mean {sum_ms:8.2f} ms | share {100.0 if total_mean else 0.0:6.2f}%")

    return total_timings, segment_timings


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 3D CUDA binding benchmark.")

    args = build_parser().parse_args()
    case = make_acoustic_3d_case(args)
    device = torch.device("cuda")

    print("Backend: CUDA binding 3D")
    print(
        f"Model shape: {case['shape']}, nt={args.nt}, "
        f"shots={case['sources'].shape[0]}, receivers={case['receivers'].shape[1]}"
    )
    print(
        "Boundary saving: "
        f"enabled={args.use_boundary_saving}, storage={args.boundary_storage}, "
        f"transfer_interval={args.transfer_interval}, pinned_memory={args.pinned_memory}"
    )
    print(
        "Checkpointing: "
        f"enabled={args.use_checkpoint}, checkpoint_mode={args.checkpoint_mode}, "
        f"checkpoint_chunks={args.checkpoint_chunks}, checkpoint_count={args.checkpoint_count}"
    )
    print_params(
        args,
        backend="cuda3d",
        device=str(device),
        shape=list(case["shape"]),
        nreceivers=int(case["receivers"].shape[1]),
    )

    call_kwargs = {}
    if args.use_boundary_saving and args.use_checkpoint:
        raise ValueError("--use-boundary-saving and --use-checkpoint cannot be used together.")
    if args.use_boundary_saving:
        call_kwargs["use_boundary_saving"] = True
        call_kwargs["boundary_saving_config"] = {
            "enabled": True,
            "storage": args.boundary_storage,
            "transfer_interval": args.transfer_interval,
            "pinned_memory": args.pinned_memory,
        }

    def build_step():
        solver = PropCUDA(
            Acoustic3D(spatial_order=args.spatial_order, device=device),
            shape=case["shape"],
            dev=device,
            dh=args.dh,
            dt=args.dt,
            source_type=["h1"],
            receiver_type=["h1"],
            abcn=args.abcn,
            free_surface=False,
            pml_type="cpmlr",
            use_ckpt=args.use_checkpoint,
            ckpt_mode=args.checkpoint_mode,
            ckpt_chunks=args.checkpoint_chunks,
            ckpt_num=args.checkpoint_count,
            nt=args.nt,
        )
        vp = torch.from_numpy(case["vp"]).to(device).requires_grad_(True)

        def run():
            if vp.grad is not None:
                vp.grad = None
            record = solver(
                case["wave"],
                case["sources"],
                case["receivers"],
                models=[vp],
                **call_kwargs,
            )
            loss = record.square().mean()
            loss.backward()
            return vp.grad

        def run_profile(profile=False):
            if not profile:
                run()
                return None

            total_start = time.perf_counter()

            zero_start = time.perf_counter()
            if vp.grad is not None:
                vp.grad = None
            zero_end = time.perf_counter()

            forward_start = time.perf_counter()
            record = solver(
                case["wave"],
                case["sources"],
                case["receivers"],
                models=[vp],
                **call_kwargs,
            )
            torch.cuda.synchronize(device)
            forward_end = time.perf_counter()

            loss_start = time.perf_counter()
            loss = record.square().mean()
            torch.cuda.synchronize(device)
            loss_end = time.perf_counter()

            backward_start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize(device)
            backward_end = time.perf_counter()

            total_end = time.perf_counter()

            zero_s = zero_end - zero_start
            forward_s = forward_end - forward_start
            loss_s = loss_end - loss_start
            backward_s = backward_end - backward_start
            total_s = total_end - total_start
            residual_s = total_s - zero_s - forward_s - loss_s - backward_s

            return {
                "total_s": total_s,
                "zero_grad_s": zero_s,
                "forward_s": forward_s,
                "loss_s": loss_s,
                "backward_s": backward_s,
                "residual_s": residual_s,
            }

        if args.profile_segments:
            return run_profile
        return run

    print("\nInversion benchmark")
    if args.profile_segments:
        if args.measure_memory:
            raise ValueError("--profile-segments and --measure-memory cannot be used together.")
        run_segment_profile(
            "CUDA binding 3D",
            build_step,
            lambda: torch.cuda.synchronize(device),
            args.warmup,
            args.repeats,
        )
    elif args.measure_memory:
        run_benchmark_with_memory(
            "CUDA binding 3D",
            build_step,
            lambda: torch.cuda.synchronize(device),
            args.warmup,
            args.repeats,
            device,
        )
    else:
        run_benchmark("CUDA binding 3D", build_step, lambda: torch.cuda.synchronize(device), args.warmup, args.repeats)


if __name__ == "__main__":
    main()
