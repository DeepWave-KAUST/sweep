import argparse

import torch
import sweep._C as _C  # noqa: F401

from sweep.equations import Acoustic3D
from sweep.propagator.cuda import PropCUDA
from _common import add_benchmark_args, make_acoustic_3d_case, print_params, run_benchmark, run_benchmark_with_memory


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark 3D acoustic inversion/gradient computation for the CUDA binding backend."
    )
    parser = add_benchmark_args(parser)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--use-boundary-saving", action="store_true")
    parser.add_argument("--boundary-storage", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--transfer-interval", type=int, default=1)
    parser.add_argument("--pinned-memory", action="store_true")
    parser.add_argument("--measure-memory", action="store_true")
    return parser


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
    print_params(
        args,
        backend="cuda3d",
        device=str(device),
        shape=list(case["shape"]),
        nreceivers=int(case["receivers"].shape[1]),
    )

    call_kwargs = {}
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
            use_ckpt=False,
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

        return run

    print("\nInversion benchmark")
    if args.measure_memory:
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
