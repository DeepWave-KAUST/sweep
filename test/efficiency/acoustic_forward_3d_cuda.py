import argparse

import torch
import sweep._C as _C  # noqa: F401

from sweep.equations import Acoustic3D
from sweep.propagator.cuda import PropCUDA
from _common import add_benchmark_args, make_acoustic_3d_case, print_params, run_benchmark


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark 3D acoustic forward modeling for the CUDA binding backend."
    )
    parser = add_benchmark_args(parser)
    parser.add_argument("--ny", type=int, default=64)
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
    print(f"Torch device: {device}")
    print_params(
        args,
        backend="cuda3d",
        device=str(device),
        shape=list(case["shape"]),
        nreceivers=int(case["receivers"].shape[1]),
    )

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
        vp = torch.from_numpy(case["vp"]).to(device)

        def run():
            syn = solver(case["wave"], case["sources"], case["receivers"], models=[vp])
            if torch.isnan(syn).any():
                raise ValueError("NaN values found in the synthetic data.")
            return syn

        return run

    print("\nForward benchmark")
    run_benchmark("CUDA binding 3D", build_step, lambda: torch.cuda.synchronize(device), args.warmup, args.repeats)


if __name__ == "__main__":
    main()
