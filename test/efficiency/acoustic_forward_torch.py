import argparse

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from _common import add_benchmark_args, make_acoustic_2d_case, print_params, run_benchmark


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark 2D acoustic forward modeling for the PyTorch backend."
    )
    parser = add_benchmark_args(parser)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def resolve_device(option):
    import torch

    if option == "cpu":
        return torch.device("cpu")
    if option == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device=cuda but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    import torch

    args = build_parser().parse_args()
    case = make_acoustic_2d_case(args)
    device = resolve_device(args.device)

    print(f"Backend: PyTorch")
    print(f"Model shape: {case['shape']}, nt={args.nt}, receivers={case['receivers'].shape[1]}")
    print(f"Torch device: {device}")
    print_params(args, backend="torch", device=str(device), shape=list(case["shape"]), nreceivers=int(case["receivers"].shape[1]))

    def build_step():
        solver = PropTorch(
            Acoustic(spatial_order=args.spatial_order, device=device, backend="torch"),
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
        )
        vp = torch.from_numpy(case["vp"]).to(device)

        def run():
            return solver(case["wave"], case["sources"], case["receivers"], models=[vp])

        return run

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    print("\nForward benchmark")
    run_benchmark("PyTorch", build_step, sync, args.warmup, args.repeats)


if __name__ == "__main__":
    main()
