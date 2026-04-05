import argparse

from _common import add_benchmark_args, make_acoustic_3d_case, print_params, run_benchmark


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark 3D acoustic inversion/gradient computation for Deepwave."
    )
    parser = add_benchmark_args(parser)
    parser.add_argument("--ny", type=int, default=64)
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

    try:
        from deepwave import scalar
    except ImportError as exc:
        raise RuntimeError("Deepwave is not installed in the current environment.") from exc

    args = build_parser().parse_args()
    case = make_acoustic_3d_case(args)
    device = resolve_device(args.device)

    print("Backend: Deepwave 3D")
    print(
        f"Model shape: {case['shape']}, nt={args.nt}, "
        f"shots={case['sources'].shape[0]}, receivers={case['receivers'].shape[1]}"
    )
    print(f"Torch device: {device}")
    print_params(
        args,
        backend="deepwave3d",
        device=str(device),
        shape=list(case["shape"]),
        nreceivers=int(case["receivers"].shape[1]),
    )

    def build_step():
        vp = torch.from_numpy(case["vp"].transpose(2, 1, 0).copy()).to(device).requires_grad_(True)
        source_amplitudes = torch.from_numpy(case["wave"][None, None, :]).to(device).float()
        source_amplitudes = source_amplitudes.repeat(case["sources"].shape[0], 1, 1)
        source_locations = torch.from_numpy(case["sources"][:, None, :].copy()).to(device).long()
        receiver_locations = torch.from_numpy(case["receivers"].copy()).to(device).long()

        def run():
            if vp.grad is not None:
                vp.grad = None
            record = scalar(
                vp,
                [args.dh, args.dh, args.dh],
                args.dt,
                source_amplitudes=source_amplitudes,
                source_locations=source_locations,
                receiver_locations=receiver_locations,
                accuracy=args.spatial_order,
                pml_width=[args.abcn] * 6,
                pml_freq=args.fm,
            )[-1]
            loss = record.square().mean()
            loss.backward()
            return vp.grad

        return run

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    print("\nInversion benchmark")
    run_benchmark("Deepwave 3D", build_step, sync, args.warmup, args.repeats)


if __name__ == "__main__":
    main()
