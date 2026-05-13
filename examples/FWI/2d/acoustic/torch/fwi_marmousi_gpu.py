import argparse

from _fwi_marmousi_common import add_common_run_args, run_fwi


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "2D acoustic Marmousi FWI on GPU. Supports both PyTorch eager CUDA "
            "and compiled CUDA implementations."
        )
    )
    parser.add_argument(
        "--impl",
        choices=("eager", "c"),
        default="eager",
        help="GPU implementation to run. Default: eager.",
    )
    add_common_run_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    run_fwi(
        backend="torch",
        impl=args.impl,
        device="cuda",
        batchsize_override=args.batchsize,
        train_shot_batchsize_override=args.train_shot_batchsize,
        forward_batchsize_override=args.forward_batchsize,
        epochs_override=args.epochs,
        use_compile_override=args.use_compile,
        use_ckpt_override=args.use_ckpt,
        ckpt_chunks_override=args.ckpt_chunks,
        use_mpi=False,
    )


if __name__ == "__main__":
    main()
