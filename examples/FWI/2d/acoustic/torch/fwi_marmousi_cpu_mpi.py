import argparse

from _fwi_marmousi_common import add_common_run_args, add_mpi_args, run_fwi


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "2D acoustic Marmousi FWI on CPU. Supports both PyTorch eager CPU "
            "and compiled C++ CPU implementations, with optional MPI shot parallelism."
        )
    )
    parser.add_argument(
        "--impl",
        choices=("eager", "c"),
        default="c",
        help="CPU implementation to run. Default: c.",
    )
    add_common_run_args(parser)
    add_mpi_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    run_fwi(
        backend="torch",
        impl=args.impl,
        device="cpu",
        batchsize_override=args.batchsize,
        train_shot_batchsize_override=args.train_shot_batchsize,
        forward_batchsize_override=args.forward_batchsize,
        epochs_override=args.epochs,
        use_compile_override=args.use_compile,
        use_ckpt_override=args.use_ckpt,
        ckpt_chunks_override=args.ckpt_chunks,
        mpi_forward_batchsize_override=args.mpi_forward_batchsize,
        use_mpi=args.mpi,
    )


if __name__ == "__main__":
    main()
