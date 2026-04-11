import argparse

from acoustic_checkpoint_compare import main as acoustic_checkpoint_main
from acoustic_gradient_compare import main as acoustic_gradient_main
from elastic_checkpoint_compare import main as elastic_checkpoint_main
from elastic_gradient_compare import main as elastic_gradient_main
from memory_growth_compare import main as memory_growth_main


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified entrypoint for acoustic/elastic comparison scripts."
    )
    parser.add_argument(
        "--equation",
        choices=("acoustic", "elastic"),
        required=True,
        help="Equation family to compare.",
    )
    parser.add_argument(
        "--task",
        choices=("gradient", "checkpoint", "memory-growth"),
        required=True,
        help="Comparison task to run.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.task == "memory-growth":
        memory_growth_main(["--equation", args.equation, *remaining])
        return

    if args.equation == "acoustic" and args.task == "gradient":
        acoustic_gradient_main(remaining)
        return
    if args.equation == "acoustic" and args.task == "checkpoint":
        acoustic_checkpoint_main(remaining)
        return
    if args.equation == "elastic" and args.task == "gradient":
        elastic_gradient_main(remaining)
        return
    if args.equation == "elastic" and args.task == "checkpoint":
        elastic_checkpoint_main(remaining)
        return

    raise ValueError(f"Unsupported combination: equation={args.equation}, task={args.task}")


if __name__ == "__main__":
    main()
