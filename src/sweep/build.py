"""``python -m sweep.build`` — compile the CUDA backend ahead of time.

Exists so the build can happen where builds are cheap. Compiling needs nvcc and
a target architecture, not a card, so this runs on a CPU node or in a CI image
and leaves a cached ``sweep_C.so`` that the GPU run picks up::

    TORCH_CUDA_ARCH_LIST=7.0 TORCH_EXTENSIONS_DIR=/scratch/ext python -m sweep.build

Without ``--no-gpu-required`` it behaves exactly like ``sweep.precompile()`` and
insists on a visible device.
"""
import argparse
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sweep.build",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-gpu-required", action="store_true",
                    help="build without a visible device; TORCH_CUDA_ARCH_LIST "
                         "must name the target architecture")
    a = ap.parse_args(argv)

    if a.no_gpu_required and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        print("TORCH_CUDA_ARCH_LIST is unset, so there is no architecture to "
              "build for. Set it to your target card, e.g. "
              "TORCH_CUDA_ARCH_LIST=8.9 (Ada) or 7.0 (V100).", file=sys.stderr)
        return 2

    import sweep

    try:
        sweep.precompile(require_gpu=not a.no_gpu_required)
    except Exception as exc:
        print(f"build failed: {exc}", file=sys.stderr)
        return 1

    from torch.utils import cpp_extension
    where = cpp_extension._get_build_directory("sweep_C", verbose=False)
    print(f"sweep._C is built and cached at {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
