#!/usr/bin/env python3
import inspect

import torch

from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch


EXPECTED_PARAMS = ("wavelet", "sources", "receivers")


def check_signature(label, solver):
    sig = inspect.signature(solver)
    names = tuple(sig.parameters.keys())
    print(f"{label}: {sig}")
    for expected in EXPECTED_PARAMS:
        assert expected in names, f"{label} missing '{expected}' in signature {sig}"


def build_torch_solver():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    equation = Acoustic(spatial_order=4, device=device, backend="torch")
    return PropTorch(
        equation,
        shape=(32, 32),
        dt=0.001,
        dh=10.0,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=10,
        free_surface=False,
        dev=device,
        use_ckpt=False,
    )


def maybe_build_cuda_solver():
    if not torch.cuda.is_available():
        print("PropCUDA: skipped (CUDA not available)")
        return None

    from sweep.propagator.cuda import PropCUDA

    device = torch.device("cuda")
    equation = Acoustic(spatial_order=4, device=device, backend="torch")
    return PropCUDA(
        equation,
        shape=(32, 32),
        dt=0.001,
        dh=10.0,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=10,
        free_surface=False,
        dev=device,
        boundary_saving_config={"enabled": False},
    )


def maybe_build_jax_solver():
    try:
        from sweep.propagator.jax import PropJax
    except Exception as exc:
        print(f"PropJax: skipped ({exc})")
        return None

    equation = Acoustic(spatial_order=4, backend="jax")
    return PropJax(
        equation,
        shape=(32, 32),
        dt=0.001,
        dh=10.0,
        source_type=["h1"],
        receiver_type=["h1"],
        abcn=10,
        free_surface=False,
        dev=None,
        use_ckpt=False,
    )


def main():
    solvers = [
        ("PropTorch", build_torch_solver()),
        ("PropCUDA", maybe_build_cuda_solver()),
        ("PropJax", maybe_build_jax_solver()),
    ]

    for label, solver in solvers:
        if solver is None:
            continue
        check_signature(label, solver)

    print("Signature smoke passed.")


if __name__ == "__main__":
    main()
