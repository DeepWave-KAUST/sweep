"""Small c/cpu forward-record and gradient smoke test.

Run from the repository root after installing the package, for example:

    PYTHONPATH=src python test/c_cpu_forward_gradient_smoke.py

The script exercises the C backend on CPU tensors.  Records are compared
against the PyTorch CPU backend.  Acoustic2D gradients use the compiled CUDA
backend as the reference when CUDA is available, because the compiled acoustic
adjoint is a handwritten GPU-style adjoint rather than PyTorch eager autograd.
Gradient comparisons also check cosine similarity so tiny gradients are not
accepted solely because their absolute errors are small.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sweep.equations import Acoustic, Acoustic3D, Elastic  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    equation_cls: type
    shape: tuple[int, ...]
    pml_type: str
    source_type: list[str]
    receiver_type: list[str]
    wavelet_scale: float = 1.0


@dataclass(frozen=True)
class TensorMetrics:
    max_abs: float
    max_rel: float
    rel_l2: float
    cosine: float
    reference_norm: float
    candidate_norm: float


CASES = {
    "acoustic2d": Case(
        name="acoustic2d",
        equation_cls=Acoustic,
        shape=(20, 22),
        pml_type="cpmlr",
        source_type=["h1"],
        receiver_type=["h1"],
        wavelet_scale=1.0,
    ),
    "elastic2d": Case(
        name="elastic2d",
        equation_cls=Elastic,
        shape=(20, 22),
        pml_type="cpmls",
        source_type=["sxx", "szz"],
        receiver_type=["vx", "vz"],
        wavelet_scale=1.0e6,
    ),
    "acoustic3d": Case(
        name="acoustic3d",
        equation_cls=Acoustic3D,
        shape=(10, 10, 10),
        pml_type="cpmlr",
        source_type=["h1"],
        receiver_type=["h1"],
        wavelet_scale=1.0,
    ),
}

MODES = ("full", "boundary")


def ricker(nt: int, dt: float, fm: float = 12.0, delay: float = 0.006, scale: float = 1.0) -> np.ndarray:
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return (scale * (1.0 - 2.0 * arg**2) * np.exp(-arg**2)).astype(np.float32)


def geometry(shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    if len(shape) == 3:
        nz, ny, nx = shape
        sources = np.array([[[nx // 2, ny // 2, nz // 2]]], dtype=np.int32)
        receivers = np.array(
            [
                [
                    [nx // 3, ny // 2, nz // 2],
                    [nx // 2, ny // 2, nz // 2],
                    [2 * nx // 3, ny // 2, nz // 2],
                ]
            ],
            dtype=np.int32,
        )
        return sources, receivers

    nz, nx = shape
    sources = np.array([[[nx // 2, nz // 2]]], dtype=np.int32)
    receivers = np.array(
        [[[nx // 3, nz // 2], [nx // 2, nz // 2], [2 * nx // 3, nz // 2]]],
        dtype=np.int32,
    )
    return sources, receivers


def model_arrays(model_names: list[str], shape: tuple[int, ...]) -> list[np.ndarray]:
    grid = np.linspace(0.0, 1.0, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
    arrays: list[np.ndarray] = []
    for name in model_names:
        if name == "vp":
            arrays.append(2200.0 + 40.0 * grid)
        elif name == "vs":
            arrays.append(1200.0 + 20.0 * grid)
        elif name == "rho":
            arrays.append(2000.0 + 10.0 * grid)
        else:
            raise KeyError(f"No synthetic model recipe for model '{name}'.")
    return [array.astype(np.float32, copy=False) for array in arrays]


def make_equation(case: Case, device: torch.device | None = None) -> object:
    return case.equation_cls(spatial_order=2, device=device or torch.device("cpu"), backend="torch")


def make_solver(case: Case, backend: str, mode: str, device: torch.device | None = None) -> PropTorch:
    device = device or torch.device("cpu")
    equation = make_equation(case, device)
    common = dict(
        shape=case.shape,
        dev=device,
        dh=10.0,
        dt=0.001,
        source_type=case.source_type,
        receiver_type=case.receiver_type,
        abcn=4,
        free_surface=False,
        pml_type=case.pml_type,
        nt=16,
        B=1,
    )
    if backend == "pytorch":
        return PropTorch(equation, backend="eager", use_ckpt=False, **common)

    if backend != "c":
        raise ValueError(f"Unknown backend '{backend}'.")

    boundary_saving_config = {"enabled": False}
    if mode == "boundary":
        boundary_saving_config = {
            "enabled": True,
            "storage": "cpu",
            "transfer_interval": 2,
            "pinned_memory": False,
        }
    elif mode != "full":
        raise ValueError(f"Unknown c/cpu mode '{mode}'.")

    # PropTorch uses backend="cuda" as the public entry point for the compiled
    # extension. CPU tensors dispatch into the C CPU binding through sweep._C.
    return PropTorch(
        equation,
        backend="cuda",
        use_ckpt=False,
        boundary_saving_config=boundary_saving_config,
        **common,
    )


def standard_record(case: Case, backend: str, record: torch.Tensor) -> torch.Tensor:
    record = record.cpu()
    if backend == "pytorch":
        return record.detach()
    if case.name in {"acoustic2d", "acoustic3d"}:
        return record.detach().transpose(1, 2).unsqueeze(-1)
    if case.name == "elastic2d":
        return record.detach().permute(1, 3, 2, 0)
    raise ValueError(case.name)


def run_once(
    case: Case,
    backend: str,
    mode: str,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = device or torch.device("cpu")
    solver = make_solver(case, backend, mode, device)
    model_names = make_equation(case).models
    models = [
        torch.tensor(array, dtype=torch.float32, device=device, requires_grad=True)
        for array in model_arrays(model_names, case.shape)
    ]
    sources, receivers = geometry(case.shape)
    wavelet = ricker(nt=16, dt=0.001, scale=case.wavelet_scale)

    record = solver(wavelet, sources, receivers, models=models)
    loss = record.pow(2).mean()
    loss.backward()

    grads = {name: model.grad.detach().cpu().clone() for name, model in zip(model_names, models)}
    return standard_record(case, backend, record), grads


def tensor_metrics(candidate: torch.Tensor, reference: torch.Tensor, *, eps: float) -> TensorMetrics:
    candidate_flat = candidate.detach().to(dtype=torch.float64).reshape(-1)
    reference_flat = reference.detach().to(dtype=torch.float64).reshape(-1)
    diff = candidate_flat - reference_flat
    max_abs = float(diff.abs().max().item()) if diff.numel() else 0.0
    max_ref = float(reference_flat.abs().max().item()) if reference_flat.numel() else 0.0
    reference_norm = float(torch.linalg.vector_norm(reference_flat).item())
    candidate_norm = float(torch.linalg.vector_norm(candidate_flat).item())
    rel_l2 = float(torch.linalg.vector_norm(diff).item() / max(reference_norm, eps))
    cosine = float("nan")
    if reference_norm > eps and candidate_norm > eps:
        cosine = float(torch.dot(candidate_flat, reference_flat).item() / (candidate_norm * reference_norm))
        cosine = max(-1.0, min(1.0, cosine))
    return TensorMetrics(
        max_abs=max_abs,
        max_rel=float(max_abs / (max_ref + eps)),
        rel_l2=rel_l2,
        cosine=cosine,
        reference_norm=reference_norm,
        candidate_norm=candidate_norm,
    )


def format_metrics(metrics: TensorMetrics, *, include_cosine: bool) -> str:
    parts = [f"max_abs={metrics.max_abs:.3e}", f"max_rel={metrics.max_rel:.3e}"]
    if include_cosine:
        cosine = "nan" if np.isnan(metrics.cosine) else f"{metrics.cosine:.8f}"
        parts.extend([f"rel_l2={metrics.rel_l2:.3e}", f"cos={cosine}"])
    return " ".join(parts)


def assert_cosine(
    label: str,
    metrics: TensorMetrics,
    *,
    threshold: float,
    eps: float,
    reference_name: str,
) -> None:
    reference_zero = metrics.reference_norm <= eps
    candidate_zero = metrics.candidate_norm <= eps
    if reference_zero and candidate_zero:
        return
    if reference_zero != candidate_zero:
        raise AssertionError(
            f"{label} gradient norm mismatch against {reference_name}: "
            f"reference_norm={metrics.reference_norm:.3e}, "
            f"candidate_norm={metrics.candidate_norm:.3e}, "
            f"{format_metrics(metrics, include_cosine=True)}"
        )
    if metrics.cosine < threshold:
        raise AssertionError(
            f"{label} gradient cosine below threshold against {reference_name}: "
            f"{format_metrics(metrics, include_cosine=True)} threshold={threshold:.8f}"
        )


def check_close(
    label: str,
    candidate: torch.Tensor,
    reference: torch.Tensor,
    rtol: float,
    atol: float,
    reference_name: str = "selected reference",
    *,
    check_cosine: bool = False,
    cosine_threshold: float = 0.999,
    cosine_eps: float = 1.0e-30,
) -> None:
    metrics = tensor_metrics(candidate, reference, eps=cosine_eps)
    torch.testing.assert_close(
        candidate,
        reference,
        rtol=rtol,
        atol=atol,
        msg=lambda msg: (
            f"{label} differs from {reference_name}\n"
            f"{format_metrics(metrics, include_cosine=check_cosine)}\n{msg}"
        ),
    )
    if check_cosine:
        assert_cosine(
            label,
            metrics,
            threshold=cosine_threshold,
            eps=cosine_eps,
            reference_name=reference_name,
        )
    print(f"  {label:<24s} {format_metrics(metrics, include_cosine=check_cosine)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="*",
        choices=sorted(CASES),
        default=sorted(CASES),
        help="Cases to run. Defaults to acoustic2d, acoustic3d, and elastic2d.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        choices=MODES,
        default=list(MODES),
        help="c/cpu modes to run. 'boundary' exercises boundary-saving backward.",
    )
    parser.add_argument("--rtol", type=float, default=5.0e-4)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.999,
        help="Minimum cosine similarity for nonzero gradient checks.",
    )
    parser.add_argument(
        "--cosine-eps",
        type=float,
        default=1.0e-30,
        help="Gradient norm below this value is treated as zero for cosine checks.",
    )
    parser.add_argument(
        "--reference",
        choices=("auto", "pytorch", "cuda"),
        default="auto",
        help=(
            "Gradient reference. 'auto' uses c/cuda for acoustic2d/acoustic3d "
            "when CUDA is available; on CPU-only nodes those modes are compared "
            "against c/cpu full mode."
        ),
    )
    parser.add_argument("--threads", type=int, default=max(1, min(8, torch.get_num_threads())))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(0)

    for case_name in args.cases:
        case = CASES[case_name]
        print(f"{case.name}: pytorch/cpu reference")
        reference_record, reference_grads = run_once(case, "pytorch", "full")
        compiled_acoustic_case = case.name in {"acoustic2d", "acoustic3d"}
        use_cuda_reference = (
            compiled_acoustic_case
            and args.reference in {"auto", "cuda"}
            and torch.cuda.is_available()
        )
        if compiled_acoustic_case and args.reference == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested --reference cuda, but CUDA is not available.")

        compiled_full_grads: dict[str, torch.Tensor] | None = None
        for mode in args.modes:
            print(f"{case.name}: c/cpu mode={mode}")
            candidate_record, candidate_grads = run_once(case, "c", mode)
            check_close(
                f"{case.name}/{mode}/record",
                candidate_record,
                reference_record,
                args.rtol,
                args.atol,
                "pytorch/cpu reference",
            )
            if use_cuda_reference:
                print(f"{case.name}: c/cuda mode={mode} gradient reference")
                _, gradient_reference = run_once(case, "c", mode, torch.device("cuda"))
                gradient_reference_name = "c/cuda reference"
            elif compiled_acoustic_case and args.reference == "auto":
                if compiled_full_grads is None:
                    compiled_full_grads = candidate_grads
                gradient_reference = compiled_full_grads
                gradient_reference_name = "c/cpu full-mode reference"
            else:
                gradient_reference = reference_grads
                gradient_reference_name = "pytorch/cpu reference"
            for name, reference_grad in gradient_reference.items():
                check_close(
                    f"{case.name}/{mode}/grad_{name}",
                    candidate_grads[name],
                    reference_grad,
                    args.rtol,
                    args.atol,
                    gradient_reference_name,
                    check_cosine=True,
                    cosine_threshold=args.cosine_threshold,
                    cosine_eps=args.cosine_eps,
                )

    print("c/cpu forward records and gradients passed the selected smoke checks.")


if __name__ == "__main__":
    main()
