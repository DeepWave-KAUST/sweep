"""Gradient consistency matrix for eager/cpu, eager/gpu, C/cpu, and C/cuda.

Run from the repository root after installing the compiled extension:

    OMP_NUM_THREADS=4 PYTHONPATH=src python test/backend_gradient_matrix.py

The eager backend supports full mode and PyTorch chunk checkpointing. Boundary
saving and recursive checkpointing are compiled-backend modes, so those eager
cells are reported as N/A.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import traceback

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TEST_ROOT = REPO_ROOT / "test"
for path in (SRC_ROOT, TEST_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from cpu_binding_gradient_consistency import (  # noqa: E402
    CASES,
    MODES,
    CUDA_MODE_REFERENCE_ONLY,
    Case,
    RunConfig,
    compare_result_to_reference,
    config_for_scale,
    make_equation,
    make_solver as make_c_solver,
    run_case,
    scaled_cases,
)
from sweep.propagator.options import CkptOptions, CUDAOptions, MemoryOptions  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402


@dataclass(frozen=True)
class BackendSpec:
    label: str
    family: str
    device: str


BACKENDS = (
    BackendSpec("eager/cpu", "eager", "cpu"),
    BackendSpec("eager/gpu", "eager", "cuda"),
    BackendSpec("c-cpu", "c", "cpu"),
    BackendSpec("c-cuda", "c", "cuda"),
)

EAGER_MODES = {"full", "ckpt_chunk"}


def make_eager_solver(case: Case, mode: str, device: torch.device, config: RunConfig) -> PropTorch:
    equation = make_equation(case, device, config)
    common = dict(
        shape=case.shape,
        abcn=config.abcn,
        dh=10.0,
        dt=config.dt,
        dev=device,
        pml_type=case.pml_type,
        nt=config.nt,
        B=1,
        receiver_type=case.receiver_type,
    )
    if mode == "full":
        return PropTorch(equation, backend="eager", use_ckpt=False, **common)
    if mode == "ckpt_chunk":
        return PropTorch(equation, backend="eager", use_ckpt=True, ckpt_mode="chunk", ckpt_chunks=config.ckpt_chunks, **common)
    raise ValueError(f"eager backend does not support mode {mode!r}")


def make_solver(case: Case, mode: str, backend: BackendSpec, config: RunConfig) -> PropTorch:
    device = torch.device(backend.device)
    if backend.family == "eager":
        return make_eager_solver(case, mode, device, config)
    return make_c_solver(case, mode, device, config)


def backend_supported(mode: str, backend: BackendSpec, cuda_available: bool) -> bool:
    if backend.device == "cuda" and not cuda_available:
        return False
    if backend.family == "eager":
        return mode in EAGER_MODES
    return True


def reference_label_for_mode(mode: str, cuda_available: bool) -> str:
    if mode in EAGER_MODES:
        return "eager/cpu"
    if cuda_available:
        return "c-cuda"
    return "c-cpu"


def one_line_error(exc: BaseException) -> str:
    text = str(exc).strip()
    if text:
        return text.splitlines()[0]
    return "".join(traceback.format_exception_only(type(exc), exc)).strip().splitlines()[0]


def run_backend(case: Case, mode: str, backend: BackendSpec, config: RunConfig):
    device = torch.device(backend.device)
    solver = make_solver(case, mode, backend, config)
    return run_case(solver, case, device, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=[case.name for case in CASES],
        choices=[case.name for case in CASES],
    )
    parser.add_argument("--modes", nargs="*", default=list(MODES), choices=list(MODES))
    parser.add_argument(
        "--scale",
        choices=("tiny", "cuda-suite"),
        default="tiny",
        help=(
            "Problem size profile. 'cuda-suite' matches the older CUDA gradient "
            "suite defaults: 2D=48x56, 3D=24x20x24, nt=120, so=4, abcn=30."
        ),
    )
    parser.add_argument("--threads", type=int, default=max(1, min(8, torch.get_num_threads())))
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--cosine-threshold", type=float, default=0.999)
    parser.add_argument("--cosine-eps", type=float, default=1e-30)
    parser.add_argument(
        "--reference-scope",
        choices=("cross", "within"),
        default="cross",
        help=(
            "cross compares every backend to one row reference "
            "(eager/cpu for eager-supported modes, otherwise c-cuda). "
            "within compares each backend mode to that same backend's full mode."
        ),
    )
    return parser.parse_args()


def run_cross_matrix(args: argparse.Namespace, selected_cases: list[Case], cuda_available: bool, config: RunConfig) -> tuple[list[dict[str, str]], list[str]]:
    selected_modes = list(dict.fromkeys(args.modes))
    rows: list[dict[str, str]] = []
    failures: list[str] = []

    for case in selected_cases:
        for mode in selected_modes:
            reference_label = reference_label_for_mode(mode, cuda_available)
            results = {}
            row = {"solver": case.name, "mode": mode, "reference": reference_label}

            for backend in BACKENDS:
                if not backend_supported(mode, backend, cuda_available):
                    row[backend.label] = "N/A"
                    continue
                try:
                    print(f"running cross {case.name} {mode} {backend.label}", flush=True)
                    results[backend.label] = run_backend(case, mode, backend, config)
                    row[backend.label] = "RUN"
                except Exception as exc:  # noqa: BLE001 - test matrix should keep going
                    row[backend.label] = "FAIL"
                    failures.append(f"{case.name}/{mode}/{backend.label}: run failed: {one_line_error(exc)}")

            reference = results.get(reference_label)
            if reference is None:
                rows.append(row)
                continue

            row[reference_label] = "REF"
            for backend in BACKENDS:
                if backend.label not in results or backend.label == reference_label:
                    continue
                ok, _, errors = compare_result_to_reference(
                    case.name,
                    f"{mode}/{backend.label}",
                    reference,
                    results[backend.label],
                    reference_label=reference_label,
                    rtol=args.rtol,
                    atol=args.atol,
                    cosine_threshold=args.cosine_threshold,
                    cosine_eps=args.cosine_eps,
                    strict=False,
                )
                if ok:
                    row[backend.label] = "PASS"
                    continue
                row[backend.label] = "FAIL"
                suffix = ""
                if (case.name, mode) in CUDA_MODE_REFERENCE_ONLY and {backend.label, reference_label} == {"c-cpu", "c-cuda"}:
                    suffix = " (mode follows c-cuda rather than c-cpu full)"
                failures.append(f"{case.name}/{mode}/{backend.label}: {errors[0] if errors else 'mismatch'}{suffix}")

            rows.append(row)

    return rows, failures


def run_within_matrix(args: argparse.Namespace, selected_cases: list[Case], cuda_available: bool, config: RunConfig) -> tuple[list[dict[str, str]], list[str]]:
    selected_modes = list(dict.fromkeys(args.modes))
    rows: list[dict[str, str]] = []
    failures: list[str] = []

    for case in selected_cases:
        full_results = {}
        for backend in BACKENDS:
            if not backend_supported("full", backend, cuda_available):
                continue
            try:
                print(f"running full reference {case.name} {backend.label}", flush=True)
                full_results[backend.label] = run_backend(case, "full", backend, config)
            except Exception as exc:  # noqa: BLE001 - test matrix should keep going
                failures.append(f"{case.name}/full/{backend.label}: reference run failed: {one_line_error(exc)}")

        for mode in selected_modes:
            row = {"solver": case.name, "mode": mode, "reference": "same backend full"}
            for backend in BACKENDS:
                if not backend_supported(mode, backend, cuda_available):
                    row[backend.label] = "N/A"
                    continue
                reference = full_results.get(backend.label)
                if reference is None:
                    row[backend.label] = "FAIL"
                    continue
                if mode == "full":
                    row[backend.label] = "REF"
                    continue
                try:
                    print(f"running mode {case.name} {mode} {backend.label}", flush=True)
                    candidate = run_backend(case, mode, backend, config)
                    ok, _, errors = compare_result_to_reference(
                        case.name,
                        f"{mode}/{backend.label}",
                        reference,
                        candidate,
                        reference_label=f"{backend.label} full",
                        rtol=args.rtol,
                        atol=args.atol,
                        cosine_threshold=args.cosine_threshold,
                        cosine_eps=args.cosine_eps,
                        strict=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    errors = [f"run failed: {one_line_error(exc)}"]
                if ok:
                    row[backend.label] = "PASS"
                else:
                    row[backend.label] = "FAIL"
                    suffix = ""
                    if (case.name, mode) in CUDA_MODE_REFERENCE_ONLY and backend.family == "c":
                        suffix = " (currently aligned to matching c-cuda/c-cpu mode, not full)"
                    failures.append(f"{case.name}/{mode}/{backend.label}: {errors[0] if errors else 'mismatch'}{suffix}")
            rows.append(row)

    return rows, failures


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    cuda_available = torch.cuda.is_available()
    config = config_for_scale(args.scale)
    selected_cases = [case for case in CASES if case.name in set(args.cases)]
    selected_cases = list(scaled_cases(config, tuple(selected_cases)))
    if args.reference_scope == "within":
        rows, failures = run_within_matrix(args, selected_cases, cuda_available, config)
    else:
        rows, failures = run_cross_matrix(args, selected_cases, cuda_available, config)

    headers = ["solver", "mode", "reference", *(backend.label for backend in BACKENDS)]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row.get(header, "") for header in headers) + " |")

    total_cells = sum(1 for row in rows for backend in BACKENDS if row.get(backend.label) not in {"", "N/A", "REF"})
    pass_cells = sum(1 for row in rows for backend in BACKENDS if row.get(backend.label) == "PASS")
    fail_cells = sum(1 for row in rows for backend in BACKENDS if row.get(backend.label) == "FAIL")
    print()
    print(f"Summary: pass={pass_cells}, fail={fail_cells}, compared_cells={total_cells}, cuda_available={cuda_available}")
    if failures:
        print()
        print("Failures:")
        for failure in failures:
            print(f"- {failure}")


if __name__ == "__main__":
    main()
