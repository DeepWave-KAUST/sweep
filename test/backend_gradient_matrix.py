"""Gradient consistency matrix for eager/cpu, eager/gpu, C/cpu, and C/cuda.

Run from the repository root after installing the compiled extension:

    OMP_NUM_THREADS=4 PYTHONPATH=src python test/backend_gradient_matrix.py \
        --backends eager/gpu c-cuda --scale cuda-suite

The eager leg is the point of this file: it is the only backend that does not
share code with the compiled backward, so it is the only reference that can see
a defect present in both C paths.  ``--backends`` exists because the c-cpu
column needs ``SWEEP_JIT_FULL=1`` (the default JIT build ships a CPU stub); drop
it to run the eager-vs-CUDA comparison on a stock build.

The eager backend supports full mode and PyTorch chunk checkpointing. Boundary
saving and recursive checkpointing are compiled-backend modes, so those eager
cells are reported as N/A.

Cell states: PASS, FAIL, ``PASS*`` (passed, but the backend declares one of the
gradients missing — see ``KNOWN_MISSING_GRADIENTS``; the declared list is
printed after the summary and a stale entry fails the run), N/A, REF.

Tolerances are peak-relative, not absolute (``--atol-scale`` /
``--record-atol-scale``): the compared tensors span ~1e-20 to ~1e9 across this
case list, and the old fixed ``atol=1e-5`` was larger than an entire elastic rho
gradient, which made those comparisons vacuous.
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
        source_type=case.source_type,
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


def selected_backends(args: argparse.Namespace) -> tuple[BackendSpec, ...]:
    return tuple(b for b in BACKENDS if b.label in args.backends)


def reference_label_for_mode(mode: str, cuda_available: bool,
                             backends: tuple[BackendSpec, ...] = BACKENDS) -> str:
    """Pick the truth leg for this mode out of the selected backends.

    Eager first, always: it is the only leg that does not share code with the
    compiled backward, so it is the only one that can see a defect present in
    both C paths.  Falling back to a C reference makes the row a C-vs-C check —
    still useful for mode consistency, useless for backend correctness.
    """
    labels = {b.label for b in backends}
    order = ["eager/cpu", "eager/gpu"] if mode in EAGER_MODES else []
    order += ["c-cuda"] if cuda_available else []
    order += ["c-cpu"]
    for label in order:
        if label in labels:
            return label
    return next(iter(labels)) if labels else "c-cpu"


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
        "--backends",
        nargs="*",
        default=[b.label for b in BACKENDS],
        choices=[b.label for b in BACKENDS],
        help=(
            "Which backends to run. The c-cpu column needs the compiled CPU "
            "tree, which the default JIT build does not include — without "
            "SWEEP_JIT_FULL=1 it raises 'CUDAGuardImpl initialized with "
            "non-CUDA DeviceType: cpu'. Pass --backends eager/gpu c-cuda to "
            "run the eager-vs-CUDA comparison on a stock build."
        ),
    )
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
        "--atol-scale", type=float, default=2e-4,
        help=(
            "Absolute tolerance is measured in units of max|reference| per "
            "tensor rather than absolutely: the compared tensors span ~1e-20 to "
            "~1e9 across this case list (a stress receiver moves the record "
            "scale by 1e6 on its own), so one fixed atol either passes "
            "everything or fails everything. 1e-5 sits above float32 "
            "accumulation noise over 120 steps (measured worst 5.5e-5) and "
            "20x below the smallest real defect this matrix currently catches "
            "(das_mu2d body-force source-cell rho, 4.3e-3). 0 keeps the fixed "
            "--atol instead."
        ),
    )
    parser.add_argument(
        "--record-atol-scale", type=float, default=1e-3,
        help=(
            "Same, for the forward record. It needs its own, looser value: the "
            "record is compared elementwise and crosses zero, so samples near a "
            "zero crossing carry peak-scaled noise with no |reference| for rtol "
            "to work against. Measured eager-vs-c record divergence over 120 "
            "steps is <=2e-5 for elastic/DAS but 5e-5..3.9e-4 for the acoustic "
            "and VRZ families; 1e-3 clears all of them with ~2.5x margin and "
            "still catches a percent-level forward difference."
        ),
    )
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
    known_gaps: list[str] = []

    backends = selected_backends(args)

    for case in selected_cases:
        for mode in selected_modes:
            reference_label = reference_label_for_mode(mode, cuda_available, backends)
            results = {}
            row = {"solver": case.name, "mode": mode, "reference": reference_label}

            for backend in backends:
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
            for backend in backends:
                if backend.label not in results or backend.label == reference_label:
                    continue
                ok, _, errors, missing = compare_result_to_reference(
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
                    atol_scale=args.atol_scale,
                    record_atol_scale=args.record_atol_scale,
                )
                known_gaps.extend(missing)
                if ok:
                    row[backend.label] = "PASS*" if missing else "PASS"
                    continue
                row[backend.label] = "FAIL"
                suffix = ""
                if (case.name, mode) in CUDA_MODE_REFERENCE_ONLY and {backend.label, reference_label} == {"c-cpu", "c-cuda"}:
                    suffix = " (mode follows c-cuda rather than c-cpu full)"
                failures.append(f"{case.name}/{mode}/{backend.label}: {errors[0] if errors else 'mismatch'}{suffix}")

            rows.append(row)

    return rows, failures, known_gaps


def run_within_matrix(args: argparse.Namespace, selected_cases: list[Case], cuda_available: bool, config: RunConfig) -> tuple[list[dict[str, str]], list[str]]:
    selected_modes = list(dict.fromkeys(args.modes))
    rows: list[dict[str, str]] = []
    failures: list[str] = []
    known_gaps: list[str] = []

    backends = selected_backends(args)

    for case in selected_cases:
        full_results = {}
        for backend in backends:
            if not backend_supported("full", backend, cuda_available):
                continue
            try:
                print(f"running full reference {case.name} {backend.label}", flush=True)
                full_results[backend.label] = run_backend(case, "full", backend, config)
            except Exception as exc:  # noqa: BLE001 - test matrix should keep going
                failures.append(f"{case.name}/full/{backend.label}: reference run failed: {one_line_error(exc)}")

        for mode in selected_modes:
            row = {"solver": case.name, "mode": mode, "reference": "same backend full"}
            for backend in backends:
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
                    ok, _, errors, missing = compare_result_to_reference(
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
                        atol_scale=args.atol_scale,
                        record_atol_scale=args.record_atol_scale,
                    )
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    errors = [f"run failed: {one_line_error(exc)}"]
                    missing = []
                known_gaps.extend(missing)
                if ok:
                    row[backend.label] = "PASS*" if missing else "PASS"
                else:
                    row[backend.label] = "FAIL"
                    suffix = ""
                    if (case.name, mode) in CUDA_MODE_REFERENCE_ONLY and backend.family == "c":
                        suffix = " (currently aligned to matching c-cuda/c-cpu mode, not full)"
                    failures.append(f"{case.name}/{mode}/{backend.label}: {errors[0] if errors else 'mismatch'}{suffix}")
            rows.append(row)

    return rows, failures, known_gaps


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
        rows, failures, known_gaps = run_within_matrix(args, selected_cases, cuda_available, config)
    else:
        rows, failures, known_gaps = run_cross_matrix(args, selected_cases, cuda_available, config)

    backends = selected_backends(args)
    headers = ["solver", "mode", "reference", *(backend.label for backend in backends)]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row.get(header, "") for header in headers) + " |")

    total_cells = sum(1 for row in rows for backend in backends if row.get(backend.label) not in {"", "N/A", "REF"})
    pass_cells = sum(1 for row in rows for backend in backends if row.get(backend.label) == "PASS")
    fail_cells = sum(1 for row in rows for backend in backends if row.get(backend.label) == "FAIL")
    print()
    star_cells = sum(1 for row in rows for backend in backends if row.get(backend.label) == "PASS*")
    pass_cells += star_cells
    print(f"Summary: pass={pass_cells} (of which {star_cells} carry a declared "
          f"missing gradient, marked PASS*), fail={fail_cells}, "
          f"compared_cells={total_cells}, cuda_available={cuda_available}")
    if known_gaps:
        print()
        print(f"Declared missing gradients ({len(known_gaps)} cells) — these are NOT compared;")
        print("they are holes in the backend, listed so they cannot be mistaken for coverage:")
        for gap in dict.fromkeys(known_gaps):
            print(f"- {gap}")
    if failures:
        print()
        print("Failures:")
        for failure in failures:
            print(f"- {failure}")


if __name__ == "__main__":
    main()
