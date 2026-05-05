#!/usr/bin/env python3
"""Compare CUDA solver-mode gradients against eager-mode gradients.

The suite is intentionally scriptable rather than a tiny pytest case because
the useful coverage here spans solver equations, memory strategies, and source
placements near finite-difference/free-surface boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / "src" / "sweep").exists() and (candidate / "test").exists():
            return candidate
    raise RuntimeError("Could not locate the repository root from this script path.")


REPO_ROOT = find_repo_root()
SRC_DIR = REPO_ROOT / "src"
SOURCE_IMPORT = os.environ.get("SWEEP_TEST_IMPORT_MODE") == "source" or "--source-import" in sys.argv
if "--source-import" in sys.argv:
    sys.argv.remove("--source-import")
if SOURCE_IMPORT and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sweep.equations import (  # noqa: E402
    Acoustic,
    Acoustic3D,
    AcousticLSRTM,
    AcousticLSRTM3D,
    AcousticVRZ,
    AcousticVRZ3D,
    Elastic,
    Elastic3D,
)
from sweep.propagator.options import (  # noqa: E402
    BoundaryOptions,
    CUDAOptions,
    CkptOptions,
    EagerOptions,
    MemoryOptions,
)
from sweep.propagator.torch import PropTorch  # noqa: E402


@dataclass(frozen=True)
class SolverSpec:
    key: str
    equation_cls: type
    ndim: int
    model_names: tuple[str, ...]
    source_type: tuple[str, ...]
    receiver_type: tuple[str, ...]
    pml_type: str
    lsrtm: bool = False
    elastic: bool = False


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    free_surface: bool
    edge_source: bool


SOLVERS = {
    "acoustic2d": SolverSpec("acoustic2d", Acoustic, 2, ("vp",), ("h1",), ("h1",), "cpmlr"),
    "acoustic3d": SolverSpec("acoustic3d", Acoustic3D, 3, ("vp",), ("h1",), ("h1",), "cpmlr"),
    "vrz2d": SolverSpec("vrz2d", AcousticVRZ, 2, ("vp", "z"), ("h1",), ("h1",), "cpmlr"),
    "vrz3d": SolverSpec("vrz3d", AcousticVRZ3D, 3, ("vp", "z"), ("h1",), ("h1",), "cpmlr"),
    "lsrtm2d": SolverSpec("lsrtm2d", AcousticLSRTM, 2, ("vp", "mp"), ("h1",), ("sh1",), "cpmlr", True),
    "lsrtm3d": SolverSpec("lsrtm3d", AcousticLSRTM3D, 3, ("vp", "mp"), ("h1",), ("sh1",), "cpmlr", True),
    "elastic2d": SolverSpec("elastic2d", Elastic, 2, ("vp", "vs", "rho"), ("sxx", "szz"), ("vx", "vz"), "cpmls", False, True),
    "elastic3d": SolverSpec("elastic3d", Elastic3D, 3, ("vp", "vs", "rho"), ("sxx", "syy", "szz"), ("vx", "vy", "vz"), "cpmls", False, True),
}

SCENARIOS = {
    "interior": ScenarioSpec("interior", free_surface=False, edge_source=False),
    "fd_edge": ScenarioSpec("fd_edge", free_surface=False, edge_source=True),
    "free_surface": ScenarioSpec("free_surface", free_surface=True, edge_source=False),
}

DEFAULT_SOLVERS = "acoustic2d,acoustic3d,vrz2d,vrz3d,lsrtm2d"
DEFAULT_MODES = "full,bs_gpu,bs_cpu,bs_cpu_pinned,bs_disk,bs_disk_async,ckpt_chunk,ckpt_recursive"
DEFAULT_SCENARIOS = "interior,fd_edge,free_surface"


def parse_csv(value: str, valid: Iterable[str], *, label: str) -> list[str]:
    valid_set = set(valid)
    if value == "all":
        return list(valid)
    out = [item.strip() for item in value.split(",") if item.strip()]
    bad = [item for item in out if item not in valid_set]
    if bad:
        raise ValueError(f"Unknown {label} entries {bad}; valid entries are {sorted(valid_set)}")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run eager-vs-CUDA gradient comparisons for full, boundary-saving, and checkpoint modes."
    )
    parser.add_argument("--solvers", default=DEFAULT_SOLVERS, help="Comma list or 'all'.")
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS, help="Comma list or 'all'.")
    parser.add_argument("--modes", default=DEFAULT_MODES, help="Comma list of CUDA modes.")
    parser.add_argument("--nz2d", type=int, default=48)
    parser.add_argument("--nx2d", type=int, default=56)
    parser.add_argument("--nz3d", type=int, default=24)
    parser.add_argument("--ny3d", type=int, default=20)
    parser.add_argument("--nx3d", type=int, default=24)
    parser.add_argument("--nt", type=int, default=120)
    parser.add_argument("--dt", type=float, default=0.0015)
    parser.add_argument("--dh", type=float, default=10.0)
    parser.add_argument("--freq", type=float, default=10.0)
    parser.add_argument("--delay", type=float, default=0.06)
    parser.add_argument("--spatial-order", type=int, default=4)
    parser.add_argument("--abcn", type=int, default=30)
    parser.add_argument("--receiver-stride2d", type=int, default=6)
    parser.add_argument("--receiver-stride3d", type=int, default=8)
    parser.add_argument("--ckpt-chunks", type=int, default=24)
    parser.add_argument("--ckpt-count", type=int, default=4)
    parser.add_argument("--transfer-interval", type=int, default=4)
    parser.add_argument("--disk-ring-buffers", type=int, default=2)
    parser.add_argument("--disk-dir", type=Path, default=None)
    parser.add_argument("--rel-l2-threshold", type=float, default=1.5)
    parser.add_argument("--cosine-threshold", type=float, default=0.8)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "test" / "test_outputs" / "solver_gradient_mode_suite")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-fail", action="store_true", help="Always exit 0 after writing summaries.")
    return parser


def require_cuda_bindings(solver_keys: list[str]):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this suite.")

    try:
        import sweep
        import sweep._C as sweep_c
    except Exception as exc:
        raise RuntimeError(
            "Could not import local sweep._C. Rebuild first, for example: "
            "`python setup_cuda.py build_ext --inplace` from the repo root."
        ) from exc

    required = []
    prefixes = {
        "acoustic2d": "acoustic2d",
        "acoustic3d": "acoustic3d",
        "vrz2d": "acoustic_vrz2d",
        "vrz3d": "acoustic_vrz3d",
        "lsrtm2d": "acoustic_lsrtm2d",
        "lsrtm3d": "acoustic_lsrtm3d",
        "elastic2d": "elastic2d",
        "elastic3d": "elastic3d",
    }
    for key in solver_keys:
        prefix = prefixes[key]
        required.extend(
            [
                f"{prefix}_forward",
                f"{prefix}_backward",
                f"{prefix}_backward_bs",
                f"{prefix}_backward_ckpt",
                f"{prefix}_backward_recursive_ckpt",
            ]
        )
    missing = sorted({name for name in required if not hasattr(sweep_c, name)})
    if missing:
        raise RuntimeError(f"The loaded sweep._C is missing CUDA bindings: {missing}")
    return sweep, sweep_c


def shape_for(spec: SolverSpec, args) -> tuple[int, ...]:
    if spec.ndim == 2:
        return (args.nz2d, args.nx2d)
    return (args.nz3d, args.ny3d, args.nx3d)


def spacing_for(spec: SolverSpec, args) -> tuple[float, ...]:
    return tuple([float(args.dh)] * spec.ndim)


def ricker(nt: int, dt: float, freq: float, delay: float) -> np.ndarray:
    t = np.arange(nt, dtype=np.float32) * dt - delay
    x = np.pi * freq * t
    return ((1.0 - 2.0 * x * x) * np.exp(-(x * x))).astype(np.float32)


def add_box(arr: np.ndarray, value: float) -> np.ndarray:
    out = arr.copy()
    nz = out.shape[0]
    if out.ndim == 2:
        nx = out.shape[1]
        out[nz // 3 : max(nz // 3 + 2, (2 * nz) // 3), nx // 4 : max(nx // 4 + 2, (3 * nx) // 4)] += value
    else:
        ny, nx = out.shape[1], out.shape[2]
        out[
            nz // 3 : max(nz // 3 + 2, (2 * nz) // 3),
            ny // 4 : max(ny // 4 + 2, (3 * ny) // 4),
            nx // 4 : max(nx // 4 + 2, (3 * nx) // 4),
        ] += value
    return out


def depth_ramp(shape: tuple[int, ...], top: float, bottom: float) -> np.ndarray:
    nz = shape[0]
    depth = np.linspace(0.0, 1.0, nz, dtype=np.float32)
    ramp = top + (bottom - top) * depth
    if len(shape) == 2:
        return np.broadcast_to(ramp[:, None], shape).astype(np.float32).copy()
    return np.broadcast_to(ramp[:, None, None], shape).astype(np.float32).copy()


def make_models(spec: SolverSpec, shape: tuple[int, ...]):
    vp_init = depth_ramp(shape, 1800.0, 2400.0)
    vp_true = add_box(vp_init, 180.0)

    if spec.lsrtm:
        mp_init = np.zeros(shape, dtype=np.float32)
        mp_true = add_box(mp_init, 0.08)
        return [vp_init, mp_true], [vp_init, mp_init], [False, True]

    if spec.elastic:
        vs_init = (vp_init / 1.73).astype(np.float32)
        vs_true = (vp_true / 1.73).astype(np.float32)
        rho_init = depth_ramp(shape, 1000.0, 1200.0)
        rho_true = add_box(rho_init, 60.0)
        return [vp_true, vs_true, rho_true], [vp_init, vs_init, rho_init], [True, True, True]

    if spec.model_names == ("vp",):
        return [vp_true], [vp_init], [True]

    z_init = depth_ramp(shape, 1.05, 1.45)
    z_true = add_box(z_init, 0.08)
    return [vp_true, z_true], [vp_init, z_init], [True, True]


def make_geometry(spec: SolverSpec, shape: tuple[int, ...], scenario: ScenarioSpec, args):
    radius = args.spatial_order // 2
    source_z = max(0, radius - 1) if scenario.edge_source else max(1, min(shape[0] - 1, shape[0] // 4))
    receiver_z = source_z if scenario.edge_source else (0 if scenario.free_surface else max(1, radius))
    margin = 0 if scenario.edge_source else max(2, radius)

    if spec.ndim == 2:
        nz, nx = shape
        source_x = nx // 2
        sources = np.array([[min(source_x, nx - 1), min(source_z, nz - 1)]], dtype=np.int32)
        rec_x = np.arange(margin, max(margin + 1, nx - margin), args.receiver_stride2d, dtype=np.int32)
        if rec_x.size == 0:
            rec_x = np.array([nx // 2], dtype=np.int32)
        rec_x = np.clip(rec_x, 0, nx - 1)
        rec_z = np.full(rec_x.size, min(receiver_z, nz - 1), dtype=np.int32)
        receivers = np.stack([rec_x, rec_z], axis=-1)[None, ...]
        return sources, receivers

    nz, ny, nx = shape
    source_x = nx // 2
    source_y = max(0, radius - 1) if scenario.edge_source else ny // 2
    sources = np.array(
        [[min(source_x, nx - 1), min(source_y, ny - 1), min(source_z, nz - 1)]],
        dtype=np.int32,
    )
    rec_x = np.arange(margin, max(margin + 1, nx - margin), args.receiver_stride3d, dtype=np.int32)
    rec_y = np.arange(margin, max(margin + 1, ny - margin), args.receiver_stride3d, dtype=np.int32)
    if rec_x.size == 0:
        rec_x = np.array([nx // 2], dtype=np.int32)
    if rec_y.size == 0:
        rec_y = np.array([ny // 2], dtype=np.int32)
    grid_y, grid_x = np.meshgrid(np.clip(rec_y, 0, ny - 1), np.clip(rec_x, 0, nx - 1), indexing="ij")
    rec_z = np.full(grid_x.size, min(receiver_z, nz - 1), dtype=np.int32)
    receivers = np.stack([grid_x.ravel(), grid_y.ravel(), rec_z], axis=-1)[None, ...]
    return sources, receivers


def build_cuda_options(mode: str, args, run_dir: Path, case_key: str) -> CUDAOptions | None:
    if mode == "full":
        return None

    if mode.startswith("bs_"):
        storage = "gpu"
        pinned = False
        async_read = False
        ring_buffers = None
        transfer_interval = None
        disk_dir = None

        if mode == "bs_gpu":
            storage = "gpu"
        elif mode == "bs_cpu":
            storage = "cpu"
            transfer_interval = args.transfer_interval
        elif mode == "bs_cpu_pinned":
            storage = "cpu"
            pinned = True
            transfer_interval = args.transfer_interval
        elif mode == "bs_disk":
            storage = "disk"
            transfer_interval = args.transfer_interval
            ring_buffers = args.disk_ring_buffers
            disk_dir = args.disk_dir or (run_dir / "disk_boundary" / case_key / mode)
        elif mode == "bs_disk_async":
            storage = "disk"
            async_read = True
            transfer_interval = args.transfer_interval
            ring_buffers = max(2, args.disk_ring_buffers)
            disk_dir = args.disk_dir or (run_dir / "disk_boundary" / case_key / mode)
        else:
            raise ValueError(f"Unsupported boundary-saving mode {mode!r}")

        if disk_dir is not None:
            disk_dir.mkdir(parents=True, exist_ok=True)

        return CUDAOptions(
            memory=MemoryOptions(
                strategy="boundary",
                boundary=BoundaryOptions(
                    storage=storage,
                    transfer_interval=transfer_interval,
                    pinned_memory=pinned,
                    disk_dir=str(disk_dir) if disk_dir is not None else None,
                    ring_buffers=ring_buffers,
                    disk_async_read=async_read,
                ),
            )
        )

    if mode == "ckpt_chunk":
        return CUDAOptions(
            memory=MemoryOptions(
                strategy="ckpt",
                ckpt=CkptOptions(mode="chunk", chunks=args.ckpt_chunks),
            )
        )
    if mode == "ckpt_recursive":
        return CUDAOptions(
            memory=MemoryOptions(
                strategy="ckpt",
                ckpt=CkptOptions(mode="recursive", count=args.ckpt_count),
            )
        )
    raise ValueError(f"Unsupported CUDA mode {mode!r}")


def build_solver(spec: SolverSpec, backend: str, mode: str, scenario: ScenarioSpec, shape, device, args, run_dir, case_key):
    equation = spec.equation_cls(spatial_order=args.spatial_order, device=device, backend="torch")
    common = dict(
        shape=shape,
        dev=device,
        dh=spacing_for(spec, args),
        dt=args.dt,
        source_type=list(spec.source_type),
        receiver_type=list(spec.receiver_type),
        abcn=args.abcn,
        pml_type=spec.pml_type,
        free_surface=scenario.free_surface,
        nt=args.nt,
        B=1,
        allow_growth=True,
    )

    if backend == "eager":
        return PropTorch(
            equation,
            backend="eager",
            eager_options=EagerOptions(use_compile=False),
            use_ckpt=False,
            **common,
        )

    cuda_options = build_cuda_options(mode, args, run_dir, case_key)
    if mode == "full":
        return PropTorch(
            equation,
            backend="cuda",
            use_ckpt=False,
            boundary_saving_config={"enabled": False},
            **common,
        )
    return PropTorch(equation, backend="cuda", cuda_options=cuda_options, **common)


def normalize_record(record: torch.Tensor, nshots: int, nreceivers: int, nt: int) -> torch.Tensor:
    if tuple(record.shape) == (nshots, nreceivers, nt):
        return record
    if tuple(record.shape) == (nshots, nt, nreceivers):
        return record.transpose(1, 2)
    if record.ndim == 4:
        if record.shape[-1] == 1:
            squeezed = record[..., 0]
            if tuple(squeezed.shape) == (nshots, nreceivers, nt):
                return squeezed
            if tuple(squeezed.shape) == (nshots, nt, nreceivers):
                return squeezed.transpose(1, 2)
        if tuple(record.shape[:3]) == (nshots, nreceivers, nt):
            return record
        if (record.shape[0], record.shape[1], record.shape[2]) == (nshots, nt, nreceivers):
            return record.transpose(1, 2)
        if tuple(record.shape[1:]) == (nshots, nreceivers, nt):
            return record.permute(1, 2, 3, 0)
        if (record.shape[1], record.shape[2], record.shape[3]) == (nshots, nt, nreceivers):
            return record.permute(1, 3, 2, 0)
    raise ValueError(
        f"Unsupported record shape {tuple(record.shape)}; expected "
        f"{(nshots, nreceivers, nt)}, {(nshots, nt, nreceivers)}, "
        f"{(nshots, nreceivers, nt, 'channels')}, or {(nshots, nt, nreceivers, 'channels')}"
    )


def tensors_from_models(models_np, grad_flags, device):
    out = []
    for arr, needs_grad in zip(models_np, grad_flags):
        tensor = torch.tensor(arr, device=device, dtype=torch.float32)
        tensor.requires_grad_(bool(needs_grad))
        out.append(tensor)
    return out


def guarded_solver_call(solver, wavelet, sources, receivers, *, models):
    sources_in = np.array(sources, dtype=np.int32, copy=True)
    receivers_in = np.array(receivers, dtype=np.int32, copy=True)
    sources_before = sources_in.copy()
    receivers_before = receivers_in.copy()

    record = solver(wavelet, sources_in, receivers_in, models=models)

    if not np.array_equal(sources_in, sources_before):
        raise RuntimeError(
            f"Solver mutated source coordinates: before={sources_before.tolist()} after={sources_in.tolist()}"
        )
    if not np.array_equal(receivers_in, receivers_before):
        raise RuntimeError("Solver mutated receiver coordinates passed into the test case.")
    return record


def run_forward(solver, wavelet, sources, receivers, models_np, device):
    models = tensors_from_models(models_np, [False] * len(models_np), device)
    with torch.no_grad():
        record = guarded_solver_call(solver, wavelet, sources, receivers, models=models)
        record = normalize_record(record, sources.shape[0], receivers.shape[1], wavelet.shape[-1])
        torch.cuda.synchronize(device)
    if not torch.isfinite(record).all():
        raise RuntimeError("Forward record contains NaN or Inf.")
    return record.detach().cpu()


def run_gradient(solver, wavelet, sources, receivers, observed, models_np, grad_flags, model_names, device):
    models = tensors_from_models(models_np, grad_flags, device)
    observed_t = observed.to(device)
    record = guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    record = normalize_record(record, sources.shape[0], receivers.shape[1], wavelet.shape[-1])
    loss = (record - observed_t).pow(2).mean()
    loss.backward()
    torch.cuda.synchronize(device)

    grads = {}
    for name, tensor, needs_grad in zip(model_names, models, grad_flags):
        if not needs_grad:
            continue
        grad = tensor.grad
        if grad is None:
            raise RuntimeError(f"{name} gradient is missing.")
        if not torch.isfinite(grad).all():
            raise RuntimeError(f"{name} gradient contains NaN or Inf.")
        grads[name] = grad.detach().cpu()
    return {
        "loss": float(loss.detach().cpu().item()),
        "record": record.detach().cpu(),
        "grads": grads,
    }


def metric_pair(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.detach().cpu().to(torch.float64).reshape(-1)
    cand = candidate.detach().cpu().to(torch.float64).reshape(-1)
    finite = torch.isfinite(ref) & torch.isfinite(cand)
    if not bool(finite.any()):
        return {
            "rel_l2": math.inf,
            "cosine": math.nan,
            "diff_l2": math.inf,
            "diff_linf": math.inf,
            "ref_l2": 0.0,
            "cand_l2": 0.0,
            "finite_fraction": 0.0,
        }
    ref = ref[finite]
    cand = cand[finite]
    diff = cand - ref
    ref_l2 = float(torch.linalg.vector_norm(ref))
    cand_l2 = float(torch.linalg.vector_norm(cand))
    diff_l2 = float(torch.linalg.vector_norm(diff))
    denom = max(ref_l2, 1e-30)
    cosine = math.nan if ref_l2 <= 0.0 or cand_l2 <= 0.0 else float(torch.dot(ref, cand) / (ref_l2 * cand_l2))
    return {
        "rel_l2": diff_l2 / denom,
        "cosine": cosine,
        "diff_l2": diff_l2,
        "diff_linf": float(diff.abs().max()),
        "ref_l2": ref_l2,
        "cand_l2": cand_l2,
        "finite_fraction": float(finite.sum().item() / finite.numel()),
    }


def percentile_limit(array: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(array)[np.isfinite(array)]
    if finite.size == 0:
        return -1.0, 1.0
    vmax = float(np.percentile(np.abs(finite), 99.5))
    if vmax <= 0:
        vmax = float(np.max(np.abs(finite)))
    if vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def flatten_points(points) -> np.ndarray:
    if points is None:
        return np.empty((0, 0), dtype=np.int32)
    arr = np.asarray(points)
    if arr.size == 0:
        return np.empty((0, 0), dtype=np.int32)
    return arr.reshape(-1, arr.shape[-1]).astype(np.int32, copy=False)


def source_reference_coord(sources: np.ndarray, shape: tuple[int, ...]) -> tuple[int, ...]:
    points = flatten_points(sources)
    if points.size == 0:
        if len(shape) == 2:
            nz, nx = shape
            return (nx // 2, nz // 2)
        nz, ny, nx = shape
        return (nx // 2, ny // 2, nz // 2)
    coord = points[0]
    if len(shape) == 2:
        nz, nx = shape
        return (int(np.clip(coord[0], 0, nx - 1)), int(np.clip(coord[1], 0, nz - 1)))
    nz, ny, nx = shape
    return (
        int(np.clip(coord[0], 0, nx - 1)),
        int(np.clip(coord[1], 0, ny - 1)),
        int(np.clip(coord[2], 0, nz - 1)),
    )


def gradient_slices(arr: np.ndarray, sources=None):
    if arr.ndim == 2:
        return [("z-x", arr, 0, 1, None, None)]
    x0, y0, z0 = source_reference_coord(sources, arr.shape)
    return [
        (f"z={z0}", arr[z0, :, :], 0, 1, 2, z0),
        (f"y={y0}", arr[:, y0, :], 0, 2, 1, y0),
        (f"x={x0}", arr[:, :, x0], 1, 2, 0, x0),
    ]


def overlay_geometry(ax, sources, receivers, x_dim: int, y_dim: int, slice_dim: int | None, slice_index: int | None):
    source_points = flatten_points(sources)
    receiver_points = flatten_points(receivers)

    def select(points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        if slice_dim is None:
            return points
        return points[points[:, slice_dim] == slice_index]

    selected_receivers = select(receiver_points)
    if selected_receivers.size:
        ax.scatter(
            selected_receivers[:, x_dim],
            selected_receivers[:, y_dim],
            s=10,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=0.45,
        )

    selected_sources = select(source_points)
    if selected_sources.size:
        ax.scatter(
            selected_sources[:, x_dim],
            selected_sources[:, y_dim],
            s=80,
            marker="*",
            c="cyan",
            edgecolors="black",
            linewidths=0.55,
        )


def save_gradient_plot(path: Path, title: str, grads: dict[str, torch.Tensor], sources=None, receivers=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(grads)
    if not names:
        return None
    ndim = grads[names[0]].ndim
    ncols = 1 if ndim == 2 else 3
    fig, axes = plt.subplots(len(names), ncols, figsize=(4.2 * ncols, 3.2 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        grad = grads[name].numpy()
        vmin, vmax = percentile_limit(grad)
        for col, (slice_name, image, x_dim, y_dim, slice_dim, slice_index) in enumerate(gradient_slices(grad, sources)):
            ax = axes[row, col]
            im = ax.imshow(image, cmap="seismic", origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
            overlay_geometry(ax, sources, receivers, x_dim, y_dim, slice_dim, slice_index)
            ax.set_title(f"{name} {slice_name}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def row_status(metrics: dict[str, dict[str, float]], args) -> tuple[str, list[str]]:
    failures = []
    for name, item in metrics.items():
        if item["finite_fraction"] < 1.0:
            failures.append(f"{name} finite_fraction={item['finite_fraction']:.6f}")
        if item["ref_l2"] <= 0.0:
            failures.append(f"{name} eager gradient is zero")
        if item["rel_l2"] > args.rel_l2_threshold:
            failures.append(f"{name} rel_l2={item['rel_l2']:.6e}>{args.rel_l2_threshold:.6e}")
        if item["cosine"] < args.cosine_threshold:
            failures.append(f"{name} cosine={item['cosine']:.6f}<{args.cosine_threshold:.6f}")
    return ("pass" if not failures else "fail"), failures


def serializable_metrics(metrics: dict[str, dict[str, float]]) -> str:
    return json.dumps(metrics, sort_keys=True)


def run_case(spec: SolverSpec, scenario: ScenarioSpec, modes: list[str], args, run_dir: Path, device):
    shape = shape_for(spec, args)
    sources, receivers = make_geometry(spec, shape, scenario, args)
    true_models, init_models, grad_flags = make_models(spec, shape)
    wavelet = torch.tensor(ricker(args.nt, args.dt, args.freq, args.delay), device=device)
    case_key = f"{spec.key}_{scenario.key}"
    case_dir = run_dir / spec.key / scenario.key
    case_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "solver": spec.key,
        "scenario": scenario.key,
        "shape": shape,
        "sources": sources.tolist(),
        "receivers_shape": tuple(receivers.shape),
        "free_surface": scenario.free_surface,
        "edge_source": scenario.edge_source,
        "model_names": spec.model_names,
        "grad_model_names": [name for name, flag in zip(spec.model_names, grad_flags) if flag],
    }
    (case_dir / "case_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"\n[{case_key}] shape={shape} sources={sources.tolist()} receivers={receivers.shape}")
    eager_solver = build_solver(spec, "eager", "eager", scenario, shape, device, args, run_dir, case_key)
    observed = run_forward(eager_solver, wavelet, sources, receivers, true_models, device)
    eager_result = run_gradient(
        eager_solver,
        wavelet,
        sources,
        receivers,
        observed,
        init_models,
        grad_flags,
        spec.model_names,
        device,
    )
    if not args.no_plot:
        save_gradient_plot(
            case_dir / f"{case_key}_eager_gradient.png",
            f"{case_key} eager",
            eager_result["grads"],
            sources,
            receivers,
        )
    del eager_solver
    torch.cuda.empty_cache()

    rows = []
    for mode in modes:
        started = time.time()
        row = {
            "solver": spec.key,
            "scenario": scenario.key,
            "mode": mode,
            "status": "error",
            "seconds": "",
            "loss_eager": f"{eager_result['loss']:.9e}",
            "loss_candidate": "",
            "metrics": "",
            "failures": "",
            "gradient_plot": "",
            "error": "",
        }
        try:
            solver = build_solver(spec, "cuda", mode, scenario, shape, device, args, run_dir, case_key)
            candidate = run_gradient(
                solver,
                wavelet,
                sources,
                receivers,
                observed,
                init_models,
                grad_flags,
                spec.model_names,
                device,
            )
            metrics = {
                name: metric_pair(eager_result["grads"][name], candidate["grads"][name])
                for name in eager_result["grads"]
            }
            status, failures = row_status(metrics, args)
            plot_path = ""
            if not args.no_plot:
                saved = save_gradient_plot(
                    case_dir / f"{case_key}_{mode}_gradient.png",
                    f"{case_key} {mode}",
                    candidate["grads"],
                    sources,
                    receivers,
                )
                plot_path = str(saved) if saved is not None else ""
            row.update(
                {
                    "status": status,
                    "loss_candidate": f"{candidate['loss']:.9e}",
                    "metrics": serializable_metrics(metrics),
                    "failures": "; ".join(failures),
                    "gradient_plot": plot_path,
                }
            )
            print(
                f"  {mode}: {status} "
                + ", ".join(
                    f"{name} rel_l2={item['rel_l2']:.3e} cos={item['cosine']:.3f}"
                    for name, item in metrics.items()
                )
            )
            del solver, candidate
        except Exception as exc:
            error_path = case_dir / f"{case_key}_{mode}_error.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            row["error"] = f"{type(exc).__name__}: {exc} ({error_path})"
            print(f"  {mode}: error {type(exc).__name__}: {exc}")
        finally:
            row["seconds"] = f"{time.time() - started:.3f}"
            rows.append(row)
            torch.cuda.empty_cache()
    return rows


def write_summaries(run_dir: Path, rows: list[dict]):
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "summary.csv"
    fieldnames = [
        "solver",
        "scenario",
        "mode",
        "status",
        "seconds",
        "loss_eager",
        "loss_candidate",
        "metrics",
        "failures",
        "gradient_plot",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path = run_dir / "summary.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path


def main():
    args = build_parser().parse_args()
    valid_modes = {
        "full",
        "bs_gpu",
        "bs_cpu",
        "bs_cpu_pinned",
        "bs_disk",
        "bs_disk_async",
        "ckpt_chunk",
        "ckpt_recursive",
    }
    solver_keys = parse_csv(args.solvers, SOLVERS, label="solver")
    scenario_keys = parse_csv(args.scenarios, SCENARIOS, label="scenario")
    modes = parse_csv(args.modes, valid_modes, label="mode")

    sweep, sweep_c = require_cuda_bindings(solver_keys)
    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    run_dir = (args.output_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"python={sys.executable}")
    print(f"sweep={sweep.__file__}")
    print(f"sweep._C={sweep_c.__file__}")
    print(f"run_dir={run_dir}")
    print(f"solvers={solver_keys}")
    print(f"scenarios={scenario_keys}")
    print(f"modes={modes}")

    device = torch.device("cuda")
    torch.manual_seed(0)
    np.random.seed(0)

    all_rows = []
    try:
        for solver_key in solver_keys:
            spec = SOLVERS[solver_key]
            for scenario_key in scenario_keys:
                rows = run_case(spec, SCENARIOS[scenario_key], modes, args, run_dir, device)
                all_rows.extend(rows)
                write_summaries(run_dir, all_rows)
    finally:
        csv_path, json_path = write_summaries(run_dir, all_rows)
        print(f"\nsummary_csv={csv_path}")
        print(f"summary_json={json_path}")

    bad = [row for row in all_rows if row["status"] != "pass"]
    if bad and not args.no_fail:
        print(f"Gradient suite finished with {len(bad)} failing/error mode(s).")
        raise SystemExit(1)
    print("Gradient suite finished.")


if __name__ == "__main__":
    main()
