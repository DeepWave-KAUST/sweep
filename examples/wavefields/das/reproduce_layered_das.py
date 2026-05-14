#!/usr/bin/env python3
"""Reproduce layered-model DAS figures and DAS method-comparison panels.

Available figure modes:
- 4: Figure-4-style vx/vz and exx_t/ezz_t common-shot gathers.
- 7: Figure-7-style vertical-well ezz_t gathers with different gauge lengths.
- 9: Figure-9-style helical-wound DAS pressure and strain-rate gathers.
- both: Figures 4 and 9.
- compare: Zhao DAS records versus Mu velocity-stress-strain DAS records.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sweep.equations import DASModeler


PAPER_CITATION = "Zhao et al., Petroleum Science, 23 (2026), 626-642 (https://doi.org/10.1016/j.petsci.2025.09.015)"
PAPER_TAG4 = "Reproduction of Figure 4 in the above paper."
PAPER_TAG7 = "Reproduction of Figure 7 in the above paper."
PAPER_TAG9 = "Reproduction of Figure 9 in the above paper."
FIGURE7_GAUGE_NOTE = (
    "Figure 7 gauge-length panels are moving averages along the vertical-well receiver line. "
    "The paper-grid mode keeps the paper-style 10/20/40 m labels but applies 11/21/41-cell "
    "windows on the modeled receiver samples; use resampled-meter mode for meter-spaced "
    "interpolation before averaging."
)
FIGURE7_EDGE_NOTE = (
    "The default Figure 7 edge mode is reflect padding. The top and bottom edge bands need "
    "receiver samples outside the simulated line, so those edge responses are padding artifacts; "
    "center traces are unaffected by this padding."
)

DEFAULT_RECORD_PATHS_FIGURE9 = [
    Path("test/test_outputs/das_paper_reproduction/layered_fig3_paper_geometry_cpml/layered_records.npz"),
    Path("test/test_outputs/das_paper_reproduction/layered_fig3_receivers_exact_cpml/layered_records.npz"),
    Path("test/test_outputs/das_paper_reproduction/layered_fig3_cpml/layered_records.npz"),
]
FIGURE7_GAUGE_LENGTHS = [
    ("gl10m", "Gauge length 10 m", "gauge_length_10m", 10.0),
    ("gl20m", "Gauge length 20 m", "gauge_length_20m", 20.0),
    ("gl40m", "Gauge length 40 m", "gauge_length_40m", 40.0),
]


def torch_impl_for_mode(mode: str) -> str:
    return "c" if mode == "cuda" else "eager"


def resolve_backend_modes(backend: str, impl: Optional[str]) -> list[str]:
    backend = str(backend).lower()
    impl = None if impl is None else str(impl).lower()

    if backend in {"eager", "pytorch"}:
        if impl not in {None, "eager"}:
            raise ValueError(f"--backend {backend} implies --impl eager, got --impl {impl}.")
        return ["eager"]
    if backend in {"cuda", "c"}:
        if impl not in {None, "c"}:
            raise ValueError(f"--backend {backend} implies --impl c, got --impl {impl}.")
        return ["cuda"]
    if backend == "cpu":
        raise ValueError("The DAS reproduction script does not currently expose the C++ CPU extension path.")
    if backend == "both":
        if impl not in {None, "both"}:
            raise ValueError(f"--backend both implies --impl both, got --impl {impl}.")
        return ["eager", "cuda"]
    if backend == "jax":
        raise ValueError("This DAS reproduction script is Torch-based. Use a JAX example for --backend jax.")
    if backend != "torch":
        raise ValueError("Unsupported backend. Use --backend torch with --impl eager, c, or both.")

    if impl in {None, "eager"}:
        return ["eager"]
    if impl == "c":
        return ["cuda"]
    if impl == "both":
        return ["eager", "cuda"]
    raise ValueError("Unsupported --impl. Expected eager, c, or both.")


def ricker(nt: int, dt: float, fm: float, delay: float) -> np.ndarray:
    t = np.arange(nt, dtype=np.float32) * np.float32(dt) - np.float32(delay)
    arg = np.pi * np.float32(fm) * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-arg**2)).astype(np.float32)


def layered_model(nz: int = 201, nx: int = 401, dh: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_m = np.arange(nz, dtype=np.float32)[:, None] * np.float32(dh)

    vp = np.empty((nz, nx), dtype=np.float32)
    vs = np.empty((nz, nx), dtype=np.float32)

    layers = [
        (z_m < 750.0, 1500.0, 1500.0/1.73),
        ((z_m >= 750.0) & (z_m < 1500.0), 2500.0, 2500.0/1.73),
        (z_m >= 1500.0, 3000.0, 3000.0/1.73),
    ]
    for mask, vp_value, vs_value in layers:
        vp[mask[:, 0], :] = vp_value
        vs[mask[:, 0], :] = vs_value

    rho = np.full_like(vp, 2100.0, dtype=np.float32)
    return vp, vs, rho


def km_to_index(value_km: float, dh: float, upper: int) -> int:
    return int(np.clip(round(float(value_km) * 1000.0 / float(dh)), 0, upper))


def build_layered_geometry(
    nz: int,
    nx: int,
    dh: float,
    source_x_km: float = 2.0,
    source_depth_km: float = 0.0,
    surface_depth_km: float = 0.0,
    horizontal_depth_km: float = 1.2,
    horizontal_x_min_km: float = 0.5,
    horizontal_x_max_km: float = 2.5,
    vertical_x_km: float = 3.0,
) -> Dict[str, np.ndarray]:
    source_x = km_to_index(source_x_km, dh, nx - 1)
    source_depth = km_to_index(source_depth_km, dh, nz - 1)
    source = np.array([[source_x, source_depth]], dtype=np.int32)

    surface_z = km_to_index(surface_depth_km, dh, nz - 1)
    surface = np.stack(
        [np.arange(nx, dtype=np.int32), np.full(nx, surface_z, dtype=np.int32)],
        axis=-1,
    )

    horizontal_z = km_to_index(horizontal_depth_km, dh, nz - 1)
    horizontal_x0 = km_to_index(horizontal_x_min_km, dh, nx - 1)
    horizontal_x1 = km_to_index(horizontal_x_max_km, dh, nx - 1)
    if horizontal_x0 > horizontal_x1:
        horizontal_x0, horizontal_x1 = horizontal_x1, horizontal_x0
    horizontal_x = np.linspace(horizontal_x0, horizontal_x1, 201, dtype=np.int32)
    horizontal = np.stack(
        [horizontal_x, np.full(horizontal_x.size, horizontal_z, dtype=np.int32)], axis=-1
    )

    vertical_x = km_to_index(vertical_x_km, dh, nx - 1)
    vertical_z = np.linspace(0, nz - 1, 201, dtype=np.int32)
    vertical = np.stack([np.full(vertical_z.size, vertical_x, dtype=np.int32), vertical_z], axis=-1)

    receivers = np.concatenate([surface, horizontal, vertical], axis=0)
    return {
        "source": source,
        "receivers": receivers,
        "slices": {
            "surface": slice(0, surface.shape[0]),
            "horizontal": slice(surface.shape[0], surface.shape[0] + horizontal.shape[0]),
            "vertical": slice(surface.shape[0] + horizontal.shape[0], receivers.shape[0]),
        },
    }


def build_dense_vertical_geometry(
    nz: int,
    nx: int,
    dh: float,
    source_x_km: float = 2.0,
    source_depth_km: float = 0.0,
    vertical_x_km: float = 3.0,
) -> Dict[str, np.ndarray]:
    source_x = km_to_index(source_x_km, dh, nx - 1)
    source_depth = km_to_index(source_depth_km, dh, nz - 1)
    source = np.array([[source_x, source_depth]], dtype=np.int32)

    vertical_x = km_to_index(vertical_x_km, dh, nx - 1)
    vertical_z = np.arange(nz, dtype=np.int32)
    vertical = np.stack([np.full(nz, vertical_x, dtype=np.int32), vertical_z], axis=-1)
    return {
        "source": source,
        "receivers": vertical,
        "slices": {
            "surface": slice(0, 0),
            "horizontal": slice(0, 0),
            "vertical": slice(0, vertical.shape[0]),
        },
    }


def clip_limits(record: np.ndarray, percentile: tuple[float, float] = (2.0, 98.0)) -> tuple[float, float]:
    finite = np.asarray(record)[np.isfinite(record)]
    if finite.size == 0:
        return -1.0, 1.0
    vmin, vmax = np.percentile(finite, percentile)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        scale = max(float(np.max(np.abs(finite))), 1.0)
        return -scale, scale
    if vmin < 0.0 < vmax:
        scale = max(abs(float(vmin)), abs(float(vmax)))
        return -scale, scale
    return float(vmin), float(vmax)


def _tensor_record_to_numpy(record: torch.Tensor) -> np.ndarray:
    if record.ndim != 4 or record.shape[0] != 1:
        raise ValueError(f"Expected record layout (1, nt, nrec, nfield), got {tuple(record.shape)}")
    return record.detach().cpu().numpy()[0].transpose(1, 0, 2)


def run_solver(
    *,
    backend: str,
    method: str = "zhao",
    geometry: Dict[str, np.ndarray],
    models: tuple[np.ndarray, np.ndarray, np.ndarray],
    wavelet: np.ndarray,
    receiver_type: list[str],
    nt: int,
    args,
) -> tuple[np.ndarray, Dict[str, int], float]:
    if args.device == "auto":
        chosen_device = "cuda:0" if backend == "cuda" or torch.cuda.is_available() else "cpu"
    else:
        chosen_device = args.device

    device = torch.device(chosen_device)
    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but CUDA is not available on this system.")
    if backend == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA backend requires a cuda device, e.g. --device cuda:0")

    vp_np, vs_np, rho_np = models
    models_t = tuple(torch.as_tensor(model, dtype=torch.float32, device=device) for model in (vp_np, vs_np, rho_np))

    solver = DASModeler(
        method=method,
        ndim=2,
        shape=(args.nz, args.nx),
        source_type=["sxx", "szz"],
        receiver_type=receiver_type,
        abcn=args.abcn,
        dh=args.dh,
        dt=args.dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
        backend="torch",
        impl="eager" if method == "mu" else torch_impl_for_mode(backend),
    )

    wavelet_t = torch.as_tensor(wavelet, dtype=torch.float32, device=device)
    source = np.asarray(geometry["source"], dtype=np.int32)
    receivers = np.asarray(geometry["receivers"], dtype=np.int32)
    if source.ndim == 2:
        source = source[None, ...]

    start = time.perf_counter()
    with torch.no_grad():
        records_t = solver(wavelet_t, sources=source, receivers=receivers[None, ...], models=list(models_t))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - start

    records = _tensor_record_to_numpy(records_t)
    channels = dict(solver.channels)
    return records, channels, elapsed_s


def run_figure4_data(
    *,
    backend: str,
    geometry: Dict[str, np.ndarray],
    models: tuple[np.ndarray, np.ndarray, np.ndarray],
    wavelet: np.ndarray,
    args,
) -> tuple[np.ndarray, Dict[str, int], float]:
    return run_solver(
        backend=backend,
        method="mu",
        geometry=geometry,
        models=models,
        wavelet=wavelet,
        receiver_type=["vx", "vz", "exx_t", "ezz_t"],
        nt=wavelet.shape[-1],
        args=args,
    )


def parse_figure9_records(npz_path: Path) -> tuple[np.ndarray, np.ndarray, Dict[str, int], Optional[float]]:
    data = np.load(npz_path, allow_pickle=True)

    required = {"records", "receiver_type", "receivers"}
    missing = [name for name in required if name not in data.files]
    if missing:
        raise RuntimeError(
            f"Input npz {npz_path} missing required fields: {', '.join(missing)}"
        )

    records = np.asarray(data["records"])
    if records.ndim == 4 and records.shape[0] == 1:
        records = records[0]
    if records.ndim != 3:
        raise RuntimeError(f"Could not normalize records with shape {records.shape}")

    receiver_type = np.asarray(data["receiver_type"], dtype=str).tolist()
    channels = {name: i for i, name in enumerate(receiver_type)}
    for required_field in ["sxx", "szz", "das35_t", "das54z_t"]:
        if required_field not in channels:
            raise RuntimeError(
                f"Required field '{required_field}' not found in {npz_path}: {receiver_type}"
            )

    receivers = np.asarray(data["receivers"])
    if receivers.ndim == 3 and receivers.shape[0] == 1:
        receivers = receivers[0]

    data_duration: Optional[float] = None
    meta_path = npz_path.with_name("metadata.json")
    if meta_path.exists():
        try:
            run_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            run_params = run_metadata.get("run_parameters", {})
            if isinstance(run_params, dict):
                duration = run_params.get("duration_s") or run_params.get("duration")
                dt = run_params.get("dt_s") or run_params.get("dt")
                nt = run_params.get("nt")
                if duration is None and dt is not None and nt is not None:
                    duration = dt * nt
                if duration is not None:
                    data_duration = float(duration)
        except Exception:
            data_duration = None

    return records, receivers, channels, data_duration


def infer_figure9_slices(receivers: np.ndarray, records: np.ndarray) -> Dict[str, slice]:
    nrec = records.shape[0]
    if nrec == 803:
        return {
            "surface": slice(0, 401),
            "horizontal": slice(401, 602),
            "vertical": slice(602, 803),
        }

    z = receivers[:, 1]
    runs = []
    start = 0
    for i in range(1, len(z) + 1):
        if i == len(z) or z[i] != z[start]:
            runs.append((start, i, z[start]))
            start = i

    if len(runs) < 2:
        raise RuntimeError("Receiver geometry does not contain enough runs for figure-9 layout")

    surface_end = runs[0][1]
    horizontal_end = surface_end
    for run_start, run_end, _ in runs[1:]:
        run_len = run_end - run_start
        if run_len > 1 and run_len < (len(z) // 2) and run_len > (horizontal_end - surface_end):
            horizontal_end = run_end

    if horizontal_end == surface_end:
        if len(runs) >= 3:
            horizontal_end = runs[1][1]
        else:
            horizontal_end = min(surface_end + max(1, len(z) // 3), len(z))

    return {
        "surface": slice(0, surface_end),
        "horizontal": slice(surface_end, horizontal_end),
        "vertical": slice(horizontal_end, nrec),
    }


def plot_figure4(
    records: np.ndarray,
    channels: Dict[str, int],
    geometry: Dict[str, np.ndarray],
    duration: float,
    out_path: Path,
) -> None:
    rows = [
        ("surface", "Surface receivers"),
        ("horizontal", "Horizontal-well receivers"),
        ("vertical", "Vertical-well receivers"),
    ]
    cols = [
        ("vx", "Particle velocity vx"),
        ("vz", "Particle velocity vz"),
        ("exx_t", "Strain-rate exx_t"),
        ("ezz_t", "Strain-rate ezz_t"),
    ]

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(16.0, 12.6), constrained_layout=True)
    for row, (geom_name, geom_title) in enumerate(rows):
        sl = geometry["slices"][geom_name]
        for col, (field, title) in enumerate(cols):
            data = records[sl, :, channels[field]]
            ax = axes[row, col]
            vmin, vmax = clip_limits(data)
            ax.imshow(
                data.T,
                cmap="gray",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                extent=[0, data.shape[0], duration, 0],
            )
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Time (s)")
            if row == 0:
                ax.set_title(f"({chr(ord('a') + col)}) {title}")
            if col == 0:
                ax.text(
                    0.02,
                    0.95,
                    geom_title,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5},
                )

    fig.suptitle("Figure 4-style common-shot gathers (vx/vz, exx_t/ezz_t)", fontsize=15)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def _figure7_display_indices(ntrace: int, display_trace_count: int) -> np.ndarray:
    if display_trace_count <= 0 or ntrace <= display_trace_count:
        return np.arange(ntrace, dtype=np.int32)
    return np.unique(np.linspace(0, ntrace - 1, display_trace_count).round().astype(np.int32))


def _figure7_gauge_windows(gauge_grid_spacing: float) -> list[tuple[str, str, str, int, float]]:
    if gauge_grid_spacing <= 0.0:
        raise ValueError(f"Gauge grid spacing must be positive, got {gauge_grid_spacing}.")
    return [
        (key, title, npz_key, max(1, int(round(length_m / gauge_grid_spacing)) + 1), length_m)
        for key, title, npz_key, length_m in FIGURE7_GAUGE_LENGTHS
    ]


def _resample_records_along_receivers(
    records: np.ndarray,
    receiver_spacing: float,
    target_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    if target_spacing <= 0.0:
        raise ValueError(f"Target receiver spacing must be positive, got {target_spacing}.")

    original_positions = np.arange(records.shape[0], dtype=np.float32) * np.float32(receiver_spacing)
    if np.isclose(target_spacing, receiver_spacing):
        return records, original_positions

    max_position = float(original_positions[-1])
    target_positions = np.arange(0.0, max_position + 0.5 * target_spacing, target_spacing, dtype=np.float32)
    target_positions = target_positions[target_positions <= max_position + 1e-5]
    scaled = target_positions / np.float32(receiver_spacing)
    left = np.floor(scaled).astype(np.int64)
    left = np.clip(left, 0, records.shape[0] - 1)
    right = np.clip(left + 1, 0, records.shape[0] - 1)
    weight = (scaled - left).astype(records.dtype)
    resampled = (1.0 - weight)[:, None] * records[left] + weight[:, None] * records[right]
    return resampled.astype(records.dtype, copy=False), target_positions


def _valid_gauge_average(record: np.ndarray, gauge_cells: int) -> tuple[np.ndarray, np.ndarray]:
    if gauge_cells <= 1:
        centers = np.arange(record.shape[0], dtype=np.int32)
        return record, centers
    if gauge_cells > record.shape[0]:
        raise ValueError(f"Gauge window {gauge_cells} is wider than receiver count {record.shape[0]}.")

    left = gauge_cells // 2
    right = gauge_cells - 1 - left
    windows = np.lib.stride_tricks.sliding_window_view(record, gauge_cells, axis=0)
    averaged = windows.mean(axis=-1)
    centers = np.arange(left, record.shape[0] - right, dtype=np.int32)
    return averaged, centers


def _partial_gauge_average(record: np.ndarray, gauge_cells: int) -> tuple[np.ndarray, np.ndarray]:
    if gauge_cells <= 1:
        centers = np.arange(record.shape[0], dtype=np.int32)
        return record, centers

    left = gauge_cells // 2
    right = gauge_cells - 1 - left
    centers = np.arange(record.shape[0], dtype=np.int32)
    starts = np.maximum(centers - left, 0)
    stops = np.minimum(centers + right + 1, record.shape[0])

    prefix = np.concatenate(
        [np.zeros((1, record.shape[1]), dtype=record.dtype), np.cumsum(record, axis=0)],
        axis=0,
    )
    summed = prefix[stops] - prefix[starts]
    counts = (stops - starts).astype(record.dtype)[:, None]
    return summed / counts, centers


def _reflect_gauge_average(record: np.ndarray, gauge_cells: int) -> tuple[np.ndarray, np.ndarray]:
    if gauge_cells <= 1:
        centers = np.arange(record.shape[0], dtype=np.int32)
        return record, centers
    if record.shape[0] < 2:
        raise ValueError("Reflect gauge edge mode requires at least two receiver traces.")

    left = gauge_cells // 2
    right = gauge_cells - 1 - left
    padded = np.pad(record, [(left, right), (0, 0)], mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, gauge_cells, axis=0)
    averaged = windows.mean(axis=-1)
    centers = np.arange(record.shape[0], dtype=np.int32)
    return averaged, centers


def _symmetric_gauge_average(record: np.ndarray, gauge_cells: int) -> tuple[np.ndarray, np.ndarray]:
    if gauge_cells <= 1:
        centers = np.arange(record.shape[0], dtype=np.int32)
        return record, centers
    if record.shape[0] < 2:
        raise ValueError("Symmetric gauge edge mode requires at least two receiver traces.")

    left = gauge_cells // 2
    right = gauge_cells - 1 - left
    padded = np.pad(record, [(left, right), (0, 0)], mode="symmetric")
    windows = np.lib.stride_tricks.sliding_window_view(padded, gauge_cells, axis=0)
    averaged = windows.mean(axis=-1)
    centers = np.arange(record.shape[0], dtype=np.int32)
    return averaged, centers


def _anti_reflect_gauge_average(record: np.ndarray, gauge_cells: int) -> tuple[np.ndarray, np.ndarray]:
    if gauge_cells <= 1:
        centers = np.arange(record.shape[0], dtype=np.int32)
        return record, centers
    if record.shape[0] < 2:
        raise ValueError("Anti-reflect gauge edge mode requires at least two receiver traces.")

    left = gauge_cells // 2
    right = gauge_cells - 1 - left
    if left >= record.shape[0] or right >= record.shape[0]:
        raise ValueError(
            f"Anti-reflect padding for gauge window {gauge_cells} needs more than "
            f"{max(left, right)} receiver traces; got {record.shape[0]}."
        )

    left_idx = np.arange(left, 0, -1, dtype=np.int64)
    right_idx = record.shape[0] - 1 - np.arange(1, right + 1, dtype=np.int64)
    left_pad = 2.0 * record[0:1] - record[left_idx]
    right_pad = 2.0 * record[-1:] - record[right_idx]
    padded = np.concatenate([left_pad, record, right_pad], axis=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, gauge_cells, axis=0)
    averaged = windows.mean(axis=-1)
    centers = np.arange(record.shape[0], dtype=np.int32)
    return averaged.astype(record.dtype, copy=False), centers


def _figure7_panel_records(
    ezz_records: np.ndarray,
    display_trace_count: int,
    receiver_spacing: float,
    gauge_grid_spacing: float,
    gauge_mode: str,
    edge_mode: str,
) -> tuple[list[tuple[str, str, np.ndarray]], Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict[str, int]]:
    if gauge_mode == "paper-grid":
        working_records = ezz_records
        receiver_positions_m = np.arange(ezz_records.shape[0], dtype=np.float32) * np.float32(receiver_spacing)
    elif gauge_mode == "resampled-meter":
        working_records, receiver_positions_m = _resample_records_along_receivers(
            ezz_records,
            receiver_spacing=receiver_spacing,
            target_spacing=gauge_grid_spacing,
        )
    else:
        raise ValueError(f"Unknown Figure 7 gauge mode {gauge_mode!r}.")

    gauge_windows = _figure7_gauge_windows(gauge_grid_spacing)
    max_left = max(gauge_cells // 2 for _key, _title, _npz_key, gauge_cells, _length_m in gauge_windows)
    max_right = max(gauge_cells - 1 - gauge_cells // 2 for _key, _title, _npz_key, gauge_cells, _length_m in gauge_windows)
    if edge_mode == "valid" and working_records.shape[0] <= max_left + max_right:
        raise ValueError(
            f"Need more than {max_left + max_right} receiver traces for Figure 7 gauge windows; "
            f"got {working_records.shape[0]}."
        )
    if edge_mode == "valid":
        common_centers = np.arange(max_left, working_records.shape[0] - max_right, dtype=np.int32)
        average_func = _valid_gauge_average
    elif edge_mode == "partial":
        common_centers = np.arange(working_records.shape[0], dtype=np.int32)
        average_func = _partial_gauge_average
    elif edge_mode == "reflect":
        common_centers = np.arange(working_records.shape[0], dtype=np.int32)
        average_func = _reflect_gauge_average
    elif edge_mode == "symmetric":
        common_centers = np.arange(working_records.shape[0], dtype=np.int32)
        average_func = _symmetric_gauge_average
    elif edge_mode == "anti-reflect":
        common_centers = np.arange(working_records.shape[0], dtype=np.int32)
        average_func = _anti_reflect_gauge_average
    else:
        raise ValueError(f"Unknown Figure 7 edge mode {edge_mode!r}.")
    display_local = _figure7_display_indices(common_centers.size, display_trace_count)
    display_centers = common_centers[display_local]
    display_positions_m = receiver_positions_m[display_centers]

    panels = [("origin", "Origin seismogram", working_records[display_centers])]
    records = {"origin": working_records[display_centers]}
    for key, title, npz_key, gauge_cells, _length_m in gauge_windows:
        averaged, centers = average_func(working_records, gauge_cells)
        center_to_local = display_centers - int(centers[0])
        data = averaged[center_to_local]
        panels.append((key, title, data))
        records[npz_key] = data
    gauge_cell_map = {key: gauge_cells for key, _title, _npz_key, gauge_cells, _length_m in gauge_windows}
    return panels, records, display_centers, display_positions_m, gauge_cell_map


def _figure7_time_slice(nt: int, duration: float, time_min: Optional[float], time_max: Optional[float]) -> tuple[slice, float, float]:
    dt = float(duration) / float(nt)
    start_t = 0.0 if time_min is None else max(0.0, float(time_min))
    stop_t = float(duration) if time_max is None else min(float(duration), float(time_max))
    if stop_t <= start_t:
        raise ValueError(f"Invalid Figure 7 time window [{start_t}, {stop_t}].")
    start_i = int(np.clip(round(start_t / dt), 0, nt - 1))
    stop_i = int(np.clip(round(stop_t / dt), start_i + 1, nt))
    return slice(start_i, stop_i), start_i * dt, stop_i * dt


def plot_figure7(
    ezz_records: np.ndarray,
    duration: float,
    dh: float,
    out_path: Path,
    display_trace_count: int = 201,
    gauge_grid_spacing: Optional[float] = None,
    gauge_mode: str = "paper-grid",
    edge_mode: str = "reflect",
    time_min: Optional[float] = None,
    time_max: Optional[float] = None,
) -> tuple[Dict[str, int], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    default_spacing = 1.0 if gauge_mode == "paper-grid" else float(dh)
    effective_gauge_spacing = float(default_spacing if gauge_grid_spacing is None else gauge_grid_spacing)
    panels, panel_records, display_indices, display_positions_m, gauge_windows = _figure7_panel_records(
        ezz_records,
        display_trace_count,
        receiver_spacing=float(dh),
        gauge_grid_spacing=effective_gauge_spacing,
        gauge_mode=gauge_mode,
        edge_mode=edge_mode,
    )
    time_sl, plot_t0, plot_t1 = _figure7_time_slice(
        panel_records["origin"].shape[1],
        duration,
        time_min,
        time_max,
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 11.8), constrained_layout=True)
    axes_flat = axes.ravel()
    for index, (_key, title, data) in enumerate(panels):
        plot_data = data[:, time_sl]
        vmin, vmax = clip_limits(plot_data)
        ax = axes_flat[index]
        ax.imshow(
            plot_data.T,
            cmap="gray",
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            extent=[0, plot_data.shape[0], plot_t1, plot_t0],
        )
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Trace")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"({chr(ord('a') + index)}) {title}")
    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)

    fig.suptitle("Fig. 7. Vertical-well z-component strain-rate with different gauge lengths", fontsize=14)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)
    return gauge_windows, panel_records, display_indices, display_positions_m


def run_figure7_data(
    *,
    backend: str,
    geometry: Dict[str, np.ndarray],
    models: tuple[np.ndarray, np.ndarray, np.ndarray],
    wavelet: np.ndarray,
    args,
) -> tuple[np.ndarray, Dict[str, int], float]:
    records, channels, elapsed = run_solver(
        backend=backend,
        method="mu",
        geometry=geometry,
        models=models,
        wavelet=wavelet,
        receiver_type=["ezz_t"],
        nt=wavelet.shape[-1],
        args=args,
    )
    vertical = geometry["slices"]["vertical"]
    return records[vertical, :, :], channels, elapsed


def plot_figure9(
    records: np.ndarray,
    channels: Dict[str, int],
    slices: Dict[str, slice],
    duration: float,
    out_path: Path,
) -> None:
    rows = [
        ("surface", "Surface receivers"),
        ("horizontal", "Horizontal-well receivers"),
        ("vertical", "Vertical-well receivers"),
    ]
    panels = [
        ("pressure", "Pressure seismogram"),
        ("das35_t", "Axial strain-rate at 35.3°"),
        ("das54z_t", "Axial strain-rate at 54.7°"),
    ]

    fig, axes = plt.subplots(len(rows), len(panels), figsize=(14.4, 12.6), constrained_layout=True)
    for row_index, (row_name, row_title) in enumerate(rows):
        sl = slices[row_name]
        for col_index, (field, title) in enumerate(panels):
            if field == "pressure":
                data = records[sl, :, channels["sxx"]] + records[sl, :, channels["szz"]]
            else:
                data = records[sl, :, channels[field]]

            ax = axes[row_index, col_index]
            vmin, vmax = clip_limits(data)
            ax.imshow(
                data.T,
                cmap="gray",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                extent=[0, data.shape[0], duration, 0],
            )
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Time (s)")
            if row_index == 0:
                ax.set_title(f"({chr(ord('a') + col_index)}) {title}")
            if col_index == 0:
                ax.text(
                    0.02,
                    0.95,
                    row_title,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
                )

    fig.suptitle("Fig. 9. Common-shot gathers of helical-wound optical fiber", fontsize=14)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


def _slice_fields(records: np.ndarray, channels: Dict[str, int], fields: list[str]) -> np.ndarray:
    return np.stack([records[:, :, channels[field]] for field in fields], axis=-1)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=np.float64) ** 2)))


def summarize_method_comparison(
    records_by_method: Dict[str, np.ndarray],
    fields: list[str],
    reference: str,
) -> Dict[str, object]:
    ref = records_by_method[reference]
    summary: Dict[str, object] = {
        "reference": reference,
        "fields": fields,
        "methods": {},
        "pairs": {},
    }
    for method, records in records_by_method.items():
        summary["methods"][method] = {
            "record_shape": list(records.shape),
            "rms": _rms(records),
            "max_abs": float(np.max(np.abs(records))),
        }
        if method == reference:
            continue
        diff = records - ref
        per_field = {}
        for index, field in enumerate(fields):
            field_ref = ref[:, :, index]
            field_diff = diff[:, :, index]
            per_field[field] = {
                "max_abs": float(np.max(np.abs(field_diff))),
                "rms_abs": _rms(field_diff),
                "relative_rms": _rms(field_diff) / (_rms(field_ref) + 1.0e-30),
            }
        summary["pairs"][f"{method}_minus_{reference}"] = {
            "max_abs": float(np.max(np.abs(diff))),
            "rms_abs": _rms(diff),
            "relative_rms": _rms(diff) / (_rms(ref) + 1.0e-30),
            "fields": per_field,
        }
    return summary


def plot_method_records(
    records_by_method: Dict[str, np.ndarray],
    fields: list[str],
    duration: float,
    out_path: Path,
) -> None:
    methods = list(records_by_method)
    fig, axes = plt.subplots(len(methods), len(fields), figsize=(3.8 * len(fields), 3.0 * len(methods)), constrained_layout=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]

    for row, method in enumerate(methods):
        records = records_by_method[method]
        for col, field in enumerate(fields):
            data = records[:, :, col]
            ax = axes[row, col]
            vmin, vmax = clip_limits(data)
            ax.imshow(
                data.T,
                cmap="gray",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                extent=[0, data.shape[0], duration, 0],
            )
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Time (s)")
            if row == 0:
                ax.set_title(field)
            if col == 0:
                ax.text(
                    0.02,
                    0.95,
                    method,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
                )

    fig.suptitle("DAS records from Zhao and Mu velocity-stress-strain equations", fontsize=14)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_method_differences(
    records_by_method: Dict[str, np.ndarray],
    fields: list[str],
    reference: str,
    duration: float,
    out_path: Path,
) -> None:
    methods = [method for method in records_by_method if method != reference]
    fig, axes = plt.subplots(len(methods), len(fields), figsize=(3.8 * len(fields), 3.0 * len(methods)), constrained_layout=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]

    ref = records_by_method[reference]
    for row, method in enumerate(methods):
        diff = records_by_method[method] - ref
        for col, field in enumerate(fields):
            data = diff[:, :, col]
            ax = axes[row, col]
            vmin, vmax = clip_limits(ref[:, :, col])
            ax.imshow(
                data.T,
                cmap="seismic",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin,
                vmax=vmax,
                extent=[0, data.shape[0], duration, 0],
            )
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.set_xlabel("Trace")
            ax.set_ylabel("Time (s)")
            if row == 0:
                ax.set_title(field)
            if col == 0:
                ax.text(
                    0.02,
                    0.95,
                    f"{method} - {reference}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
                )

    fig.suptitle(f"DAS method errors (color scale: {reference} records)", fontsize=14)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_das_method_comparison(
    *,
    geometry: Dict[str, np.ndarray],
    models: tuple[np.ndarray, np.ndarray, np.ndarray],
    wavelet: np.ndarray,
    args,
    output_dir: Path,
) -> Dict[str, object]:
    fields = ["exx_t", "ezz_t", "das35_t", "das54x_t", "das54z_t"]
    method_specs = [
        ("zhao_cuda", "zhao", "cuda"),
        ("zhao_eager", "zhao", "eager"),
        ("mu_eager_gpu", "mu", "eager"),
    ]

    records_by_method: Dict[str, np.ndarray] = {}
    results: Dict[str, object] = {}
    for method_name, method, backend in method_specs:
        records, channels, elapsed = run_solver(
            backend=backend,
            method=method,
            geometry=geometry,
            models=models,
            wavelet=wavelet,
            receiver_type=fields,
            nt=wavelet.shape[-1],
            args=args,
        )
        records = _slice_fields(records, channels, fields)

        records_by_method[method_name] = records
        out_npz = output_dir / f"{method_name}_records.npz"
        np.savez_compressed(
            out_npz,
            records=records,
            receiver_type=np.array(fields, dtype="U"),
            source=geometry["source"][None, ...],
            receivers=geometry["receivers"][None, ...],
            duration=args.duration,
            dt=args.dt,
            peak_frequency=args.peak_frequency,
        )
        results[method_name] = {
            "backend": backend,
            "method": method,
            "elapsed_s": elapsed,
            "record_shape": list(records.shape),
            "receiver_fields": fields,
            "records": str(out_npz),
        }

    records_png = output_dir / "das_method_records.png"
    diff_png = output_dir / "das_method_differences.png"
    reference = "zhao_cuda"
    plot_method_records(
        {
            "Zhao": records_by_method[reference],
            "Mu": records_by_method["mu_eager_gpu"],
        },
        fields,
        args.duration,
        records_png,
    )
    plot_method_differences(records_by_method, fields, reference, args.duration, diff_png)
    summary = summarize_method_comparison(records_by_method, fields, reference)
    metadata = {
        "paper": PAPER_CITATION,
        "purpose": "Compare Zhao DAS CUDA/eager records and Mu velocity-stress-strain DAS records.",
        "geometry": {
            "nz": args.nz,
            "nx": args.nx,
            "dh": args.dh,
            "duration": args.duration,
            "dt": args.dt,
            "spatial_order": args.spatial_order,
            "abcn": args.abcn,
        },
        "runs": results,
        "comparison": summary,
        "figures": {
            "records": str(records_png),
            "differences": str(diff_png),
        },
    }
    (output_dir / "das_method_comparison_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def resolve_records_path(explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit if explicit.exists() else None

    for path in DEFAULT_RECORD_PATHS_FIGURE9:
        if path.exists():
            return path
    return None


def resolve_backends(args: argparse.Namespace) -> Iterable[str]:
    return resolve_backend_modes(args.backend, args.impl)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce layered DAS Figure 4, Figure 7, Figure 9, or method-comparison panels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  PYTHONPATH=src python examples/wavefields/das/reproduce_layered_das.py --figure both --backend torch --impl eager --device cuda\n"
            "  PYTHONPATH=src python examples/wavefields/das/reproduce_layered_das.py --figure 7 --backend torch --impl eager --device cuda "
            "--figure7-edge-mode reflect --figure7-time-min 0.68 --figure7-time-max 2.15\n"
            "  PYTHONPATH=src python examples/wavefields/das/reproduce_layered_das.py --figure compare --backend torch --impl eager --device cuda\n"
        ),
    )
    parser.add_argument(
        "--figure",
        choices=("4", "7", "9", "both", "compare"),
        default="both",
        help="Figure set to generate. 'both' runs Figure 4 and Figure 9.",
    )
    parser.add_argument(
        "--backend",
        metavar="{torch,jax}",
        default="torch",
        help="Array/programming backend. Legacy aliases eager, cuda, and both are still accepted.",
    )
    parser.add_argument(
        "--impl",
        metavar="{eager,c,both}",
        default="both",
        help=(
            "Torch implementation to run. The default preserves the previous behavior and runs both "
            "eager and c modes."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for generated PNG, NPZ, and JSON artifacts.",
    )
    parser.add_argument("--records-path", type=Path, default=None, help="Figure 9: path to existing layered_records.npz")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nz", type=int, default=201)
    parser.add_argument("--nx", type=int, default=401)
    parser.add_argument("--dh", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--peak-frequency", type=float, default=10.0)
    parser.add_argument("--delay", type=float, default=0.08)
    parser.add_argument("--spatial-order", type=int, default=8)
    parser.add_argument("--abcn", type=int, default=50)
    parser.add_argument("--figure7-trace-count", type=int, default=201)
    parser.add_argument(
        "--figure7-gauge-mode",
        choices=("paper-grid", "resampled-meter"),
        default="paper-grid",
        help="Figure 7 gauge interpretation: paper-grid uses paper-style cell windows; resampled-meter interpolates receivers before averaging.",
    )
    parser.add_argument(
        "--figure7-edge-mode",
        choices=("reflect", "symmetric", "anti-reflect", "partial", "valid"),
        default="reflect",
        help="Figure 7 edge handling for gauge windows. reflect keeps all traces with mirror padding.",
    )
    parser.add_argument("--figure7-gauge-grid-spacing", type=float, default=None)
    parser.add_argument("--figure7-time-min", type=float, default=None, help="Figure 7 display start time in seconds.")
    parser.add_argument("--figure7-time-max", type=float, default=None, help="Figure 7 display end time in seconds.")
    parser.add_argument("--check-backward", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = build_layered_geometry(args.nz, args.nx, args.dh)
    figure7_geometry = build_dense_vertical_geometry(args.nz, args.nx, args.dh)
    nt = int(round(args.duration / args.dt))
    wavelet = ricker(nt, args.dt, args.peak_frequency, args.delay).reshape(1, 1, nt)
    vp_np, vs_np, rho_np = layered_model(args.nz, args.nx, args.dh)

    models = (vp_np, vs_np, rho_np)
    run_figure4 = args.figure in {"4", "both"}
    run_figure7 = args.figure == "7"
    run_figure9 = args.figure in {"9", "both"}
    run_compare = args.figure == "compare"
    multi_mode = args.figure == "both"

    if run_compare:
        metadata = run_das_method_comparison(
            geometry=geometry,
            models=models,
            wavelet=wavelet,
            args=args,
            output_dir=output_dir,
        )
        print(json.dumps(metadata["runs"], indent=2))
        return

    if run_figure4:
        results = {}
        for backend in resolve_backends(args):
            records, channels, elapsed = run_figure4_data(
                backend=backend,
                geometry=geometry,
                models=models,
                wavelet=wavelet,
                args=args,
            )
            out_png = output_dir / f"figure4_{backend}.png"
            plot_figure4(records, channels, geometry, args.duration, out_png)

            out_npz = output_dir / f"figure4_records_{backend}.npz"
            np.savez_compressed(
                out_npz,
                records=records,
                channels=np.array(list(channels.keys()), dtype="U"),
                source=geometry["source"],
                receivers=geometry["receivers"],
                duration=args.duration,
            )

            results[backend] = {
                "backend": backend,
                "elapsed_s": elapsed,
                "record_shape": list(records.shape),
                "receiver_fields": list(channels.keys()),
                "figure": str(out_png),
                "records": str(out_npz),
            }

        metadata = {
            "paper": PAPER_CITATION,
            "paper_tag": PAPER_TAG4,
            "geometry": {
                "nz": args.nz,
                "nx": args.nx,
                "dh": args.dh,
                "duration": args.duration,
                "dt": args.dt,
                "spatial_order": args.spatial_order,
                "abcn": args.abcn,
            },
            "runs": results,
        }
        metadata_path = output_dir / ("figure4_metadata.json" if multi_mode else "metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))

    if run_figure7:
        results = {}
        for backend in resolve_backends(args):
            records, channels, elapsed = run_figure7_data(
                backend=backend,
                geometry=figure7_geometry,
                models=models,
                wavelet=wavelet,
                args=args,
            )
            if "ezz_t" not in channels:
                raise RuntimeError(f"Figure 7 requires ezz_t records, got {list(channels)}")
            ezz_t = records[:, :, channels["ezz_t"]]
            out_png = output_dir / f"figure7_{backend}.png"
            gauge_cells, figure7_records, display_indices, display_positions_m = plot_figure7(
                ezz_t,
                args.duration,
                args.dh,
                out_png,
                args.figure7_trace_count,
                args.figure7_gauge_grid_spacing,
                args.figure7_gauge_mode,
                args.figure7_edge_mode,
                args.figure7_time_min,
                args.figure7_time_max,
            )
            vertical_x = int(figure7_geometry["receivers"][figure7_geometry["slices"]["vertical"]][0, 0])
            receiver_z_indices = np.rint(display_positions_m / float(args.dh)).astype(np.int32)
            displayed_receivers = np.stack(
                [np.full(receiver_z_indices.size, vertical_x, dtype=np.int32), receiver_z_indices],
                axis=-1,
            )
            effective_gauge_grid_spacing = float(
                (1.0 if args.figure7_gauge_mode == "paper-grid" else args.dh)
                if args.figure7_gauge_grid_spacing is None
                else args.figure7_gauge_grid_spacing
            )
            out_npz = output_dir / f"figure7_records_{backend}.npz"
            np.savez_compressed(
                out_npz,
                origin=figure7_records["origin"],
                **{key: value for key, value in figure7_records.items() if key != "origin"},
                receiver_type=np.array(["ezz_t"], dtype="U"),
                source=figure7_geometry["source"][None, ...],
                receivers=displayed_receivers[None, ...],
                receiver_positions_m=display_positions_m,
                dense_receiver_count=ezz_t.shape[0],
                display_indices=display_indices,
                duration=args.duration,
                dt=args.dt,
                gauge_cells=json.dumps(gauge_cells),
                edge_mode=args.figure7_edge_mode,
                gauge_note=FIGURE7_GAUGE_NOTE,
                edge_note=FIGURE7_EDGE_NOTE,
            )

            results[backend] = {
                "backend": backend,
                "elapsed_s": elapsed,
                "dense_record_shape": list(ezz_t.shape),
                "display_record_shape": list(figure7_records["origin"].shape),
                "receiver_field": "ezz_t",
                "gauge_cells": gauge_cells,
                "gauge_mode": args.figure7_gauge_mode,
                "gauge_grid_spacing": effective_gauge_grid_spacing,
                "gauge_average_mode": args.figure7_edge_mode,
                "time_window": [args.figure7_time_min, args.figure7_time_max],
                "figure": str(out_png),
                "records": str(out_npz),
            }

        metadata = {
            "paper": PAPER_CITATION,
            "paper_tag": PAPER_TAG7,
            "geometry": {
                "nz": args.nz,
                "nx": args.nx,
                "dh": args.dh,
                "duration": args.duration,
                "dt": args.dt,
                "spatial_order": args.spatial_order,
                "abcn": args.abcn,
                "figure7_trace_count": args.figure7_trace_count,
                "figure7_gauge_mode": args.figure7_gauge_mode,
                "figure7_edge_mode": args.figure7_edge_mode,
                "figure7_gauge_grid_spacing": args.figure7_gauge_grid_spacing,
                "figure7_time_window": [args.figure7_time_min, args.figure7_time_max],
                "figure7_gauge_average_mode": args.figure7_edge_mode,
            },
            "notes": {
                "gauge_length": FIGURE7_GAUGE_NOTE,
                "edge_padding": FIGURE7_EDGE_NOTE,
            },
            "runs": results,
        }
        (output_dir / "figure7_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        return

    if not run_figure9:
        return

    # Figure 9
    records_path = resolve_records_path(args.records_path)
    if records_path is not None:
        records, receivers, channels, replay_duration = parse_figure9_records(records_path)
        slices = infer_figure9_slices(receivers, records)
        duration = args.duration if replay_duration is None else replay_duration
        out_png = output_dir / "figure9.png"
        plot_figure9(records, channels, slices, duration, out_png)
        metadata = {
            "paper": PAPER_CITATION,
            "paper_tag": PAPER_TAG9,
            "records_path": str(records_path),
            "receiver_fields": list(channels.keys()),
            "duration": duration,
            "dt": args.dt,
            "geometry": {"nz": args.nz, "nx": args.nx, "dh": args.dh},
        }
        metadata_path = output_dir / ("figure9_metadata.json" if multi_mode else "metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata, indent=2))
        return

    results = {}
    for backend in resolve_backends(args):
        records, channels, elapsed = run_solver(
            backend=backend,
            method="mu",
            geometry=geometry,
            models=models,
            wavelet=wavelet,
            receiver_type=["sxx", "szz", "das35_t", "das54x_t", "das54z_t"],
            nt=wavelet.shape[-1],
            args=args,
        )
        slices = infer_figure9_slices(geometry["receivers"], records)
        out_png = output_dir / f"figure9_{backend}.png"
        plot_figure9(records, channels, slices, args.duration, out_png)

        out_npz = output_dir / f"figure9_records_{backend}.npz"
        np.savez_compressed(
            out_npz,
            records=records,
            receiver_type=np.array(list(channels.keys()), dtype="U"),
            source=geometry["source"][None, ...],
            receivers=geometry["receivers"][None, ...],
            duration=args.duration,
            dt=args.dt,
            peak_frequency=args.peak_frequency,
        )

        results[backend] = {
            "backend": backend,
            "elapsed_s": elapsed,
            "record_shape": list(records.shape),
            "receiver_fields": list(channels.keys()),
            "figure": str(out_png),
            "records": str(out_npz),
        }

    metadata = {
        "paper": PAPER_CITATION,
        "paper_tag": PAPER_TAG9,
        "runs": results,
        "geometry": {
            "nz": args.nz,
            "nx": args.nx,
            "dh": args.dh,
            "duration": args.duration,
            "dt": args.dt,
            "spatial_order": args.spatial_order,
            "abcn": args.abcn,
        },
    }
    metadata_path = output_dir / ("figure9_metadata.json" if multi_mode else "metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
