#!/usr/bin/env python3
"""Reproduce layered-model DAS figures (Figure 4 and Figure 9) in one script."""

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

from sweep.equations import Elastic, DASElastic
from sweep.propagator.torch import PropTorch


PAPER_CITATION = "Zhao et al., Petroleum Science, 23 (2026), 626-642 (https://doi.org/10.1016/j.petsci.2025.09.015)"
PAPER_TAG4 = "Reproduction of Figure 4 in the above paper."
PAPER_TAG9 = "Reproduction of Figure 9 in the above paper."

DEFAULT_RECORD_PATHS_FIGURE9 = [
    Path("test/test_outputs/das_paper_reproduction/layered_fig3_paper_geometry_cpml/layered_records.npz"),
    Path("test/test_outputs/das_paper_reproduction/layered_fig3_receivers_exact_cpml/layered_records.npz"),
    Path("test/test_outputs/das_paper_reproduction/layered_fig3_cpml/layered_records.npz"),
]


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


def normalize_record(record: torch.Tensor, nreceiver: int, nt: int) -> np.ndarray:
    arr = record.detach().cpu().numpy()

    if arr.ndim == 4:
        for axis in range(arr.ndim - 1, -1, -1):
            if arr.ndim <= 3:
                break
            if arr.shape[axis] == 1:
                arr = np.squeeze(arr, axis=axis)

    if arr.ndim != 3:
        raise ValueError(f"Unexpected record tensor layout {arr.shape}")

    candidates = [
        (0, 1, 2),
        (1, 0, 2),
        (0, 2, 1),
        (2, 0, 1),
        (2, 1, 0),
        (1, 2, 0),
    ]
    for perm in candidates:
        if arr.shape[perm[0]] == nreceiver and arr.shape[perm[1]] == nt:
            return arr.transpose(perm)

    raise ValueError(f"Could not map record shape {arr.shape} to (nreceiver, nt, nfield)")


def run_solver(
    *,
    backend: str,
    equation_cls,
    geometry: Dict[str, np.ndarray],
    models: tuple[np.ndarray, np.ndarray, np.ndarray],
    wavelet: np.ndarray,
    receiver_type: list[str],
    nt: int,
    args,
) -> tuple[np.ndarray, Dict[str, int], float]:
    if args.device == "auto":
        chosen_device = "cuda:0"# if backend == "cuda" else "cpu"
    else:
        chosen_device = args.device

    device = torch.device(chosen_device)
    if backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA backend requested but CUDA is not available on this system.")
    if backend == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA backend requires a cuda device, e.g. --device cuda:0")

    vp_np, vs_np, rho_np = models
    models_t = tuple(torch.as_tensor(model, dtype=torch.float32, device=device) for model in (vp_np, vs_np, rho_np))

    equation = equation_cls(spatial_order=args.spatial_order, device=device, backend="torch")
    solver = PropTorch(
        equation,
        shape=(args.nz, args.nx),
        source_type=["sxx", "szz"],
        receiver_type=receiver_type,
        abcn=args.abcn,
        dh=args.dh,
        dt=args.dt,
        dev=device,
        pml_type="cpmls",
        use_ckpt=False,
        backend=backend,
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

    records = normalize_record(records_t, nreceiver=receivers.shape[0], nt=nt)
    channels = {name: i for i, name in enumerate(solver.receiver_type)}
    return records, channels, elapsed_s


def run_figure4_data(
    *,
    backend: str,
    geometry: Dict[str, np.ndarray],
    models: tuple[np.ndarray, np.ndarray, np.ndarray],
    wavelet: np.ndarray,
    args,
) -> tuple[np.ndarray, Dict[str, int], float]:
    nt = wavelet.shape[-1]
    nrec = geometry["receivers"].shape[0]

    elastic_records, _, elastic_elapsed = run_solver(
        backend=backend,
        equation_cls=Elastic,
        geometry=geometry,
        models=models,
        wavelet=wavelet,
        receiver_type=["vx", "vz"],
        nt=nt,
        args=args,
    )

    das_records, _, das_elapsed = run_solver(
        backend=backend,
        equation_cls=DASElastic,
        geometry=geometry,
        models=models,
        wavelet=wavelet,
        receiver_type=["exx", "ezz"],
        nt=nt,
        args=args,
    )

    if elastic_records.shape[:2] != das_records.shape[:2]:
        raise RuntimeError(
            f"Elastic and DAS record shapes do not align: elastic={elastic_records.shape}, das={das_records.shape}"
        )

    records = np.concatenate([elastic_records[:, :, :2], das_records[:, :, :2]], axis=2)
    channels = {"vx": 0, "vz": 1, "exx": 2, "ezz": 3}
    return records, channels, max(elastic_elapsed, das_elapsed)


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
    for required_field in ["sxx", "szz", "das35", "das54z"]:
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
        ("exx", "Strain-rate exx"),
        ("ezz", "Strain-rate ezz"),
    ]

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(16.0, 8.5), constrained_layout=True)
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

    fig.suptitle("Figure 4-style common-shot gathers (vx/vz, exx/ezz)", fontsize=15)
    fig.savefig(out_path, dpi=260)
    plt.close(fig)


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
        ("das35", "Axial strain-rate at 35.3°"),
        ("das54z", "Axial strain-rate at 54.7°"),
    ]

    fig, axes = plt.subplots(len(rows), len(panels), figsize=(14.4, 9.2), constrained_layout=True)
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


def resolve_records_path(explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit if explicit.exists() else None

    for path in DEFAULT_RECORD_PATHS_FIGURE9:
        if path.exists():
            return path
    return None


def resolve_backends(choice: str) -> Iterable[str]:
    if choice in {"eager", "cuda"}:
        return [choice]
    return ["eager", "cuda"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce layered DAS Figure 4/9 in one script.")
    parser.add_argument("--figure", choices=("4", "9", "both"), default="both")
    parser.add_argument("--backend", choices=("eager", "cuda", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry = build_layered_geometry(args.nz, args.nx, args.dh)
    nt = int(round(args.duration / args.dt))
    wavelet = ricker(nt, args.dt, args.peak_frequency, args.delay).reshape(1, 1, nt)
    vp_np, vs_np, rho_np = layered_model(args.nz, args.nx, args.dh)

    models = (vp_np, vs_np, rho_np)
    run_figure4 = args.figure in {"4", "both"}
    run_figure9 = args.figure in {"9", "both"}
    multi_mode = args.figure == "both"

    if run_figure4:
        results = {}
        for backend in resolve_backends(args.backend):
            records, channels, elapsed = run_figure4_data(
                backend=backend,
                geometry=geometry,
                models=models,
                wavelet=wavelet,
                args=args,
            )
            out_png = output_dir / f"figure4_{backend}.png"
            plot_figure4(records, channels, geometry, args.duration, out_png)

            np.savez_compressed(
                output_dir / f"records_{backend}.npz",
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
            }

        metadata = {
            "paper": PAPER_CITATION,
            "paper_tag": PAPER_TAG4,
            "geometry": {"nz": args.nz, "nx": args.nx, "dh": args.dh},
            "runs": results,
        }
        metadata_path = output_dir / ("figure4_metadata.json" if multi_mode else "metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))

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
    for backend in resolve_backends(args.backend):
        records, channels, elapsed = run_solver(
            backend=backend,
            equation_cls=DASElastic,
            geometry=geometry,
            models=models,
            wavelet=wavelet,
            receiver_type=["sxx", "szz", "das35", "das54x", "das54z"],
            nt=nt,
            args=args,
        )
        slices = infer_figure9_slices(geometry["receivers"], records)
        out_png = output_dir / f"figure9_{backend}.png"
        plot_figure9(records, channels, slices, args.duration, out_png)

        np.savez_compressed(
            output_dir / f"records_{backend}.npz",
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
