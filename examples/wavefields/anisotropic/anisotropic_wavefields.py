import argparse

import numpy as np
import torch

from common import IMPORT_MODE, OUTPUT_DIR, plot_record_grid, plot_snapshot_grid
from sweep.equations import Acoustic, AcousticTariq, AcousticTTI, AcousticVTI
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


PHYSICAL_SIZE = (960.0, 960.0)
DH = (5.0, 15.0)
DT = 0.001
NT = 1100
ABCN = 20
DOM_FREQ = 15.0
SPATIAL_ORDER = 8
SNAPSHOT_TIMES = [220, 320, 460]
RECEIVER_Z_M = 90.0
RECEIVER_X_MIN_M = 90.0
RECEIVER_X_MAX_M = 870.0
RECEIVER_COUNT = 33
SOURCE_COORD_M = (PHYSICAL_SIZE[1] * 0.5, PHYSICAL_SIZE[0] * 0.5)


def shape_from_extent(physical_size, dh):
    lz, lx = physical_size
    dz, dx = dh
    return (int(round(lz / dz)) + 1, int(round(lx / dx)) + 1)


def physical_to_grid(x_m, z_m, dh):
    dz, dx = dh
    return [int(round(x_m / dx)), int(round(z_m / dz))]


def build_geometry(dh):
    rec_x_m = np.linspace(RECEIVER_X_MIN_M, RECEIVER_X_MAX_M, RECEIVER_COUNT, dtype=np.float32)
    receivers = np.array(
        [physical_to_grid(x_m, RECEIVER_Z_M, dh) for x_m in rec_x_m],
        dtype=np.int64,
    )[None, ...]
    sources = np.array([physical_to_grid(*SOURCE_COORD_M, dh)], dtype=np.int64)
    return sources, receivers


def make_wavelet():
    t = np.arange(NT, dtype=np.float32) * DT - 0.12
    return (1e6 * ricker(t, f=DOM_FREQ)).astype(np.float32)


def to_record_panel(record):
    data = record.detach().cpu().numpy()
    if data.ndim == 4:
        return data[0, :, :, 0]
    if data.ndim == 3:
        return data[0]
    raise ValueError(f"Unexpected record shape {data.shape}")


def crop_panel(panel, shape):
    nz, nx = shape
    return panel[ABCN : ABCN + nz, ABCN : ABCN + nx]


def boundary_ratio(panel, border_cells):
    border_cells = max(2, min(border_cells, panel.shape[0] // 4, panel.shape[1] // 4))
    edge_mask = np.zeros_like(panel, dtype=bool)
    edge_mask[:border_cells, :] = True
    edge_mask[-border_cells:, :] = True
    edge_mask[:, :border_cells] = True
    edge_mask[:, -border_cells:] = True
    edge_vals = np.abs(panel[edge_mask])
    inner_vals = np.abs(panel[~edge_mask])
    edge_amp = float(np.max(edge_vals)) if edge_vals.size else 0.0
    inner_amp = float(np.max(inner_vals)) if inner_vals.size else 1e-6
    return edge_amp / max(inner_amp, 1e-6)


def array_health_stats(data):
    arr = np.asarray(data)
    return {
        "nan": int(np.isnan(arr).sum()),
        "posinf": int(np.isposinf(arr).sum()),
        "neginf": int(np.isneginf(arr).sum()),
        "finite_max": float(np.nanmax(np.where(np.isfinite(arr), np.abs(arr), np.nan))) if np.isfinite(arr).any() else float("nan"),
    }


def build_models(case_key, shape, dev):
    nz, nx = shape
    parameter_sets = {
        "A": {"delta": 0.3, "epsilon": 0.3, "vp": 2500.0},
        "B": {"delta": 0.1, "epsilon": 0.3, "vp": 2500.0},
        "C": {"delta": 0.3, "epsilon": 0.1, "vp": 2500.0},
    }
    if case_key == "Acoustic":
        return [
            torch.full((nz, nx), 2500.0, dtype=torch.float32, device=dev),
        ]
    if case_key == "Tariq":
        return [
            torch.full((nz, nx), 2300.0, dtype=torch.float32, device=dev),
            torch.full((nz, nx), 2100.0, dtype=torch.float32, device=dev),
            torch.full((nz, nx), 0.15, dtype=torch.float32, device=dev),
        ]
    if case_key.startswith("VTI_"):
        key = case_key.split("_", 1)[1]
        params = parameter_sets[key]
        return [
            torch.full((nz, nx), params["vp"], dtype=torch.float32, device=dev),
            torch.full((nz, nx), params["epsilon"], dtype=torch.float32, device=dev),
            torch.full((nz, nx), params["delta"], dtype=torch.float32, device=dev),
        ]
    if case_key.startswith("TTI_"):
        key = case_key.split("_", 1)[1]
        params = parameter_sets[key]
        return [
            torch.full((nz, nx), params["vp"], dtype=torch.float32, device=dev),
            torch.full((nz, nx), params["epsilon"], dtype=torch.float32, device=dev),
            torch.full((nz, nx), params["delta"], dtype=torch.float32, device=dev),
            torch.full((nz, nx), 20.0, dtype=torch.float32, device=dev),
        ]
    raise ValueError(f"Unknown equation name {case_key}")


def case_fields(case_key):
    if case_key == "Tariq":
        return ["h1"], ["f1"]
    return ["h1"], ["h1"]


def build_solver(equation, shape, dev, case_key):
    source_type, receiver_type = case_fields(case_key)
    return PropTorch(
        equation,
        shape=shape,
        dev=dev,
        dh=DH,
        dt=DT,
        nt=NT,
        abcn=ABCN,
        free_surface=False,
        source_type=source_type,
        receiver_type=receiver_type,
        pml_type="cpmlr",
        backend="eager",
        eager_options=EagerOptions(use_compile=False),
        use_ckpt=False,
    )


def run_case(case_key, equation_cls, dev):
    shape = shape_from_extent(PHYSICAL_SIZE, DH)
    solver = build_solver(
        equation_cls(spatial_order=SPATIAL_ORDER, device=dev, backend="torch"),
        shape,
        dev,
        case_key,
    )
    sources, receivers = build_geometry(DH)
    record, snapshots = solver(
        make_wavelet(),
        sources,
        receivers,
        models=build_models(case_key, shape, dev),
        return_wavefield=True,
        snapshot_times=SNAPSHOT_TIMES,
    )

    panels = [crop_panel(snapshots[i, 0, 0, 0].detach().cpu().numpy(), shape) for i in range(len(SNAPSHOT_TIMES))]
    ratios = [boundary_ratio(panel, ABCN) for panel in panels]
    record_panel = to_record_panel(record)
    return panels, record_panel, ratios, shape, array_health_stats(record_panel)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize anisotropic qP wavefields and inspect CPML boundary behavior.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run on.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dev = torch.device(args.device)
    cases = [
        ("Acoustic (vp=2500)", "Acoustic", Acoustic),
        ("Tariq", "Tariq", AcousticTariq),
        ("VTI-A (delta=0.3, epsilon=0.3, vp=2500)", "VTI_A", AcousticVTI),
        ("VTI-B (delta=0.1, epsilon=0.3, vp=2500)", "VTI_B", AcousticVTI),
        ("VTI-C (delta=0.3, epsilon=0.1, vp=2500)", "VTI_C", AcousticVTI),
        ("TTI-A (delta=0.3, epsilon=0.3, vp=2500)", "TTI_A", AcousticTTI),
        ("TTI-B (delta=0.1, epsilon=0.3, vp=2500)", "TTI_B", AcousticTTI),
        ("TTI-C (delta=0.3, epsilon=0.1, vp=2500)", "TTI_C", AcousticTTI),
    ]
    groups = [
        ("acoustic_tariq", "Acoustic and Tariq Wavefields", "Acoustic and Tariq Seismograms", {"Acoustic", "Tariq"}),
        ("vti", "VTI Wavefields", "VTI Seismograms", {"VTI_A", "VTI_B", "VTI_C"}),
        ("tti", "TTI Wavefields", "TTI Seismograms", {"TTI_A", "TTI_B", "TTI_C"}),
    ]

    results = []
    shape = None

    for title, case_key, equation_cls in cases:
        print(f"Running {title} ({IMPORT_MODE}) on {dev}...")
        panels, record, ratios, shape, health = run_case(case_key, equation_cls, dev)
        results.append((title, case_key, panels, record, ratios, health))

    for group_key, snapshot_title, record_title, members in groups:
        group_results = [item for item in results if item[1] in members]
        plot_snapshot_grid(
            [item[2] for item in group_results],
            [item[0] for item in group_results],
            SNAPSHOT_TIMES,
            OUTPUT_DIR / f"{group_key}_snapshots.png",
            f"{snapshot_title} | dh={DH}",
            shape,
            DH,
        )
        plot_record_grid(
            [item[3] for item in group_results],
            [item[0] for item in group_results],
            OUTPUT_DIR / f"{group_key}_records.png",
            record_title,
        )

    metrics_path = OUTPUT_DIR / "boundary_metrics.txt"
    with metrics_path.open("w", encoding="ascii") as f:
        f.write(f"import_mode={IMPORT_MODE}\n")
        f.write(f"device={dev}\n")
        f.write(f"dh={DH}\n")
        f.write(f"snapshot_times={SNAPSHOT_TIMES}\n")
        for group_key, _, _, members in groups:
            f.write(f"[{group_key}]\n")
            for title, case_key, _, _, ratios, health in results:
                if case_key not in members:
                    continue
                ratio_str = ", ".join(f"t={t}: edge_ratio={r:.4f}" for t, r in zip(SNAPSHOT_TIMES, ratios))
                f.write(
                    f"{title}: {ratio_str}; "
                    f"nan={health['nan']}, +inf={health['posinf']}, -inf={health['neginf']}, "
                    f"finite_max={health['finite_max']:.6e}\n"
                )

    print(f"Saved anisotropic figures to {OUTPUT_DIR}")
    print(f"Saved boundary metrics to {metrics_path}")


if __name__ == "__main__":
    main()
