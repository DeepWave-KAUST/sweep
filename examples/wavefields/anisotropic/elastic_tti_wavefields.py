import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import IMPORT_MODE, OUTPUT_DIR, percentile_clip
from sweep.equations import ElasticTTI, ElasticTTISG
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


FULL_SHAPE = (600, 600)
QUICK_SHAPE = (180, 180)
DH = 5.0
DT = 0.00025
FULL_NT = 1800
QUICK_NT = 520
FULL_ABCN = 60
QUICK_ABCN = 30
DOM_FREQ = 30.0
SOURCE_DELAY = 0.04
SOURCE_RADIUS = 2
SOURCE_SIGMA = 1.2
SPATIAL_ORDER = 8
FULL_SNAPSHOT_TIMES = (500, 1000, 1600)
QUICK_SNAPSHOT_TIMES = (160, 320, 480)

ROTATION_CASES = (
    ("VTI 0/0", 0.0, 0.0),
    ("TTI 35/0", 35.0, 0.0),
    ("TTI 35/45", 35.0, 45.0),
)

FREE_SURFACE_CASES = (
    ("RSG absorbing", ElasticTTI, False),
    ("RSG free-surface", ElasticTTI, True),
    ("SG absorbing", ElasticTTISG, False),
    ("SG free-surface", ElasticTTISG, True),
)


def make_wavelet(nt, source_weights):
    t = np.arange(nt, dtype=np.float32) * DT - SOURCE_DELAY
    wavelet = (1e3 * ricker(t, f=DOM_FREQ)).astype(np.float32)
    weights = np.asarray(source_weights, dtype=np.float32)
    return (weights[None, :, None] * wavelet[None, None, :]).astype(np.float32)


def build_models(shape, dev, theta_deg, phi_deg):
    nz, nx = shape
    return [
        torch.full((nz, nx), 2400.0, dtype=torch.float32, device=dev),
        torch.full((nz, nx), 1200.0, dtype=torch.float32, device=dev),
        torch.full((nz, nx), 2200.0, dtype=torch.float32, device=dev),
        torch.full((nz, nx), 0.35, dtype=torch.float32, device=dev),
        torch.full((nz, nx), 0.05, dtype=torch.float32, device=dev),
        torch.full((nz, nx), 0.20, dtype=torch.float32, device=dev),
        torch.full((nz, nx), np.deg2rad(theta_deg), dtype=torch.float32, device=dev),
        torch.full((nz, nx), np.deg2rad(phi_deg), dtype=torch.float32, device=dev),
    ]


def build_solver(shape, nt, abcn, dev, equation_cls, free_surface):
    is_sg = equation_cls is ElasticTTISG
    return PropTorch(
        equation_cls(spatial_order=SPATIAL_ORDER, device=dev, backend="torch"),
        shape=shape,
        dev=dev,
        dh=DH,
        dt=DT,
        nt=nt,
        abcn=abcn,
        free_surface=free_surface,
        source_type=["sxx", "szz"],
        receiver_type=["vx", "vy", "vz"],
        pml_type="cpmls" if is_sg else "cpmlr",
        backend="torch",
        impl="eager",
        eager_options=EagerOptions(use_compile=False),
        use_ckpt=False,
    )


def build_source(shape, source_z):
    nz, nx = shape
    cx = nx // 2
    points = []
    weights = []
    for dz in range(-SOURCE_RADIUS, SOURCE_RADIUS + 1):
        for dx in range(-SOURCE_RADIUS, SOURCE_RADIUS + 1):
            x = cx + dx
            z = source_z + dz
            if 0 <= x < nx and 0 <= z < nz:
                points.append([x, z])
                weights.append(np.exp(-0.5 * (dx * dx + dz * dz) / (SOURCE_SIGMA * SOURCE_SIGMA)))
    weights = np.asarray(weights, dtype=np.float32)
    weights /= weights.sum()
    return np.asarray(points, dtype=np.int64)[None, ...], weights


def crop_panel(panel, shape, abcn, free_surface):
    nz, nx = shape
    if free_surface:
        return panel[:nz, abcn : abcn + nx]
    return panel[abcn : abcn + nz, abcn : abcn + nx]


def run_forward(
    shape,
    nt,
    abcn,
    snapshot_times,
    dev,
    *,
    equation_cls,
    free_surface,
    source_z,
    theta_deg,
    phi_deg,
):
    solver = build_solver(shape, nt, abcn, dev, equation_cls, free_surface)
    source, weights = build_source(shape, source_z)
    receivers = np.array([[[shape[1] // 2, source_z]]], dtype=np.int64)
    _, snapshots = solver(
        make_wavelet(nt, weights),
        source,
        receivers,
        models=build_models(shape, dev, theta_deg, phi_deg),
        source_encoding=True,
        return_wavefield=True,
        snapshot_times=snapshot_times,
    )
    panels = {
        "vx": [
            crop_panel(snapshots[i, 0, 0, 0].cpu().numpy(), shape, abcn, free_surface)
            for i in range(len(snapshot_times))
        ],
        "vy": [
            crop_panel(snapshots[i, 1, 0, 0].cpu().numpy(), shape, abcn, free_surface)
            for i in range(len(snapshot_times))
        ],
        "vz": [
            crop_panel(snapshots[i, 2, 0, 0].cpu().numpy(), shape, abcn, free_surface)
            for i in range(len(snapshot_times))
        ],
    }
    if not all(np.isfinite(panel).all() for values in panels.values() for panel in values):
        raise FloatingPointError("Non-finite ElasticTTI wavefield detected.")
    return panels


def infer_free_surface_plot_nz(shape, snapshot_times, source_z, plot_zmax):
    if plot_zmax is not None:
        return min(shape[0], max(1, int(np.ceil(float(plot_zmax) / DH))))
    wave_radius = int(np.ceil(1.15 * 3000.0 * max(snapshot_times) * DT / DH))
    return min(shape[0], max(source_z + wave_radius + 20, source_z + 40))


def infer_active_window(results, field, shape, margin=36):
    stack = np.stack([np.asarray(panel) for _, panels in results for panel in panels[field]], axis=0)
    amplitude = np.nanmax(np.abs(stack), axis=0)
    max_amp = float(np.nanmax(amplitude))
    if not np.isfinite(max_amp) or max_amp <= 0.0:
        return (0, shape[0], 0, shape[1])

    threshold = max(max_amp * 0.015, float(np.nanpercentile(amplitude, 99.0)) * 0.05)
    coords = np.argwhere(amplitude >= threshold)
    if coords.size == 0:
        return (0, shape[0], 0, shape[1])

    z0, x0 = coords.min(axis=0)
    z1, x1 = coords.max(axis=0) + 1
    z0 = max(0, int(z0) - margin)
    x0 = max(0, int(x0) - margin)
    z1 = min(shape[0], int(z1) + margin)
    x1 = min(shape[1], int(x1) + margin)
    return (z0, z1, x0, x1)


def window_extent(window):
    z0, z1, x0, x1 = window
    return (x0 * DH, (x1 - 1) * DH, (z1 - 1) * DH, z0 * DH)


def plot_panel_grid(results, field, snapshot_times, shape, out_path, title, window=None):
    nrows = len(results)
    ncols = len(snapshot_times)
    window = (0, shape[0], 0, shape[1]) if window is None else window
    z0, z1, x0, x1 = window
    display_shape = (z1 - z0, x1 - x0)
    panel_width = 2.85
    panel_height = panel_width * display_shape[0] / display_shape[1]
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_width * ncols + 0.35, panel_height * nrows + 0.55),
        squeeze=False,
    )
    extent = window_extent(window)
    display_panels = [panel[z0:z1, x0:x1] for _, panels in results for panel in panels[field]]
    vmin, vmax = percentile_clip(display_panels)

    for row, (label, panels) in enumerate(results):
        for col, panel in enumerate(panels[field]):
            ax = axes[row, col]
            ax.imshow(panel[z0:z1, x0:x1], cmap="seismic", aspect="equal", vmin=vmin, vmax=vmax, extent=extent)
            if row == 0:
                ax.set_title(f"t={snapshot_times[col]}", fontsize=8, pad=2)
            if row == nrows - 1:
                ax.set_xlabel("X (m)", fontsize=7, labelpad=1)
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(label, fontsize=7, labelpad=3)
            else:
                ax.tick_params(labelleft=False)
            ax.tick_params(axis="both", labelsize=7, pad=1, length=2)

    fig.suptitle(title, fontsize=10, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955), pad=0.12, h_pad=0.18, w_pad=0.12)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_metrics(path, blocks):
    with Path(path).open("w", encoding="ascii") as f:
        f.write(f"import_mode={IMPORT_MODE}\n")
        f.write(f"dh={DH}\n")
        for block_name, results in blocks:
            f.write(f"[{block_name}]\n")
            for label, panels in results:
                stats = {
                    field: float(np.nanmax(np.abs(panels[field][-1])))
                    for field in ("vx", "vy", "vz")
                }
                text = ", ".join(f"{field}_max={value:.6e}" for field, value in stats.items())
                f.write(f"{label}: {text}\n")


def run_rotation(shape, nt, abcn, snapshot_times, dev, output_dir, fields):
    source_z = shape[0] // 2
    results = []
    for label, theta_deg, phi_deg in ROTATION_CASES:
        print(f"  rotation {label}...")
        panels = run_forward(
            shape,
            nt,
            abcn,
            snapshot_times,
            dev,
            equation_cls=ElasticTTI,
            free_surface=False,
            source_z=source_z,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
        )
        results.append((label, panels))

    for field in fields:
        window = infer_active_window(results, field, shape)
        plot_panel_grid(
            results,
            field,
            snapshot_times,
            shape,
            output_dir / f"elastic_tti_rotation_{field}_snapshots.png",
            f"Elastic TTI rotation: {field}",
            window=window,
        )
    return results


def run_free_surface(shape, nt, abcn, snapshot_times, dev, output_dir, fields, plot_zmax):
    source_z = max(8, shape[0] // 12)
    plot_nz = infer_free_surface_plot_nz(shape, snapshot_times, source_z, plot_zmax)
    results = []
    for label, equation_cls, free_surface in FREE_SURFACE_CASES:
        print(f"  free-surface {label}...")
        panels = run_forward(
            shape,
            nt,
            abcn,
            snapshot_times,
            dev,
            equation_cls=equation_cls,
            free_surface=free_surface,
            source_z=source_z,
            theta_deg=35.0,
            phi_deg=45.0,
        )
        results.append((label, panels))

    for field in fields:
        plot_panel_grid(
            results,
            field,
            snapshot_times,
            shape,
            output_dir / f"elastic_tti_free_surface_{field}_snapshots.png",
            f"Elastic TTI free-surface: {field}",
            window=(0, plot_nz, 0, shape[1]),
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Representative Elastic TTI wavefield experiments.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="Use a smaller grid for a fast smoke run.")
    parser.add_argument(
        "--experiment",
        choices=("rotation", "free-surface", "all"),
        default="all",
        help="Which experiment group to run.",
    )
    parser.add_argument(
        "--fields",
        default="vx,vy,vz",
        help="Comma-separated fields to plot. Use 'vz' for a compact docs-style run.",
    )
    parser.add_argument("--plot-zmax", type=float, default=None, help="Maximum displayed depth in meters for free-surface plots.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    dev = torch.device(args.device)
    shape = QUICK_SHAPE if args.quick else FULL_SHAPE
    nt = QUICK_NT if args.quick else FULL_NT
    abcn = QUICK_ABCN if args.quick else FULL_ABCN
    snapshot_times = QUICK_SNAPSHOT_TIMES if args.quick else FULL_SNAPSHOT_TIMES
    fields = tuple(field.strip() for field in args.fields.split(",") if field.strip())
    unknown = set(fields) - {"vx", "vy", "vz"}
    if unknown:
        raise ValueError(f"Unknown fields: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running ElasticTTI wavefield experiments ({IMPORT_MODE}) on {dev}...")
    print(f"shape={shape}, dh={DH}, nt={nt}, snapshots={snapshot_times}, fields={fields}")

    blocks = []
    if args.experiment in {"rotation", "all"}:
        blocks.append(("rotation", run_rotation(shape, nt, abcn, snapshot_times, dev, args.output_dir, fields)))
    if args.experiment in {"free-surface", "all"}:
        blocks.append(("free-surface", run_free_surface(shape, nt, abcn, snapshot_times, dev, args.output_dir, fields, args.plot_zmax)))

    metrics_path = args.output_dir / "elastic_tti_wavefields_metrics.txt"
    write_metrics(metrics_path, blocks)
    print(f"Saved ElasticTTI figures to {args.output_dir}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
