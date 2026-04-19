from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sweep.equations import Elastic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker


OUTPUT_DIR = Path(__file__).resolve().parent / "elastic_free_surface_view"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PHYS_SHAPE = (80, 120)
ABCN = 20
SPATIAL_ORDER = 6
DT = 0.001
DH = 5.0
NT = 500
DOM_FREQ = 20.0


def make_solver(free_surface):
    dev = torch.device("cuda")
    return PropTorch(
        Elastic(spatial_order=SPATIAL_ORDER, device=dev, backend="torch"),
        shape=PHYS_SHAPE,
        dev=dev,
        dh=DH,
        dt=DT,
        nt=NT,
        abcn=ABCN,
        source_type=["sxx", "szz"],
        receiver_type=["vz"],
        free_surface=free_surface,
        pml_type="cpmls",
        use_ckpt=False,
    )


def make_inputs(nt, shape):
    nz, nx = shape
    vp = torch.full((nz, nx), 2200.0, dtype=torch.float32)
    vs = torch.full((nz, nx), 1200.0, dtype=torch.float32)
    rho = torch.full((nz, nx), 1800.0, dtype=torch.float32)

    t = np.arange(nt, dtype=np.float32) * DT - 0.20
    wavelet = 5e6 * ricker(t, f=DOM_FREQ)

    sources = np.array([[nx // 2, 6]], dtype=np.int64)
    receivers = np.stack(
        [np.arange(12, nx - 12, dtype=np.int64), np.full(nx - 24, 6, dtype=np.int64)],
        axis=1,
    )[None, ...]
    return wavelet, sources, receivers, [vp, vs, rho]


def snapshot_times():
    return [100, 200, 300, 450]


def crop_wavefield_panel(panel, free_surface):
    if free_surface:
        return panel[: PHYS_SHAPE[0], ABCN : ABCN + PHYS_SHAPE[1]]
    return panel[ABCN : ABCN + PHYS_SHAPE[0], ABCN : ABCN + PHYS_SHAPE[1]]


def crop_snapshots(snapshots, free_surface):
    cropped = np.empty(
        (snapshots.shape[0], snapshots.shape[1]) + PHYS_SHAPE,
        dtype=snapshots.dtype,
    )
    for it in range(snapshots.shape[0]):
        for iw in range(snapshots.shape[1]):
            cropped[it, iw] = crop_wavefield_panel(snapshots[it, iw, 0, 0], free_surface)
    return cropped


def run_case(free_surface):
    solver = make_solver(free_surface)
    wavelet, sources, receivers, models = make_inputs(solver.nt, solver.shape_nopad)
    times = snapshot_times()
    record, snapshots = solver(
        wavelet,
        sources,
        receivers,
        models=models,
        return_wavefield=True,
        snapshot_times=times,
    )
    return record.detach().cpu().numpy(), times, crop_snapshots(snapshots.detach().cpu().numpy(), free_surface)


def percentile_clip(data, pct=99.0):
    amp = np.percentile(np.abs(data), pct)
    return max(float(amp), 1e-6)


def plot_wavefield_comparison(fs_snapshots, abs_snapshots, times, out_path):
    names = ["vz", "szz", "sxz"]
    field_index = {"vz": 1, "szz": 3, "sxz": 4}

    fig, axes = plt.subplots(len(names), len(times) * 2, figsize=(18, 9), squeeze=False)

    for row, name in enumerate(names):
        idx = field_index[name]
        all_panels = []
        for snap_idx, _ in enumerate(times):
            all_panels.append(fs_snapshots[snap_idx, idx])
            all_panels.append(abs_snapshots[snap_idx, idx])
        amp = percentile_clip(np.stack(all_panels))

        for col, it in enumerate(times):
            fs_ax = axes[row, 2 * col]
            abs_ax = axes[row, 2 * col + 1]

            fs_ax.imshow(fs_snapshots[col, idx], cmap="seismic", aspect="auto", vmin=-amp, vmax=amp)
            fs_ax.set_title(f"{name} fs t={it}")
            fs_ax.axhline(0, color="k", linewidth=0.6)

            abs_ax.imshow(abs_snapshots[col, idx], cmap="seismic", aspect="auto", vmin=-amp, vmax=amp)
            abs_ax.set_title(f"{name} no-fs t={it}")
            abs_ax.axhline(0, color="k", linewidth=0.6)

    for ax in axes.ravel():
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_single_case_snapshots(snapshots, times, label, out_path):
    names = ["vz", "szz", "sxz"]
    field_index = {"vz": 1, "szz": 3, "sxz": 4}

    fig, axes = plt.subplots(len(names), len(times), figsize=(14, 9), squeeze=False)

    for row, name in enumerate(names):
        idx = field_index[name]
        amp = percentile_clip(np.stack([snapshots[snap_idx, idx] for snap_idx, _ in enumerate(times)]))
        for col, it in enumerate(times):
            ax = axes[row, col]
            ax.imshow(snapshots[col, idx], cmap="seismic", aspect="auto", vmin=-amp, vmax=amp)
            ax.set_title(f"{label} {name} t={it}")
            ax.axhline(0, color="k", linewidth=0.6)
            ax.set_xlabel("x")
            ax.set_ylabel("z")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_top_boundary(fs_snapshots, abs_snapshots, times, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    panels = [
        ("FS top szz", fs_snapshots[:, 3, 0, :]),
        ("No-FS top szz", abs_snapshots[:, 3, 0, :]),
        ("FS top sxz", fs_snapshots[:, 4, 0, :]),
        ("No-FS top sxz", abs_snapshots[:, 4, 0, :]),
    ]

    for ax, (title, data) in zip(axes.ravel(), panels):
        amp = percentile_clip(data)
        ax.imshow(data, cmap="seismic", aspect="auto", vmin=-amp, vmax=amp)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("snapshot time")
        ax.set_yticks(range(len(times)))
        ax.set_yticklabels(times)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_record_comparison(fs_record, abs_record, out_path):
    fs_panel = fs_record[0, :, :, 0]
    abs_panel = abs_record[0, :, :, 0]
    diff_panel = fs_panel - abs_panel

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    amp = percentile_clip(np.stack([fs_panel, abs_panel]))
    diff_amp = percentile_clip(diff_panel)

    axes[0, 0].imshow(fs_panel, cmap="seismic", aspect="auto", vmin=-amp, vmax=amp)
    axes[0, 0].set_title("Receiver vz with free surface")
    axes[0, 1].imshow(abs_panel, cmap="seismic", aspect="auto", vmin=-amp, vmax=amp)
    axes[0, 1].set_title("Receiver vz without free surface")
    axes[0, 2].imshow(diff_panel, cmap="seismic", aspect="auto", vmin=-diff_amp, vmax=diff_amp)
    axes[0, 2].set_title("Difference")

    for ax in axes.ravel():
        ax.set_xlabel("receiver")
        ax.set_ylabel("time")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    fs_record, fs_times, fs_snapshots = run_case(True)
    abs_record, abs_times, abs_snapshots = run_case(False)
    if fs_times != abs_times:
        raise RuntimeError("Snapshot times do not match between cases.")

    plot_single_case_snapshots(
        fs_snapshots,
        fs_times,
        "free-surface",
        OUTPUT_DIR / "elastic_free_surface_only_snapshots.png",
    )
    plot_single_case_snapshots(
        abs_snapshots,
        abs_times,
        "no-free-surface",
        OUTPUT_DIR / "elastic_no_free_surface_only_snapshots.png",
    )
    plot_wavefield_comparison(
        fs_snapshots,
        abs_snapshots,
        fs_times,
        OUTPUT_DIR / "elastic_free_surface_snapshots.png",
    )
    plot_top_boundary(
        fs_snapshots,
        abs_snapshots,
        fs_times,
        OUTPUT_DIR / "elastic_free_surface_top_boundary.png",
    )
    plot_record_comparison(
        fs_record,
        abs_record,
        OUTPUT_DIR / "elastic_free_surface_records.png",
    )

    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
