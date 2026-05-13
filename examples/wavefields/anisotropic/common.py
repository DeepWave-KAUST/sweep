from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def find_examples_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if candidate.name == "examples":
            return candidate
    raise RuntimeError("Could not locate the examples directory.")


EXAMPLES_DIR = find_examples_root()
sys.path.insert(0, str(EXAMPLES_DIR / "_shared"))
from bootstrap import configure_example_imports

IMPORT_MODE = configure_example_imports(EXAMPLES_DIR)


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def percentile_clip(data, percentiles=(1.0, 99.0)):
    values = np.asarray(data)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1e-6, 1e-6
    vmin, vmax = np.percentile(finite, percentiles)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return -1e-6, 1e-6
    if vmin == vmax:
        eps = max(abs(float(vmin)) * 1e-6, 1e-6)
        return float(vmin) - eps, float(vmax) + eps
    return float(vmin), float(vmax)


def extent_meters(shape, dh):
    nz, nx = shape
    dz, dx = dh
    return (0.0, (nx - 1) * dx, (nz - 1) * dz, 0.0)


def plot_snapshot_grid(panels, row_titles, col_titles, out_path, figure_title, shape, dh):
    nrows = len(panels)
    ncols = len(panels[0])
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.8 * nrows), squeeze=False)
    ext = extent_meters(shape, dh)

    for i, row in enumerate(panels):
        vmin, vmax = percentile_clip(row)
        for j, panel in enumerate(row):
            ax = axes[i, j]
            ax.imshow(panel, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax, extent=ext)
            ax.set_title(f"{row_titles[i]} | t={col_titles[j]}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")

    fig.suptitle(figure_title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_record_grid(records, titles, out_path, figure_title):
    fig, axes = plt.subplots(1, len(records), figsize=(5.2 * len(records), 4.4), squeeze=False)
    for ax, record, title in zip(axes[0], records, titles):
        vmin, vmax = percentile_clip(record)
        ax.imshow(record, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Receiver Index")
        ax.set_ylabel("Time Sample")

    fig.suptitle(figure_title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
