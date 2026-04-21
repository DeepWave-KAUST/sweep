from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


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


def percentile_clip(data, percentile=99.0):
    amp = np.percentile(np.abs(np.asarray(data)), percentile)
    return max(float(amp), 1e-6)


def extent(shape, dh):
    nz, nx = shape
    return (0.0, (nx - 1) * dh, (nz - 1) * dh, 0.0)


def plot_model(data, sources, receivers, out_path, title, dh, cbar_label):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    im = ax.imshow(data, cmap="viridis", aspect="auto", extent=extent(data.shape, dh))
    ax.scatter(receivers[0, :, 0] * dh, receivers[0, :, 1] * dh, s=10, c="white", marker="v", edgecolors="black", linewidths=0.3)
    ax.scatter(sources[:, 0] * dh, sources[:, 1] * dh, s=90, c="gold", marker="*", edgecolors="black", linewidths=0.5)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    fig.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_panel_grid(panels, row_titles, col_titles, out_path, title, dh):
    nrows = len(panels)
    ncols = len(panels[0])
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
    flat_panels = [panel for row in panels for panel in row]
    vmax = percentile_clip(flat_panels)
    ext = extent(panels[0][0].shape, dh)

    for i, row in enumerate(panels):
        for j, panel in enumerate(row):
            ax = axes[i, j]
            ax.imshow(panel, cmap="seismic", aspect="auto", vmin=-vmax, vmax=vmax, extent=ext)
            ax.set_title(f"{row_titles[i]} | t={col_titles[j]}")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_record_comparison(records, row_titles, out_path, title):
    fig, axes = plt.subplots(1, len(records), figsize=(5.2 * len(records), 4.2), squeeze=False)
    vmax = percentile_clip(records)

    for i, panel in enumerate(records):
        ax = axes[0, i]
        ax.imshow(panel, cmap="seismic", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_title(row_titles[i])
        ax.set_xlabel("Receiver Index")
        ax.set_ylabel("Time Sample")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
