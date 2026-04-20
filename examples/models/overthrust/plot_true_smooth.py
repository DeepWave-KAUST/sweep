#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def percentile_range(*arrays: np.ndarray, pct: float = 2.0) -> tuple[float, float]:
    joined = np.concatenate([np.ravel(arr) for arr in arrays])
    return float(np.percentile(joined, pct)), float(np.percentile(joined, 100.0 - pct))


def load_optional(path: Path) -> np.ndarray | None:
    return np.load(path).astype(np.float32) if path.exists() else None


def middle_slice(volume: np.ndarray, axis: int) -> tuple[np.ndarray, int]:
    index = volume.shape[axis] // 2
    return np.take(volume, index, axis=axis), index


def plot_3d_slices(true_3d: np.ndarray, smooth_3d: np.ndarray, output_path: Path) -> None:
    planes = [
        ("z", 0),
        ("y", 1),
        ("x", 2),
    ]

    fig, axes = plt.subplots(len(planes), 2, figsize=(10, 12), squeeze=False)
    vmin, vmax = percentile_range(true_3d, smooth_3d)

    for row, (name, axis) in enumerate(planes):
        true_slice, true_idx = middle_slice(true_3d, axis)
        smooth_slice, smooth_idx = middle_slice(smooth_3d, axis)

        im_true = axes[row, 0].imshow(true_slice, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"3D True {name}-slice @ {true_idx}")
        im_smooth = axes[row, 1].imshow(smooth_slice, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"3D Smooth {name}-slice @ {smooth_idx}")
        fig.colorbar(im_true, ax=axes[row, 0], shrink=0.85)
        fig.colorbar(im_smooth, ax=axes[row, 1], shrink=0.85)

    for ax in axes.ravel():
        ax.set_xlabel("horizontal index")
        ax.set_ylabel("vertical index")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_2d_models(true_2d: np.ndarray, smooth_2d: np.ndarray, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    vmin, vmax = percentile_range(true_2d, smooth_2d)

    im_true = axes[0, 0].imshow(true_2d, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("2D True")
    im_smooth = axes[0, 1].imshow(smooth_2d, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("2D Smooth")
    fig.colorbar(im_true, ax=axes[0, 0], shrink=0.85)
    fig.colorbar(im_smooth, ax=axes[0, 1], shrink=0.85)

    for ax in axes.ravel():
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Overthrust true/smooth models for 3D slices and 2D sections.")
    root = Path(__file__).resolve().parent
    parser.add_argument("--true-3d", default=str(root / "true_3d.npy"))
    parser.add_argument("--smooth-3d", default=str(root / "smooth_3d.npy"))
    parser.add_argument("--true-2d", default=str(root / "true_2d.npy"))
    parser.add_argument("--smooth-2d", default=str(root / "smooth_2d.npy"))
    parser.add_argument("--out-3d", default=str(root / "true_smooth_3d_slices.png"))
    parser.add_argument("--out-2d", default=str(root / "true_smooth_2d.png"))
    args = parser.parse_args()

    true_3d = load_optional(Path(args.true_3d))
    smooth_3d = load_optional(Path(args.smooth_3d))
    true_2d = load_optional(Path(args.true_2d))
    smooth_2d = load_optional(Path(args.smooth_2d))

    if true_3d is not None and smooth_3d is not None:
        plot_3d_slices(true_3d, smooth_3d, Path(args.out_3d))
        print(f"Saved {args.out_3d}")
    else:
        print("Skip 3D plot: true_3d.npy or smooth_3d.npy is missing.")

    if true_2d is not None and smooth_2d is not None:
        plot_2d_models(true_2d, smooth_2d, Path(args.out_2d))
        print(f"Saved {args.out_2d}")
    else:
        print("Skip 2D plot: true_2d.npy or smooth_2d.npy is missing.")


if __name__ == "__main__":
    main()
