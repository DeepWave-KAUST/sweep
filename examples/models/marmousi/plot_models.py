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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Marmousi true/smooth/linear models.")
    root = Path(__file__).resolve().parent
    parser.add_argument("--true", default=str(root / "true.npy"))
    parser.add_argument("--smooth", default=str(root / "smooth.npy"))
    parser.add_argument("--linear", default=str(root / "linear.npy"))
    parser.add_argument("--output", default=str(root / "true_smooth_linear.png"))
    args = parser.parse_args()

    true_model = np.load(args.true).astype(np.float32)
    smooth_model = np.load(args.smooth).astype(np.float32)
    linear_model = np.load(args.linear).astype(np.float32)
    vmin, vmax = percentile_range(true_model, smooth_model, linear_model)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    panels = [
        ("True", true_model),
        ("Smooth", smooth_model),
        ("Linear", linear_model),
    ]
    for ax, (title, data) in zip(axes[0], panels):
        im = ax.imshow(data, cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.tight_layout()
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
