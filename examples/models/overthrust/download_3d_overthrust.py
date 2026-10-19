#!/usr/bin/env python3
"""Write the SEG/EAGE Overthrust 3-D cubes the 3-D examples load.

- ``true_3d.npy``: the official velocity grid, 187 x 801 x 801 (z, y, x) at
  25 m, in m/s, from ``sweep.datasets``. It downloads and checks the SEG/EAGE
  CD on first use (re-hosted on Zenodo, doi:10.5281/zenodo.23469292) and reuses
  a CD already in its cache.
- ``smooth_3d.npy``: the initial model, the true model smoothed with
  ``make_smooth_model.py`` (by default radii 6,6,6 and 3 passes).

Needs ``pip install sweep-solver[datasets]``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from make_smooth_model import parse_radii, smooth_nd


def main() -> None:
    parser = argparse.ArgumentParser(description="Write true_3d.npy and smooth_3d.npy for the 3-D Overthrust examples.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Where to write the two files. Default: this script's directory.",
    )
    parser.add_argument("--radii", default="6,6,6", help="Smoothing radii for smooth_3d.npy, one value or z,y,x.")
    parser.add_argument("--passes", type=int, default=3, help="Smoothing passes for smooth_3d.npy.")
    parser.add_argument("--force", action="store_true", help="Rewrite the files even if they already exist.")
    args = parser.parse_args()

    true_path = args.out_dir / "true_3d.npy"
    smooth_path = args.out_dir / "smooth_3d.npy"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if true_path.exists() and not args.force:
        print(f"Skip: {true_path} already exists.")
        true_model = np.load(true_path).astype(np.float32)
    else:
        from sweep.datasets import load

        true_model = load("overthrust", "3d-acoustic")["vp"]
        np.save(true_path, true_model)
        print(f"Saved {true_path} shape={true_model.shape} "
              f"vp {float(true_model.min()):.1f}-{float(true_model.max()):.1f} m/s")

    if smooth_path.exists() and not args.force:
        print(f"Skip: {smooth_path} already exists.")
        return
    radii = parse_radii(args.radii, true_model.ndim)
    smooth = true_model
    for _ in range(args.passes):
        smooth = smooth_nd(smooth, radii)
    np.save(smooth_path, smooth)
    print(f"Saved {smooth_path} (radii {radii}, {args.passes} passes)")


if __name__ == "__main__":
    main()
