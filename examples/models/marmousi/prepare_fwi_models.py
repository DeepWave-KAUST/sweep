#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def smooth1d_along_axis(array: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius <= 0:
        return array.copy()

    pad = [(0, 0)] * array.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(array, pad, mode="edge")

    moved = np.moveaxis(padded, axis, 0)
    out = np.empty_like(np.moveaxis(array, axis, 0), dtype=np.float32)
    window = 2 * radius + 1

    running = moved[:window].sum(axis=0, dtype=np.float64)
    out[0] = running / window
    for i in range(1, out.shape[0]):
        running += moved[i + window - 1]
        running -= moved[i - 1]
        out[i] = running / window

    return np.moveaxis(out, 0, axis)


def smooth2d(array: np.ndarray, radii: tuple[int, int], passes: int) -> np.ndarray:
    result = array.astype(np.float32, copy=True)
    for _ in range(passes):
        result = smooth1d_along_axis(result, radii[0], axis=0)
        result = smooth1d_along_axis(result, radii[1], axis=1)
    return result


def make_linear_model(true_model: np.ndarray) -> np.ndarray:
    top = float(true_model[0].mean())
    bottom = float(true_model[-1].mean())
    profile = np.linspace(top, bottom, true_model.shape[0], dtype=np.float32)
    return np.repeat(profile[:, None], true_model.shape[1], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Marmousi FWI-ready true/smooth/linear models from the 1.25 m Vp grid.")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "npy" / "vp_1p25m.npy"),
        help="Input Vp model at 1.25 m spacing.",
    )
    parser.add_argument(
        "--source-dh",
        type=float,
        default=1.25,
        help="Grid spacing of the source Vp model in meters.",
    )
    parser.add_argument(
        "--target-dh",
        type=float,
        default=12.5,
        help="Target grid spacing for the FWI examples in meters.",
    )
    parser.add_argument(
        "--radii",
        default="16,16",
        help="Smoothing radii on the downsampled 2D model, formatted like z,x. "
             "Defaults to 16,16 which is ~200 m physical width at target-dh=12.5.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="Number of smoothing passes for the smooth initial model.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input model: {input_path}")

    ratio = args.target_dh / args.source_dh
    stride = int(round(ratio))
    if abs(stride * args.source_dh - args.target_dh) > 1e-6:
        raise ValueError("target-dh must be an integer multiple of source-dh.")

    parts = [int(part.strip()) for part in args.radii.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("Expected exactly two radii values formatted as z,x.")
    radii = (parts[0], parts[1])

    vp = np.load(input_path).astype(np.float32)
    true_model = vp[::stride, ::stride]
    smooth_model = smooth2d(true_model, radii=radii, passes=args.passes)
    linear_model = make_linear_model(true_model)

    np.save(root / "true.npy", true_model)
    np.save(root / "smooth.npy", smooth_model)
    np.save(root / "linear.npy", linear_model)

    print(f"Input: {input_path} shape={vp.shape}")
    print(f"Source dh: {args.source_dh} m")
    print(f"Target dh: {args.target_dh} m")
    print(f"Stride: {stride}")
    print(f"true.npy shape: {true_model.shape}")
    print(f"smooth radii: {radii}, passes: {args.passes}")
    print(f"Saved: {root / 'true.npy'}")
    print(f"Saved: {root / 'smooth.npy'}")
    print(f"Saved: {root / 'linear.npy'}")


if __name__ == "__main__":
    main()
