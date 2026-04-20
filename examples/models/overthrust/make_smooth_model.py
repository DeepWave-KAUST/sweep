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


def smooth_nd(array: np.ndarray, radii: tuple[int, ...]) -> np.ndarray:
    result = array.astype(np.float32, copy=True)
    for axis, radius in enumerate(radii):
        result = smooth1d_along_axis(result, radius, axis)
    return result


def parse_radii(text: str, ndim: int) -> tuple[int, ...]:
    parts = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) == 1:
        return tuple(parts * ndim)
    if len(parts) != ndim:
        raise ValueError(f"Expected 1 or {ndim} radii values, got {parts}")
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a smoothed initial model from a 2D or 3D NumPy velocity model.")
    parser.add_argument("--input", required=True, help="Input .npy model path.")
    parser.add_argument("--output", required=True, help="Output .npy model path.")
    parser.add_argument(
        "--radii",
        default="10",
        help="Smoothing radii. Use one value for all axes or comma-separated per axis, e.g. 10 or 6,6,3.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="How many smoothing passes to apply with the same radii.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    model = np.load(input_path).astype(np.float32)
    radii = parse_radii(args.radii, model.ndim)
    smooth = model
    for _ in range(args.passes):
        smooth = smooth_nd(smooth, radii)

    np.save(output_path, smooth)
    print(f"Input: {input_path} shape={model.shape}")
    print(f"Radii: {radii}")
    print(f"Passes: {args.passes}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
