#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


AXIS_TO_INDEX = {
    "z": 0,
    "y": 1,
    "x": 2,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a 2D slice from a 3D NumPy model for 2D FWI experiments.")
    parser.add_argument("--input", required=True, help="Input 3D .npy model path.")
    parser.add_argument("--output", required=True, help="Output 2D .npy path.")
    parser.add_argument(
        "--axis",
        choices=sorted(AXIS_TO_INDEX),
        default="y",
        help="Axis to slice away. We assume the Overthrust volume is stored as (z, y, x).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Slice index along the selected axis. Defaults to the middle slice.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    volume = np.load(input_path).astype(np.float32)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D array, got shape {volume.shape}")

    axis = AXIS_TO_INDEX[args.axis]
    index = volume.shape[axis] // 2 if args.index is None else args.index
    if index < 0 or index >= volume.shape[axis]:
        raise IndexError(f"Slice index {index} is out of range for axis {args.axis} with size {volume.shape[axis]}")

    slice_2d = np.take(volume, indices=index, axis=axis)
    np.save(output_path, slice_2d)

    print(f"Input: {input_path} shape={volume.shape}")
    print(f"Axis: {args.axis} (index {axis})")
    print(f"Slice index: {index}")
    print(f"Output shape: {slice_2d.shape}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
