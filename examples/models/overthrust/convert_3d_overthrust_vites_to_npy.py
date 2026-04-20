#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("Overthrust_3D_CD1/3-D_Overthrust_Model_Disk1/3D-Velocity-Grid")
DEFAULT_BIN = "overthrust.vites"
N1 = 801
N2 = 801
N3 = 187
DTYPE = ">f4"
SHAPE = (N3, N2, N1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the official Overthrust 3D VITES grid to a NumPy file.")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent / DEFAULT_ROOT),
        help="Directory containing overthrust.vites and its metadata files.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "true_3d.npy"),
        help="Output .npy path.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    bin_path = root / DEFAULT_BIN

    if not bin_path.exists():
        raise FileNotFoundError(f"Missing binary model file: {bin_path}")

    expected_size = int(np.prod(SHAPE))
    array = np.fromfile(bin_path, dtype=DTYPE)
    if array.size != expected_size:
        raise RuntimeError(
            f"Unexpected element count in {bin_path}: got {array.size}, expected {expected_size}."
        )
    array = array.reshape(SHAPE, order="C")

    output_path = Path(args.output)
    np.save(output_path, array)

    print(f"Input binary: {bin_path}")
    print(f"Dtype: {DTYPE}")
    print(f"Shape: {SHAPE}")
    print(f"Value range: min={float(array.min()):.3f}, max={float(array.max()):.3f}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
