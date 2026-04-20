#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    import segyio
except ImportError as exc:
    raise SystemExit(
        "This script requires segyio. Install it with `pip install segyio` and rerun."
    ) from exc


SEGY_FILES = {
    "vp": "MODEL_P-WAVE_VELOCITY_1.25m.segy",
    "vs": "MODEL_S-WAVE_VELOCITY_1.25m.segy",
    "rho": "MODEL_DENSITY_1.25m.segy",
}


def load_segy(path: Path) -> np.ndarray:
    with segyio.open(path, "r", ignore_geometry=True) as segyfile:
        num_traces = segyfile.tracecount
        num_samples = segyfile.samples.size
        data = np.empty((num_samples, num_traces), dtype=np.float32)
        for i in range(num_traces):
            data[:, i] = segyfile.trace[i]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Marmousi SEG-Y model files into NumPy arrays.")
    parser.add_argument(
        "--which",
        choices=["vp", "vs", "rho", "all"],
        default="all",
        help="Choose which SEG-Y files to convert.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    segy_dir = root / "segy"
    output_dir = root / "npy"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = list(SEGY_FILES) if args.which == "all" else [args.which]
    for key in selected:
        segy_path = segy_dir / SEGY_FILES[key]
        if not segy_path.exists():
            raise FileNotFoundError(f"Missing SEG-Y file: {segy_path}")
        array = load_segy(segy_path)
        output_path = output_dir / f"{key}_1p25m.npy"
        np.save(output_path, array)
        print(f"Converted {segy_path} -> {output_path} shape={array.shape}")


if __name__ == "__main__":
    main()
