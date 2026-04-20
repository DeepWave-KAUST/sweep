#!/usr/bin/env python3
from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


MODEL_TARS = {
    "vp": "MODEL_P-WAVE_VELOCITY_1.25m.segy.tar.gz",
    "vs": "MODEL_S-WAVE_VELOCITY_1.25m.segy.tar.gz",
    "rho": "MODEL_DENSITY_1.25m.segy.tar.gz",
}


def extract_tar(archive_path: Path, output_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(output_dir)
    print(f"Extracted {archive_path.name} into {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the Marmousi model SEG-Y files from the official archive contents.")
    parser.add_argument(
        "--which",
        choices=["vp", "vs", "rho", "all"],
        default="all",
        help="Choose which model archives to extract.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    model_dir = root / "elastic-marmousi-model" / "model"
    output_dir = root / "segy"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = list(MODEL_TARS) if args.which == "all" else [args.which]
    for key in selected:
        archive_path = model_dir / MODEL_TARS[key]
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing model archive: {archive_path}")
        extract_tar(archive_path, output_dir)


if __name__ == "__main__":
    main()
