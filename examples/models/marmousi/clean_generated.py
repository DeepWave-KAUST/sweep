#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


FILES_TO_REMOVE = [
    "elastic-marmousi-model.tar.gz",
    "true.npy",
    "smooth.npy",
    "linear.npy",
    "true_smooth_linear.png",
]

DIRS_TO_REMOVE = [
    "elastic-marmousi-model",
    "segy",
    "npy",
    "__pycache__",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove downloaded and generated Marmousi artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting anything.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    for name in FILES_TO_REMOVE:
        path = root / name
        if not path.exists():
            continue
        if args.dry_run:
            print(f"Would remove file: {path}")
        else:
            path.unlink()
            print(f"Removed file: {path}")

    for name in DIRS_TO_REMOVE:
        path = root / name
        if not path.exists():
            continue
        if args.dry_run:
            print(f"Would remove directory: {path}")
        else:
            shutil.rmtree(path)
            print(f"Removed directory: {path}")


if __name__ == "__main__":
    main()
