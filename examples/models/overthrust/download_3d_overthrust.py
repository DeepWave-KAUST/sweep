#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path


ARCHIVE_URL = (
    "https://s3.amazonaws.com/open.source.geoscience/open_data/seg_eage_models_cd/"
    "Overthrust_3D_CD1.tar.gz"
)
ARCHIVE_NAME = "Overthrust_3D_CD1.tar.gz"


def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    def report(block_count: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_count * block_size, total_size)
        pct = downloaded * 100.0 / total_size
        print(f"\rDownloading {dest.name}: {pct:6.2f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=report)
    print()


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(output_dir)
    print(f"Extracted {archive_path.name} into {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and optionally extract the official SEG/EAGE Overthrust 3D CD1 archive.")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the archive after download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if the archive already exists.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    archive_path = root / ARCHIVE_NAME
    extract_dir = root / "Overthrust_3D_CD1"

    if archive_path.exists() and not args.force:
        print(f"Skip download: {archive_path} already exists (md5 {file_md5(archive_path)}).")
    else:
        print(f"Fetching {ARCHIVE_URL}")
        download(ARCHIVE_URL, archive_path)
        print(f"Saved {archive_path} (md5 {file_md5(archive_path)})")

    if args.extract:
        if extract_dir.exists():
            print(f"Archive already extracted at {extract_dir}")
        else:
            extract_archive(archive_path, root)


if __name__ == "__main__":
    main()
