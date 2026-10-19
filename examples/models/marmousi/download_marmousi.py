#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path


ORIGINAL_URL = (
    "https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/"
    "elastic-marmousi-model.tar.gz"
)
# The SEG open-data bucket stopped serving anonymous downloads in 2026. The
# same file is re-hosted on Zenodo (doi:10.5281/zenodo.23469300) and Hugging
# Face, and kept by the Internet Archive; the original URL stays last.
ARCHIVE_URLS = (
    "https://zenodo.org/records/23469300/files/elastic-marmousi-model.tar.gz",
    "https://huggingface.co/datasets/shaowinw/seismic-benchmark-models/resolve/main/"
    "marmousi2_elastic/elastic-marmousi-model.tar.gz",
    "https://web.archive.org/web/20240328082553id_/" + ORIGINAL_URL,
    ORIGINAL_URL,
)
ARCHIVE_NAME = "elastic-marmousi-model.tar.gz"
ARCHIVE_SHA256 = "b6fe8d10ce6a6dfadf203a8d0e1a040ce9f2c6cb52c1366193d35113d61797fe"


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
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


def fetch_archive(dest: Path) -> None:
    """Download from the first URL that serves the original file (checked by sha256)."""
    part = dest.with_name(dest.name + ".part")
    for url in ARCHIVE_URLS:
        print(f"Fetching {url}")
        try:
            download(url, part)
        except OSError as e:
            print(f"  failed: {e}")
            continue
        got = file_sha256(part)
        if got == ARCHIVE_SHA256:
            part.replace(dest)
            print(f"Saved {dest} (sha256 {got})")
            return
        print(f"  wrong file (sha256 {got}, expected {ARCHIVE_SHA256})")
    part.unlink(missing_ok=True)
    raise SystemExit(f"Could not download {ARCHIVE_NAME} from any of: {', '.join(ARCHIVE_URLS)}")


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(output_dir)
    print(f"Extracted {archive_path.name} into {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and optionally extract the official Elastic Marmousi archive.")
    parser.add_argument("--extract", action="store_true", help="Extract the archive after download.")
    parser.add_argument("--force", action="store_true", help="Redownload even if the archive already exists.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    archive_path = root / ARCHIVE_NAME
    extract_dir = root / "elastic-marmousi-model"

    if archive_path.exists() and not args.force and file_sha256(archive_path) == ARCHIVE_SHA256:
        print(f"Skip download: {archive_path} already exists (sha256 matches).")
    else:
        if archive_path.exists() and not args.force:
            print(f"{archive_path} does not match its sha256; downloading it again.")
        fetch_archive(archive_path)

    if args.extract:
        if extract_dir.exists():
            print(f"Archive already extracted at {extract_dir}")
        else:
            extract_archive(archive_path, root)


if __name__ == "__main__":
    main()
