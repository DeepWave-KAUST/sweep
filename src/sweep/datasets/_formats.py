"""Decompression + model-file parsing helpers for the download benchmarks.

The public open-data benchmarks ship in a zoo of formats — raw big-endian
grids, gzipped SEG-Y, tarballs of (nested) SEG-Y, and raw binary inside a
nested ZIP. This module centralises the "get a float32 ndarray out of
whatever was downloaded" logic. It is **numpy-only** (a self-contained SEG-Y
rev-1 reader is inlined below) so the datasets package has no parser
dependency beyond numpy.
"""

from __future__ import annotations

import gzip
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np

_SEGY_SUFFIXES = {".segy", ".sgy"}


def normalize_downsample(factor, ndim: int) -> tuple[int, ...]:
    """Normalize a downsample ``factor`` to a per-axis tuple of length ``ndim``.

    ``factor`` may be a single int (applied to every axis) or a per-axis
    sequence (list/tuple) of length ``ndim``.
    """
    if isinstance(factor, int):
        out = (factor,) * ndim
    else:
        out = tuple(int(f) for f in factor)
    if len(out) != ndim:
        raise ValueError(
            f"downsample {factor!r} has {len(out)} axes but the model is {ndim}-D"
        )
    if any(f < 1 for f in out):
        raise ValueError(f"downsample factors must be >= 1; got {out}")
    return out


def decimate(arr: np.ndarray, factor) -> tuple[np.ndarray, tuple[int, ...]]:
    """Strided-decimate ``arr`` per axis. Returns ``(arr_out, factor_tuple)``.

    ``factor`` is an int (uniform) or a per-axis sequence. The returned tuple
    lets the caller scale grid spacing ``dh`` axis-by-axis.
    """
    factor = normalize_downsample(factor, arr.ndim)
    if all(f == 1 for f in factor):
        return arr, factor
    sl = tuple(slice(None, None, f) for f in factor)
    return arr[sl], factor


# --------------------------------------------------------------- decompress
def gunzip(src: Path, dest: Path | None = None) -> Path:
    """Decompress a ``.gz`` file to ``dest`` (default: strip the ``.gz``).

    Skips the work if ``dest`` already exists.
    """
    src = Path(src)
    if dest is None:
        dest = src.with_suffix("") if src.suffix == ".gz" else src.with_name(src.name + ".out")
    dest = Path(dest)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with gzip.open(src, "rb") as fin, open(tmp, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=1 << 22)
    tmp.replace(dest)
    return dest


def extract_tar_member(tar_path: Path, member_suffix: str, dest_dir: Path) -> Path:
    """Extract the first tar member whose name ends with ``member_suffix``.

    Returns the extracted file path. Skips extraction if it already exists.
    """
    tar_path = Path(tar_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        match = next(
            (m for m in tf.getmembers()
             if m.isfile() and m.name.lower().endswith(member_suffix.lower())),
            None,
        )
        if match is None:
            raise FileNotFoundError(
                f"no member ending in {member_suffix!r} inside {tar_path.name}"
            )
        out = dest_dir / Path(match.name).name
        if out.exists():
            return out
        with tf.extractfile(match) as fin, open(out, "wb") as fout:  # type: ignore[arg-type]
            shutil.copyfileobj(fin, fout, length=1 << 22)
    return out


def extract_nested_tar_segy(outer_tar: Path, member_suffix: str, dest_dir: Path) -> Path:
    """Two-level extraction: pull ``member_suffix`` (a ``.segy.tar.gz``) out of
    ``outer_tar``, then untar that inner archive to its single ``.segy``/``.sgy``.

    Used by Marmousi2, whose ``elastic-marmousi-model.tar.gz`` stores each model
    parameter as its own gzipped-tar of one SEG-Y.
    """
    dest_dir = Path(dest_dir)
    inner_tgz = extract_tar_member(outer_tar, member_suffix, dest_dir)
    with tarfile.open(inner_tgz, "r:*") as tf:
        seg = next(
            (m for m in tf.getmembers()
             if m.isfile() and m.name.lower().endswith((".segy", ".sgy"))),
            None,
        )
        if seg is None:
            raise FileNotFoundError(f"no .segy/.sgy inside {Path(inner_tgz).name}")
        out = dest_dir / Path(seg.name).name
        if out.exists():
            return out
        with tf.extractfile(seg) as fin, open(out, "wb") as fout:  # type: ignore[arg-type]
            shutil.copyfileobj(fin, fout, length=1 << 22)
    return out


def extract_zip_member(zip_path: Path, member_suffix: str, dest_dir: Path) -> Path:
    """Extract the first zip member whose name ends with ``member_suffix``."""
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        match = next(
            (n for n in zf.namelist() if n.lower().endswith(member_suffix.lower())),
            None,
        )
        if match is None:
            raise FileNotFoundError(
                f"no member ending in {member_suffix!r} inside {zip_path.name}"
            )
        out = dest_dir / Path(match).name
        if out.exists():
            return out
        with zf.open(match) as fin, open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=1 << 22)
    return out


# ---------------------------------------------------------------- SEG-Y read
# Self-contained SEG-Y rev-1 reader — numpy only, no external dependency.
_SEGY_TEXT_HEADER = 3200
_SEGY_BIN_HEADER = 400
_SEGY_TRACE_HEADER = 240
# sample-format code (binary header bytes 25-26) -> bytes per sample
_SEGY_BYTES_PER_SAMPLE = {1: 4, 2: 4, 3: 2, 5: 4, 6: 8, 8: 1}


def _ibm_to_ieee(u32: np.ndarray) -> np.ndarray:
    """Decode IBM 32-bit floats (uint32 bit-pattern) to IEEE float32."""
    u32 = np.ascontiguousarray(u32, dtype=np.uint32)
    sign = (u32 >> 31).astype(np.int8)
    expo = ((u32 >> 24) & 0x7F).astype(np.int32)
    mant = (u32 & 0x00FFFFFF).astype(np.float32)
    out = np.ldexp(mant, (expo - 64) * 4 - 24).astype(np.float32)
    return np.where(sign != 0, -out, out)


def segy_to_array(path: Path) -> np.ndarray:
    """Read every trace of a SEG-Y file into a ``(n_traces, n_samples)`` array.

    Numpy-only SEG-Y rev-1 reader (no ``segyio`` / ``sweep-io``); handles
    IBM/IEEE float and integer sample formats (big-endian). The caller
    reshapes/transposes to the physical model grid.
    """
    import os
    import struct

    path = Path(path)
    with open(path, "rb") as f:
        f.seek(_SEGY_TEXT_HEADER)
        binhdr = f.read(_SEGY_BIN_HEADER)
    n_samples = struct.unpack(">H", binhdr[20:22])[0]
    fmt = struct.unpack(">H", binhdr[24:26])[0]
    if fmt not in _SEGY_BYTES_PER_SAMPLE:
        raise NotImplementedError(f"{path.name}: SEG-Y sample format {fmt} not supported")
    bps = _SEGY_BYTES_PER_SAMPLE[fmt]

    trace_bytes = _SEGY_TRACE_HEADER + n_samples * bps
    body = os.path.getsize(path) - _SEGY_TEXT_HEADER - _SEGY_BIN_HEADER
    n_traces = body // trace_bytes

    raw = np.memmap(
        path, dtype=np.uint8, mode="r",
        offset=_SEGY_TEXT_HEADER + _SEGY_BIN_HEADER,
        shape=(n_traces, trace_bytes),
    )
    data = np.ascontiguousarray(raw[:, _SEGY_TRACE_HEADER:])  # drop per-trace headers
    del raw
    if fmt == 1:                                     # IBM float32
        return _ibm_to_ieee(data.view(">u4").reshape(n_traces, n_samples))
    if fmt == 5:                                     # IEEE float32
        return data.view(">f4").reshape(n_traces, n_samples).astype(np.float32, copy=False)
    if fmt == 6:                                     # IEEE float64
        return data.view(">f8").reshape(n_traces, n_samples).astype(np.float32)
    if fmt == 2:                                     # int32
        return data.view(">i4").reshape(n_traces, n_samples).astype(np.float32)
    if fmt == 3:                                     # int16
        return data.view(">i2").reshape(n_traces, n_samples).astype(np.float32)
    return data.view("i1").reshape(n_traces, n_samples).astype(np.float32)  # int8


# --------------------------------------------------------------- dispatch
def read_model(
    path: Path,
    *,
    shape: Sequence[int] | None = None,
    dtype: str = "float32",
    order: str = "C",
) -> np.ndarray:
    """Load a model file into a float32 ndarray, dispatching by extension.

    ``.segy/.sgy`` go through :func:`segy_to_array`; ``.npy/.npz`` through
    ``numpy``; ``.bin/.raw`` are raw binary (need ``shape``). Numpy-only, no
    external parser dependency.

    For SEG-Y, ``segy_to_array`` returns ``(n_traces, n_samples) = (nx, nz)``;
    this is **transposed** to ``(nz, nx)`` (never flat-reshaped — that would
    scramble the grid). If ``shape`` is given it is used as a strict
    ``(nz, nx)`` check and a mismatch raises, so a wrong assumed grid fails
    loudly instead of returning garbage.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext in _SEGY_SUFFIXES:
        arr = segy_to_array(path).T  # (n_traces, n_samples) -> (nz, nx)
        if shape is not None and tuple(arr.shape) != tuple(shape):
            raise ValueError(
                f"{path.name}: SEG-Y grid {arr.shape} != expected {tuple(shape)}"
            )
        return np.ascontiguousarray(arr, dtype="float32")
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        with np.load(path) as z:
            arr = z[list(z.keys())[0]]
    elif ext in (".bin", ".raw"):
        if shape is None:
            raise ValueError(f"{path.name}: raw binary load requires shape=...")
        arr = np.fromfile(path, dtype=np.dtype(dtype)).reshape(tuple(shape), order=order)
    else:
        raise ValueError(f"{path.name}: unsupported model format {ext!r}")
    return np.asarray(arr).astype("float32", copy=False)


__all__ = [
    "gunzip",
    "extract_tar_member",
    "extract_nested_tar_segy",
    "extract_zip_member",
    "segy_to_array",
    "read_model",
    "normalize_downsample",
    "decimate",
]
