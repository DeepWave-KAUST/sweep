"""nvCOMP for ``BoundarySaving(compress='bitcomp')``: find the library, hand it
to the core once per process, and answer the sizing queries the propagator
needs to allocate the compression workspace.

The library is a runtime dependency of the compress option only: the core is
built without nvCOMP headers and ``dlopen``s it (csrc/cuda/common/boundary/
nvcomp_shim.cuh).  Where it comes from, in order:

* ``SWEEP_NVCOMP_LIB`` -- the path of a ``libnvcomp.so.6`` (a machine-specific
  location, hence an environment variable and not a BoundarySaving field);
* the ``nvidia-nvcomp-cu12`` wheel (``import nvidia.libnvcomp``), whose
  ``lib64/libnvcomp.so.6`` is what pip installed.

Nothing falls back: a missing library raises, since a run that was asked to
compress and silently did not would report a different memory footprint than
it had.
"""
from __future__ import annotations

import ctypes
import os

from .loader import _ERR_CAP, _raise_core, any_core_lib, core_lib, kind_for_device

_LIB_NAME = "libnvcomp.so.6"


def find_library() -> str:
    """The ``libnvcomp.so.6`` to load, or raise ImportError saying how to get one."""
    env = os.environ.get("SWEEP_NVCOMP_LIB", "").strip()
    if env:
        if not os.path.isfile(env):
            raise ImportError(f"SWEEP_NVCOMP_LIB={env} is not a file")
        return env
    try:
        import nvidia.libnvcomp as pkg
    except ImportError as exc:
        raise ImportError(
            "BoundarySaving(compress='bitcomp') needs nvCOMP 6: `pip install "
            "nvidia-nvcomp-cu12` (the nvidia.libnvcomp package), or point "
            f"SWEEP_NVCOMP_LIB at a {_LIB_NAME}.") from exc
    roots = list(getattr(pkg, "__path__", [])) or [os.path.dirname(pkg.__file__)]
    for root in roots:
        for sub in ("lib64", "lib"):
            cand = os.path.join(root, sub, _LIB_NAME)
            if os.path.isfile(cand):
                return cand
    raise ImportError(
        f"nvidia.libnvcomp is installed at {roots} but holds no {sub}/{_LIB_NAME}; "
        "this layer is written against nvCOMP 6 (nvidia-nvcomp-cu12 >= 6).")


def _lib(device=None):
    return core_lib(kind_for_device(device)) if device is not None else any_core_lib()


def ensure_loaded(device=None) -> str:
    """Load nvCOMP into the core that serves ``device`` (once); returns the path."""
    lib = _lib(device)
    path = find_library()
    if lib.sweep_nvcomp_loaded():
        return path
    err = ctypes.create_string_buffer(_ERR_CAP)
    rc = lib.sweep_nvcomp_load(path.encode(), err, _ERR_CAP)
    if rc != 0:
        _raise_core(err, "nvcomp load", rc)
    return path


def piece_bytes(device=None) -> int:
    """Bytes of one uncompressed piece (the core's constant)."""
    return int(_lib(device).sweep_nvcomp_piece_bytes())


def max_piece_out(elem_size: int, device=None) -> int:
    """8-aligned bound on one compressed piece for a ring of ``elem_size``-byte elements."""
    lib = _lib(device)
    err = ctypes.create_string_buffer(_ERR_CAP)
    out = int(lib.sweep_nvcomp_max_piece_out(int(elem_size), err, _ERR_CAP))
    if out < 0:
        _raise_core(err, "nvcomp max piece size", -1)
    return out


def decompress_temp_bytes(max_pieces: int, device=None) -> int:
    lib = _lib(device)
    err = ctypes.create_string_buffer(_ERR_CAP)
    out = int(lib.sweep_nvcomp_decompress_temp_bytes(int(max_pieces), err, _ERR_CAP))
    if out < 0:
        _raise_core(err, "nvcomp decompress temp size", -1)
    return out
