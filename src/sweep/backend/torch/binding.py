"""Capability helpers for sweep's CUDA backend (``sweep._C``, ``impl='c'``).

``sweep._C`` reaches a process one of three ways.  By default it is the ctypes
layer (``sweep.backend.c``) over the prebuilt core the wheel ships, loaded on
first use (see ``sweep/backend/c/jit.py``), so a plain ``import sweep._C``
always succeeds and says nothing about whether the core can run here.  ``SWEEP_JIT_FULL=1`` makes
it the compiled pybind shim instead, JIT-compiled against your torch on first
use -- the developer path.  And a wheel built with ``SWEEP_BUILD_CUDA=1`` -- or
``setup.py build_ext --inplace`` -- ships a real compiled extension, which
needs nothing at run time.  These helpers report the real state across all
three without loading or compiling anything.
"""

from __future__ import annotations

import os
import sys


def _jit_full() -> bool:
    try:
        from sweep.backend.c import jit
        return jit.jit_full()
    except Exception:
        return False


def _core_loaded() -> bool:
    """Whether the ctypes loader already holds the core library -- asked
    without importing it (never imported: nothing is loaded).
    ``sweep.backend.c.loader._lib`` is its module-level cache, set by
    ``loader.core_lib()``."""
    loader = sys.modules.get("sweep.backend.c.loader")
    return loader is not None and getattr(loader, "_lib", None) is not None


def is_available() -> bool:
    """True when the ``sweep._C`` backend is **usable** here -- an ahead-of-time
    extension is on disk, or PyTorch, a CUDA GPU and a CUDA core (the wheel's
    prebuilt one, nothing to compile; else a suitable ``nvcc`` -- torch's CUDA
    major, >= 12.4 for CUDA 12, >= 12.8 for a Blackwell target, or a CUDA 13
    nvcc -- to build one) are present.  Does NOT load or compile anything."""
    try:
        from sweep import is_torch_binding_available
        return is_torch_binding_available()
    except Exception:
        return False


def is_compiled() -> bool:
    """True when the first ``impl='c'`` use costs nothing: an ahead-of-time
    extension; the ctypes shim already holding the core library in this
    process; or, under ``SWEEP_JIT_FULL``, a module compiled in this process
    or a cached ``sweep_C.so`` from a previous JIT run."""
    try:
        from sweep import _prebuilt_binding_present
        if _prebuilt_binding_present():
            return True
        if not _jit_full():
            return _core_loaded()
        from sweep.backend.c import jit
        if jit._module is not None:
            return True
        from torch.utils import cpp_extension
        build_dir = cpp_extension._get_build_directory("sweep_C", verbose=False)
        return os.path.exists(os.path.join(build_dir, "sweep_C.so"))
    except Exception:
        return False


def diagnostics() -> dict:
    """Diagnostics for the compiled backend -- usable / why-not / nvcc / built.

    ``shim`` names which ``sweep._C`` this process gets: ``"ctypes"`` (the
    default -- ``sweep.backend.c`` over the prebuilt core, nothing compiles) or
    ``"pybind"`` (the compiled developer shim: ``SWEEP_JIT_FULL=1``, or an
    ahead-of-time extension, which shadows the ctypes layer).
    ``prebuilt`` and ``shipped_core`` explain the otherwise confusing pairs:
    with an ahead-of-time extension, or with the wheel's prebuilt CUDA core
    fitting this torch and GPU, the backend is usable even though ``cuda_home``
    is None, because nothing needs nvcc.  ``shipped_core`` carries ``path``
    and ``reason`` (the core this process would use, or why none), ``tag``
    (the ``lib/cu<major>/`` drawer torch's CUDA major points at) and
    ``available`` (the sorted drawers the install holds a core in, e.g.
    ``["cu12", "cu13"]``, so a mismatch between the two is visible).
    """
    shim = "pybind" if _jit_full() else "ctypes"
    try:
        from sweep import _prebuilt_binding_present
        from sweep.backend.c import jit
        prebuilt = _prebuilt_binding_present()
        if prebuilt:
            shim = "pybind"   # a compiled sweep._C is the pybind shim, whatever SWEEP_JIT_FULL says
        can_jit, reason = jit.can_build()
        return {
            "usable": prebuilt or can_jit,   # can impl='c' be used at all?
            "reason": "ok (pre-built extension)" if prebuilt else reason,
            "shim": shim,
            "cuda_home": jit._find_cuda_home(),
            "already_compiled": is_compiled(),
            "prebuilt": prebuilt,            # compiled ahead of time, no toolkit needed
            "shipped_core": jit.shipped_core_info(),   # {path, reason, tag, available}
        }
    except Exception as exc:  # pragma: no cover
        return {"usable": False, "reason": f"{type(exc).__name__}: {exc}", "shim": shim,
                "cuda_home": None, "already_compiled": False, "prebuilt": False,
                "shipped_core": {"path": None, "reason": "not probed", "tag": "",
                                 "available": []}}


__all__ = ["diagnostics", "is_available", "is_compiled"]
