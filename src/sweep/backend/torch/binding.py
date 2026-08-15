"""Compiled PyTorch CUDA binding capability helpers.

``sweep._C`` is JIT-compiled from source on first use (see ``sweep/_jit.py``), so a
plain ``import sweep._C`` always succeeds regardless of whether the compile can or
did happen. These helpers therefore report the real state without triggering a
compile.
"""

import os


def is_available() -> bool:
    """True when the compiled ``sweep._C`` backend is **usable** — i.e. PyTorch,
    a CUDA GPU and a suitable ``nvcc`` (>=12.4) are present, so it can be (or
    already is) JIT-compiled. Does NOT trigger the compile."""
    try:
        from sweep import _jit
        return _jit.can_build()[0]
    except Exception:
        return False


def is_compiled() -> bool:
    """True when the backend is already built — compiled in this process, or a
    cached ``.so`` from a previous run — so the first ``impl='c'`` use is instant."""
    try:
        from sweep import _jit
        if _jit._module is not None:
            return True
        from torch.utils import cpp_extension
        build_dir = cpp_extension._get_build_directory("sweep_C", verbose=False)
        return os.path.exists(os.path.join(build_dir, "sweep_C.so"))
    except Exception:
        return False


def diagnostics() -> dict:
    """Diagnostics for the compiled backend — usable / why-not / nvcc / built."""
    try:
        from sweep import _jit
        usable, reason = _jit.can_build()
        return {
            "usable": usable,             # can impl='c' be used (built now / on first use)?
            "reason": reason,             # explanation when usable is False
            "cuda_home": _jit._find_cuda_home(),
            "already_compiled": is_compiled(),
        }
    except Exception as exc:  # pragma: no cover
        return {"usable": False, "reason": f"{type(exc).__name__}: {exc}",
                "cuda_home": None, "already_compiled": False}


__all__ = ["diagnostics", "is_available", "is_compiled"]
