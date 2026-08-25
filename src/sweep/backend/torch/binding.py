"""Compiled PyTorch CUDA binding capability helpers.

``sweep._C`` reaches a process one of two ways. Normally it is JIT-compiled from
source on first use (see ``sweep/_jit.py``), so a plain ``import sweep._C``
always succeeds and says nothing about whether the compile can happen. But a
wheel built with ``SWEEP_BUILD_CUDA=1`` — or ``setup.py build_ext --inplace`` —
ships a real compiled extension instead, and then no CUDA toolkit is needed at
run time at all. These helpers report the real state across both, without
triggering a compile.
"""

import os


def is_available() -> bool:
    """True when the compiled ``sweep._C`` backend is **usable** — either it is
    already compiled on disk (needing no toolkit), or PyTorch, a CUDA GPU and a
    suitable ``nvcc`` (>=12.4) are present so it can be JIT-compiled on first
    use. Does NOT trigger the compile."""
    try:
        from sweep import is_torch_binding_available
        return is_torch_binding_available()
    except Exception:
        return False


def is_compiled() -> bool:
    """True when the backend is already built — an ahead-of-time extension, one
    compiled in this process, or a cached ``.so`` from a previous JIT run — so
    the first ``impl='c'`` use is instant."""
    try:
        from sweep import _prebuilt_binding_present
        if _prebuilt_binding_present():
            return True
        from sweep import _jit
        if _jit._module is not None:
            return True
        from torch.utils import cpp_extension
        build_dir = cpp_extension._get_build_directory("sweep_C", verbose=False)
        return os.path.exists(os.path.join(build_dir, "sweep_C.so"))
    except Exception:
        return False


def diagnostics() -> dict:
    """Diagnostics for the compiled backend — usable / why-not / nvcc / built.

    ``prebuilt`` is the one that explains an otherwise confusing pair: with an
    ahead-of-time extension the backend is usable even though ``cuda_home`` is
    None, because nothing has to be compiled.
    """
    try:
        from sweep import _jit, _prebuilt_binding_present
        prebuilt = _prebuilt_binding_present()
        can_jit, reason = _jit.can_build()
        return {
            "usable": prebuilt or can_jit,   # can impl='c' be used at all?
            "reason": "ok (pre-built extension)" if prebuilt else reason,
            "cuda_home": _jit._find_cuda_home(),
            "already_compiled": is_compiled(),
            "prebuilt": prebuilt,            # compiled ahead of time, no toolkit needed
        }
    except Exception as exc:  # pragma: no cover
        return {"usable": False, "reason": f"{type(exc).__name__}: {exc}",
                "cuda_home": None, "already_compiled": False, "prebuilt": False}


__all__ = ["diagnostics", "is_available", "is_compiled"]
