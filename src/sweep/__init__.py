"""Top-level package helpers for sweep."""

from importlib import import_module
from importlib.util import find_spec


_LAZY_SUBMODULES = {
    "backend",
    "equations",
    "jax",
    "memory",
    "operators",
    "propagator",
    "receivers",
    "signal",
    "sources",
    "torch",
    "utils",
}


def is_torch_binding_available():
    """Return True when PyTorch and the compiled ``sweep._C`` binding are usable."""
    if find_spec("torch") is None:
        return False

    try:
        import torch
    except Exception:
        return False

    if not torch.cuda.is_available():
        return False

    try:
        import_module("sweep._C")
    except Exception:
        return False

    return True


def __getattr__(name):
    if name in _LAZY_SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(set(globals()) | _LAZY_SUBMODULES)


__all__ = ["is_torch_binding_available", *_LAZY_SUBMODULES]
