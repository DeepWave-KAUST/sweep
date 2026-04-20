"""Top-level package helpers for sweep."""

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path


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


def _extend_package_path_with_build_outputs():
    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parents[1]
    build_dir = repo_root / "build"

    if not build_dir.exists():
        return

    package_path = globals().get("__path__")
    if package_path is None:
        return

    for candidate in sorted(build_dir.glob("lib*/sweep")):
        candidate_str = str(candidate)
        if candidate.is_dir() and candidate_str not in package_path:
            package_path.append(candidate_str)


_extend_package_path_with_build_outputs()


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
