"""Compiled PyTorch CUDA binding capability helpers."""

from importlib import import_module


def is_available():
    """Return ``True`` when the compiled ``sweep._C`` binding is importable."""
    try:
        import_module('torch')
        import_module("sweep._C")
    except Exception:
        return False
    return True


def diagnostics():
    """Return a small diagnostics dictionary for compiled binding availability."""
    return {
        "binding_importable": is_available(),
    }


__all__ = ["diagnostics", "is_available"]
