"""PyTorch backend capability helpers."""

from . import cuda


def is_available():
    """Return ``True`` when the PyTorch backend is importable."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


__all__ = ["cuda", "is_available"]
