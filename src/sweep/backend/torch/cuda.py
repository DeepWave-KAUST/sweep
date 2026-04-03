"""CUDA capability helpers for the PyTorch backend."""


def is_available():
    """Return ``True`` when PyTorch reports CUDA support is available."""
    try:
        import torch
    except Exception:
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


__all__ = ["is_available"]
