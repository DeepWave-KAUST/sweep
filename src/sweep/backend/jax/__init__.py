"""JAX backend capability helpers."""

from . import cuda


def is_available():
    """Return ``True`` when the JAX backend is importable."""
    try:
        import jax  # noqa: F401
    except Exception:
        return False
    return True


__all__ = ["cuda", "is_available"]
