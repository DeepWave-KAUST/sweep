"""CUDA capability helpers for the JAX backend."""


def is_available():
    """Return ``True`` when JAX can see at least one GPU device."""
    try:
        import jax
    except Exception:
        return False

    try:
        return any(device.platform == "gpu" for device in jax.devices())
    except Exception:
        return False


__all__ = ["is_available"]
