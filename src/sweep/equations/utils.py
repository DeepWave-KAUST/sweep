import numpy as np


def to_backend(arr, backend, device=None):
    if backend == 'torch':
        import torch
        if isinstance(arr, list):
            return [torch.from_numpy(a).float().to(device) for a in arr]
        return torch.from_numpy(arr).float().to(device)
    elif backend == 'jax':
        import jax.numpy as jnp
        if isinstance(arr, list):
            return [jnp.array(a, dtype=jnp.float32) for a in arr]
        return jnp.array(arr, dtype=jnp.float32)
    elif backend == 'cuda':
        return arr.astype(np.float32)


def backend_concat(arrays, axis):
    first = arrays[0]
    if hasattr(first, "device") and hasattr(first, "dtype"):
        import torch

        return torch.cat(arrays, dim=axis)
    try:
        import jax.numpy as jnp

        if type(first).__module__.startswith("jax"):
            return jnp.concatenate(arrays, axis=axis)
    except Exception:
        pass
    return np.concatenate(arrays, axis=axis)


def zero_edge_halo(field, halo, axis, side="low"):
    """Zero the ``halo`` cells at one end of ``axis``.

    ``side='low'`` zeros ``[0, halo)`` (the historical top-of-z behaviour);
    ``side='high'`` zeros ``[n-halo, n)``.  Used to enforce a pressure-release
    (``p = 0``) free surface for the 2nd-order acoustic wavefield at any face —
    the free-surface face is padded with 0 PML so its ``halo`` band sits at the
    physical boundary and the antisymmetric image gives the reflection.
    """
    if halo <= 0:
        return field
    ndim = len(field.shape)
    axis = axis if axis >= 0 else ndim + axis
    n = field.shape[axis]
    band = slice(0, halo) if side == "low" else slice(n - halo, None)
    if hasattr(field, "clone") and hasattr(field, "device") and hasattr(field, "dtype"):
        out = field.clone()
        sl = [slice(None)] * ndim
        sl[axis] = band
        out[tuple(sl)] = 0
        return out
    # Backend-agnostic (jax / numpy) fallback: rebuild via concat with the band zeroed.
    band_slice = [slice(None)] * ndim
    band_slice[axis] = band
    if side == "low":
        rest_slice = [slice(None)] * ndim
        rest_slice[axis] = slice(halo, None)
        parts = [field[tuple(band_slice)] * 0, field[tuple(rest_slice)]]
    else:
        rest_slice = [slice(None)] * ndim
        rest_slice[axis] = slice(0, n - halo)
        parts = [field[tuple(rest_slice)], field[tuple(band_slice)] * 0]
    return backend_concat(parts, axis=axis)


def zero_top_halo(field, halo, axis):
    """Backward-compatible alias: zero the low (top-of-z) ``halo`` band."""
    return zero_edge_halo(field, halo, axis, side="low")


def zero_top_halo_fields(fields, halo, axis):
    return tuple(zero_top_halo(field, halo, axis=axis) for field in fields)


def zero_edge_halo_fields(fields, halo, axis, side="low"):
    return tuple(zero_edge_halo(field, halo, axis=axis, side=side) for field in fields)
