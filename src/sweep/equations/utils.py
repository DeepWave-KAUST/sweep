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


def zero_top_halo(field, halo, axis):
    if halo <= 0:
        return field
    ndim = len(field.shape)
    axis = axis if axis >= 0 else ndim + axis
    top_slice = [slice(None)] * ndim
    rest_slice = [slice(None)] * ndim
    top_slice[axis] = slice(0, halo)
    rest_slice[axis] = slice(halo, None)
    return backend_concat(
        [field[tuple(top_slice)] * 0, field[tuple(rest_slice)]],
        axis=axis,
    )


def zero_top_halo_fields(fields, halo, axis):
    return tuple(zero_top_halo(field, halo, axis=axis) for field in fields)
