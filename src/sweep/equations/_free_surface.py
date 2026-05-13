import numpy as np


def _flip(u, axis):
    module = np
    if hasattr(u, "flip"):
        try:
            return u.flip((axis,))
        except TypeError:
            pass
    return module.flip(u, axis=axis)


def _concat(arrays, axis):
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


def _slice_axis(u, axis, start=None, stop=None):
    axis = axis if axis >= 0 else u.ndim + axis
    slices = [slice(None)] * u.ndim
    slices[axis] = slice(start, stop)
    return u[tuple(slices)]


def _zero_axis_index(u, axis, index):
    if hasattr(u, "clone") and hasattr(u, "device") and hasattr(u, "dtype"):
        axis = axis if axis >= 0 else u.ndim + axis
        out = u.clone()
        slices = [slice(None)] * u.ndim
        slices[axis] = index
        out[tuple(slices)] = 0
        return out
    return None


def extend_top_free_surface(u, halo, odd, axis):
    if halo <= 0:
        return u
    ghost = _flip(_slice_axis(u, axis, halo + 1, 2 * halo + 1), axis=axis)
    if odd:
        ghost = -ghost
    return _concat([ghost, _slice_axis(u, axis, halo, None)], axis=axis)


def top_free_surface_derivative(u, deriv, halo, odd, axis):
    return deriv(extend_top_free_surface(u, halo, odd, axis))


def extend_top_free_surface_cell_centered(u, halo, odd, axis):
    if halo <= 0:
        return u
    ghost = _flip(_slice_axis(u, axis, halo, 2 * halo), axis=axis)
    if odd:
        ghost = -ghost
    return _concat([ghost, _slice_axis(u, axis, halo, None)], axis=axis)


def top_free_surface_cell_derivative(u, deriv, halo, odd, axis):
    return deriv(extend_top_free_surface_cell_centered(u, halo, odd, axis))


def zero_top_row(u, halo, axis):
    row = 0 if halo <= 0 else halo
    out = _zero_axis_index(u, axis, row)
    if out is not None:
        return out
    if halo <= 0:
        return _concat([_slice_axis(u, axis, 0, 1) * 0, _slice_axis(u, axis, 1, None)], axis=axis)
    return _concat(
        [
            _slice_axis(u, axis, None, halo),
            _slice_axis(u, axis, halo, halo + 1) * 0,
            _slice_axis(u, axis, halo + 1, None),
        ],
        axis=axis,
    )
