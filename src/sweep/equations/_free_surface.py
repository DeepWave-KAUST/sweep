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


# -----------------------------------------------------------------------------
# Per-column (irregular topography) image-method helpers.
#
# These generalise the flat-surface mirror to a column-dependent surface row
# ``iz_surf[..., ix]`` (and ``[..., iy, ix]`` in 3D). The reflection rule is
# ``mirror_z(ix) = 2*iz_surf(ix) - z``; air cells (``z < iz_surf(ix)``) are
# overwritten by the reflected interior value, with an optional sign flip for
# z-anti-symmetric (odd-parity) fields.
#
# Torch-only for now; called only on the eager Python backend.
# -----------------------------------------------------------------------------


def _broadcast_topo(u, iz_surf, axis):
    """Build ``(z, surf, ax, nz)`` broadcasting tensors for per-column mirror.

    ``iz_surf`` must describe the surface for every non-z dim AFTER the z-axis,
    e.g. for a 2D field ``u`` of shape ``(..., nz, nx)`` with ``axis=-2`` the
    expected ``iz_surf`` shape is ``(nx,)``; for a 3D field
    ``(..., nz, ny, nx)`` with ``axis=-3`` it is ``(ny, nx)``. Leading
    batch/channel dims of ``u`` are implicitly broadcast (single surface
    shared across the batch).
    """
    import torch

    if not isinstance(u, torch.Tensor):
        raise NotImplementedError(
            "topography free-surface helpers currently support torch tensors only; "
            f"got {type(u).__name__}"
        )
    if not isinstance(iz_surf, torch.Tensor):
        iz_surf = torch.as_tensor(iz_surf, dtype=torch.long, device=u.device)

    nz = u.shape[axis]
    ax = axis if axis >= 0 else u.ndim + axis

    z_shape = [1] * u.ndim
    z_shape[ax] = nz
    z = torch.arange(nz, device=u.device).view(z_shape)

    extra_dims = u.ndim - ax - 1
    if iz_surf.ndim != extra_dims:
        raise ValueError(
            f"iz_surf.ndim={iz_surf.ndim} must equal u.ndim-z_axis-1={extra_dims} "
            f"(u.shape={tuple(u.shape)}, axis={axis}, iz_surf.shape={tuple(iz_surf.shape)})"
        )
    s_shape = [1] * u.ndim
    for i, size in enumerate(iz_surf.shape):
        s_shape[ax + 1 + i] = size
    surf = iz_surf.view(s_shape).to(dtype=torch.long, device=u.device)
    return z, surf, ax, nz


def extend_top_free_surface_topo(u, halo, odd, axis, iz_surf):
    """Per-column image-method mirror across an irregular surface.

    Parameters
    ----------
    u : torch.Tensor
        Field of shape ``(..., nz, [ny,] nx)`` on a padded grid.
    halo : int
        Stencil half-width. ``halo <= 0`` returns ``u`` unchanged.
    odd : bool
        If True the mirrored ghost is negated (vz-like, anti-symmetric fields).
    axis : int
        Z-axis position in ``u`` (e.g. ``-2`` for 2D, ``-3`` for 3D).
    iz_surf : torch.Tensor or array_like
        Long-tensor giving the surface row per column in padded-grid
        coordinates. Shape must match ``u``'s non-z dims after the z-axis;
        see :func:`_broadcast_topo`.

    Notes
    -----
    The ``halo`` argument is kept for API symmetry with
    :func:`extend_top_free_surface`; the per-column mirror itself does not
    need ``halo`` (the reflection works for any number of air cells). It is
    still honoured as the trivial early-out ``halo <= 0 -> u``.
    """
    if halo <= 0:
        return u
    import torch

    z, surf, ax, nz = _broadcast_topo(u, iz_surf, axis)
    above = z < surf
    mirror_z = torch.where(above, 2 * surf - z, z).clamp(0, nz - 1)
    mirror_z = mirror_z.expand_as(u)
    out = u.gather(ax, mirror_z)
    if odd:
        out = torch.where(above.expand_as(u), -out, out)
    return out


def top_free_surface_derivative_topo(u, deriv, halo, odd, axis, iz_surf):
    """Apply ``deriv`` to the topo-mirrored field. Mirror of
    :func:`top_free_surface_derivative` for the irregular-surface path."""
    return deriv(extend_top_free_surface_topo(u, halo, odd, axis, iz_surf))


def zero_above_topo(u, iz_surf, axis):
    """Zero cells strictly above the surface in each column (the "air"
    region). Replaces :func:`zero_top_row` on the topo path."""
    import torch

    z, surf, ax, nz = _broadcast_topo(u, iz_surf, axis)
    mask = (z < surf).expand_as(u)
    return u.masked_fill(mask, 0.0)
