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
    # Dispatch order matters under torch.compile + CUDA:
    #   - ``np.ndarray`` first short-circuits the eager numpy path
    #     before touching torch; numpy >= 1.25 exposes ``.device`` per
    #     the Array API so a generic duck-type check on ``first`` would
    #     mis-route plain ndarrays into ``torch.cat``.
    #   - ``torch.cat`` is then attempted unconditionally rather than
    #     gated by ``isinstance(first, torch.Tensor)``. Under
    #     ``torch.compile`` Dynamo can specialise such an isinstance
    #     guard into the numpy branch and bake "can't convert cuda
    #     tensor to numpy" into the compiled graph; calling
    #     ``torch.cat`` directly lets Dynamo recognise the tensor op
    #     and emit ``aten.cat`` in the FX graph. For non-torch inputs
    #     ``torch.cat`` raises ``TypeError`` and we fall through.
    first = arrays[0]
    if isinstance(first, np.ndarray):
        return np.concatenate(arrays, axis=axis)
    try:
        import torch

        return torch.cat(arrays, dim=axis)
    except (ImportError, TypeError):
        pass
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


def blend_top_n(top, bottom, n, axis):
    """Use ``top`` for the first ``n`` rows along ``axis``, ``bottom`` for the rest.
    Used for near-surface order reduction (low-order z-derivative in the top band)."""
    if n <= 0:
        return bottom
    return _concat([_slice_axis(top, axis, None, n), _slice_axis(bottom, axis, n, None)], axis=axis)


def overwrite_top_row(base, repl, halo, axis):
    """Return ``base`` with its free-surface row (index ``halo``) replaced by the
    corresponding row of ``repl`` (same shape). Backend-agnostic (slice+concat)."""
    row = 0 if halo <= 0 else halo
    pre = _slice_axis(base, axis, None, row) if row > 0 else None
    surf = _slice_axis(repl, axis, row, row + 1)
    post = _slice_axis(base, axis, row + 1, None)
    parts = ([pre] if pre is not None else []) + [surf, post]
    return _concat(parts, axis=axis)


def get_o2_pd(pd):
    """Cached order-2 staggered-derivative twin of ``pd`` (near-surface order
    reduction). The order-2 image derivative is unconditionally stable at the
    free surface; blending it into the top band tames the high-order
    FS-∩-CPML-corner instability without disturbing the interior accuracy."""
    p2 = getattr(pd, "_o2", None)
    if p2 is None:
        from sweep.operators.general import StaggeredDerivative
        from .utils import to_backend as _tb

        p2 = StaggeredDerivative(2, pd.device, pd.backend, ndim=pd.ndim)
        p2.to_backend(_tb)
        p2._spacing = pd._spacing
        pd._o2 = p2
    return p2


def fs_deriv(field, full_op, o2_op, halo, odd, axis, n_o2):
    """Free-surface z-derivative with optional near-surface order reduction:
    order-2 image-derivative for the top ``n_o2`` rows, full-order below."""
    full = top_free_surface_derivative(field, full_op, halo, odd=odd, axis=axis)
    if n_o2 <= 0:
        return full
    o2 = top_free_surface_derivative(field, o2_op, halo, odd=odd, axis=axis)
    return blend_top_n(o2, full, n_o2, axis)


def near_surface_o2_count(top_halo, env_value):
    """Map the ``SWEEP_FS_NEARSURF_O2`` env string to a top-band row count.
    ``"0"`` disables; ``"1"`` (default) uses ``2*top_halo`` rows; any other
    integer string is taken literally."""
    if env_value == "0":
        return 0
    if env_value == "1":
        return 2 * top_halo
    return int(env_value)


# -----------------------------------------------------------------------------
# Per-edge (axis, side) generalisation of the flat top-only image method.
#
# ``side='low'`` is the historical top-of-axis surface (reflect about index
# ``halo``); ``side='high'`` is the far end (reflect about ``n-1-halo``).  The
# high side is obtained by flipping the axis, applying the low-side operation,
# and flipping back — geometrically exact and reusing the audited low-side code.
# Parities (``odd``) and the forward/backward derivative operator are unchanged
# between the two ends of an axis (the free-surface BC is the same at both);
# only the mirror location moves.  ``sides`` lists let a single axis carry a
# free surface at both ends (a closed / waveguide box) with one derivative call.
# -----------------------------------------------------------------------------


def sides_from_fs_faces_2d(fs_faces):
    """Split a 2-D ``fs_faces`` tuple ``(z_lo, z_hi, x_lo, x_hi)`` = (top,
    bottom, left, right) into ``(z_sides, x_sides)`` lists of ``'low'``/``'high'``."""
    z_sides = []
    if fs_faces[0]:
        z_sides.append("low")
    if fs_faces[1]:
        z_sides.append("high")
    x_sides = []
    if fs_faces[2]:
        x_sides.append("low")
    if fs_faces[3]:
        x_sides.append("high")
    return z_sides, x_sides


def extend_free_surface(u, halo, odd, axis, side="low"):
    """Image-method mirror at one end of ``axis``.  ``side='low'`` is the
    historical :func:`extend_top_free_surface`; ``side='high'`` mirrors about
    ``n-1-halo`` via flip-apply-flip (exact, reuses the low-side code)."""
    if halo <= 0:
        return u
    if side == "low":
        return extend_top_free_surface(u, halo, odd, axis)
    uf = _flip(u, axis)
    ef = extend_top_free_surface(uf, halo, odd, axis)
    return _flip(ef, axis)


def blend_side_n(near, far, n, axis, side="low"):
    """Use ``near`` for the ``n`` cells nearest the ``side`` surface, ``far`` for
    the rest (near-surface order reduction at either end of the axis)."""
    if n <= 0:
        return far
    if side == "low":
        return blend_top_n(near, far, n, axis)
    ax = axis if axis >= 0 else far.ndim + axis
    nsz = far.shape[ax]
    return _concat(
        [_slice_axis(far, ax, None, nsz - n), _slice_axis(near, ax, nsz - n, None)],
        axis=ax,
    )


def fs_deriv_1side(field, op, op_swapped, halo, odd, axis, side):
    """Free-surface staggered derivative near ONE surface.

    ``side='low'`` mirrors about index ``halo`` and applies ``op`` (the
    historical top path).  ``side='high'`` uses the flip identity so the
    STAGGERED operator direction is correct at the far surface::

        high = -flip( op_swapped( extend_top(flip(field)) ) )

    because flipping the axis turns a forward staggered difference into a
    (negated) backward one — ``F∘op_swapped∘F = -op`` — so this reproduces
    ``op(field)`` in the interior while enforcing the free surface at the high
    end.  ``op_swapped`` is the opposite-direction operator (z_forward↔z_backward)."""
    if side == "low":
        return op(extend_top_free_surface(field, halo, odd, axis))
    ff = _flip(field, axis)
    d = op_swapped(extend_top_free_surface(ff, halo, odd, axis))
    return -_flip(d, axis)


def fs_deriv_axis(field, op, op_swapped, halo, odd, axis, sides):
    """Full-order FS derivative honouring every surface in ``sides``.  Each
    single-side derivative equals the plain derivative away from its own
    surface, so a two-sided (closed / waveguide) axis is stitched from the low
    result (correct at the low end) with the high end's band overwritten by the
    high result (correct at the high end)."""
    if not sides:
        return op(field)
    if sides == ["low"] or sides == ["high"]:
        return fs_deriv_1side(field, op, op_swapped, halo, odd, axis, sides[0])
    d_low = fs_deriv_1side(field, op, op_swapped, halo, odd, axis, "low")
    d_high = fs_deriv_1side(field, op, op_swapped, halo, odd, axis, "high")
    ax = axis if axis >= 0 else field.ndim + axis
    n = d_low.shape[ax]
    band = 2 * halo
    return _concat(
        [_slice_axis(d_low, ax, None, n - band), _slice_axis(d_high, ax, n - band, None)],
        axis=ax,
    )


def fs_deriv_edges(field, op, op_swapped, o2, o2_swapped, halo, odd, axis, sides, n_o2):
    """FS derivative honouring every surface in ``sides`` with optional
    near-surface order-2 reduction blended in at each side.  With
    ``sides=['low']`` this is bit-for-bit :func:`fs_deriv`."""
    full = fs_deriv_axis(field, op, op_swapped, halo, odd, axis, sides)
    if n_o2 <= 0 or o2 is None:
        return full
    o2full = fs_deriv_axis(field, o2, o2_swapped, halo, odd, axis, sides)
    out = full
    for side in sides:
        out = blend_side_n(o2full, out, n_o2, axis, side)
    return out


def zero_surface_row(u, halo, axis, side="low"):
    """Zero the single surface row/column: index ``halo`` for ``side='low'``
    (== :func:`zero_top_row`), ``n-1-halo`` for ``side='high'``."""
    if side == "low":
        return zero_top_row(u, halo, axis)
    ax = axis if axis >= 0 else u.ndim + axis
    n = u.shape[ax]
    row = n - 1 - (halo if halo > 0 else 0)
    out = _zero_axis_index(u, ax, row)
    if out is not None:
        return out
    parts = [_slice_axis(u, ax, None, row), _slice_axis(u, ax, row, row + 1) * 0]
    if row + 1 < n:
        parts.append(_slice_axis(u, ax, row + 1, None))
    return _concat(parts, axis=ax)


def overwrite_surface_row(base, repl, halo, axis, side="low"):
    """Replace the surface row/column of ``base`` with that of ``repl``: index
    ``halo`` for ``side='low'`` (== :func:`overwrite_top_row`), ``n-1-halo`` for
    ``side='high'``."""
    if side == "low":
        return overwrite_top_row(base, repl, halo, axis)
    ax = axis if axis >= 0 else base.ndim + axis
    n = base.shape[ax]
    row = n - 1 - (halo if halo > 0 else 0)
    parts = [_slice_axis(base, ax, None, row), _slice_axis(repl, ax, row, row + 1)]
    if row + 1 < n:
        parts.append(_slice_axis(base, ax, row + 1, None))
    return _concat(parts, axis=ax)


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


def overwrite_at_topo(base, repl, iz_surf, axis):
    """Per-column version of :func:`overwrite_top_row`: set the surface row
    (``z == iz_surf[ix]``) of ``base`` to the corresponding entry of ``repl``;
    all other cells are left untouched.  Used for the Robertsson surface-sigma_xx
    modified coefficient under irregular topography."""
    import torch

    z, surf, ax, nz = _broadcast_topo(base, iz_surf, axis)
    mask = (z == surf).expand_as(base)
    return torch.where(mask, repl, base)


def fs_sxx_correction(sxx, sxx_pre, dt, c_surf, vx_x, top_halo, axis=-2, topo_rows=None):
    """Robertsson free-surface sigma_xx correction, shared across the staggered
    elastic equations (Elastic, DASMu, ElasticVR).

    At a free surface sigma_zz = 0 forces the surface-row sigma_xx to
    ``sxx_pre + dt * c_surf * vx_x`` -- the image-method mirror alone leaves it
    ~mu/lambda too large.  ``c_surf`` is the modified coefficient and ``vx_x`` the
    matching x velocity-gradient, both expressed in each caller's own variables:
      * Elastic / DASMu : c_surf = 4 mu (lam+mu)/(lam+2mu),  vx_x = d vx/dx
      * ElasticVR (native, momentum px = rho*vx, no explicit rho needed):
                          c_surf = 4 vs^2 (vp^2 - vs^2)/vp^2,  vx_x = d px/dx
    Overwrites the flat top row (``overwrite_top_row``) or, when ``topo_rows`` is
    given, the per-column surface row along topography (``overwrite_at_topo``).
    Env-gated by ``SWEEP_FS_MOD_SXX`` (default on); returns ``sxx`` unchanged off.
    """
    import os as _os

    if _os.environ.get("SWEEP_FS_MOD_SXX", "1") != "1":
        return sxx
    sxx_surf = sxx_pre + dt * c_surf * vx_x
    if topo_rows is not None:
        return overwrite_at_topo(sxx, sxx_surf, topo_rows, axis=axis)
    return overwrite_top_row(sxx, sxx_surf, top_halo, axis=axis)


def zero_at_topo(u, iz_surf, axis):
    """Zero exactly the surface row per column (one cell per ``ix`` at
    ``z == iz_surf[ix]``).

    For staggered first-order elastic FS the image method enforces
    ``σ_zz = σ_xz = 0`` at the surface row; this is the topo
    generalisation of :func:`zero_top_row` (which zeros the constant row
    ``halo``). Cells above and below the surface are untouched — the
    mirror operation done one step earlier already replaces air cells
    next time the derivative is taken."""
    import torch

    z, surf, ax, nz = _broadcast_topo(u, iz_surf, axis)
    mask = (z == surf).expand_as(u)
    return u.masked_fill(mask, 0.0)
