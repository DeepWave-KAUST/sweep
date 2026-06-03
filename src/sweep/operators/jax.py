import jax
import jax.numpy as jnp
from jax import lax


def _resolve_spacing_for_axis(h, axis, ndim):
    if hasattr(h, "ndim"):
        if h.ndim == 0:
            return h
        spatial_ndim = h.shape[0]
    elif isinstance(h, (tuple, list)):
        spatial_ndim = len(h)
    else:
        return h

    normalized_axis = axis if axis >= 0 else ndim + axis
    spatial_axis = normalized_axis - (ndim - spatial_ndim)
    if spatial_axis < 0 or spatial_axis >= spatial_ndim:
        raise ValueError(
            f"Axis {axis} is incompatible with spacing of length {spatial_ndim} for tensor ndim={ndim}."
        )
    return h[spatial_axis]


def _to_nchw(u):
    if u.ndim == 2:
        return u[None, None, ...], lambda x: x[0, 0]
    if u.ndim == 3:
        return u[:, None, ...], lambda x: x[:, 0]
    if u.ndim == 4:
        return u, lambda x: x
    raise ValueError(f"Expected 2D/3D/4D input for gradient kernel, got shape {u.shape}")


def _zero_halo(out, padding):
    if isinstance(padding, int):
        pad_z = pad_x = padding
    else:
        pad_z, pad_x = padding
    if pad_z > 0:
        out = out.at[..., :pad_z, :].set(0)
        out = out.at[..., -pad_z:, :].set(0)
    if pad_x > 0:
        out = out.at[..., :, :pad_x].set(0)
        out = out.at[..., :, -pad_x:].set(0)
    return out

@jax.jit
def separable_d2_2d(u, k1d, hz=1.0, hx=1.0):
    """Separable per-axis 2nd derivatives of a 2-D wavefield (JAX).

    Naming: ``d2`` = second derivative (∂²); ``2d`` = 2-D wavefield.
    Returns the **components**, not their sum — sum them yourself for an
    isotropic Laplacian or use :func:`laplacian_2d` for the shortcut.

    Args:
        u: Input wavefield, shape ``(B, 1, nz, nx)``.
        k1d: 1-D kernel of length ``2M+1``.
        hz, hx: Grid spacings along z and x.

    Returns:
        ``(d2u_dz2, d2u_dx2)`` — two arrays of the same shape as ``u``.
    """
    # k1d = jnp.asarray(k1d, dtype=u.dtype)
    kz = k1d[:, None, None, None]  # (k,1,1,1)
    kx = k1d[None, :, None, None]  # (1,k,1,1)

    d2z = lax.conv_general_dilated(
        u, kz,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "HWIO", "NCHW"),
    )

    d2x = lax.conv_general_dilated(
        u, kx,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "HWIO", "NCHW"),
    )

    return d2z / (hz * hz), d2x / (hx * hx)


def separable_d2_3d(u, k1d, hz, hy, hx):
    """Separable per-axis 2nd derivatives of a 3-D wavefield (JAX).

    Naming: ``d2`` = second derivative (∂²); ``3d`` = 3-D wavefield.
    Returns the **components**, not their sum — sum them yourself for an
    isotropic Laplacian or use :func:`laplacian_3d` for the shortcut.

    Args:
        u: Input wavefield, shape ``(B, 1, nz, ny, nx)``.
        k1d: 1-D kernel of length ``2M+1``.
        hz, hy, hx: Grid spacings along z, y, x.

    Returns:
        ``(d2u_dz2, d2u_dy2, d2u_dx2)`` — three arrays of the same
        shape as ``u``.
    """
    kz = k1d[None, None, :, None, None]  # (1,1,k,1,1)
    ky = k1d[None, None, None, :, None]  # (1,1,1,k,1)
    kx = k1d[None, None, None, None, :]  # (1,1,1,1,k)
    dn = jax.lax.conv_dimension_numbers(u.shape, kz.shape,
                                        ('NCDHW', 'OIDHW', 'NCDHW'))
    kwargs = {'window_strides': (1,1,1),
              'padding': 'SAME',
              'lhs_dilation': (1,1,1),
              'rhs_dilation': (1,1,1),
              'dimension_numbers': dn}
    d2z = lax.conv_general_dilated(u, kz, **kwargs)/ (hz ** 2)
    d2y = lax.conv_general_dilated(u, ky, **kwargs)/ (hy ** 2)
    d2x = lax.conv_general_dilated(u, kx, **kwargs)/ (hx ** 2)
    return d2z, d2y, d2x


@jax.jit
def laplacian_2d(u, k1d, hz=1.0, hx=1.0):
    """Isotropic 2-D Laplacian via separable kernels: ``d2z + d2x``."""
    d2z, d2x = separable_d2_2d(u, k1d, hz=hz, hx=hx)
    return d2z + d2x


@jax.jit
def laplacian_3d(u, k1d, hz=1.0, hy=1.0, hx=1.0):
    """Isotropic 3-D Laplacian via separable kernels: ``d2z + d2y + d2x``."""
    d2z, d2y, d2x = separable_d2_3d(u, k1d, hz=hz, hy=hy, hx=hx)
    return d2z + d2y + d2x


def _normalize_2d_kernel_bank(kernels):
    kernels = jnp.asarray(kernels)
    if kernels.ndim == 3:
        return kernels[:, None, :, :]
    if kernels.ndim == 4:
        return kernels
    raise ValueError(f"Expected 2D kernels with shape (K, kh, kw) or (K, 1, kh, kw), got {kernels.shape}")


def _normalize_3d_kernel_bank(kernels):
    kernels = jnp.asarray(kernels)
    if kernels.ndim == 4:
        return kernels[:, None, :, :, :]
    if kernels.ndim == 5:
        return kernels
    raise ValueError(
        f"Expected 3D kernels with shape (K, kD, kH, kW) or (K, 1, kD, kH, kW), got {kernels.shape}"
    )


@jax.jit
def apply_kernels(u, kernels):
    # u: (B, 1, H, W)
    # kernels: (K, kh, kw) or (K, 1, kh, kw)
    kernel_bank = _normalize_2d_kernel_bank(kernels)
    kernel_bank = jnp.flip(kernel_bank, axis=(-2, -1))
    _, _, kh, kw = kernel_bank.shape
    padding = ((kh // 2, kh // 2), (kw // 2, kw // 2))
    conv_out = lax.conv_general_dilated(
        lhs=u,
        rhs=kernel_bank,
        window_strides=(1, 1),
        padding=padding,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return conv_out if conv_out.shape[1] == 1 else conv_out.sum(axis=1, keepdims=True)


@jax.jit
def apply_kernels_3d(u, kernels):
    # u: (B, 1, D, H, W)
    # kernels: (K, kD, kH, kW) or (K, 1, kD, kH, kW)
    kernel_bank = _normalize_3d_kernel_bank(kernels)
    kernel_bank = jnp.flip(kernel_bank, axis=(-3, -2, -1))
    _, _, kd, kh, kw = kernel_bank.shape
    padding = ((kd // 2, kd // 2), (kh // 2, kh // 2), (kw // 2, kw // 2))
    conv_out = lax.conv_general_dilated(
        lhs=u,
        rhs=kernel_bank,
        window_strides=(1, 1, 1),
        padding=padding,
        dimension_numbers=("NCDHW", "OIDHW", "NCDHW"),
    )
    return conv_out if conv_out.shape[1] == 1 else conv_out.sum(axis=1, keepdims=True)


def gradient(u, h, axis, kernels=None):
    h_axis = _resolve_spacing_for_axis(h, axis, u.ndim)
    if kernels is not None:
        if axis not in kernels:
            raise ValueError(f"No gradient kernel configured for axis={axis}.")
        u_nchw, restore = _to_nchw(u)
        rhs = jnp.transpose(kernels[axis] / h_axis, (2, 3, 1, 0))[::-1, ::-1, :, :]
        out = lax.conv_general_dilated(
            u_nchw,
            rhs,
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NCHW', 'HWIO', 'NCHW'),
        )
        out = _zero_halo(out, (kernels[axis].shape[-2] // 2, kernels[axis].shape[-1] // 2))
        return restore(out)
    return jnp.gradient(u, h_axis, axis=axis)
