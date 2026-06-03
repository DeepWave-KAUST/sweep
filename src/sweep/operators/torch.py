import torch
import torch.nn.functional as F


def _resolve_spacing_for_axis(h, axis, ndim):
    if isinstance(h, torch.Tensor):
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
        return u.unsqueeze(0).unsqueeze(0), lambda x: x.squeeze(0).squeeze(0)
    if u.ndim == 3:
        return u.unsqueeze(1), lambda x: x.squeeze(1)
    if u.ndim == 4:
        return u, lambda x: x
    raise ValueError(f"Expected 2D/3D/4D input for gradient kernel, got shape {tuple(u.shape)}")


def _to_ncdhw(u):
    if u.ndim == 3:
        return u.unsqueeze(0).unsqueeze(0), lambda x: x.squeeze(0).squeeze(0)
    if u.ndim == 4:
        return u.unsqueeze(1), lambda x: x.squeeze(1)
    if u.ndim == 5:
        return u, lambda x: x
    raise ValueError(f"Expected 3D/4D/5D input for 3D gradient kernel, got shape {tuple(u.shape)}")


def _zero_halo(out, padding):
    if isinstance(padding, int):
        pad_z = pad_x = padding
    else:
        pad_z, pad_x = padding
    if pad_z > 0:
        out[..., :pad_z, :] = 0
        out[..., -pad_z:, :] = 0
    if pad_x > 0:
        out[..., :, :pad_x] = 0
        out[..., :, -pad_x:] = 0
    return out


def _zero_halo_3d(out, padding):
    if isinstance(padding, int):
        pad_z = pad_y = pad_x = padding
    else:
        pad_z, pad_y, pad_x = padding
    if pad_z > 0:
        out[..., :pad_z, :, :] = 0
        out[..., -pad_z:, :, :] = 0
    if pad_y > 0:
        out[..., :, :pad_y, :] = 0
        out[..., :, -pad_y:, :] = 0
    if pad_x > 0:
        out[..., :, :, :pad_x] = 0
        out[..., :, :, -pad_x:] = 0
    return out

def separable_d2_2d(u, k1d, hz=1.0, hx=1.0):
    """Separable per-axis 2nd derivatives of a 2-D wavefield.

    Naming: ``d2`` = second derivative (∂²); ``2d`` = 2-D wavefield.
    Returns the **components**, not their sum — sum them yourself for an
    isotropic Laplacian or use :func:`laplacian_2d` for the shortcut.

    Implementation: two separable 1-D ``conv2d`` calls, one per axis.

    Args:
        u: Input wavefield, shape ``(B, 1, nz, nx)``.
        k1d: 1-D kernel of length ``2M+1`` (or a tuple ``(kz, kx)`` of
            pre-shaped 4-D kernels for the cached path).
        hz, hx: Grid spacings along z and x.

    Returns:
        ``(d2u_dz2, d2u_dx2)`` — two tensors of the same shape as ``u``.
    """
    if isinstance(k1d, tuple):
        kz, kx = k1d
        pad = max(kz.shape[-3], kx.shape[-1]) // 2
    else:
        kz = k1d[None, None, :, None]  # (1,1,k,1)
        kx = k1d[None, None, None, :]  # (1,1,1,k)
        pad = k1d.shape[-1] // 2
    lapx = F.conv2d(u, kx, padding=(0, pad)) / (hx*hx)
    lapz = F.conv2d(u, kz, padding=(pad, 0)) / (hz*hz)
    return lapz, lapx

def separable_d2_3d(u, k1d, hz=1.0, hy=1.0, hx=1.0):
    """Separable per-axis 2nd derivatives of a 3-D wavefield.

    Naming: ``d2`` = second derivative (∂²); ``3d`` = 3-D wavefield.
    Returns the **components**, not their sum — sum them yourself for an
    isotropic Laplacian or use :func:`laplacian_3d` for the shortcut.

    Implementation: three separable 1-D ``conv3d`` calls, one per axis.

    Args:
        u: Input wavefield, shape ``(B, 1, nz, ny, nx)``.
        k1d: 1-D kernel of length ``2M+1`` (or a tuple ``(kz, ky, kx)``
            of pre-shaped 5-D kernels for the cached path).
        hz, hy, hx: Grid spacings along z, y, and x.

    Returns:
        ``(d2u_dz2, d2u_dy2, d2u_dx2)`` — three tensors of the same
        shape as ``u``.
    """
    if isinstance(k1d, tuple):
        kz, ky, kx = k1d
        pad = max(kz.shape[-3], ky.shape[-2], kx.shape[-1]) // 2
    else:
        pad = k1d.shape[-1] // 2
        kz = k1d.view(1, 1, -1, 1, 1)
        ky = k1d.view(1, 1, 1, -1, 1)
        kx = k1d.view(1, 1, 1, 1, -1)

    lapz = F.conv3d(u, kz, padding=(pad, 0, 0)) / (hz * hz)
    lapy = F.conv3d(u, ky, padding=(0, pad, 0)) / (hy * hy)
    lapx = F.conv3d(u, kx, padding=(0, 0, pad)) / (hx * hx)

    return lapz, lapy, lapx

# Plain Python — no @torch.jit.script. TorchScript and Dynamo are separate
# JITs and a @torch.jit.script function is opaque to Dynamo: every PD call
# inside the equation step becomes a graph break, which neutralises
# torch.compile on Elastic / DAS (their step calls these kernels 8+ times).
# Measured on RTX 6000 Ada, removing the decorator also makes eager Elastic
# ~6% faster, so the decorator was net cost even without torch.compile.
def apply_kernels(u, kernels):
    # u: (B, 1, H, W). kernels: (1, 1, kh, kw) or (K, 1, kh, kw).
    _, _, KH, KW = kernels.shape
    padding = (KH // 2, KW // 2)
    conv_out = F.conv2d(u, kernels, padding=padding)
    return conv_out if conv_out.shape[1] == 1 else conv_out.sum(dim=1, keepdim=True)


def apply_kernels_3d(u, kernels):
    # u: (B, 1, D, H, W). kernels: (1, 1, kD, kH, kW) or (K, 1, kD, kH, kW).
    _, _, KD, KH, KW = kernels.shape
    padding = (KD // 2, KH // 2, KW // 2)
    conv_out = F.conv3d(u, kernels, padding=padding)
    return conv_out if conv_out.shape[1] == 1 else conv_out.sum(dim=1, keepdim=True)


def laplacian_2d(u, k1d, hz=1.0, hx=1.0):
    """Isotropic 2-D Laplacian via separable kernels: ``d2z + d2x``.

    Convenience wrapper around :func:`separable_d2_2d` for callers that
    only need the scalar Laplacian ``∇²u`` (the common isotropic-acoustic
    case). Anisotropic equations should keep using
    :func:`separable_d2_2d` and combine the two components themselves.
    """
    d2z, d2x = separable_d2_2d(u, k1d, hz=hz, hx=hx)
    return d2z + d2x


def laplacian_3d(u, k1d, hz=1.0, hy=1.0, hx=1.0):
    """Isotropic 3-D Laplacian via separable kernels: ``d2z + d2y + d2x``.

    Convenience wrapper around :func:`separable_d2_3d`; see
    :func:`laplacian_2d` for the rationale.
    """
    d2z, d2y, d2x = separable_d2_3d(u, k1d, hz=hz, hy=hy, hx=hx)
    return d2z + d2y + d2x


def gradient(u, h, axis, kernels=None):
    h_axis = _resolve_spacing_for_axis(h, axis, u.ndim)
    if kernels is not None:
        if axis not in kernels:
            raise ValueError(f"No gradient kernel configured for axis={axis}.")
        kernel = kernels[axis]
        if kernel.ndim == 4:
            padding = (kernel.shape[-2] // 2, kernel.shape[-1] // 2)
            u_nchw, restore = _to_nchw(u)
            out = F.conv2d(u_nchw, kernel / h_axis, padding=padding)
            out = _zero_halo(out, padding)
        elif kernel.ndim == 5:
            padding = (kernel.shape[-3] // 2, kernel.shape[-2] // 2, kernel.shape[-1] // 2)
            u_ncdhw, restore = _to_ncdhw(u)
            out = F.conv3d(u_ncdhw, kernel / h_axis, padding=padding)
            out = _zero_halo_3d(out, padding)
        else:
            raise ValueError(f"Expected 2D or 3D gradient kernel, got shape {tuple(kernel.shape)}")
        return restore(out)
    return torch.gradient(u, spacing=h_axis, dim=axis)[0]
