import jax, torch
import numpy as np
import jax.numpy as jnp
import torch.nn.functional as F
from sweep.scalars import staggered_grid_coes


def _resolve_spacing_for_axis(h, axis, ndim):
    if isinstance(h, torch.Tensor):
        if h.ndim == 0:
            return h
        spatial_ndim = h.shape[0]
    elif hasattr(h, "ndim"):
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

# @torch.jit.script
def laplace(u: torch.Tensor, 
            h: float | torch.Tensor, 
            kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    return torch.nn.functional.conv2d(u, kernel, padding=padding) / (h*h)

def laplace1d_sep(u, k1d, hz=1.0, hx=1.0):
    kz = k1d[None, None, :, None]  # (k,1,1,1)
    kx = k1d[None, None, None, :]  # (1,k,1,1)
    pad = k1d.shape[-1] // 2
    lapx = torch.nn.functional.conv2d(u, kx, padding=(0, pad)) / (hx*hx)
    lapz = torch.nn.functional.conv2d(u, kz, padding=(pad, 0)) / (hz*hz)
    return lapz, lapx

# @torch.jit.script
def gradient(u: torch.Tensor, 
             h: torch.Tensor, 
             kernel: torch.Tensor) -> torch.Tensor:
    """Gradient operator.

    Args:
        u (torch.Tensor): Wavefield (batch, 1, nz, nx).
        h (torch.Tensor): Grid spacing.
        kernel (torch.Tensor): Gradient kernel (FD coefficients).

    Returns:
        torch.Tensor: Gradient result
    """
    operator = (h) ** (-1) * kernel
    padding = kernel.shape[-1] // 2
    return torch.nn.functional.conv2d(u, operator, padding=padding)

def gradientO2(u, h, axis):
    h_axis = _resolve_spacing_for_axis(h, axis, u.ndim)
    if isinstance(u, torch.Tensor):
        return torch.gradient(u, spacing=h_axis, dim=axis)[0]
    return jnp.gradient(u, h_axis, axis=axis)
