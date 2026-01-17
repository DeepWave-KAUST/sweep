import jax, torch
import numpy as np
import jax.numpy as jnp
import torch.nn.functional as F
from sweep.scalars import staggered_grid_coes

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
    if isinstance(u, torch.Tensor):
        return torch.gradient(u, spacing=h, dim=axis)[0]
    return jnp.gradient(u, h, axis=axis)

