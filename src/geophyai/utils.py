import jax
import jax.numpy as jnp

import numpy as np

import torch
import torch.nn.functional as F

def to_tensor(array):
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    elif isinstance(array, torch.Tensor):
        return array
    
class EdgePadding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, pad):
        ctx.pad = pad
        return F.pad(input_tensor.unsqueeze(0), pad=pad, mode='replicate').squeeze(0)

    @staticmethod
    def backward(ctx, grad_output):
        pad_left, pad_right, pad_top, pad_bottom = ctx.pad
        grad_input = grad_output[..., pad_top:-pad_bottom, pad_left:-pad_right]
        return grad_input, None
    
def edge_pad_base(u, pad_width):
    """Pad the edges of the input data.

    Args:
        u (jnp.array): The input data with shape (batch_size, 1, nz, nx).
        pad_width (int): The width of the padding.

    Returns:
        jnp.array: Padded data.
    """
    return jnp.pad(u, pad_width, mode='edge')

def edge_pad_fwd(u, pad_width):
    """Forward function of edge_pad.

    Args:
        u (jnp.array): The input data with shape (batch_size, 1, nz, nx).
        pad_width (int): The width of the padding.

    Returns:
        jnp.array: Padded data.
    """
    u = jnp.pad(u, pad_width, mode='edge')
    return u, pad_width

def edge_pad_bwd(res, g):
    """Backward function of edge_pad.

    Args:
        res (jnp.array): The input data with shape (batch_size, 1, nz, nx).
        g (jnp.array): The gradient.

    Returns:
        jnp.array: The gradient of the input data.
    """
    pad_left, pad_right = res[1]
    pad_top, pad_bottom = res[0]
    # g = g.at[..., pad_top:-pad_bottom, pad_left:-pad_right].set(0.)
    return g[..., pad_top:-pad_bottom, pad_left:-pad_right], None

edge_pad = jax.custom_vjp(edge_pad_base)
edge_pad.defvjp(edge_pad_fwd, edge_pad_bwd)