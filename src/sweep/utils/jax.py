
import jax
import jax.numpy as jnp

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

def edge_pad_bwd(pad_width, g):
    """Backward function of edge_pad.

    Args:
        pad_width (jnp.array): The padding width for each dimension.
        g (jnp.array): The gradient.

    Returns:
        jnp.array: The gradient of the input data.
    """    
    slices = [
        slice(p0, g.shape[i] - p1)
        for i, (p0, p1) in enumerate(pad_width)
    ]
    return g[tuple(slices)], None

edge_pad = jax.custom_vjp(edge_pad_base)
edge_pad.defvjp(edge_pad_fwd, edge_pad_bwd)