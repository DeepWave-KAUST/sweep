
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
    return jnp.pad(u, pad_width, mode='edge'), None

def edge_pad_bwd(pad_width, res, g):
    """Backward function of edge_pad.

    Args:
        pad_width: The padding width pairs per dimension (static).
        res: Residuals from the forward function (unused).
        g (jnp.array): The gradient.

    Returns:
        tuple: The gradient of the input data.
    """
    slices = [
        slice(p0, g.shape[i] - p1)
        for i, (p0, p1) in enumerate(pad_width)
    ]
    return (g[tuple(slices)],)

# pad_width must be a static (nondiff) argument: without nondiff_argnums a
# pure-forward jit traces the primal with every argument lifted to a tracer,
# and jnp.pad cannot take a traced pad_width (ConcretizationTypeError).
edge_pad = jax.custom_vjp(edge_pad_base, nondiff_argnums=(1,))
edge_pad.defvjp(edge_pad_fwd, edge_pad_bwd)