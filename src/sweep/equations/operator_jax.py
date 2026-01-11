import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.signal import convolve2d as conv2d
from jax import lax

@jax.jit
def laplace1d_sep(u, k1d, hz=1.0, hx=1.0):
    """
    Anisotropic spacing: Laplace = d2/dz2 / hz^2 + d2/dx2 / hx^2
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

def _laplace(image, kernel):
    # Expected input shape: (height, width)
    return conv2d(image, kernel, mode='same')

batch_convolve2d = vmap(vmap(_laplace, in_axes=(0, None)), in_axes=(0, None))

@jax.jit
def laplace(u, h=1.0, kernel=None):
    return batch_convolve2d(u, kernel) / (h ** 2)

def laplace3d(u, kernel=None, h=1.0):
    """ 3D Laplace operator using JAX.
     Args:
         u (jnp.ndarray): Input wavefield of shape (batch, 1, depth, height, width).
         h (float): Grid spacing.
         kernel (jnp.ndarray): 3D convolution kernel of shape (1, 1, kD, kH, kW).

     Returns:
         jnp.ndarray: Resulting wavefield after applying the Laplace operator.
     """
    dn = jax.lax.conv_dimension_numbers(u.shape, kernel.shape,
                                        ('NCDHW', 'OIDHW', 'NCDHW'))
    out = jax.lax.conv_general_dilated(u,    # lhs = image tensor
                                       kernel,  # rhs = conv kernel tensor
                                       (1,1,1), # window strides
                                       'SAME',  # padding mode
                                       (1,1,1), # lhs/image dilation
                                       (1,1,1), # rhs/kernel dilation
                                       dn)      # dimension_numbers
    return out / (h ** 2)
