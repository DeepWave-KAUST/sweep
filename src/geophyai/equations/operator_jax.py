import jax
import jax.numpy as jnp
from jax import vmap
from geophyai.scalars import staggered_grid_coes_torch
from jax.scipy.signal import convolve2d as conv2d


def _laplace(image, kernel):
    # Expected input shape: (height, width)
    return conv2d(image, kernel, mode='same')

batch_convolve2d = vmap(vmap(_laplace, in_axes=(0, None)), in_axes=(0, None))

def laplace(u, h=1, kernel=None):
    return batch_convolve2d(u, kernel) / (h ** 2)

def laplace3d(u, h=1, kernel=None):
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
