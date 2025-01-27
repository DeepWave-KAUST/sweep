from jax import vmap
from jax.scipy.signal import convolve2d as conv2d

def _laplace(image, kernel):
    # Expected input shape: (height, width)
    return conv2d(image, kernel, mode='same')

batch_convolve2d = vmap(vmap(_laplace, in_axes=(0, None)), in_axes=(0, None))

def laplace(u, h, kernel):
    return batch_convolve2d(u, kernel) / (h ** 2)