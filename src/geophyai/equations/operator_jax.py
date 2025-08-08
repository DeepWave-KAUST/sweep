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

class PartialDerivative:

    def __init__(self, spatial_order:int=4):
        self.coes = jnp.array(staggered_grid_coes_torch(int(spatial_order//2)), dtype=jnp.float32)
        num_kernels = spatial_order // 2 # max length is 2*num_kernels+1
        max_length = 2 * num_kernels + 1
        # self.coes = jnp.ones_like(self.coes)

        self.kxf = -pad_kernels(create_kernels(num_kernels, self.coes, axis='x', forward=True), (1, max_length))
        self.kxb = -pad_kernels(create_kernels(num_kernels, self.coes, axis='x', forward=False), (1, max_length))
        self.kzf = -pad_kernels(create_kernels(num_kernels, self.coes, axis='z', forward=True), (max_length, 1))
        self.kzb = -pad_kernels(create_kernels(num_kernels, self.coes, axis='z', forward=False), (max_length, 1))
    
    def x_forward(self, u):
        return apply_kernels(u, self.kxf)

    def x_backward(self, u):
        return apply_kernels(u, self.kxb)
    
    def z_forward(self, u):
        return apply_kernels(u, self.kzf)
    
    def z_backward(self, u):
        return apply_kernels(u, self.kzb)

def pad_kernels(kernels, target_shape):
    # pads 2D kernels to same shape (H, W)
    padded = []
    for k in kernels:
        pad_y = (target_shape[0] - k.shape[0]) // 2
        pad_x = (target_shape[1] - k.shape[1]) // 2
        padded.append(jnp.pad(k, ((pad_y, pad_y), (pad_x, pad_x))))
    return jnp.stack(padded)

def make_kernel(length, axis, flip=False):
    k = jnp.zeros((length, 1) if axis == 'z' else (1, length), dtype=jnp.float32)
    if not flip:
        k = k.at[0, 0].set(-1) if axis == 'z' else k.at[0, 0].set(-1)
        k = k.at[-2, 0].set(1) if axis == 'z' else k.at[0, -2].set(1)
    else:
        k = k.at[1, 0].set(-1) if axis == 'z' else k.at[0, 1].set(-1)
        k = k.at[-1, 0].set(1) if axis == 'z' else k.at[0, -1].set(1)
    return k

def create_kernel(length, axis='x', forward=True):
    if axis == 'x':
        kernel = jnp.zeros((1, length), dtype=jnp.float32)
        if forward:
            kernel = kernel.at[..., 0].set(-1)
            kernel = kernel.at[..., -2].set(1)
        else:
            kernel = kernel.at[..., 1].set(-1)
            kernel = kernel.at[..., -1].set(1)
    else:  # axis == 'z'
        kernel = jnp.zeros((length, 1), dtype=jnp.float32)
        if forward:
            kernel = kernel.at[0, :].set(-1)
            kernel = kernel.at[-2, :].set(1)
        else:
            kernel = kernel.at[1, :].set(-1)
            kernel = kernel.at[-1, :].set(1)
    return kernel

def create_kernels(num_kernels, scale, axis='x', forward=True):
    return [create_kernel(2 * i + 1, axis, forward)*scale[i-1] for i in range(1, num_kernels + 1)]

def apply_kernels(u, kernels):
    # u: (b, 1, h, w)
    # kernels: (k, kh, kw)
    B, C, H, W = u.shape
    K, KH, KW = kernels.shape
    kernels_exp = kernels[:, None, ::-1, ::-1]  # (K, 1, kh, kw), need reverse for lax conv
    def single_conv():
        return jax.lax.conv_general_dilated(
            lhs=u,  # (b, k, h, w)
            rhs=kernels_exp, # (1, k, kh, kw)
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'), 
        )  # → (b, k, 1, h, w)
    conv_out = single_conv()  # → (b, k, h, w)
    return jnp.sum(conv_out, axis=1, keepdims=True)  # → (b, 1, h, w)

