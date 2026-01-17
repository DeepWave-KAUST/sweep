import numpy as np
from sweep.scalars import staggered_grid_coes
from typing import Callable

class PartialDerivative:

    def __init__(self, spatial_order:int=4, device='cpu', backend='torch'):
        self.coes = staggered_grid_coes(int(spatial_order//2))
        num_kernels = spatial_order // 2 # max length is 2*num_kernels+1
        max_length = 2 * num_kernels + 1

        self.kxf = -pad_kernels(create_kernels(num_kernels, self.coes, axis='x', forward=True), (1, max_length))
        self.kxb = -pad_kernels(create_kernels(num_kernels, self.coes, axis='x', forward=False), (1, max_length))
        self.kzf = -pad_kernels(create_kernels(num_kernels, self.coes, axis='z', forward=True), (max_length, 1))
        self.kzb = -pad_kernels(create_kernels(num_kernels, self.coes, axis='z', forward=False), (max_length, 1))

        self.backend = backend
        self.get_ops()

        self.device = device

    def get_ops(self):
        if self.backend == 'jax':
            from sweep.operators.jax import apply_kernels_jax as apply_kernels
        if self.backend == 'torch':
            from sweep.operators.torch import apply_kernels_torch as apply_kernels
        self.apply_kernels = apply_kernels

    def to_backend(self, to: Callable, *args, **kwargs):
        self.kxf = to(self.kxf, self.backend, self.device)
        self.kxb = to(self.kxb, self.backend, self.device)
        self.kzf = to(self.kzf, self.backend, self.device)
        self.kzb = to(self.kzb, self.backend, self.device)

    def x_forward(self, u):
        return self.apply_kernels(u, self.kxf)

    def x_backward(self, u):
        return self.apply_kernels(u, self.kxb)
    
    def z_forward(self, u):
        return self.apply_kernels(u, self.kzf)
    
    def z_backward(self, u):
        return self.apply_kernels(u, self.kzb)

def pad_kernels(kernels, target_shape):
    # pads 2D kernels to same shape (H, W)
    padded = []
    for k in kernels:
        pad_y = (target_shape[0] - k.shape[0]) // 2
        pad_x = (target_shape[1] - k.shape[1]) // 2
        padded.append(np.pad(k, ((pad_y, pad_y), (pad_x, pad_x))))
    return np.stack(padded)

def create_kernel(length, axis='x', forward=True):
    if axis == 'x':
        kernel = np.zeros((1, length), dtype=np.float32)
        if forward:
            kernel[..., 0] = -1
            kernel[..., -2] = 1
        else:
            kernel[..., 1] = -1
            kernel[..., -1] = 1
    else:  # axis == 'z'
        kernel = np.zeros((length, 1), dtype=np.float32)
        if forward:
            kernel[0, :] = -1
            kernel[-2, :] = 1
        else:
            kernel[1, :] = -1
            kernel[-1, :] = 1
    return kernel

def create_kernels(num_kernels, scale, axis='x', forward=True):
    return [create_kernel(2 * i + 1, axis, forward)*scale[i-1] for i in range(1, num_kernels + 1)]