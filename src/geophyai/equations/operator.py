import torch
from typing import List
import numpy as np
import torch.nn.functional as F
from geophyai.scalars import staggered_grid_coes

@torch.jit.script
def laplace(u: torch.Tensor, 
            h: torch.Tensor, 
            kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    return torch.nn.functional.conv2d(u, kernel, padding=padding) / (h**2)

@torch.jit.script
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

class PartialDerivative:

    def __init__(self, spatial_order:int=4, device='cpu'):
        self.coes = staggered_grid_coes(int(spatial_order//2))
        num_kernels = spatial_order // 2 # max length is 2*num_kernels+1
        max_length = 2 * num_kernels + 1
        # self.coes = cp.ones_like(self.coes)

        self.kxf = -torch.from_numpy(pad_kernels(create_kernels(num_kernels, self.coes, axis='x', forward=True), (1, max_length))).cuda()
        self.kxb = -torch.from_numpy(pad_kernels(create_kernels(num_kernels, self.coes, axis='x', forward=False), (1, max_length))).cuda()
        self.kzf = -torch.from_numpy(pad_kernels(create_kernels(num_kernels, self.coes, axis='z', forward=True), (max_length, 1))).cuda()
        self.kzb = -torch.from_numpy(pad_kernels(create_kernels(num_kernels, self.coes, axis='z', forward=False), (max_length, 1))).cuda()

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
        padded.append(np.pad(k, ((pad_y, pad_y), (pad_x, pad_x))))
    return np.stack(padded)

def make_kernel(length, axis, flip=False):
    k = np.zeros((length, 1) if axis == 'z' else (1, length), dtype=np.float32)
    if not flip:
        if axis == 'z':
            k[0, 0] = -1
            k[-2, 0] = 1
        else:  # axis == 'x'
            k[0, 0] = -1
            k[0, -2] = 1
    else:
        if axis == 'z':
            k[1, 0] = -1
            k[-1, 0] = 1
        else:  # axis == 'x'
            k[0, 1] = -1
            k[0, -1] = 1
    return k

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

def apply_kernels(u, kernels):
    # u: (B, 1, H, W), torch.Tensor
    # kernels: (K, kh, kw), torch.Tensor

    B, C, H, W = u.shape
    K, KH, KW = kernels.shape

    kernels_exp = kernels.unsqueeze(1)  # (K, 1, kh, kw)

    padding = (KH // 2, KW // 2) 

    conv_out = F.conv2d(u, kernels_exp, padding=padding)  # (B, K, H, W)

    out = conv_out.sum(dim=1, keepdim=True)  # (B, 1, H, W)

    return out