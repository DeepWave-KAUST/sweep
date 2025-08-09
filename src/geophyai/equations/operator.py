import jax, torch
import numpy as np
import jax.numpy as jnp
import torch.nn.functional as F
from geophyai.scalars import staggered_grid_coes

# @torch.jit.script
def laplace(u: torch.Tensor, 
            h: float | torch.Tensor, 
            kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    return torch.nn.functional.conv2d(u, kernel, padding=padding) / (h*h)

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

class PartialDerivative:

    def __init__(self, spatial_order:int=4, device='cpu', backend='torch'):
        self.coes = staggered_grid_coes(int(spatial_order//2))
        num_kernels = spatial_order // 2 # max length is 2*num_kernels+1
        max_length = 2 * num_kernels + 1
        # self.coes = cp.ones_like(self.coes)

        self.kxf = -pad_kernels(create_kernels(num_kernels, self.coes, axis='x', forward=True), (1, max_length))
        self.kxb = -pad_kernels(create_kernels(num_kernels, self.coes, axis='x', forward=False), (1, max_length))
        self.kzf = -pad_kernels(create_kernels(num_kernels, self.coes, axis='z', forward=True), (max_length, 1))
        self.kzb = -pad_kernels(create_kernels(num_kernels, self.coes, axis='z', forward=False), (max_length, 1))

        self.device = device
        self.to_backend(backend)
        self.apply_kernels = apply_kernels_torch if backend == 'torch' else apply_kernels_jax

    def to_backend(self, backend: str):
        to = {'torch': lambda d: torch.tensor(d, device=self.device, dtype=torch.float32),
              'jax': lambda d: jnp.array(d, dtype=jnp.float32)}[backend]
        self.kxf = to(self.kxf)
        self.kxb = to(self.kxb)
        self.kzf = to(self.kzf)
        self.kzb = to(self.kzb)

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

def apply_kernels_torch(u, kernels):
    # u: (B, 1, H, W), torch.Tensor
    # kernels: (K, kh, kw), torch.Tensor

    B, C, H, W = u.shape
    K, KH, KW = kernels.shape

    kernels_exp = kernels.unsqueeze(1)  # (K, 1, kh, kw)

    padding = (KH // 2, KW // 2) 

    conv_out = F.conv2d(u, kernels_exp, padding=padding)  # (B, K, H, W)

    out = conv_out.sum(dim=1, keepdim=True)  # (B, 1, H, W)

    return out

def apply_kernels_jax(u, kernels):
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