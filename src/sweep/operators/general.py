import numpy as np
from sweep.scalars import staggered_grid_coes
from typing import Callable


def _prepare_torch_kernel_bank(kernel):
    import torch

    if not isinstance(kernel, torch.Tensor):
        return kernel
    if kernel.ndim == 3:
        # The derivative operator is linear: summing the per-offset responses is
        # equivalent to convolving once with the summed stencil.
        return kernel.sum(dim=0, keepdim=True).flip(-1, -2).unsqueeze(1).contiguous()
    if kernel.ndim == 4:
        return kernel.sum(dim=0, keepdim=True).flip(-1, -2, -3).unsqueeze(1).contiguous()
    return kernel


def _prepare_jax_kernel_bank(kernel):
    try:
        import jax.numpy as jnp
    except Exception:
        return kernel

    if type(kernel).__module__.startswith("jax"):
        if kernel.ndim == 3:
            return jnp.sum(kernel, axis=0, keepdims=True)
        if kernel.ndim == 4:
            return jnp.sum(kernel, axis=0, keepdims=True)
    return kernel


class PartialDerivative:

    def __init__(self, spatial_order:int=4, device='cpu', backend='torch', ndim=2):
        self.coes = staggered_grid_coes(int(spatial_order//2))
        num_kernels = spatial_order // 2 # max length is 2*num_kernels+1
        max_length = 2 * num_kernels + 1
        self.ndim = ndim
        pad_func = pad_kernels3d if ndim == 3 else pad_kernels2d

        if ndim == 2:
            self.kxf = -pad_func(create_kernels(num_kernels, self.coes, axis='x', forward=True, ndim=ndim), (1, max_length))
            self.kxb = -pad_func(create_kernels(num_kernels, self.coes, axis='x', forward=False, ndim=ndim), (1, max_length))
            self.kzf = -pad_func(create_kernels(num_kernels, self.coes, axis='z', forward=True, ndim=ndim), (max_length, 1))
            self.kzb = -pad_func(create_kernels(num_kernels, self.coes, axis='z', forward=False, ndim=ndim), (max_length, 1))
        else:
            self.kxf = -pad_func(create_kernels(num_kernels, self.coes, axis='x', forward=True, ndim=ndim), (1, 1, max_length))
            self.kxb = -pad_func(create_kernels(num_kernels, self.coes, axis='x', forward=False, ndim=ndim), (1, 1, max_length))
            self.kyf = -pad_func(create_kernels(num_kernels, self.coes, axis='y', forward=True, ndim=ndim), (1, max_length, 1))
            self.kyb = -pad_func(create_kernels(num_kernels, self.coes, axis='y', forward=False, ndim=ndim), (1, max_length, 1))
            self.kzf = -pad_func(create_kernels(num_kernels, self.coes, axis='z', forward=True, ndim=ndim), (max_length, 1, 1))
            self.kzb = -pad_func(create_kernels(num_kernels, self.coes, axis='z', forward=False, ndim=ndim), (max_length, 1, 1))

        self.backend = backend
        self.get_ops()

        self.device = device
        if ndim == 2:
            self._spacing = {"z": 1.0, "x": 1.0}
        else:
            self._spacing = {"z": 1.0, "y": 1.0, "x": 1.0}

    def get_ops(self):
        if self.backend == 'jax':
            from sweep.operators.jax import apply_kernels_jax as apply_kernels
            from sweep.operators.jax import apply_kernels_jax3d as apply_kernels3d
        if self.backend == 'torch':
            from sweep.operators.torch import apply_kernels_torch as apply_kernels
            from sweep.operators.torch import apply_kernels_torch3d as apply_kernels3d

        self.apply_kernels = apply_kernels3d if self.ndim == 3 else apply_kernels

    def to_backend(self, to: Callable, *args, **kwargs):
        self.kxf = to(self.kxf, self.backend, self.device)
        self.kxb = to(self.kxb, self.backend, self.device)
        self.kzf = to(self.kzf, self.backend, self.device)
        self.kzb = to(self.kzb, self.backend, self.device)
        if self.ndim == 3:
            self.kyf = to(self.kyf, self.backend, self.device)
            self.kyb = to(self.kyb, self.backend, self.device)
        if self.backend == 'jax':
            self.kxf = _prepare_jax_kernel_bank(self.kxf)
            self.kxb = _prepare_jax_kernel_bank(self.kxb)
            self.kzf = _prepare_jax_kernel_bank(self.kzf)
            self.kzb = _prepare_jax_kernel_bank(self.kzb)
            if self.ndim == 3:
                self.kyf = _prepare_jax_kernel_bank(self.kyf)
                self.kyb = _prepare_jax_kernel_bank(self.kyb)
        if self.backend == 'torch':
            self.kxf = _prepare_torch_kernel_bank(self.kxf)
            self.kxb = _prepare_torch_kernel_bank(self.kxb)
            self.kzf = _prepare_torch_kernel_bank(self.kzf)
            self.kzb = _prepare_torch_kernel_bank(self.kzb)
            if self.ndim == 3:
                self.kyf = _prepare_torch_kernel_bank(self.kyf)
                self.kyb = _prepare_torch_kernel_bank(self.kyb)

    def set_spacing(self, spacing):
        if np.isscalar(spacing):
            value = float(spacing)
            if self.ndim == 2:
                self._spacing = {"z": value, "x": value}
            else:
                self._spacing = {"z": value, "y": value, "x": value}
            return

        spacing = tuple(float(v) for v in spacing)
        if len(spacing) != self.ndim:
            raise ValueError(f"Expected spacing length {self.ndim}, got {len(spacing)}.")
        if self.ndim == 2:
            self._spacing = {"z": spacing[0], "x": spacing[1]}
        else:
            self._spacing = {"z": spacing[0], "y": spacing[1], "x": spacing[2]}

    def _apply(self, u, kernels, h=None, axis=None):
        out = self.apply_kernels(u, kernels)
        if h is None and axis is not None:
            h = self._spacing[axis]
        if h is None:
            return out
        return out / h

    def x_forward(self, u, h=None):
        return self._apply(u, self.kxf, h=h, axis="x")

    def x_backward(self, u, h=None):
        return self._apply(u, self.kxb, h=h, axis="x")

    def y_forward(self, u, h=None):
        return self._apply(u, self.kyf, h=h, axis="y")

    def y_backward(self, u, h=None):
        return self._apply(u, self.kyb, h=h, axis="y")

    def z_forward(self, u, h=None):
        return self._apply(u, self.kzf, h=h, axis="z")

    def z_backward(self, u, h=None):
        return self._apply(u, self.kzb, h=h, axis="z")

def pad_kernels2d(kernels, target_shape):
    # pads 2D kernels to same shape (H, W)
    padded = []
    for k in kernels:
        pad_y = (target_shape[0] - k.shape[0]) // 2
        pad_x = (target_shape[1] - k.shape[1]) // 2
        padded.append(np.pad(k, ((pad_y, pad_y), (pad_x, pad_x))))
    return np.stack(padded)

def pad_kernels3d(kernels, target_shape):
    # pads 3D kernels to same shape (D, H, W)
    padded = []
    for k in kernels:
        pad_d = (target_shape[0] - k.shape[0]) // 2
        pad_h = (target_shape[1] - k.shape[1]) // 2
        pad_w = (target_shape[2] - k.shape[2]) // 2
        padded.append(np.pad(k, ((pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w))))
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
    if axis == 'z':  # axis == 'z'
        kernel = np.zeros((length, 1), dtype=np.float32)
        if forward:
            kernel[0, :] = -1
            kernel[-2, :] = 1
        else:
            kernel[1, :] = -1
            kernel[-1, :] = 1
    return kernel

def create_kernel3d(length, axis='x', forward=True):
    if axis == 'x':
        kernel = np.zeros((1, 1, length), dtype=np.float32)
        if forward:
            kernel[..., 0] = -1
            kernel[..., -2] = 1
        else:
            kernel[..., 1] = -1
            kernel[..., -1] = 1
    if axis == 'y':
        kernel = np.zeros((1, length, 1), dtype=np.float32)
        if forward:
            kernel[:, 0, :] = -1
            kernel[:, -2, :] = 1
        else:
            kernel[:, 1, :] = -1
            kernel[:, -1, :] = 1
    if axis == 'z':  # axis == 'z'
        kernel = np.zeros((length, 1, 1), dtype=np.float32)
        if forward:
            kernel[0, ...] = -1
            kernel[-2, ...] = 1
        else:
            kernel[1, ...] = -1
            kernel[-1, ...] = 1
    return kernel

def create_kernels(num_kernels, scale, axis='x', forward=True, ndim=2):
    if ndim == 3:
        return [create_kernel3d(2 * i + 1, axis, forward)*scale[i-1] for i in range(1, num_kernels + 1)]
    else:
        return [create_kernel(2 * i + 1, axis, forward)*scale[i-1] for i in range(1, num_kernels + 1)]
