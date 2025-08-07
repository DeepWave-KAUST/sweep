import torch, jax
import jax.numpy as jnp
from .operator_jax import laplace
from .operator_jax import laplace3d as laplace3d_jax
import numpy as np
from .utils import to_backend
from .operator_jax import PartialDerivative as PartialDerivativeJax

from typing import Tuple, Optional, Union, List
from .habc_jax import habc, bound_mask
from geophyai.scalars import generate_convolution_kernel, generate_convolution_kernel3d

def step(u_now, u_pre, vp, rx, rz, dt, h, b, lap_u_now, dpdx, dpdz, dvpdx, dvpdz, pd, habc_masks=None):
    a = 1 / (1 + b * dt)
    u_next = 2 * u_now - u_pre + vp**2*dt**2*lap_u_now + vp*(dvpdx*dpdx + dvpdz*dpdz)*dt**2 - 2*vp**2*(rx*dpdx + rz*dpdz)*dt**2
    u_next = a * u_next + (1 - a) * u_now
    return u_next, u_now

# def step_habc(u_now, u_pre, vp, dt, h, b, lap_u_now, habc_mask):
#     u_next = 2*u_now - u_pre + vp**2*dt**2 * lap_u_now
#     u_next = habc(u_next, u_now, u_pre, vp, b, dt, h, maskidx=habc_mask)
#     return u_next, u_now

class Acoustic:
    """
    Parameter order: vp, rx, rz

    Wavefields: (h1, h2)

    Reference: 10.3997/2214-4609.202010332
    """
    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=2):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        
        kernel = generate_convolution_kernel(spatial_order)
        # First order gradient kernel (first derivative)
        gkernel_x = generate_convolution_kernel(spatial_order, derivative_order=1, mode='x', no_center=True, grid='normal', sign=-1)
        gkernel_z = generate_convolution_kernel(spatial_order, derivative_order=1, mode='z', no_center=True, grid='normal', sign=-1)
        
        self.gkernel_x = to_backend(gkernel_x, backend, device)
        self.gkernel_z = to_backend(gkernel_z, backend, device)
        self.kernel = to_backend(kernel, backend, device)

        self.pd = PartialDerivativeJax(spatial_order)


    def init_habc(self, shape, abcn, free_surface=False, batchsize=1, use_habc=False):
        habc_masks = bound_mask(*shape, abcn, batchsize, return_idx=True, free_surface=free_surface)
        self.habc_masks = tuple([np.array(mask) if mask is not None else mask for mask in habc_masks])
        self.use_habc = use_habc

    @property
    def models(self):
        return ['vp', 'rx', 'rz']
    
    @property
    def wavefields(self):
        return ['h1', 'h2']

    # def func(self, *args, **kwargs):
    #     lap_u_now = laplace(args[0], args[4], self.kernel)
    #     return step_pml(*args, lap_u_now)
    
    def func_jax(self, *args, **kwargs):
        lap_u_now = laplace(args[0], args[6], self.kernel)
        dvpdx = jnp.gradient(args[2],args[6], axis=-1) # 2nd Center Difference
        dvpdz = jnp.gradient(args[2],args[6], axis=-2)
        dpdx = jnp.gradient(args[0], args[6], axis=-1)
        dpdz = jnp.gradient(args[0], args[6], axis=-2)
        return step(*args, lap_u_now, dpdx, dpdz, dvpdx, dvpdz, self.pd, self.habc_masks)