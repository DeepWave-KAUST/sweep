import torch, math
import numpy as np
from .operator_jax import laplace
import jax.numpy as jnp
from typing import Tuple, Optional, Union, List
from geophyai.scalars import generate_convolution_kernel
from .utils import to_backend
from .habc_jax import habc, bound_mask

def step(u_now, u_pre, f_now, f_next, # wavefields
         vv, v, eta, # Model parameters 
         dt, h, b,  # Auxiliary parameters
         nabla_x, nabla_z, dpdx2dz2, # Partial derivatives
         habc_masks=None):
    
    a = 1 / (1 + b * dt)

    # Equation 22
    u_next = 2*u_now - u_pre + (1+2*eta)*v**2*nabla_x*dt**2 + vv**2*nabla_z*dt**2 - 2*eta*vv**2*v**2*dpdx2dz2*dt**2
    # Equation 26
    f_next = 2*f_now - f_next + dt**2*u_now

    u_next = a * u_next + (1 - a) * u_now
    f_next = a * f_next + (1 - a) * f_now

    return u_next, u_now, f_next, f_now


class AcousticTariq:
    """Parameter order: vv, v, eta.
    
       Wavefields: (h1, h2, f1, f2)

       Reference: Alkhalifah Tariq, 10.1190/1.1444815
    """
    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=2):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        
        # Second order laplace kernel (Second derivative), Full kernel
        lkernel_x = generate_convolution_kernel(spatial_order, mode='x', no_center=False, grid='normal')
        lkernel_z = generate_convolution_kernel(spatial_order, mode='z', no_center=False, grid='normal')
        # First order gradient kernel (first derivative)
        self.lkernel_x = to_backend(lkernel_x, backend, device)
        self.lkernel_z = to_backend(lkernel_z, backend, device)

        self.backend = backend

    def init_habc(self, shape, abcn, free_surface=False, batchsize=1, use_habc=False):
        habc_masks = bound_mask(*shape, abcn, batchsize, return_idx=True, free_surface=free_surface)
        self.habc_masks = tuple([np.array(mask) if mask is not None else mask for mask in habc_masks])
        self.use_habc = use_habc
    
    @property
    def models(self):
        return ['vv', 'v', 'eta']
    
    @property
    def wavefields(self):
        return ['h1', 'h2', 'f1', 'f2']
    
    def func_jax(self, *args, **kwargs):
        nabla_x = laplace(args[0], args[8], self.lkernel_x)
        nabla_z = laplace(args[0], args[8], self.lkernel_z)
        dpdx2 = laplace(args[2], args[8], self.lkernel_x)
        dpdx2dz2 = laplace(dpdx2, args[8], self.lkernel_z)
        return step(*args, nabla_x, nabla_z, dpdx2dz2, **kwargs)
