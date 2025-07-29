import torch, math
import numpy as np
from .operator_jax import laplace
import jax.numpy as jnp
from typing import Tuple, Optional, Union, List
from geophyai.scalars import generate_convolution_kernel
from .utils import to_backend
from .habc_jax import habc, bound_mask

def step(u_now, u_pre, # Wavefields
         vp, epsilon, delta, # Model parameters
         dt, h, b,  # Auxiliary parameters
         nabla_x, nabla_z, dpdx, dpdz,# Partial derivatives
         habc_masks=None):

        # 10.1190/geo2022-0292.1 EQ(19) from 10.1190/geo2014-0242.1
        numerator = -2*(epsilon-delta)*dpdx**2*dpdz**2 
        denominator = (1+2*epsilon)*dpdx**4+dpdz**4+2*(1+delta)*dpdx**2*dpdz**2
        sk = numerator*((denominator+1e-26)**-1)

        vp2dt2 = vp**2*dt**2

        a1 = 1+b*dt
        a2 = 1-b*dt
        
        # 10.1190/geo2022-0292.1 EQ(22) No ABC
        u_next = 2*a1**-1*u_now-a2*a1**(-1)*u_pre + a1**-1*(vp2dt2*((1+2*epsilon)+sk)*nabla_x + vp2dt2*(1+sk)*nabla_z)

        return u_next, u_now


class AcousticVTI:
    """
    Parameter order: (vp, epsilon, delta)

    Wavefields: (u_now, u_pre)
    
    Reference: Liang K., et.al, 10.1190/geo2022-0292.1 (EQUATION A-7)
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
        gkernel_x = generate_convolution_kernel(spatial_order, derivative_order=1, mode='x', no_center=True, grid='normal', sign=-1)
        gkernel_z = generate_convolution_kernel(spatial_order, derivative_order=1, mode='z', no_center=True, grid='normal', sign=-1)

        self.lkernel_x = to_backend(lkernel_x, backend, device)
        self.lkernel_z = to_backend(lkernel_z, backend, device)
        self.gkernel_x = to_backend(gkernel_x, backend, device)
        self.gkernel_z = to_backend(gkernel_z, backend, device)

        self.backend = backend

    def init_habc(self, shape, abcn, free_surface=False, batchsize=1, use_habc=False):
        habc_masks = bound_mask(*shape, abcn, batchsize, return_idx=True, free_surface=free_surface)
        self.habc_masks = tuple([np.array(mask) if mask is not None else mask for mask in habc_masks])
        self.use_habc = use_habc

    @property
    def need_init(self):
        return True

    def init(self, shape, device, h):
        assert shape is not None, "shape must be provided to calculate the wavenumbers!!!"
        kz = np.fft.fftfreq(shape[0], d=h) * 2 * np.pi
        kx = np.fft.fftfreq(shape[1], d=h) * 2 * np.pi
        kzz, kxx = np.meshgrid(kz, kx, indexing='ij')
        self.kx = to_backend(kxx, self.backend, device)
        self.kz = to_backend(kzz, self.backend, device)
    
    @property
    def models(self):
        return ['vp', 'epsilon', 'delta']
    
    @property
    def wavefields(self):
        return ['h1', 'h2']
    
    def func_jax(self, *args, **kwargs):
        nabla_x = laplace(args[0], 1.0, self.lkernel_x/(args[6]**2))
        nabla_z = laplace(args[0], 1.0, self.lkernel_z/(args[6]**2))
        dpdx = laplace(args[0], 1.0, self.gkernel_x/args[6])
        dpdz = laplace(args[0], 1.0, self.gkernel_z/args[6])
        return step(*args, nabla_x, nabla_z, dpdx, dpdz, **kwargs)
