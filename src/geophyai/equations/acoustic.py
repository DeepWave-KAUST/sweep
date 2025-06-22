import torch, jax
import jax.numpy as jnp
from .operator import laplace
from .operator_jax import laplace as laplace_jax
from typing import Tuple, Optional, Union, List
from .habc import habc
from geophyai.scalars import generate_convolution_kernel

def step(u_now, u_pre, vp, dt, h, b, lap_u_now):
    a = (dt**-2 + b * dt**-1)**(-1)
    u_next = a*(2. / dt**2 * u_now - (dt**-2-b*dt**-1)*u_pre + vp**2*lap_u_now)
    return u_next, u_now

class Acoustic:

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch'):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        if backend == 'torch':
            self.kernel = torch.from_numpy(generate_convolution_kernel(spatial_order)).to(device)
        else:
            self.kernel = jnp.array(generate_convolution_kernel(spatial_order), dtype=jnp.float32).squeeze()
        
    @property
    def models(self):
        return ['vp']
    
    @property
    def wavefields(self):
        return ['h1', 'h2']

    def func(self, *args, **kwargs):
        lap_u_now = laplace(args[0], args[4], self.kernel)
        return step(*args, lap_u_now)
    
    def func_jax(self, *args, **kwargs):
        lap_u_now = laplace_jax(args[0], args[4], self.kernel)
        return step(*args, lap_u_now)
