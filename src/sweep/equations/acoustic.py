from .base import SecondOrderEquation
from .operator_jax import laplace3d as laplace3d_jax
import numpy as np

from .habc_jax import habc, bound_mask

def step_pml(u_now, u_pre, vp, dt, h, b, lap_u_now, habc_mask=None):    
    a = 1 / (1 + b * dt)
    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * lap_u_now
    u_next = a * u_next + (1 - a) * u_now
    return u_next, u_now

def step_habc(u_now, u_pre, vp, dt, h, b, lap_u_now, habc_mask):
    u_next = 2*u_now - u_pre + vp**2*dt**2 * lap_u_now
    u_next = habc(u_next, u_now, u_pre, vp, b, dt, h, maskidx=habc_mask)
    return u_next, u_now

class Acoustic(SecondOrderEquation):

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=2):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        super().__init__(spatial_order, device, backend, dim=dim)
        super().init_laplace(ltype='1dsep')

    def init_habc(self, shape, abcn, free_surface=False, batchsize=1, use_habc=False):
        habc_masks = bound_mask(*shape, abcn, batchsize, return_idx=True, free_surface=free_surface)
        self.habc_masks = tuple([np.array(mask) if mask is not None else mask for mask in habc_masks])
        self.use_habc = use_habc

    @property
    def models(self):
        return ['vp']
    
    @property
    def wavefields(self):
        return ['h1', 'h2']

    def func(self, *args, **kwargs):
        lap_u_now = self.laplace(args[0], self.kernel, args[4], args[4])
        step = step_habc if self.use_habc else step_pml
        return step(*args, lap_u_now, self.habc_masks)

    def func_jax3d(self, *args, **kwargs):
        lap_u_now = laplace3d_jax(args[0], self.kernel[None, None, ...], args[4])
        return step_pml(*args, lap_u_now)