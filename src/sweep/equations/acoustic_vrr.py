import torch
import jax.numpy as jnp
import numpy as np
from .base import SecondOrderEquation

def step(u_now, u_pre, vp, rx, rz, dt, h, b, lap_u_now, dpdx, dpdz, dvpdx, dvpdz, pd, habc_masks=None):
    a = 1 / (1 + b * dt)
    u_next = 2 * u_now - u_pre + vp**2*dt**2*lap_u_now + vp*(dvpdx*dpdx + dvpdz*dpdz)*dt**2 - 2*vp**2*(rx*dpdx + rz*dpdz)*dt**2
    u_next = a * u_next + (1 - a) * u_now
    return u_next, u_now

# def step_habc(u_now, u_pre, vp, dt, h, b, lap_u_now, habc_mask):
#     u_next = 2*u_now - u_pre + vp**2*dt**2 * lap_u_now
#     u_next = habc(u_next, u_now, u_pre, vp, b, dt, h, maskidx=habc_mask)
#     return u_next, u_now

class AcousticVRR(SecondOrderEquation):
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
        
        super().__init__(spatial_order, device, backend, other_kernels=True)

    @property
    def models(self):
        return ['vp', 'rx', 'rz']
    
    @property
    def wavefields(self):
        return ['h1', 'h2']
    
    def func(self, *args, **kwargs):
        _func = {'torch': self.func_torch, 'jax': self.func_jax}[self.backend]
        return _func(*args, **kwargs)

    def func_jax(self, *args, **kwargs):
        lap_u_now = self.laplace(args[0], args[6], self.kernel)
        dvpdx = jnp.gradient(args[2],args[6], axis=-1) # 2nd Center Difference
        dvpdz = jnp.gradient(args[2],args[6], axis=-2)
        dpdx = jnp.gradient(args[0], args[6], axis=-1)
        dpdz = jnp.gradient(args[0], args[6], axis=-2)
        return step(*args, lap_u_now, dpdx, dpdz, dvpdx, dvpdz, self.pd, self.habc_masks)
      
    def func_torch(self, *args, **kwargs):
        lap_u_now = self.laplace(args[0], args[6], self.kernel)
        dvpdx = torch.gradient(args[2],args[6], axis=-1)[0] # 2nd Center Difference
        dvpdz = torch.gradient(args[2],args[6], axis=-2)[0]
        dpdx = torch.gradient(args[0], args[6], axis=-1)[0]
        dpdz = torch.gradient(args[0], args[6], axis=-2)[0]
        return step(*args, lap_u_now, dpdx, dpdz, dvpdx, dvpdz, self.pd, self.habc_masks)
