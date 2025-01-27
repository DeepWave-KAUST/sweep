import torch, jax
import jax.numpy as jnp
from .operator import laplace
from .operator_jax import laplace as laplace_jax
from typing import Tuple, Optional, Union, List
from .habc import habc
from geophyai.scalars import generate_convolution_kernel


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
        return Acoustic.step(*args, kernel=self.kernel, **kwargs)
    
    def func_jax(self, *args, **kwargs):
        return Acoustic.step_jax(*args, self.kernel)

    @torch.jit.script
    def step(u_now: torch.Tensor, # u_{n} 
             u_pre: torch.Tensor, # u_{n-1}
             c: torch.Tensor,     # velocity
             dt: torch.Tensor,    # time step
             h: torch.Tensor,     # grid spacing
             b: torch.Tensor,     # ABC coefficient
             kernel: torch.Tensor,# convolution kernel (FD coefficients)
             habc_masks: Optional[Union[None, List[torch.Tensor]]]=None)  -> Tuple[torch.Tensor, torch.Tensor]:

        ### PML
        u_next = (dt**-2 + b * dt**-1).pow(-1) * ( 
            2 / dt**2 * u_now - (dt**-2 - b * dt**-1) * u_pre
            + c.pow(2) * laplace(u_now, h, kernel)
        )
        ### HABC
        # with out boundary
        # u_next = 2*u_now - u_pre + c.pow(2) *dt**2 * laplace(u_now, h, kernel)
        # apply habc
        # u_next = habc(u_next, u_now, u_pre, c, b, dt, h, maskidx=habc_masks)
        return u_next, u_now
    
    @jax.jit
    def step_jax(*args):

        u_now, u_pre, vp, dt, h, b, kernel = args
        _laplace_u = laplace_jax(u_now, h, kernel)
        a = (dt**-2 + b * dt**-1)**(-1)
        u_next = a*(2. / dt**2 * u_now - (dt**-2-b*dt**-1)*u_pre + vp**2*_laplace_u)
        
        return u_next, u_now
