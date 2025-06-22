import torch
import jax.numpy as jnp
from .operator import laplace
from .operator_jax import laplace as laplace_jax
from typing import Tuple, Optional, Union, List
from geophyai.scalars import generate_convolution_kernel
from .acoustic import Acoustic

def step(u_now, u_pre, su_now, su_pre, vp, ref, dt, h, b, lap_u_now, lap_su_now, habc_masks=None):
    """Step function for the acoustic LSRTM solver.

    Args:
        u_now (torch.Tensor): Current background wavefield.
        u_pre (torch.Tensor): Previous background wavefield.
        su_now (torch.Tensor): Current scatter wavefield.
        su_pre (torch.Tensor): Previous scatter wavefield.
        vp (torch.Tensor): Velocity model.
        ref (torch.Tensor): Reflectivity model.
        dt (torch.Tensor): Time step size.
        h (torch.Tensor): Grid spacing.
        b (torch.Tensor): Absorbing boundary condition coefficient.
        kernel (torch.Tensor): Convolution kernel for finite difference.
        habc_masks (Optional[Union[None, List[torch.Tensor]]]): Masks for absorbing boundary conditions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Updated wavefields.
    """
    a = (dt**-2 + b * dt**-1)**(-1)

    # background wavefield
    vp2_nabla_p0 = vp**2*lap_u_now
    u_next = a*(2. / dt**2 * u_now - (dt**-2-b*dt**-1)*u_pre + vp2_nabla_p0)
    
    # scatter wavefield
    vp2_nabla_sh0 = vp**2*lap_su_now
    su_next = a*(2. / dt**2 * su_now - (dt**-2-b*dt**-1)*su_pre + vp2_nabla_sh0 + ref*vp2_nabla_p0)

    return u_next, u_now, su_next, su_now

class AcousticLSRTM:

    def __init__(self, spatial_order=4, device='cpu', backend='torch'):
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
        return ['vp', 'ref']
    
    @property
    def wavefields(self):
        return ['h1', 'h2', 'sh1', 'sh2']
    
    def func(self, *args, **kwargs):
        lap_u_now = laplace(args[0], args[7], self.kernel)
        lap_su_now = laplace(args[2], args[7], self.kernel)
        return step(*args, lap_u_now, lap_su_now)
    
    def func_jax(self, *args, **kwargs):
        lap_u_now = laplace_jax(args[0], args[7], self.kernel)
        lap_su_now = laplace_jax(args[2], args[7], self.kernel)
        return step(*args, lap_u_now, lap_su_now)
    
