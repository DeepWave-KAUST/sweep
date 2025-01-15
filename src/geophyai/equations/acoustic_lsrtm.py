import torch
from .operator import laplace
from typing import Tuple, Optional, Union, List
from geophyai.scalars import generate_convolution_kernel
from .acoustic import Acoustic

class AcousticLSRTM:

    def __init__(self, spatial_order=4, device='cpu'):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        self.kernel = torch.from_numpy(generate_convolution_kernel(spatial_order)).to(device)
    
    @property
    def models(self):
        return ['vp', 'ref']
    
    @property
    def wavefields(self):
        return ['h1', 'h2', 'sh1', 'sh2']
    
    def func(self, *args, **kwargs):
        return AcousticLSRTM.step(*args, kernel=self.kernel, **kwargs)

    @torch.jit.script
    def step(u_now: torch.Tensor, #
             u_pre: torch.Tensor, #
             su_now: torch.Tensor, #
             su_pre: torch.Tensor, #
             vp: torch.Tensor,     # velocity
             ref: torch.Tensor,    # reflectivity
             dt: torch.Tensor,    # time step
             h: torch.Tensor,     # grid spacing
             b: torch.Tensor,     # ABC coefficient
             kernel: torch.Tensor,# convolution kernel (FD coefficients)
             habc_masks: Optional[Union[None, List[torch.Tensor]]]=None)  -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        a = (dt**-2 + b * dt**-1)**(-1)

        # background wavefield
        vp2_nabla_p0 = vp**2*laplace(u_now, h, kernel)
        u_next = a*(2. / dt**2 * u_now - (dt**-2-b*dt**-1)*u_pre + vp2_nabla_p0)
        
        # scatter wavefield
        vp2_nabla_sh0 = vp**2*laplace(su_now, h, kernel)
        su_next = a*(2. / dt**2 * su_now - (dt**-2-b*dt**-1)*su_pre + vp2_nabla_sh0 + ref*vp2_nabla_p0)

        return u_next, u_now, su_next, su_now