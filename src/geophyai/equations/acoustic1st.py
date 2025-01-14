import torch
from .operator import PartialDerivative
from typing import Tuple, Optional, Union, List, Any

class Acoustic:

    def __init__(self, spatial_order=4, device='cpu'):
        self.pd = torch.jit.script(PartialDerivative(spatial_order, device))
    
    @property
    def models(self):
        return ['vp', 'rho']
    
    @property
    def wavefields(self):
        return ['p', 'vx', 'vz']
    
    def func(self, *args, **kwargs):
        return Acoustic.step(*args, pd=self.pd, **kwargs)

    @torch.jit.script
    def step(p: torch.Tensor, #
             vx: torch.Tensor, #
             vz: torch.Tensor, #
             vp: torch.Tensor,     # velocity
             rho: torch.Tensor,    # density
             dt: torch.Tensor,    # time step
             h: torch.Tensor,     # grid spacing
             b: torch.Tensor,     # ABC coefficient
             pd: PartialDerivative, # partial derivative operator
             habc_masks: Optional[Union[None, List[torch.Tensor]]]=None)  -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        c = 0.5*dt*b

        p_x = pd.x_backward(p)
        p_z = pd.z_backward(p)

        y_vx = (1+c)**-1*(dt * rho.pow(-1)* p_x / h + (1-c)*vx)
        y_vz = (1+c)**-1*(dt * rho.pow(-1)* p_z / h + (1-c)*vz)

        vx_x = pd.x_forward(y_vx)
        vz_z = pd.z_forward(y_vz)

        y_p = (1+c)**-1*(vp**2*dt*rho*h.pow(-1)*(vx_x+vz_z)+(1-c)*p)

        return y_p, y_vx, y_vz