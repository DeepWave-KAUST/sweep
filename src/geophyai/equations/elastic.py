import torch, jax
from functools import partial
from .operator import PartialDerivative
from .operator_jax import PartialDerivative as PartialDerivativeJax
from typing import Tuple, Optional, Union, List, Any

class Elastic:

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch'):
        if backend == 'torch':
            self.pd = torch.jit.script(PartialDerivative(spatial_order, device))
        else:
            self.pd = PartialDerivativeJax(spatial_order)

    @property
    def models(self):
        return ['vp', 'vs', 'rho']
    
    @property
    def wavefields(self):
        return ['vx', 'vz', 'txx', 'tzz', 'txz']
    
    def func(self, *args, **kwargs):
        return Elastic.step(*args, pd=self.pd, **kwargs)
    
    def func_jax(self, *args, **kwargs):
        return Elastic.step_jax(*args, pd=self.pd, **kwargs)

    @torch.jit.script
    def step(vx: torch.Tensor, #
             vz: torch.Tensor, #
             txx: torch.Tensor, #
             tzz: torch.Tensor, #
             txz: torch.Tensor, #
             vp: torch.Tensor, #
             vs: torch.Tensor, #
             rho: torch.Tensor, #
             dt: torch.Tensor,    # time step
             h: torch.Tensor,     # grid spacing
             b: torch.Tensor,     # ABC coefficient
             pd: PartialDerivative,
             habc_masks: Optional[Union[None, List[torch.Tensor]]]=None)  -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        ### PML
        lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
        lame_mu = rho*(vs.pow(2))
        c = 0.5*dt*b

        vx_x = pd.x_forward(vx)
        vz_z = pd.z_backward(vz)
        vx_z = pd.z_forward(vx)
        vz_x = pd.x_backward(vz)

        y_txx  = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
        y_tzz  = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
        y_txz = (1+c)**-1*(dt*lame_mu*h**(-1)*(vz_x+vx_z)+(1-c)*txz)

        txx_x = pd.x_backward(y_txx)
        txz_z = pd.z_backward(y_txz)
        tzz_z = pd.z_forward(y_tzz)
        txz_x = pd.x_forward(y_txz)

        y_vx = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txx_x+txz_z)+(1-c)*vx)
        y_vz = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txz_x+tzz_z)+(1-c)*vz)

        return y_vx, y_vz, y_txx, y_tzz, y_txz

    @partial(jax.jit, static_argnums=(11,))
    def step_jax(vx, vz, txx, tzz, txz,
                 vp, vs, rho, 
                 dt, h, b, pd=None):

        lame_lambda = rho*(vp**2-2*vs**2)
        lame_mu = rho*vs**2
        c = 0.5*dt*b

        vx_x = pd.x_forward(vx)
        vz_z = pd.z_backward(vz)
        vx_z = pd.z_forward(vx)
        vz_x = pd.x_backward(vz)

        y_txx  = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
        y_tzz  = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
        y_txz = (1+c)**-1*(dt*lame_mu*h**(-1)*(vz_x+vx_z)+(1-c)*txz)

        txx_x = pd.x_backward(y_txx)
        txz_z = pd.z_backward(y_txz)
        tzz_z = pd.z_forward(y_tzz)
        txz_x = pd.x_forward(y_txz)

        y_vx = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txx_x+txz_z)+(1-c)*vx)
        y_vz = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txz_x+tzz_z)+(1-c)*vz)

        return y_vx, y_vz, y_txx, y_tzz, y_txz
