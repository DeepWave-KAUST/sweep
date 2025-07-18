import torch, jax
from functools import partial
from .operator import PartialDerivative
from .operator_jax import PartialDerivative as PartialDerivativeJax
from typing import Tuple, Optional, Union, List, Any

def step(p, vx, vz, txx, tzz, txz,
         vp, vs, rho, 
         dt, h, b, pd):

    lame_lambda = rho*(vp**2-2*vs**2)
    lame_mu = rho*vs**2
    c = 0.5*dt*b

    txx_x = pd.x_forward(txx)
    txz_z = pd.z_forward(txz)
    tzz_z = pd.z_backward(tzz)
    txz_x = pd.x_backward(txz)

    p_x = pd.x_forward(p)
    p_z = pd.z_backward(p)

    y_vx = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txx_x+txz_z-p_x)+(1-c)*vx)
    y_vz = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txz_x+tzz_z-p_z)+(1-c)*vz)

    vx_x = pd.x_backward(y_vx)
    vz_z = pd.z_forward(y_vz)
    vx_z = pd.z_backward(y_vx)
    vz_x = pd.x_forward(y_vz)

    y_txx = (1+c)**-1*(dt*lame_mu*h**(-1)*(vx_x-vz_z)+(1-c)*txx)
    y_tzz = (1+c)**-1*(dt*lame_mu*h**(-1)*(vz_z-vx_x)+(1-c)*tzz)
    y_txz = (1+c)**-1*(dt*lame_mu*h**(-1)*(vz_x+vx_z)+(1-c)*txz)
    y_p = (1+c)**-1*(-dt*(lame_lambda+lame_mu)*h**(-1)*(vx_x+vz_z)+(1-c)*p)

    return y_p, y_vx, y_vz, y_txx, y_tzz, y_txz

class AcousticElasticCoupled:
    """
    Parameter order:vp, vs, rho.

    Wavefields: (p, vx, vz, txx, tzz, txz)
    """
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
        return ['p', 'vx', 'vz', 'txx', 'tzz', 'txz']
    
    def func(self, *args, **kwargs):
        return step(*args, pd=self.pd, **kwargs)
    
    def func_jax(self, *args, **kwargs):
        return step(*args, self.pd, **kwargs)