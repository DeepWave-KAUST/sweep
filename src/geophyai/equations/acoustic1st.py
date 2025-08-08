import torch
from .operator import PartialDerivative
from typing import Tuple, Optional, Union, List, Any

def step(p, vx, vz, vp, rho, dt, h, b, pd):

    c = 0.5*dt*b

    p_x = pd.x_backward(p)
    p_z = pd.z_backward(p)

    y_vx = (1+c)**-1*(-dt * rho**(-1)* p_x / h + (1-c)*vx)
    y_vz = (1+c)**-1*(-dt * rho**(-1)* p_z / h + (1-c)*vz)

    vx_x = pd.x_forward(y_vx)
    vz_z = pd.z_forward(y_vz)

    y_p = (1+c)**-1*(-vp**2*dt*rho / h*(vx_x+vz_z)+(1-c)*p)

    return y_p, y_vx, y_vz

class Acoustic:
    """
    Parameter order: vp, rho

    Wavefields: (p, vx, vz).

    References: 10.1190/GEO2011-0345.1
    """
    def __init__(self, spatial_order=4, device='cpu', backend='torch'):
        self.pd = PartialDerivative(spatial_order, device, backend)

    @property
    def models(self):
        return ['vp', 'rho']
    
    @property
    def wavefields(self):
        return ['p', 'vx', 'vz']
    
    def func(self, *args, **kwargs):
        return step(*args, pd=self.pd, **kwargs)
    
    def func_jax(self, *args, **kwargs):
        return step(*args, pd=self.pd, **kwargs)


