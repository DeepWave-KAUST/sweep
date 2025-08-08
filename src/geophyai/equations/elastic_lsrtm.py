import torch, jax
from functools import partial
from .operator import PartialDerivative
from typing import Tuple, Optional, Union, List, Any

# 10.1190/GEO2016-0254.1
def step(vx, vz, txx, tzz, txz,
         vxs, vzs, txxs, tzzs, txzs,
         mp, ms, vp, vs, rho, 
         dt, h, b, pd):

    lame_lambda = rho*(vp**2-2*vs**2)
    lame_mu = rho*vs**2
    Ip = rho*vp
    Is = rho*vs
    c = 0.5*dt*b

    # Elastic wavefields
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

    # Perturbed wavefields

    dlame_lambda = 2*(Ip**2*mp-2*Is**2*ms)/rho
    dlame_mu = 2*Is**2*ms/rho

    vxs_x = pd.x_forward(vxs)
    vzs_z = pd.z_backward(vzs)
    vxs_z = pd.z_forward(vxs)
    vzs_x = pd.x_backward(vzs)

    y_txxs  = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vxs_x+lame_lambda*vzs_z \
                                     +dlame_lambda*(vx_x+vz_z)+2*dlame_mu*vx_x)+(1-c)*txxs)
    y_tzzs  = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vzs_z+lame_lambda*vxs_x \
                                     +dlame_lambda*(vx_x+vz_z)+2*dlame_mu*vz_z)+(1-c)*tzzs)
    y_txzs = (1+c)**-1*(dt*h**(-1)*(lame_mu*(vzs_x+vxs_z)+dlame_mu*(vz_x+vx_z))+(1-c)*txzs)

    txxs_x = pd.x_backward(y_txxs)
    txzs_z = pd.z_backward(y_txzs)
    tzzs_z = pd.z_forward(y_tzzs)
    txzs_x = pd.x_forward(y_txzs)

    y_vxs = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txxs_x+txzs_z)+(1-c)*vxs)
    y_vzs = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txzs_x+tzzs_z)+(1-c)*vzs)

    return y_vx, y_vz, y_txx, y_tzz, y_txz, y_vxs, y_vzs, y_txxs, y_tzzs, y_txzs

class ElasticLSRTM:
    """Parameter order: mp, ms, vp, vs, rho.
    
       Wavefields: (vx, vz, txx, tzz, txz), (vxs, vzs, txxs, tzzs, txzs)

       Reference: Feng & Schuster, 10.1190/geo2016-0254.1
    """
    def __init__(self, spatial_order=4, device='cpu', backend = 'torch'):
        self.pd = PartialDerivative(spatial_order, device, backend)

    @property
    def models(self):
        return ['mp', 'ms', 'vp', 'vs', 'rho']
    
    @property
    def wavefields(self):
        return ['vx', 'vz', 'txx', 'tzz', 'txz', 'vxs', 'vzs', 'txxs', 'tzzs', 'txzs']
    
    def func(self, *args, **kwargs):
        return step(*args, pd=self.pd, **kwargs)
    
    def func_jax(self, *args, **kwargs):
        return step(*args, self.pd, **kwargs)