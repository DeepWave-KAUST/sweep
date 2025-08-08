import torch
import jax.numpy as jnp
import numpy as np
from .operator import PartialDerivative
from .operator_jax import PartialDerivative as PartialDerivativeJax
from .habc_jax import habc1st, bound_mask


def step(vx, vz, txx, tzz, txz,
         vp, vs, rho, 
         dt, h, b, pd, 
         habc_masks=None):

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

def step_habc(vx, vz, txx, tzz, txz,
              vp, vs, rho, 
              dt, h, b, pd, habc_masks):

    lame_lambda = rho*(vp**2-2*vs**2)
    lame_mu = rho*vs**2

    vx_x = pd.x_forward(vx)
    vz_z = pd.z_backward(vz)
    vx_z = pd.z_forward(vx)
    vz_x = pd.x_backward(vz)

    y_txx  = dt*h**(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+txx
    y_tzz  = dt*h**(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+tzz
    y_txz = dt*lame_mu*h**(-1)*(vz_x+vx_z)+txz

    y_txx = habc1st(y_txx, txx, vp, vs, b, dt, h, maskidx=habc_masks)
    y_tzz = habc1st(y_tzz, tzz, vp, vs, b, dt, h, maskidx=habc_masks)
    y_txz = habc1st(y_txz, txz, vp, vs, b, dt, h, maskidx=habc_masks)

    txx_x = pd.x_backward(y_txx)
    txz_z = pd.z_backward(y_txz)
    tzz_z = pd.z_forward(y_tzz)
    txz_x = pd.x_forward(y_txz)

    y_vx = dt*rho**(-1)*h**(-1)*(txx_x+txz_z)+vx
    y_vz = dt*rho**(-1)*h**(-1)*(txz_x+tzz_z)+vz

    y_vx = habc1st(y_vx, vx, vp, vs, b, dt, h, maskidx=habc_masks)
    y_vz = habc1st(y_vz, vz, vp, vs, b, dt, h, maskidx=habc_masks)

    return y_vx, y_vz, y_txx, y_tzz, y_txz

class Elastic:
    """Parameter order: vp, vs, rho.
    
       Wavefields: (vx, vz, txx, tzz, txz)

       Reference: Jean Virieux, 10.1190/1.1442147
    """
    def __init__(self, spatial_order=4, device='cpu', backend = 'torch'):
        if backend == 'torch':
            self.pd = PartialDerivative(spatial_order, device)
        else:
            self.pd = PartialDerivativeJax(spatial_order)
        self.use_habc = False

    def init_habc(self, shape, abcn, free_surface=False, batchsize=1, use_habc=False):
        habc_masks = bound_mask(*shape, abcn, batchsize, return_idx=True, free_surface=free_surface)
        self.habc_masks = tuple([np.array(mask) if mask is not None else mask for mask in habc_masks])
        self.use_habc = use_habc

    @property
    def models(self):
        return ['vp', 'vs', 'rho']
    
    @property
    def wavefields(self):
        return ['vx', 'vz', 'txx', 'tzz', 'txz']
    
    def func(self, *args, **kwargs):
        return step(*args, pd=self.pd, **kwargs)
    
    def func_jax(self, *args, **kwargs):
        _step = step_habc if self.use_habc else step
        return _step(*args, self.pd, habc_masks=self.habc_masks, **kwargs)