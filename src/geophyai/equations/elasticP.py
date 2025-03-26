import torch, jax
import jax.numpy as jnp
from functools import partial
from torch.fft import fft2, ifft2
from .operator import PartialDerivative
from .operator_jax import PartialDerivative as PartialDerivativeJax
from typing import Tuple, Optional, Union, List, Any
from geophyai.scalars import generate_convolution_kernel

class ElasticP:
    """This class implements the 2D elastic wave equation for pure P-mode waves.
       Mu and Alkhalifah, 2024, 10.1111/1365-2478.13610
    """

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch'):
        self.backend = backend
        if backend == 'torch':
            self.pd = torch.jit.script(PartialDerivative(spatial_order, device))
        else:
            self.pd = PartialDerivativeJax(spatial_order)

    @property
    def need_init(self):
        return True

    def init(self, shape, device, h):
        if self.backend == 'torch':
            assert shape is not None, "shape must be provided to calculate the wavenumbers!!!"
            kz = torch.fft.fftfreq(shape[0], d=h, dtype=torch.float32).to(device) * 2 * torch.pi
            kx = torch.fft.fftfreq(shape[1], d=h, dtype=torch.float32).to(device) * 2 * torch.pi
            kz, kx = torch.meshgrid(kz, kx, indexing='ij')
            self.ksquared = kx**2 + kz**2
            self.ksquared = self.ksquared.unsqueeze(0).unsqueeze(0)
        else:
            kz = jax.numpy.fft.fftfreq(shape[0], d=h).astype(jax.numpy.float32) * 2 * jax.numpy.pi
            kx = jax.numpy.fft.fftfreq(shape[1], d=h).astype(jax.numpy.float32) * 2 * jax.numpy.pi
            kz, kx = jax.numpy.meshgrid(kz, kx, indexing='ij')
            self.ksquared = kx**2 + kz**2
            self.ksquared = self.ksquared[None, None, ...]

    @property
    def models(self):
        return ['vp', 'vs', 'rho']
    
    @property
    def wavefields(self):
        return ['vx', 'vz', 'txx', 'tzz', 'txz']
    
    def func(self, *args, **kwargs):
        wavenumbers = [self.ksquared]
        return ElasticP.step(*args, self.pd, *wavenumbers, **kwargs)
    
    def func_jax(self, *args, **kwargs):
        return ElasticP.step_jax(*args, self.ksquared, pd=self.pd, **kwargs)

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
             ksquared: torch.Tensor,
             habc_masks: Optional[Union[None, List[torch.Tensor]]]=None)  -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        lame_lambda = rho*(vp.pow(2)-2*vs.pow(2))
        lame_mu = rho*(vs.pow(2))
        c = 0.5*dt*b*3.

        ############  Update Vx and Vz  ############
        txx_x = pd.x_backward(txx)
        txz_z = pd.z_backward(txz)
        tzz_z = pd.z_forward(tzz)
        txz_x = pd.x_forward(txz)

        y_vx = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txx_x+txz_z)+(1-c)*vx)
        y_vz = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txz_x+tzz_z)+(1-c)*vz)

        ############  Update Vxp and Vzp  ############
        fvz = fft2(y_vz * torch.exp(-c))
        fvx = fft2(y_vx * torch.exp(-c))
        
        fvwz = torch.where(ksquared == 0., torch.zeros_like(fvz), fvz / (-ksquared))
        fvwx = torch.where(ksquared == 0., torch.zeros_like(fvx), fvx / (-ksquared))
 
        vwz = ifft2(fvwz).real
        vwx = ifft2(fvwx).real

        xx = pd.x_forward(vwx)*h**(-1)
        zz = pd.z_backward(vwz)*h**(-1)

        div = xx + zz

        vxp = pd.x_backward(div)*h**(-1) * torch.exp(-c)
        vzp = pd.z_forward(div)*h**(-1) * torch.exp(-c)

        ############### Update P ###############
        vx_x = pd.x_forward(vxp)
        vz_z = pd.z_backward(vzp)
        vx_z = pd.z_forward(vxp)
        vz_x = pd.x_backward(vzp)

        # Equation
        y_txx = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
        y_tzz = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
        y_txz = (1+c)**-1*(dt*h**(-1)*lame_mu*(vx_z+vz_x)+(1-c)*txz)

        return y_vx, y_vz, y_txx, y_tzz, y_txz


    @partial(jax.jit, static_argnums=(12,))
    def step_jax(vx, vz, txx, tzz, txz,
                 vp, vs, rho, 
                 dt, h, b, ksquared, pd=None):

        lame_lambda = rho*(vp**2-2*vs**2)
        lame_mu = rho*vs**2
        c = 0.5*dt*b*3.

        ############  Update Vx and Vz  ############
        txx_x = pd.x_backward(txx)
        txz_z = pd.z_backward(txz)
        tzz_z = pd.z_forward(tzz)
        txz_x = pd.x_forward(txz)

        y_vx = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txx_x+txz_z)+(1-c)*vx)
        y_vz = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txz_x+tzz_z)+(1-c)*vz)

        ############  Update Vxp and Vzp  ############
        fvz = jnp.fft.fft2(y_vz * jnp.exp(-c))
        fvx = jnp.fft.fft2(y_vx * jnp.exp(-c))
        
        fvwz = jnp.where(ksquared == 0., jnp.zeros_like(fvz), fvz / (-ksquared))
        fvwx = jnp.where(ksquared == 0., jnp.zeros_like(fvx), fvx / (-ksquared))
 
        vwz = jnp.fft.ifft2(fvwz).real
        vwx = jnp.fft.ifft2(fvwx).real

        xx = pd.x_forward(vwx)*h**(-1)
        zz = pd.z_backward(vwz)*h**(-1)

        div = xx + zz

        vxp = pd.x_backward(div)*h**(-1) * jnp.exp(-c)
        vzp = pd.z_forward(div)*h**(-1) * jnp.exp(-c)

        ############### Update P ###############
        vx_x = pd.x_forward(vxp)
        vz_z = pd.z_backward(vzp)
        vx_z = pd.z_forward(vxp)
        vz_x = pd.x_backward(vzp)

        # Equation
        y_txx = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
        y_tzz = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
        y_txz = (1+c)**-1*(dt*h**(-1)*lame_mu*(vx_z+vz_x)+(1-c)*txz)

        return y_vx, y_vz, y_txx, y_tzz, y_txz