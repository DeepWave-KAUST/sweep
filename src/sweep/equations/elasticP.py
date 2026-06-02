import torch
import jax.numpy as jnp
from .base import FirstOrderEquation
from .fields import FieldSpec, ModelSpec

def step(vx, vz, txx, tzz, txz,
         vp, vs, rho, 
         dt, h, b, 
         ksquared, 
         pd, 
         backend):

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
    fvz = backend.fft.fft2(y_vz * backend.exp(-c))
    fvx = backend.fft.fft2(y_vx * backend.exp(-c))
    
    fvwz = backend.where(ksquared == 0., backend.zeros_like(fvz), fvz / (-ksquared))
    fvwx = backend.where(ksquared == 0., backend.zeros_like(fvx), fvx / (-ksquared))

    vwz = backend.fft.ifft2(fvwz).real
    vwx = backend.fft.ifft2(fvwx).real

    xx = pd.x_forward(vwx)*h**(-1)
    zz = pd.z_backward(vwz)*h**(-1)

    div = xx + zz

    vxp = pd.x_backward(div)*h**(-1) * backend.exp(-c)
    vzp = pd.z_forward(div)*h**(-1) * backend.exp(-c)

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

class ElasticP(FirstOrderEquation):
    """This class implements the 2D elastic wave equation for pure P-mode waves.
       Mu and Alkhalifah, 2024, 10.1111/1365-2478.13610
    """
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("p_velocity",), description="Elastic P-wave velocity model.", unit="m/s"),
        ModelSpec("vs", aliases=("s_velocity",), description="Elastic S-wave velocity model.", unit="m/s"),
        ModelSpec("rho", aliases=("density",), description="Density model.", unit="kg/m^3"),
    )
    FIELD_SPECS = (
        FieldSpec("vx", aliases=("velocity_x",), description="Particle velocity in the x direction.", supports_receiver=True),
        FieldSpec("vz", aliases=("velocity_z",), description="Particle velocity in the z direction.", supports_receiver=True),
        FieldSpec("txx", description="Normal stress component in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("tzz", description="Normal stress component in the z direction.", supports_source=True, supports_receiver=True),
        FieldSpec("txz", description="Shear stress component.", supports_source=True),
    )

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch'):
        self.backend = backend
        self.op = {'torch': torch, 'jax': jnp}[backend]
        super().__init__(spatial_order, device, backend)

    @property
    def need_init(self):
        return True

    def func(self, wavefields, models, dt, h, b, **kwargs):
        return step(*wavefields, *models, dt, h, b, self.k, self.pd, self.op, **kwargs)
