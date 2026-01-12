import numpy as np
import torch
import jax.numpy as jnp
from .base import SecondOrderEquation

def step(u_now, u_pre, # Wavefields
         vp, epsilon, delta, theta, # Model parameters 
         dt, h, b,  # Auxiliary parameters
         dpdx, dpdz, # Wavenumbers
         nabla_x, nabla_z, dpdx_dz, # Partial derivatives
         op=None,  # Operator (Jax or Torch)
         ):

    # from degree to radian
    theta = op.deg2rad(theta)
    sin0 = op.sin(theta)
    cos0 = op.cos(theta)
    sin20 = op.sin(2*theta)

    # 10.1190/geo2022-0292.1 EQ(A-5)
    # numerator = -2*(epsilon-delta)*(kx*cos0-kz*sin0)**2*(kx*sin0+kz*cos0)**2
    # denominator = (1+2*epsilon)*(kx*cos0-kz*sin0)**4+(kx*sin0+kz*cos0)**4+2*(1+delta)*(kx*cos0-kz*sin0)**2*(kx*sin0+kz*cos0)**2
    # sk = numerator*((denominator+1e-26)**-1)

    numerator = -2*(epsilon-delta)*(dpdx*cos0-dpdz*sin0)**2*(dpdx*sin0+dpdz*cos0)**2
    denominator = (1+2*epsilon)*(dpdx*cos0-dpdz*sin0)**4+(dpdx*sin0+dpdz*cos0)**4+2*(1+delta)*(dpdx*cos0-dpdz*sin0)**2*(dpdx*sin0+dpdz*cos0)**2
    sk = numerator*((denominator+1e-26)**-1)

    vp2dt2 = vp**2*dt**2

    a1 = 1+b*dt
    a2 = 1-b*dt
    
    # 10.1190/geo2022-0292.1 EQ(A-7)
    # u_next = 2*u_now-u_pre + vp2dt2*((1+2*epsilon)*cos0**2+sin0**2+sk)*nabla_x + vp2dt2*((1+2*epsilon)*sin0**2+cos0**2+sk)*nabla_z-2*epsilon*vp2dt2*sin20*dpdx_dz
    u_next = 2*a1**-1*u_now-a2*a1**(-1)*u_pre + a1**-1*(vp2dt2*((1+2*epsilon)*cos0**2+sin0**2+sk)*nabla_x + vp2dt2*((1+2*epsilon)*sin0**2+cos0**2+sk)*nabla_z-2*epsilon*vp2dt2*sin20*dpdx_dz)

    return u_next, u_now


class AcousticTTI(SecondOrderEquation):
    """Parameter order: vp, epsilon, delta, theta.
    
       Wavefields: (h1, h2)

       Reference: Liang K., et.al, 10.1190/geo2022-0292.1
    """
    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=2):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        super().__init__(spatial_order, device, backend, other_kernels=True)
        self.op = {'torch': torch, 'jax': jnp}[backend]
    
    @property
    def models(self):
        return ['vp', 'epsilon', 'delta', 'theta']
    
    @property
    def wavefields(self):
        return ['h1', 'h2']
    
    def func(self, *args, **kwargs):
        nabla_x = self.laplace(args[0], 1.0, self.lkernel_x/(args[7]**2))
        nabla_z = self.laplace(args[0], 1.0, self.lkernel_z/(args[7]**2))
        dpdx = self.laplace(args[0], 1.0, self.gkernel_x/args[7]) # 10.1190/geo2022-0292.1 EQ(21)
        dpdz = self.laplace(args[0], 1.0, self.gkernel_z/args[7])
        dpdx_dz = self.laplace(dpdx, 1.0, self.gkernel_z/args[7])
        return step(*args, dpdx, dpdz, nabla_x, nabla_z, dpdx_dz, self.op, **kwargs)
