import numpy as np
from .base import SecondOrderEquation

def step(u_now, u_pre, f_now, f_next, # wavefields
         vv, v, eta, # Model parameters 
         dt, h, b,  # Auxiliary parameters
         nabla_x, nabla_z, dpdx2dz2, # Partial derivatives
         ):
    
    a = 1 / (1 + b * dt)

    # Equation 22
    u_next = 2*u_now - u_pre + (1+2*eta)*v**2*nabla_x*dt**2 + vv**2*nabla_z*dt**2 - 2*eta*vv**2*v**2*dpdx2dz2*dt**2
    # Equation 26
    f_next = 2*f_now - f_next + dt**2*u_now

    u_next = a * u_next + (1 - a) * u_now
    f_next = a * f_next + (1 - a) * f_now

    return u_next, u_now, f_next, f_now


class AcousticTariq(SecondOrderEquation):
    """Parameter order: vv, v, eta.
    
       Wavefields: (h1, h2, f1, f2)

       Reference: Alkhalifah Tariq, 10.1190/1.1444815
    """
    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=2):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        super().__init__(spatial_order, device, backend, other_kernels=True)
        super().init_laplace(ltype='1dsep', backend=backend)

    @property
    def models(self):
        return ['vv', 'v', 'eta']
    
    @property
    def wavefields(self):
        return ['h1', 'h2', 'f1', 'f2']
    
    def func(self, *args, **kwargs):
        hz, hx = self._spacings_2d(args[8])
        nabla_z, nabla_x = self.laplace1d_sep(args[0], self.laplace_kernels, hz, hx)
        _, dpdx2 = self.laplace1d_sep(args[2], self.laplace_kernels, hz, hx)
        dpdx2dz2, _ = self.laplace1d_sep(dpdx2, self.laplace_kernels, hz, hx)
        return step(*args, nabla_x, nabla_z, dpdx2dz2, **kwargs)
