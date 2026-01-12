import numpy as np
from .base import SecondOrderEquation

def step(u_now, u_pre, # Wavefields
         vp, epsilon, delta, # Model parameters
         dt, h, b,  # Auxiliary parameters
         nabla_x, nabla_z, dpdx, dpdz,# Partial derivatives
         ):

        # 10.1190/geo2022-0292.1 EQ(19) from 10.1190/geo2014-0242.1
        numerator = -2*(epsilon-delta)*dpdx**2*dpdz**2 
        denominator = (1+2*epsilon)*dpdx**4+dpdz**4+2*(1+delta)*dpdx**2*dpdz**2
        sk = numerator*((denominator+1e-26)**-1)

        vp2dt2 = vp**2*dt**2

        a1 = 1+b*dt
        a2 = 1-b*dt
        
        # 10.1190/geo2022-0292.1 EQ(22) No ABC
        u_next = 2*a1**-1*u_now-a2*a1**(-1)*u_pre + a1**-1*(vp2dt2*((1+2*epsilon)+sk)*nabla_x + vp2dt2*(1+sk)*nabla_z)

        return u_next, u_now


class AcousticVTI(SecondOrderEquation):
    """Parameter order: vp, epsilon, delta.
    
       Wavefields: (h1, h2)

       Reference: Liang K., et.al, 10.1190/geo2022-0292.1
    """
    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=2):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        
        super().__init__(spatial_order, device, backend, other_kernels=True)
    
    @property
    def models(self):
        return ['vp', 'epsilon', 'delta']
    
    @property
    def wavefields(self):
        return ['h1', 'h2']
    
    def func(self, *args, **kwargs):
        nabla_x = self.laplace(args[0], 1.0, self.lkernel_x/(args[6]**2))
        nabla_z = self.laplace(args[0], 1.0, self.lkernel_z/(args[6]**2))
        dpdx = self.laplace(args[0], 1.0, self.gkernel_x/args[6])
        dpdz = self.laplace(args[0], 1.0, self.gkernel_z/args[6])
        return step(*args, nabla_x, nabla_z, dpdx, dpdz, **kwargs)
