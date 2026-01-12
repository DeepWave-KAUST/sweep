from .base import SecondOrderEquation
from .operator_jax import laplace3d as laplace3d_jax
import numpy as np

from .operator import gradientO2
import jax.numpy as jnp

def step_cpml(
        u_now, u_pre, psix, psiz, zetax, zetaz, 
        vp, dt, h, b, 
        lap_x, lap_z, 
        dudx, dudz
        ):

    az, bz, dbzdz, ax, bx, dbxdx = b

    w_sum = 0.

    # Z direction
    tmpz = ((1+bz)*lap_z + dbzdz * dudz) + gradientO2(az*psiz, h, -2)
    w_sum += (1+bz) * tmpz + az * zetaz

    psiyn = bz * dudz + az * psiz
    zetaz = bz * tmpz + az * zetaz

    # X direction
    tmpx = ((1+bx)*lap_x + dbxdx * dudx) + gradientO2(ax*psix, h, -1)
    w_sum += (1+bx) * tmpx + ax * zetax
    psixn = bx * dudx + ax * psix
    zetax = bx * tmpx + ax * zetax

    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * w_sum

    return u_next, u_now, psixn, psiyn, zetax, zetaz

class Acoustic(SecondOrderEquation):

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=2):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        super().__init__(spatial_order, device, backend, dim=dim)
        self._wavefields = []
        super().init_laplace(ltype='1dsep', backend=backend)

    @property
    def models(self):
        return ['vp']
    
    @property
    def wavefields(self):
        return ['h1', 'h2', 'psix', 'psiz', 'zetax', 'zetaz']

    def func(self, *args, **kwargs):
        dh = args[8]
        dudz = gradientO2(args[0], dh, -2)
        dudx = gradientO2(args[0], dh, -1)
        lap_u_now_z, lap_u_now_x = self.laplace(args[0], self.kernel, dh, dh)
        return step_cpml(*args, lap_u_now_x, lap_u_now_z, dudx, dudz)