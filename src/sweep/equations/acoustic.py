from .base import SecondOrderEquation
from .operator_jax import laplace3d as laplace3d_jax
import numpy as np

from .habc_jax import habc, bound_mask
import jax.numpy as jnp

def step_pml(u_now, u_pre, vp, dt, h, b, lap_u_now, habc_mask=None, habc_width=0):    
    a = 1 / (1 + b * dt)
    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * lap_u_now
    u_next = a * u_next + (1 - a) * u_now
    return u_next, u_now

def step_habc(u_now, u_pre, vp, dt, h, b, lap_x, lap_z, habc_mask=None, habc_width=50):
    u_next = 2*u_now - u_pre + vp**2*dt**2 * (lap_x + lap_z)
    u_next = habc(u_next, u_now, u_pre, vp, b, dt, h, maskidx=habc_mask, w=habc_width)
    return u_next, u_now

def step_cpml(u_now, u_pre, psix, psiz, zetax, zetaz, vp, dt, h, b, lap_x, lap_z, habc_mask=None, habc_width=50):

    az, bz, dbzdz, ax, bx, dbxdx = b

    w_sum = 0.

    # Z direction
    dwfcdy = jnp.gradient(u_now, h, axis=-2)
    tmpz = ((1+bz)*lap_z + dbzdz * dwfcdy) + jnp.gradient(az*psiz, h, axis=-2)
    w_sum += (1+bz) * tmpz + az * zetaz

    psiyn = bz * dwfcdy + az * psiz
    zetaz = bz * tmpz + az * zetaz

    # X direction
    dwfcdx = jnp.gradient(u_now, h, axis=-1) # x
    tmpx = ((1+bx)*lap_x + dbxdx * dwfcdx) + jnp.gradient(ax*psix, h, axis=-1)
    w_sum += (1+bx) * tmpx + ax * zetax
    psixn = bx * dwfcdx + ax * psix
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
        self._wavefields = ['h1', 'h2']

        self.use_habc = False
        self.use_cpml = False
        if backend == 'jax':
            super().init_laplace(ltype='1dsep')

    def init_habc(self, shape, abcn, free_surface=False, batchsize=1, use_habc=False):
        habc_masks = bound_mask(*shape, abcn, batchsize, return_idx=True, free_surface=free_surface)
        self.habc_masks = tuple([np.array(mask) if mask is not None else mask for mask in habc_masks])
        self.use_habc = use_habc
        self.abcn = abcn

    def init_cpml(self, abcn, free_surface=False):
        self.use_cpml = True
        self.wavefields += ['psix', 'psiz', 'zetax', 'zetaz']
        self.abcn = abcn

    @property
    def models(self):
        return ['vp']
    
    @property
    def wavefields(self):
        return self._wavefields

    @wavefields.setter
    def wavefields(self, value):
        self._wavefields = list(value)

    def func(self, *args, **kwargs):
        dh = args[4] if self.use_habc else args[8]
        lap_u_now_z, lap_u_now_x = self.laplace(args[0], self.kernel, dh, dh)
        step = step_habc if self.use_habc else step_cpml
        # return step(*args, lap_u_now_z + lap_u_now_x, self.habc_masks, habc_width=self.abcn)
        return step(*args, lap_u_now_x, lap_u_now_z, self.habc_masks, habc_width=self.abcn)

    def func_jax3d(self, *args, **kwargs):
        lap_u_now = laplace3d_jax(args[0], self.kernel[None, None, ...], args[4])
        return step_pml(*args, lap_u_now)