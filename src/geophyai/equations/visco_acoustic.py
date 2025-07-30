import jax
import numpy as np
from .operator_jax import laplace
import jax.numpy as jnp
from typing import Tuple, Optional, Union, List
from geophyai.scalars import generate_convolution_kernel
from .utils import to_backend
from .habc_jax import habc, bound_mask

def step(u_now, u_pre, # Wavefields
         vp, Q, omega,# Model parameters 
         dt, h, b,  # Auxiliary parameters
         k, # Wavenumbers
         laplace_u_now, # Partial derivatives
         phase_shift=True, # Phase shift
         amplitude_damping=True, # Amplitude damping
         habc_masks=None):
    
    a = 1 / (1 + b * dt)

    t_sigma = omega**-1*(jnp.sqrt(1+(Q**-2))-Q**-1)
    t_epslion = (omega**2 * t_sigma)**-1.
    t = t_epslion/(t_sigma-1e-8) - 1.

    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * laplace_u_now

    def linear(u_next):
        return u_next

    def phase(u_next):
        u_next = u_next - ((1-jnp.sqrt(Q**2+1))*Q**-2)*vp**2*laplace_u_now*dt**2
        return u_next
    
    def amplitude(u_next):
        dudt = (u_now-u_pre)/dt
        fft_dudt = jnp.fft.fft2(dudt, axes=(-2, -1))
        temp = jnp.fft.ifft2(k*fft_dudt, axes=(-2, -1)).real
        u_next = u_next - (dt**2*t*vp/2)*temp        
        return u_next

    u_next = jax.lax.cond(phase_shift, phase, linear, u_next)
    u_next = jax.lax.cond(amplitude_damping, amplitude, linear, u_next)

    u_next = a * u_next + (1 - a) * u_pre

    return u_next, u_now


class ViscoAcoustic:
    """This class is the implementation of the acoustic wave equation solver with TTI media.
       Reference: Liang K., et.al, 10.1190/geo2022-0292.1 (EQUATION A-7)
    """
    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=2, phase_shift=True, amplitude_damping=True):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        
        # Second order laplace kernel (Second derivative), Full kernel
        kernel = generate_convolution_kernel(spatial_order)
        self.kernel = to_backend(kernel, backend, device)

        self.backend = backend

        self.phase_shift = phase_shift
        self.amplitude_damping = amplitude_damping

    def init_habc(self, shape, abcn, free_surface=False, batchsize=1, use_habc=False):
        habc_masks = bound_mask(*shape, abcn, batchsize, return_idx=True, free_surface=free_surface)
        self.habc_masks = tuple([np.array(mask) if mask is not None else mask for mask in habc_masks])
        self.use_habc = use_habc

    @property
    def need_init(self):
        return True

    def init(self, shape, device, h):
        assert shape is not None, "shape must be provided to calculate the wavenumbers!!!"
        kz = np.fft.fftfreq(shape[0], d=h) * 2 * np.pi
        kx = np.fft.fftfreq(shape[1], d=h) * 2 * np.pi
        kzz, kxx = np.meshgrid(kz, kx, indexing='ij')
        self.k = np.sqrt(kxx**2 + kzz**2)
        self.kx = to_backend(kxx, self.backend, device)
        self.kz = to_backend(kzz, self.backend, device)
        self.k = to_backend(self.k, self.backend, device)
    
    @property
    def models(self):
        return ['vp', 'Q', 'omega']
    
    @property
    def wavefields(self):
        return ['h1', 'h2']
    
    def func_jax(self, *args, **kwargs):
        laplace_u_now = laplace(args[0], args[6], self.kernel)
        return step(*args, self.k, laplace_u_now, self.phase_shift, self.amplitude_damping, **kwargs)
