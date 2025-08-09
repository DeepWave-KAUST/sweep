import jax, torch
import numpy as np
import jax.numpy as jnp
from .base import SecondOrderEquation
from .utils import to_backend
from .habc_jax import habc, bound_mask

def cond_torch(pred, true_fn, false_fn, args):
    # Make sure args is a tuple
    if not isinstance(args, tuple):
        args = (args,)
    return torch.cond(pred, true_fn, false_fn, args)

def step(u_now, u_pre, # Wavefields
         vp, Q, omega,# Model parameters 
         dt, h, b,  # Auxiliary parameters
         k, # Wavenumbers
         laplace_u_now, # Partial derivatives
         phase_shift=True, # Phase shift
         amplitude_damping=True, # Amplitude damping
         op=None,  # Operator (jax or torch)
         cond_op = None, # Operator (jax.lax.cond or torch.cond)
         habc_masks=None):
    
    a = 1 / (1 + b * dt)

    t_sigma = omega**-1*(op.sqrt(1+(Q**-2))-Q**-1)
    t_epslion = (omega**2 * t_sigma)**-1.
    t = t_epslion/(t_sigma-1e-8) - 1.

    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * laplace_u_now

    def linear(u_next):
        return u_next

    def phase(u_next):
        u_next = u_next - ((1-op.sqrt(Q**2+1))*Q**-2)*vp**2*laplace_u_now*dt**2
        return u_next
    
    def amplitude(u_next):
        dudt = (u_now-u_pre)/dt
        fft_dudt = op.fft.fft2(dudt, u_next.shape[-2:], (-2, -1))
        temp = op.fft.ifft2(k*fft_dudt, u_next.shape[-2:], (-2, -1)).real
        u_next = u_next - (dt**2*t*vp/2)*temp        
        return u_next

    u_next = cond_op(phase_shift, phase, linear, u_next)
    u_next = cond_op(amplitude_damping, amplitude, linear, u_next)

    u_next = a * u_next + (1 - a) * u_now

    return u_next, u_now

class ViscoAcoustic(SecondOrderEquation):
    """Parameter order: vp, Q, omega.
    
       Wavefields: (h1, h2)

       Reference: Wang Enjiang, Thesis.
    """
    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=2, phase_shift=True, amplitude_damping=True):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        
        # Second order laplace kernel (Second derivative), Full kernel
        super().__init__(spatial_order, device, backend, dim=dim)

        self.backend = backend

        self.phase_shift = phase_shift
        self.amplitude_damping = amplitude_damping

        if backend == 'torch':
            self.phase_shift = torch.tensor(phase_shift, dtype=torch.bool, device=device)
            self.amplitude_damping = torch.tensor(amplitude_damping, dtype=torch.bool, device=device)

        self.op = {'torch': torch, 'jax': jnp}[backend]
        self.cond_op = {'torch': cond_torch, 'jax': jax.lax.cond}[backend]

    def init_habc(self, shape, abcn, free_surface=False, batchsize=1, use_habc=False):
        habc_masks = bound_mask(*shape, abcn, batchsize, return_idx=True, free_surface=free_surface)
        self.habc_masks = tuple([np.array(mask) if mask is not None else mask for mask in habc_masks])
        self.use_habc = use_habc

    @property
    def need_init(self):
        return True
    
    @property
    def models(self):
        return ['vp', 'Q', 'omega']
    
    @property
    def wavefields(self):
        return ['h1', 'h2']
    
    def func(self, *args, **kwargs):
        laplace_u_now = self.laplace(args[0], args[6], self.kernel)
        return step(*args, self.k, laplace_u_now, self.phase_shift, self.amplitude_damping, self.op, self.cond_op, **kwargs)
