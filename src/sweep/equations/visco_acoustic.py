import numpy as np
from .base import SecondOrderEquation
from .fields import FieldSpec, ModelSpec
from .utils import to_backend
from .acoustic import step_cpml

class ViscoAcoustic(SecondOrderEquation):
    """Parameter order: vp, Q, omega.
    
       Wavefields: (h1, h2)

       Reference: Wang Enjiang, Thesis.
    """
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="Visco-acoustic wave velocity model.", unit="m/s"),
        ModelSpec("Q", description="Quality factor controlling attenuation."),
        ModelSpec("omega", description="Angular reference frequency for visco-acoustic attenuation.", unit="rad/s"),
    )
    FIELD_SPECS = (
        FieldSpec("h1", aliases=("pressure", "p"), description="Primary visco-acoustic pressure-like wavefield.", supports_source=True, supports_receiver=True),
        FieldSpec("h2", aliases=("pressure_prev",), description="Previous-step pressure-like wavefield.", internal=True),
        FieldSpec("psix", description="CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiz", description="CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("zetax", description="CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetaz", description="CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
    )
    default_pml_type = "cpmlr"
    
    def __init__(self, spatial_order=4, device='cpu', backend='torch', dim=2,
                 phase_shift=True, amplitude_damping=True):
        """Visco-acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
            phase_shift (bool, optional): Apply the (non-dissipative) dispersion
                term that shifts the wavefront. Defaults to True.
            amplitude_damping (bool, optional): Apply the dissipative attenuation
                term that decays the wavefront. Defaults to True.
                Both off = acoustic; both on = full visco-acoustic.
        """

        super().__init__(spatial_order, device, backend, dim=dim)
        super().init_separable_laplace()

        if backend == 'torch' and 'cuda' in str(device):
            super().init_grad_kernels()

        self.backend = backend
        self.phase_shift = phase_shift
        self.amplitude_damping = amplitude_damping

        if backend == 'torch':
            import torch
            self.op = torch
        elif backend == 'jax':
            import jax
            import jax.numpy as jnp
            self.op = jnp
        else:
            raise ValueError(f"Unknown backend: {backend}")

        

    @property
    def need_init(self):
        return True
    
    def func(self, wavefields, models, dt, h, b, **kwargs):
        u_now, u_pre = wavefields[0], wavefields[1]
        vp, Q, omega = models
        hz, hx = self._spacings_2d(h)

        # Rebuild the FFT wavenumber grid to match the wavefield the FFT sees
        # (PML pad + stencil halo), once, then cache it. self.k as built by
        # need_init is sized for the un-padded grid and mismatches under CPML.
        fft_shape = tuple(int(s) for s in u_now.shape[-2:])
        if getattr(self, "_k_fft_shape", None) != fft_shape:
            from .base import init_wavenumbers
            h_scalar = float(np.asarray(hz).reshape(-1)[0])
            k_np, _, _ = init_wavenumbers(fft_shape, h_scalar)
            self.k = to_backend(k_np, self.backend, str(vp.device))
            self._k_fft_shape = fft_shape
        lap_u_now_z, lap_u_now_x = self.separable_d2_2d(u_now, self.laplace_kernels, hz, hx)
        laplace_u_now = lap_u_now_x + lap_u_now_z


        # CPML base step

        out = step_cpml(*wavefields, vp, dt, h, b, lap_u_now_x, lap_u_now_z, self.b, self.gradient, self.grad_kernels)
        u_next = out[0]

        # visco-acoustic attenuation corrections (each independently toggleable)
        op = self.op

        # phase shift (dispersion: moves the wavefront, non-dissipative)
        if self.phase_shift:
            u_next = u_next - ((1 - op.sqrt(Q**2 + 1)) * Q**-2) * vp**2 * laplace_u_now * dt**2

        # amplitude damping (attenuation: decays the wavefront, dissipative)
        if self.amplitude_damping:
            t_sigma = omega**-1 * (op.sqrt(1 + Q**-2) - Q**-1)
            t_eps   = (omega**2 * t_sigma)**-1
            tt      = t_eps / (t_sigma - 1e-8) - 1.
            dudt    = (u_now - u_pre) / dt
            fft_dudt = op.fft.fft2(dudt, u_next.shape[-2:], (-2, -1))
            temp    = op.fft.ifft2(self.k * fft_dudt, u_next.shape[-2:], (-2, -1)).real
            u_next  = u_next - (dt**2 * tt * vp / 2) * temp

        return (u_next, out[1], out[2], out[3], out[4], out[5])