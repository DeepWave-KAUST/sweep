import numpy as np
from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from ._free_surface import zero_above_topo
from .utils import to_backend
from .acoustic import step_cpml


def step_visco_cpml(
        u_now, u_pre, psix, psiz, zetax, zetaz,
        vp_step, A, dt, h, b,
        lap_x, lap_z,
        pml,
        grad_op,
        grad_kernels=None,
        k=None,
        op=None,
        amplitude_damping=True,
        ):
    """Nearly constant-Q visco-acoustic step on top of the CPML acoustic step.

    Attenuation enters as two decoupled terms (Zhu & Harris, 2014) that switch
    independently: both off reduces to :func:`~sweep.equations.acoustic.step_cpml`,
    both on gives the full visco-acoustic update.  Both material inputs are the
    PREPARED forms from :meth:`ViscoAcoustic.prepare_models` (shared with the
    CUDA backend): ``vp_step`` has the dispersion (phase-shift) term folded in
    as an effective velocity, so it rides through the CPML machinery unchanged;
    ``A = tt * vp / 2`` is the amplitude-damping coefficient.

    ``k`` is the FFT wavenumber grid and must match ``u_now``'s trailing shape
    (PML pad plus stencil halo); ``op`` supplies the backend FFT namespace.
    """
    u_next, u_prev, psixn, psizn, zetaxn, zetazn = step_cpml(
        u_now, u_pre, psix, psiz, zetax, zetaz,
        vp_step, dt, h, b,
        lap_x, lap_z,
        pml,
        grad_op,
        grad_kernels,
    )

    # Amplitude loss: dissipative, driven by du/dt through a |k| operator.
    # A carries the reference vp, not the dispersion-corrected vp_step.
    if amplitude_damping:
        dudt = (u_now - u_pre) / dt
        fft_dudt = op.fft.fft2(dudt, u_next.shape[-2:], (-2, -1))
        temp = op.fft.ifft2(k * fft_dudt, u_next.shape[-2:], (-2, -1)).real
        u_next = u_next - (dt**2 * A) * temp

    return u_next, u_prev, psixn, psizn, zetaxn, zetazn


class ViscoAcoustic(SecondOrderEquation):
    """Second-order 2-D nearly constant-Q visco-acoustic wave equation.

    An attenuating counterpart to :class:`~sweep.equations.acoustic.Acoustic`:
    the pressure-like field ``h1`` is driven by ``vp**2 · Laplace(u)`` and
    absorbed by the same CPML formulation (``cpmlr`` by default), with
    attenuation added on top as two *decoupled* terms:

    * **phase shift** -- velocity dispersion. Makes the phase velocity
      frequency dependent (higher frequencies travel slightly faster), so it
      moves the wavefront. Non-dissipative: it costs no energy. Numerically it
      is exactly an effective velocity ``vp·sqrt(1-c)``, so it is folded into
      ``vp`` and rides through the CPML with the base step.
    * **amplitude damping** -- attenuation. The dissipative term: it removes
      energy so the wavefront decays, absorbing high frequencies more strongly
      (the medium acts as a distance-dependent low-pass). It weakens the
      wavefront without moving it. Evaluated in the wavenumber domain, so the
      amplitude term is the only part of the step needing an FFT.

    The two switch independently via ``phase_shift`` / ``amplitude_damping``
    (both default on). Both off reduces bit-exactly to ``Acoustic``; both on
    gives the full visco-acoustic response.

    Because the decoupling is what makes the switches meaningful, note that the
    two effects are not physically independent -- causality ties dispersion and
    attenuation together through the Kramers-Kronig relations. The decoupled
    form is an approximation that buys the ability to treat them separately;
    the fully coupled constant-Q equation is more faithful but harder to solve.

    Parameter order: ``vp``, ``Q``, ``omega``. Wavefields: ``(h1, h2)`` plus
    the CPML auxiliaries.

    Backends: eager (CPU/GPU, torch + jax) and compiled CUDA (``impl='c'``).
    The CUDA path reuses the acoustic2d CPML kernels with the prepared
    ``vp_step`` and applies the amplitude damping per step via ATen/cuFFT;
    gradient memory strategies are ``full`` and ``ckpt`` (chunk or recursive).
    Boundary saving is not supported — the damping term is dissipative and
    global, so the reverse-time reconstruction that boundary saving relies on
    does not exist; the impl='c' default silently falls back to ``full`` and
    an explicit boundary request raises.  Per-edge free surfaces work on both
    backends; topography and domain decomposition are eager-only.

    References:
        Zhu, T. and Harris, J. M., 2014, Modeling acoustic wave propagation in
        heterogeneous attenuating media using decoupled fractional Laplacians:
        Geophysics, 79(3), T105-T116 -- the decoupled form implemented here,
        derived from Kjartansson's constant-Q stress-strain relation.

        Wang Enjiang, Thesis.
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

    # Eager ``func`` honours a per-edge free surface (any subset of the 4 faces,
    # each pressure-release) via ``self.fs_faces``; the zeroing is the shared
    # ``SecondOrderEquation._apply_free_surface`` (bit-identical to the old
    # top-only ``zero_top_halo_fields`` when only the top face is active).
    supports_per_edge_free_surface = True

    # The CUDA backend reuses the acoustic2d CPML kernels with the
    # dispersion-folded ``vp_step`` (see ``prepare_models``) plus a per-step
    # ATen-FFT amplitude-damping correction, so per-edge free surfaces come
    # along for free (region logic + post-damping halo zeroing).
    supports_per_edge_free_surface_c = True

    # Same kernels as Acoustic: per-shot batched models stride per batch index.
    supports_batched_models = True

    # The amplitude damping is dissipative (reverse-time reconstruction
    # amplifies) and global (the |k| filter reads the whole padded grid, which
    # boundary strips cannot restore), so boundary saving is refused; the
    # impl='c' default gradient-memory strategy falls back to 'full'.
    supports_boundary_saving_c = False

    # impl='c' consumes the PREPARED model pair (vp_step, A), not (vp, Q,
    # omega); autograd chains the returned (grad_vp_step, grad_A) back through
    # ``prepare_models`` to the user's three models.
    prepare_models_for_c = True

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

        # Only the FFT in the amplitude term needs a backend namespace; the rest
        # of the step is plain arithmetic and works on either backend.
        if backend == 'torch':
            import torch
            self.op = torch
        elif backend == 'jax':
            import jax.numpy as jnp
            self.op = jnp
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def prepare_models(self, models):
        """Map the user models (vp, Q, omega) onto what the CUDA kernels
        consume: ``vp_step`` (dispersion folded into an effective velocity,
        exactly as in :func:`step_visco_cpml`) and the damping coefficient
        ``A = tt * vp / 2`` (the forward multiplies ``dt**2``).  All ops are
        differentiable, so autograd routes the C-returned (grad_vp_step,
        grad_A) back to vp / Q / omega."""
        vp, Q, omega = models
        if self.phase_shift:
            c = (1.0 - (Q**2 + 1.0)**0.5) * Q**-2
            vp_step = vp * (1.0 - c)**0.5
        else:
            vp_step = vp
        if self.amplitude_damping:
            t_sigma = omega**-1 * ((1.0 + Q**-2)**0.5 - Q**-1)
            t_eps = (omega**2 * t_sigma)**-1
            tt = t_eps / (t_sigma - 1e-8) - 1.0
            A = tt * vp / 2
        else:
            # No damping: A is a dead input (the C backward returns grad_A=0),
            # and skipping the tt chain avoids injecting spurious zero-grad
            # graph nodes (or NaNs from degenerate Q / omega) into autograd.
            A = self.op.zeros_like(vp)
        return (vp_step, A)

    def c_eq_aux(self, prop):
        """ForwardInput.eq_aux for the CUDA kernels: the |k| FFT multiplier on
        the runtime grid, present iff amplitude damping is active.  Built with
        the same :func:`~sweep.equations.base.init_wavenumbers` rule as the
        eager path (scalar h = first grid-spacing entry) and cached per
        (shape, h)."""
        if not self.amplitude_damping:
            return ()
        import torch
        from .base import init_wavenumbers
        shape = tuple(int(v) for v in prop.shape_cuda)[-2:]
        h = float(np.asarray(prop._grid_spacing).reshape(-1)[0])
        key = (shape, h, str(prop.dev))
        cached = getattr(self, "_c_kmul_cache", None)
        if cached is None or cached[0] != key:
            k_np, _, _ = init_wavenumbers(shape, h)
            t = torch.as_tensor(k_np, dtype=torch.float32, device=prop.dev)
            self._c_kmul_cache = (key, t.contiguous())
        return (self._c_kmul_cache[1],)

    def _C(self):
        # CUDA IMPLEMENTATION
        from sweep._C import (
            visco_acoustic2d_forward,
            visco_acoustic2d_backward,
            visco_acoustic2d_backward_bs,
            visco_acoustic2d_backward_ckpt,
            visco_acoustic2d_backward_recursive_ckpt,
        )
        return (
            visco_acoustic2d_forward,
            visco_acoustic2d_backward,
            visco_acoustic2d_backward_bs,
            visco_acoustic2d_backward_ckpt,
            visco_acoustic2d_backward_recursive_ckpt,
        )

    def _C_rtm(self):
        from sweep._C import visco_acoustic2d_rtm

        return visco_acoustic2d_rtm

    @property
    def cuda_layout(self):
        # Identical to Acoustic: same wavefield state (the damping correction
        # is memoryless in (u_now, u_prev) and runs through ATen).
        return CUDALayoutSpec(
            base_nvar=3,
            pml_nvar=6,
            adjoint_extra_nvar=2,
            last_two_nvar=2,
            last_two_storage_nvar=1,
            checkpoint_nvar=6,
            pml_slot_axes=("x", "z", "x", "z", "x", "z"),
            checkpoint_slot_axes=(None, None, "x", "z", "x", "z"),
            boundary_save_nvar=1,
            backward_workspace_nvar=1,
        )

    def init_abc(self, type='cpml', **kwargs):
        """Set up the PML profiles, then the FFT wavenumber grid.

        ``shape`` here is the runtime grid the step actually sees (PML pad plus
        stencil halo) and ``grid_spacing`` is a plain Python value, so the
        wavenumber grid is built entirely outside any jit trace.
        """
        from .base import init_wavenumbers

        super().init_abc(type=type, **kwargs)

        shape = tuple(int(s) for s in kwargs['shape'])[-2:]
        h_scalar = float(np.asarray(kwargs['grid_spacing']).reshape(-1)[0])
        k_np, _, _ = init_wavenumbers(shape, h_scalar)
        # Same rule as the PML profiles above: keep numpy on jax, where this
        # runs inside the user's trace and a jnp array would leak as a tracer.
        self.k = k_np if self.backend == 'jax' else to_backend(k_np, self.backend, self.device)

    def func(self, wavefields, models, dt, h, b, **kwargs):
        u_now = wavefields[0]
        # The torch-eager and CUDA propagators hand the PREPARED (vp_step, A)
        # pair (their prepare_models hooks run once, outside the time loop);
        # PropJax — and any caller without a prepare hook — hands the raw
        # (vp, Q, omega), prepared here inside the trace (pure pointwise ops).
        if len(models) == 3:
            models = self.prepare_models(models)
        vp_step, A = models
        hz, hx = self._spacings_2d(h)
        lap_u_now_z, lap_u_now_x = self.separable_d2_2d(u_now, self.laplace_kernels, hz, hx)
        out = step_visco_cpml(
            *wavefields, vp_step, A, dt, h, b,
            lap_u_now_x, lap_u_now_z,
            self.b, self.gradient, self.grad_kernels,
            k=self.k, op=self.op,
            amplitude_damping=self.amplitude_damping,
        )
        if getattr(self, "free_surface", False):
            topo_rows = getattr(self, "_topo_rows_runtime", None)
            if topo_rows is not None:
                # Irregular surface: zero ALL air cells per column.
                # In the flat-degenerate case (topo_rows == halo*ones) this
                # matches ``zero_top_halo_fields`` bit-for-bit.
                out = tuple(zero_above_topo(field, topo_rows, axis=-2) for field in out)
            else:
                out = self._apply_free_surface(out)
        return out
