from ._cpml import cpml_axis_update
from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec, history_plain, record_single

from .fields import FieldSpec, ModelSpec
from .utils import zero_top_halo_fields
from ._registry import register_equation

def step(u_now, u_pre, psix, psiz, zetax, zetaz, 
         su_now, su_pre, spsix, spsiz, szetax, szetaz, 
         vp, ref, dt, h, b, 
         lap_ux, lap_uz, lap_sux, lap_suz,
         pml,
         grad_op,
         ):
    
    az, bz, dbzdz, ax, bx, dbxdx = pml

    # Calcualte gradients based on 2nd order central finite difference
    dudz = grad_op(u_now, h, -2)
    dudx = grad_op(u_now, h, -1)
    dsudz = grad_op(su_now, h, -2)
    dsudx = grad_op(su_now, h, -1)

    # Background wavefield
    w_sum = 0.
    # ``psiyn`` is the Z memory variable -- historical name, correct slot.
    contrib_z, psiyn, zetaz = cpml_axis_update(
        lap_uz, dudz, psiz, zetaz, az, bz, dbzdz, h, -2, grad_op)
    w_sum += contrib_z

    contrib_x, psixn, zetax = cpml_axis_update(
        lap_ux, dudx, psix, zetax, ax, bx, dbxdx, h, -1, grad_op)
    w_sum += contrib_x

    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * w_sum

    # Scatter wavefield
    w_sum_s = 0.
    contrib_sz, spsiyn, szetaz = cpml_axis_update(
        lap_suz, dsudz, spsiz, szetaz, az, bz, dbzdz, h, -2, grad_op)
    w_sum_s += contrib_sz

    contrib_sx, spsixn, szetax = cpml_axis_update(
        lap_sux, dsudx, spsix, szetax, ax, bx, dbxdx, h, -1, grad_op)
    w_sum_s += contrib_sx
    su_next = 2 * su_now - su_pre + vp**2 * dt**2 * w_sum_s + ref * vp**2 * dt**2 * w_sum

    # # background wavefield
    # vp2_nabla_p0 = vp**2*lap_u_now*dt**2
    # u_next = 2 * u_now - u_pre + vp2_nabla_p0
    # u_next = a * u_next + (1 - a) * u_now
    
    # # scatter wavefield
    # vp2_nabla_sh0 = vp**2*lap_su_now*dt**2
    # su_next = 2 * su_now - su_pre + vp2_nabla_sh0 + ref*vp2_nabla_p0
    # su_next = a * su_next + (1 - a) * su_now

    return u_next, u_now, psixn, psiyn, zetax, zetaz, \
            su_next, su_now, spsixn, spsiyn, szetax, szetaz


# The checkpoint replay state (u_prev, u_now, u_next, psix, psiz, zetax, zetaz of
# the background field, AcousticWavefieldTensor bind order): one acoustic state
# set. The LSRTM forward slot list is two acoustic layouts back to back, so the
# generic "forward slots minus shadows" rule does not apply and the count is
# declared explicitly (``checkpoint_state_nvar``); the recursive backward bisects
# each segment and keeps one scratch set per level (``recursive_state_depth``).
_CKPT_REPLAY_STATE_NVAR = 7


def _adjoint_workspace_shapes(B, nt, shape, mode):
    """The compiled backward's scratch (acoustic_lsrtm2d/backward.cu WorkspaceSlot),
    one padded grid per shot each: the vp^2*lambda grid of every adjoint step,
    plus the replayed step's background u_tt in the recursive-checkpoint mode.
    """
    n = 2 if mode == "recursive" else 1
    return n * [[B, 1, *shape]]

@register_equation()
class AcousticLSRTM(SecondOrderEquation):
    """Second-order 2-D acoustic Born / LSRTM wave equation.

    Two coupled scalar wave equations: a *background* pressure-like field
    ``h1`` propagating through the smooth velocity ``vp``, and a
    *scattered* pressure-like field ``sh1`` driven by the reflectivity
    perturbation ``mp`` acting on the background Laplacian (linearised
    Born scattering). Both fields share an independent set of CPML
    memory variables. Defaults: source on the background field ``h1``,
    receivers on the scattered field ``sh1`` — the standard layout for
    least-squares reverse-time migration.

    
    """

    C_NAME = "acoustic_lsrtm2d"

    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="Background acoustic velocity model.", unit="m/s"),
        ModelSpec("mp", aliases=("reflectivity", "ref"), description="Acoustic reflectivity perturbation used for LSRTM."),
    )
    FIELD_SPECS = (
        FieldSpec("h1", aliases=("pressure", "p", "background"), description="Background acoustic pressure-like wavefield.", supports_source=True),
        FieldSpec("h2", aliases=("pressure_prev", "background_prev"), description="Previous-step background wavefield.", internal=True),
        FieldSpec("psix", description="Background CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiz", description="Background CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("zetax", description="Background CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetaz", description="Background CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
        FieldSpec("sh1", aliases=("scattered", "scattered_pressure", "data"), description="Scattered acoustic wavefield used for LSRTM data prediction.", supports_receiver=True),
        FieldSpec("sh2", aliases=("scattered_prev",), description="Previous-step scattered wavefield.", internal=True),
        FieldSpec("spsix", description="Scattered-wave CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("spsiz", description="Scattered-wave CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("szetax", description="Scattered-wave CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("szetaz", description="Scattered-wave CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmlr"

    def __init__(self, spatial_order=4, device='cpu', backend='torch'):
        """Build the 2-D acoustic LSRTM equation operator.

        Args:
            spatial_order: FD accuracy order of the spatial Laplacians applied to both the background and scattered fields — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding). Must be an even integer
                (``2, 4, 6, 8, 10, …``).
                **Performance note (`impl='c'` on CUDA):** the compiled
                kernels ship template specialisations only for
                ``spatial_order ∈ {2, 4, 6, 8}``. Above 8 the dispatcher
                drops to a generic runtime path (``order = -1`` in
                ``src/sweep/csrc/cuda/equations/acoustic_lsrtm2d/forward.cu``)
                which uses more registers and runs noticeably slower.
                The PyTorch eager path is unaffected. Defaults to 4.
            device: Device for the operator's static kernels. Use
                ``'cuda'`` / a ``torch.device`` for GPU runs so the
                propagator can follow without a host↔device copy.
                Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or
                ``'jax'``. When you later want ``impl='c'``, leave this
                on ``'torch'``. Defaults to ``'torch'``.
        """
        super().__init__(spatial_order, device, backend)
        super().init_separable_laplace()
    
    @property
    def default_source_fields(self):
        return ["h1"]

    @property
    def default_receiver_fields(self):
        return ["sh1"]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        hz, hx = self._spacings_2d(h)
        lap_uz, lap_ux = self.separable_d2_2d(wavefields[0], self.laplace_kernels, hz, hx)
        lap_suz, lap_sux = self.separable_d2_2d(wavefields[6], self.laplace_kernels, hz, hx)
        out = step(*wavefields, *models, dt, h, b, lap_ux, lap_uz, lap_sux, lap_suz, self.b, self.gradient)
        if getattr(self, "free_surface", False):
            out = zero_top_halo_fields(out, self.so // 2, axis=-2)
        return out

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            record_shape=record_single(),
            # chunk_forward: the replayed chunk's background u_tt
            # chunk_forward, the replayed chunk's background u_tt: chunk mode only (the
            # recursive leaf images from its u_this workspace slot)
            checkpoint_replay_shapes=lambda B, nt, grid, seg, mode: [(seg, B, *grid)] if mode == "ckpt" else [],
            backward_workspace_shapes=_adjoint_workspace_shapes,
            bs_reconstruction_nvar=3,   # u_prev, u_now, u_next of the background field
            checkpoint_state_nvar=_CKPT_REPLAY_STATE_NVAR,   # one acoustic state set per replay / recursion level
            recursive_state_depth=True,
            save_all_shape=history_plain(),   # bg_utt_all
            # BackwardOutput.grads = {grad_wavelet, <model grads>}; the
            # propagator sizes grads_out from this.
            grads_out_has_wavelet=True,
            base_nvar=6,
            # 2 wavefields (bg+sc); each: psix,psiz,zetax,zetaz (4) + psixn,psizn (2)
            # for the race-free forward psi double-buffer -> 2*(4+2)=12.
            pml_nvar=12,
            last_two_nvar=2,
            last_two_storage_nvar=1,
            checkpoint_nvar=6,
            boundary_save_nvar=1,
        )
    
