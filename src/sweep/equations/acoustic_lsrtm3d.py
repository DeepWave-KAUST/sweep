from ._cpml import cpml_axis_update
from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec, history_plain, record_single

from .fields import FieldSpec, ModelSpec
from .utils import zero_top_halo_fields
from ._registry import register_equation


def step_cpml(
    u_now,
    u_pre,
    psix,
    psiy,
    psiz,
    zetax,
    zetay,
    zetaz,
    su_now,
    su_pre,
    spsix,
    spsiy,
    spsiz,
    szetax,
    szetay,
    szetaz,
    vp,
    ref,
    dt,
    h,
    b,
    lap_x,
    lap_y,
    lap_z,
    lap_sx,
    lap_sy,
    lap_sz,
    pml,
    grad_op,
):
    az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx = pml

    dudz = grad_op(u_now, h, -3)
    dudy = grad_op(u_now, h, -2)
    dudx = grad_op(u_now, h, -1)
    dsudz = grad_op(su_now, h, -3)
    dsudy = grad_op(su_now, h, -2)
    dsudx = grad_op(su_now, h, -1)

    w_sum = 0.0

    contrib_z, psizn, zetaz = cpml_axis_update(
        lap_z, dudz, psiz, zetaz, az, bz, dbzdz, h, -3, grad_op)
    w_sum += contrib_z

    contrib_y, psiyn, zetay = cpml_axis_update(
        lap_y, dudy, psiy, zetay, ay, by, dbydy, h, -2, grad_op)
    w_sum += contrib_y

    contrib_x, psixn, zetax = cpml_axis_update(
        lap_x, dudx, psix, zetax, ax, bx, dbxdx, h, -1, grad_op)
    w_sum += contrib_x

    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * w_sum

    sw_sum = 0.0

    contrib_sz, spsizn, szetaz = cpml_axis_update(
        lap_sz, dsudz, spsiz, szetaz, az, bz, dbzdz, h, -3, grad_op)
    sw_sum += contrib_sz

    contrib_sy, spsiyn, szetay = cpml_axis_update(
        lap_sy, dsudy, spsiy, szetay, ay, by, dbydy, h, -2, grad_op)
    sw_sum += contrib_sy

    contrib_sx, spsixn, szetax = cpml_axis_update(
        lap_sx, dsudx, spsix, szetax, ax, bx, dbxdx, h, -1, grad_op)
    sw_sum += contrib_sx

    su_next = 2 * su_now - su_pre + vp**2 * dt**2 * sw_sum + ref * vp**2 * dt**2 * w_sum

    return (
        u_next,
        u_now,
        psixn,
        psiyn,
        psizn,
        zetax,
        zetay,
        zetaz,
        su_next,
        su_now,
        spsixn,
        spsiyn,
        spsizn,
        szetax,
        szetay,
        szetaz,
    )



# The checkpoint replay state (u_prev, u_now, u_next, psix, psiz, zetax, zetaz,
# psiy, zetay of the background field, AcousticWavefieldTensor 3-D bind order):
# one acoustic state set, declared explicitly because the LSRTM forward slot list
# is two acoustic layouts back to back (``checkpoint_state_nvar``); the recursive
# backward keeps one scratch set per bisection level (``recursive_state_depth``).
_CKPT_REPLAY_STATE_NVAR = 9


def _adjoint_workspace_shapes(B, nt, shape, mode):
    """The compiled backward's scratch (acoustic_lsrtm3d/backward.cu WorkspaceSlot),
    one padded grid per shot each: the vp^2*lambda grid of every adjoint step,
    plus one grid in the boundary-saving mode (the forward step) and the
    recursive-checkpoint mode (the replayed step's field).
    """
    n = 2 if mode in ("bs", "recursive") else 1
    return n * [[B, 1, *shape]]

@register_equation()
class AcousticLSRTM3D(SecondOrderEquation):
    """Second-order 3-D acoustic Born / LSRTM wave equation.

    Three-dimensional generalisation of :class:`AcousticLSRTM`: a
    *background* pressure-like field ``h1`` propagating through the
    smooth velocity ``vp``, and a *scattered* pressure-like field
    ``sh1`` driven by the reflectivity perturbation ``mp`` acting on
    the background Laplacian (linearised Born scattering). Both fields
    carry their own CPML memory variables on every face. Defaults:
    source on ``h1``, receivers on ``sh1``.

    
    """

    C_NAME = "acoustic_lsrtm3d"

    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="Background 3D acoustic velocity model.", unit="m/s"),
        ModelSpec("mp", aliases=("reflectivity", "ref"), description="3D acoustic reflectivity perturbation used for LSRTM."),
    )
    FIELD_SPECS = (
        FieldSpec("h1", aliases=("pressure", "p", "background"), description="Background 3D acoustic pressure-like wavefield.", supports_source=True),
        FieldSpec("h2", aliases=("pressure_prev", "background_prev"), description="Previous-step background wavefield.", internal=True),
        FieldSpec("psix", description="Background CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiy", description="Background CPML memory variable for the y-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiz", description="Background CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("zetax", description="Background CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetay", description="Background CPML auxiliary wavefield for the y-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetaz", description="Background CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
        FieldSpec("sh1", aliases=("scattered", "scattered_pressure", "data"), description="Scattered 3D acoustic wavefield used for LSRTM data prediction.", supports_receiver=True),
        FieldSpec("sh2", aliases=("scattered_prev",), description="Previous-step scattered wavefield.", internal=True),
        FieldSpec("spsix", description="Scattered-wave CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("spsiy", description="Scattered-wave CPML memory variable for the y-derivative term.", internal=True, boundary_related=True),
        FieldSpec("spsiz", description="Scattered-wave CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("szetax", description="Scattered-wave CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("szetay", description="Scattered-wave CPML auxiliary wavefield for the y-direction update.", internal=True, boundary_related=True),
        FieldSpec("szetaz", description="Scattered-wave CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmlr"

    def __init__(self, spatial_order=4, device="cpu", backend="torch"):
        """Build the 3-D acoustic LSRTM equation operator.

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
                ``src/sweep/csrc/cuda/equations/acoustic_lsrtm3d/forward.cu``)
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
        super().__init__(spatial_order, device, backend, dim=3)
        super().init_separable_laplace()

    @property
    def default_source_fields(self):
        return ["h1"]

    @property
    def default_receiver_fields(self):
        return ["sh1"]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        hz, hy, hx = self._spacings_3d(h)
        lap_z, lap_y, lap_x = self.separable_d2_3d(wavefields[0], self.laplace_kernels, hz, hy, hx)
        lap_sz, lap_sy, lap_sx = self.separable_d2_3d(wavefields[8], self.laplace_kernels, hz, hy, hx)
        out = step_cpml(
            *wavefields,
            *models,
            dt,
            h,
            b,
            lap_x,
            lap_y,
            lap_z,
            lap_sx,
            lap_sy,
            lap_sz,
            self.b,
            self.gradient,
        )
        if getattr(self, "free_surface", False):
            out = zero_top_halo_fields(out, self.so // 2, axis=-3)
        return out

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            record_shape=record_single(),
            # chunk_forward: the replayed chunk's background field
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
            # 2 wavefields (bg+sc); each: psix,psiy,psiz,zetax,zetay,zetaz (6) +
            # psixn,psiyn,psizn (3) for the race-free forward double-buffer -> 2*9=18.
            pml_nvar=18,
            last_two_nvar=2,
            last_two_storage_nvar=1,
            checkpoint_nvar=8,
            boundary_save_nvar=1,
        )
