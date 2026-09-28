from ._cpml import cpml_axis_update
from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec, history_fields, record_single

from . import slot_table
from .fields import FieldSpec, ModelSpec
from .utils import to_backend, zero_top_halo_fields
from sweep.scalars import fd_coefficients
import numpy as np
from ._registry import register_equation


def _gradient_kernel3d(spatial_order, axis, sign=-1):
    if axis not in (-3, -2, -1):
        raise ValueError("3D gradient kernel axis must be one of -3, -2, or -1.")
    coes = fd_coefficients(1, spatial_order).astype(np.float32)
    size = spatial_order + 1
    center = spatial_order // 2
    kernel = np.zeros((1, 1, size, size, size), dtype=np.float32)
    if axis == -3:
        kernel[0, 0, center + 1:, center, center] = coes
        kernel[0, 0, :center, center, center] = sign * coes[::-1]
    elif axis == -2:
        kernel[0, 0, center, center + 1:, center] = coes
        kernel[0, 0, center, :center, center] = sign * coes[::-1]
    else:
        kernel[0, 0, center, center, center + 1:] = coes
        kernel[0, 0, center, center, :center] = sign * coes[::-1]
    return kernel

def step_cpml(u_now, u_pre, psix, psiz, zetax, zetaz, 
              vp, z, dt, h, b, 
              lap_x, lap_z,
              pml, grad_op, grad_kernels=None
              ):

    az, bz, dbzdz, ax, bx, dbxdx = pml

    w_sum = 0.

    dpdx = grad_op(u_now, h, axis=-1, kernels=grad_kernels)
    dpdz = grad_op(u_now, h, axis=-2, kernels=grad_kernels)
    inv_z = 1.0 / z
    model_b = vp * inv_z
    kappa = z * vp
    # ∇b via the product rule on (vp, 1/z) — matches the C kernel exactly:
    #   dbdx = (∂x vp)·(1/z) + vp·∂x(1/z)   [NOT ∂x(vp/z), the field-gradient form]
    # The two discretisations differ at O(h²); using the product rule keeps the
    # eager forward operator identical to the compiled CUDA kernel so impl='c'
    # and impl='eager' produce the same wavefields and the same gradients.
    dvpdx = grad_op(vp, h, axis=-1, kernels=grad_kernels)
    dvpdz = grad_op(vp, h, axis=-2, kernels=grad_kernels)
    dinvzdx = grad_op(inv_z, h, axis=-1, kernels=grad_kernels)
    dinvzdz = grad_op(inv_z, h, axis=-2, kernels=grad_kernels)
    dbdx = dvpdx * inv_z + vp * dinvzdx
    dbdz = dvpdz * inv_z + vp * dinvzdz

    # ``psiyn`` is the Z memory variable -- historical name, correct slot.
    contrib_z, psiyn, zetaz = cpml_axis_update(
        lap_z, dpdz, psiz, zetaz, az, bz, dbzdz, h, -2, grad_op, grad_kernels)
    w_sum += contrib_z

    contrib_x, psixn, zetax = cpml_axis_update(
        lap_x, dpdx, psix, zetax, ax, bx, dbxdx, h, -1, grad_op, grad_kernels)
    w_sum += contrib_x

    dpdx_cpml = dpdx + psixn
    dpdz_cpml = dpdz + psiyn

    u_next = 2 * u_now - u_pre + dt**2 * kappa * (
        model_b * w_sum + dbdx * dpdx_cpml + dbdz * dpdz_cpml
    )

    return u_next, u_now, psixn, psiyn, zetax, zetaz


def step_cpml_3d(
        u_now, u_pre, psix, psiy, psiz, zetax, zetay, zetaz,
        vp, z, dt, h, b,
        lap_x, lap_y, lap_z,
        pml, grad_op, grad_kernels=None
        ):

    az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx = pml

    w_sum = 0.

    dpdx = grad_op(u_now, h, axis=-1, kernels=grad_kernels)
    dpdy = grad_op(u_now, h, axis=-2, kernels=grad_kernels)
    dpdz = grad_op(u_now, h, axis=-3, kernels=grad_kernels)
    inv_z = 1.0 / z
    model_b = vp * inv_z
    kappa = z * vp
    # ∇b via the product rule on (vp, 1/z) — matches the C kernel exactly
    # (see the 2-D step_cpml for why the field-gradient form is avoided).
    dvpdx = grad_op(vp, h, axis=-1, kernels=grad_kernels)
    dvpdy = grad_op(vp, h, axis=-2, kernels=grad_kernels)
    dvpdz = grad_op(vp, h, axis=-3, kernels=grad_kernels)
    dinvzdx = grad_op(inv_z, h, axis=-1, kernels=grad_kernels)
    dinvzdy = grad_op(inv_z, h, axis=-2, kernels=grad_kernels)
    dinvzdz = grad_op(inv_z, h, axis=-3, kernels=grad_kernels)
    dbdx = dvpdx * inv_z + vp * dinvzdx
    dbdy = dvpdy * inv_z + vp * dinvzdy
    dbdz = dvpdz * inv_z + vp * dinvzdz

    contrib_z, psizn, zetaz = cpml_axis_update(
        lap_z, dpdz, psiz, zetaz, az, bz, dbzdz, h, -3, grad_op, grad_kernels)
    w_sum += contrib_z

    contrib_y, psiyn, zetay = cpml_axis_update(
        lap_y, dpdy, psiy, zetay, ay, by, dbydy, h, -2, grad_op, grad_kernels)
    w_sum += contrib_y

    contrib_x, psixn, zetax = cpml_axis_update(
        lap_x, dpdx, psix, zetax, ax, bx, dbxdx, h, -1, grad_op, grad_kernels)
    w_sum += contrib_x

    dpdx_cpml = dpdx + psixn
    dpdy_cpml = dpdy + psiyn
    dpdz_cpml = dpdz + psizn

    u_next = 2 * u_now - u_pre + dt**2 * kappa * (
        model_b * w_sum + dbdx * dpdx_cpml + dbdy * dpdy_cpml + dbdz * dpdz_cpml
    )

    return u_next, u_now, psixn, psiyn, psizn, zetax, zetay, zetaz


def _adjoint_workspace_shapes(B, nt, shape, mode):
    """The compiled 2-D backward's scratch (acoustic_vrz2d/driver_traits.cuh
    WorkspaceSlot), one padded grid per shot each, in the order the DD runner
    binds the family (coupling grids first, adjoint coefficients last):
    [0-3] = c_x, c_z, e_x, e_z, the split-gradient coupling scratch
            (lambda*vp*grad p and lambda*vp^2*z*grad p; order >= 6 only),
    [4-6] = C0, Cx, Cz, the time-invariant adjoint coefficients
            (vp^2, dx b*kappa, dz b*kappa), built once per backward.
    Every memory mode takes the same seven: full and boundary-saving through
    the template driver, the checkpoint modes through the hand-written
    backward_ckpt (acoustic_vrz2d/backward.cu), which binds the same slots.
    """
    return 7 * [[B, 1, *shape]]


@register_equation()
class AcousticVRZ(SecondOrderEquation):
    """Second-order 2-D acoustic wave equation in variable-density VRZ form.

    Pressure-only scalar acoustics with explicit density coupling through
    an impedance-like auxiliary parameter ``z``. The Laplacian carries an
    extra term ``∇b · ∇p`` (with ``b = vp / z``, ``κ = z · vp``), so the
    propagator is a single second-order PDE in ``h1`` that correctly
    refracts at sharp impedance contrasts without needing a staggered
    velocity field. Absorbing boundaries via split-step CPML (``cpmlr``).

    Reference: 10.3997/2214-4609.202010332.

    
    """
    C_NAME = "acoustic_vrz2d"

    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="Acoustic velocity model.", unit="m/s"),
        ModelSpec("z", description="Auxiliary parameter used by the VRZ formulation."),
    )
    FIELD_SPECS = (
        FieldSpec("h1", aliases=("pressure", "p"), description="Primary VRZ acoustic pressure-like wavefield.", supports_source=True, supports_receiver=True),
        FieldSpec("h2", aliases=("pressure_prev",), description="Previous-step VRZ acoustic pressure-like wavefield.", internal=True),
        FieldSpec("psix", description="CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiz", description="CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("zetax", description="CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetaz", description="CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmlr"

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=2):
        """Build the 2-D VRZ acoustic equation operator.

        Args:
            spatial_order: FD accuracy order of the spatial Laplacian and the auxiliary first-derivative kernels used by the ``∇b · ∇p`` term — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding).
                Must be an even integer (``2, 4, 6, 8, 10, …``).
                **Performance note (`impl='c'` on CUDA):** the compiled
                kernels ship template specialisations only for
                ``spatial_order ∈ {2, 4, 6, 8}``. Above 8 the dispatcher
                drops to a generic runtime path (``order = -1`` in
                ``src/sweep/csrc/cuda/equations/acoustic_vrz2d/forward.cu``)
                which uses more registers and runs noticeably slower.
                The PyTorch eager path is unaffected. Defaults to 4.
            device: Device for the operator's static gradient kernels.
                Use ``'cuda'`` / a ``torch.device`` for GPU runs so the
                propagator can follow without a host↔device copy.
                Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or ``'jax'``.
                When you later want ``impl='c'``, leave this on
                ``'torch'`` — the compiled CUDA kernels go through the
                Torch binding. Defaults to ``'torch'``.
            dim: Stored dimensionality. Always ``2`` for this class; use
                :class:`AcousticVRZ3D` for 3-D. Defaults to 2.
        """
        super().__init__(spatial_order, device, backend)
        super().init_separable_laplace()
        super().init_grad_kernels()

    @property
    def default_source_fields(self):
        return ["h1"]

    @property
    def default_receiver_fields(self):
        return ["h1"]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        u_now = wavefields[0]
        hz, hx = self._spacings_2d(h)
        lap_u_now_z, lap_u_now_x = self.separable_d2_2d(u_now, self.laplace_kernels, hz, hx)
        out = step_cpml(*wavefields, *models, dt, h, b, lap_u_now_x, lap_u_now_z, self.b, self.gradient, self.grad_kernels)
        if getattr(self, "free_surface", False):
            out = zero_top_halo_fields(out, self.so // 2, axis=-2)
        return out

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            record_shape=record_single(),
            # chunk_forward: the replayed segment's pressure, (steps, B, 1, grid)
            checkpoint_replay_shapes=lambda B, nt, grid, seg, mode: [(seg, B, 1, *grid)],
            # u, psix, psiz, zetax, zetaz; the singleton channel axis is the
            # driver's own layout (acoustic_vrz2d allt_shape).
            save_all_shape=lambda B, nt, grid: (nt, 5, B, 1, *grid),
            # The compiled backward's scratch (acoustic_vrz2d/driver_traits.cuh
            # WorkspaceSlot): four c/e coupling grids of the split gradient and
            # the three adjoint coefficients C0/Cx/Cz -- the same 7 in every
            # memory mode (see _adjoint_workspace_shapes).
            backward_workspace_shapes=_adjoint_workspace_shapes,
            derived_model_nvar=1,   # 1/z (common/derived_models.h VrzSlot)
            base_nvar=3,
            # psix,psiz,zetax,zetaz (4) + psixn,psizn (2): race-free forward psi
            # double-buffer (read psi, write psi*n, swap_pml).
            pml_nvar=6,
            last_two_nvar=2,
            last_two_storage_nvar=1,
            checkpoint_nvar=6,
            boundary_tangent_pad=self.so // 2,
            boundary_save_nvar=1,
            slots=slot_table.ACOUSTIC_VRZ2D,
            stepped=True,
            grads_out_has_wavelet=True,
            grads_out_wavelet_written=False,
            illum_nvar=2,
            # Same divergence-form gradient as the 3-D sibling.  DD still
            # refuses this class -- no longer for want of stepped kernels
            # (the template driver made it stepped), but because its backward
            # lacks the coupling-exchange phases (dd_backward_phases).
            dd_coupling_nvar=6,
            dd_adjoint_coeff_nvar=4,
        )


@register_equation()
class AcousticVRZ3D(SecondOrderEquation):
    """Second-order 3-D acoustic wave equation in variable-density VRZ form.

    Three-dimensional generalisation of :class:`AcousticVRZ`: a single
    pressure-like field ``h1`` is propagated with an extra ``∇b · ∇p``
    coupling term (with ``b = vp / z``, ``κ = z · vp``) so that
    impedance contrasts refract correctly without needing a separate
    velocity field. Absorbing boundaries on every face via split-step
    CPML (``cpmlr``).

    Reference: 10.3997/2214-4609.202010332.

    
    """
    C_NAME = "acoustic_vrz3d"

    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="3D acoustic velocity model.", unit="m/s"),
        ModelSpec("z", description="Auxiliary parameter used by the 3D VRZ formulation."),
    )
    FIELD_SPECS = (
        FieldSpec("h1", aliases=("pressure", "p"), description="Primary 3D VRZ acoustic pressure-like wavefield.", supports_source=True, supports_receiver=True),
        FieldSpec("h2", aliases=("pressure_prev",), description="Previous-step 3D VRZ acoustic pressure-like wavefield.", internal=True),
        FieldSpec("psix", description="CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiy", description="CPML memory variable for the y-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiz", description="CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("zetax", description="CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetay", description="CPML auxiliary wavefield for the y-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetaz", description="CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmlr"

    def __init__(self, spatial_order=4, device='cpu', backend='torch', dim=3):
        """Build the 3-D VRZ acoustic equation operator.

        Args:
            spatial_order: FD accuracy order of the spatial Laplacian and the auxiliary first-derivative kernels used by the ``∇b · ∇p`` term — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding).
                Must be an even integer (``2, 4, 6, 8, 10, …``).
                **Performance note (`impl='c'` on CUDA):** the compiled
                kernels ship template specialisations only for
                ``spatial_order ∈ {2, 4, 6, 8}``. Above 8 the dispatcher
                drops to a generic runtime path (``order = -1`` in
                ``src/sweep/csrc/cuda/equations/acoustic_vrz3d/forward.cu``)
                which uses more registers and runs noticeably slower.
                The PyTorch eager path is unaffected. Defaults to 4.
            device: Device for the operator's static gradient kernels.
                Use ``'cuda'`` / a ``torch.device`` for GPU runs so the
                propagator can follow without a host↔device copy.
                Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or ``'jax'``.
                When you later want ``impl='c'``, leave this on
                ``'torch'`` — the compiled CUDA kernels go through the
                Torch binding. Defaults to ``'torch'``.
            dim: Stored dimensionality. Always ``3`` for this class; use
                :class:`AcousticVRZ` for 2-D. Defaults to 3.
        """
        super().__init__(spatial_order, device, backend, dim=dim)
        super().init_separable_laplace()
        if backend == 'torch':
            self.grad_kernels = {
                -3: to_backend(_gradient_kernel3d(spatial_order, -3), backend=backend, device=device),
                -2: to_backend(_gradient_kernel3d(spatial_order, -2), backend=backend, device=device),
                -1: to_backend(_gradient_kernel3d(spatial_order, -1), backend=backend, device=device),
            }
        else:
            self.grad_kernels = None

    @property
    def default_source_fields(self):
        return ["h1"]

    @property
    def default_receiver_fields(self):
        return ["h1"]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        u_now = wavefields[0]
        hz, hy, hx = self._spacings_3d(h)
        lap_z, lap_y, lap_x = self.separable_d2_3d(u_now, self.laplace_kernels, hz, hy, hx)
        out = step_cpml_3d(*wavefields, *models, dt, h, b, lap_x, lap_y, lap_z, self.b, self.gradient, self.grad_kernels)
        if getattr(self, "free_surface", False):
            out = zero_top_halo_fields(out, self.so // 2, axis=-3)
        return out

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            record_shape=record_single(),
            # chunk_forward: the replayed segment's pressure, (steps, B, 1, grid)
            checkpoint_replay_shapes=lambda B, nt, grid, seg, mode: [(seg, B, 1, *grid)],
            save_all_shape=history_fields(7),   # u + 3 psi + 3 zeta
            # The compiled backward's scratch (acoustic_vrz3d/backward.cu WorkspaceSlot):
            # six c/e coupling grids of the split gradient and the four adjoint
            # coefficients C0/Cx/Cy/Cz -- the same ten the DD runner binds.
            backward_workspace_nvar=10,
            derived_model_nvar=1,   # 1/z (common/derived_models.h VrzSlot)
            base_nvar=3,
            # psix,psiy,psiz,zetax,zetay,zetaz (6) + psixn,psiyn,psizn (3): race-free
            # forward psi double-buffer (read psi, write psi*n, swap_pml).
            pml_nvar=9,
            last_two_nvar=2,
            last_two_storage_nvar=1,
            checkpoint_nvar=8,
            boundary_tangent_pad=self.so // 2,
            boundary_save_nvar=1,
            slots=slot_table.ACOUSTIC_VRZ3D,
            stepped=True,
            dd_backward_phases=True,
            grads_out_has_wavelet=True,
            grads_out_wavelet_written=False,
            illum_nvar=2,
            # DD: the variable-density gradient is div(c/e) with
            # c/e = lambda*vp*grad(p), so a cut seam needs the
            # neighbour's coupling field before the divergence.
            dd_coupling_nvar=6,
            dd_adjoint_coeff_nvar=4,
        )
