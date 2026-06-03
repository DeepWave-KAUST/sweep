from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from .utils import to_backend, zero_top_halo_fields
from sweep.scalars import fd_coefficients
import numpy as np


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

    # Z direction
    tmpz = ((1+bz)*lap_z + dbzdz * dpdz) + grad_op(az * psiz, h, axis=-2, kernels=grad_kernels)
    w_sum += (1+bz) * tmpz + az * zetaz

    psiyn = bz * dpdz + az * psiz
    zetaz = bz * tmpz + az * zetaz

    # X direction
    tmpx = ((1+bx)*lap_x + dbxdx * dpdx) + grad_op(ax * psix, h, axis=-1, kernels=grad_kernels)
    w_sum += (1+bx) * tmpx + ax * zetax

    psixn = bx * dpdx + ax * psix
    zetax = bx * tmpx + ax * zetax

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

    # Z direction
    tmpz = ((1 + bz) * lap_z + dbzdz * dpdz) + grad_op(az * psiz, h, axis=-3, kernels=grad_kernels)
    w_sum += (1 + bz) * tmpz + az * zetaz

    psizn = bz * dpdz + az * psiz
    zetaz = bz * tmpz + az * zetaz

    # Y direction
    tmpy = ((1 + by) * lap_y + dbydy * dpdy) + grad_op(ay * psiy, h, axis=-2, kernels=grad_kernels)
    w_sum += (1 + by) * tmpy + ay * zetay

    psiyn = by * dpdy + ay * psiy
    zetay = by * tmpy + ay * zetay

    # X direction
    tmpx = ((1 + bx) * lap_x + dbxdx * dpdx) + grad_op(ax * psix, h, axis=-1, kernels=grad_kernels)
    w_sum += (1 + bx) * tmpx + ax * zetax

    psixn = bx * dpdx + ax * psix
    zetax = bx * tmpx + ax * zetax

    dpdx_cpml = dpdx + psixn
    dpdy_cpml = dpdy + psiyn
    dpdz_cpml = dpdz + psizn

    u_next = 2 * u_now - u_pre + dt**2 * kappa * (
        model_b * w_sum + dbdx * dpdx_cpml + dbdy * dpdy_cpml + dbdz * dpdz_cpml
    )

    return u_next, u_now, psixn, psiyn, psizn, zetax, zetay, zetaz


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
        super().__init__(spatial_order, device, backend, other_kernels=True)
        super().init_separable_laplace()
        self.grad_kernels = {-2: self.gkernel_z, -1: self.gkernel_x}

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

    def _C(self):
        import torch
        from sweep._C import (
            acoustic_vrz2d_forward,
            acoustic_vrz2d_backward,
            acoustic_vrz2d_backward_bs,
            acoustic_vrz2d_backward_ckpt,
            acoustic_vrz2d_backward_recursive_ckpt,
        )

        return (
            acoustic_vrz2d_forward,
            acoustic_vrz2d_backward,
            acoustic_vrz2d_backward_bs,
            acoustic_vrz2d_backward_ckpt,
            acoustic_vrz2d_backward_recursive_ckpt,
        )

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            base_nvar=3,
            pml_nvar=4,
            last_two_nvar=2,
            last_two_storage_nvar=1,
            checkpoint_nvar=6,
            boundary_tangent_pad=self.so // 2,
            boundary_save_nvar=1,
        )


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

    def _C(self):
        import torch
        from sweep._C import (
            acoustic_vrz3d_forward,
            acoustic_vrz3d_backward,
            acoustic_vrz3d_backward_bs,
            acoustic_vrz3d_backward_ckpt,
            acoustic_vrz3d_backward_recursive_ckpt,
        )

        return (
            acoustic_vrz3d_forward,
            acoustic_vrz3d_backward,
            acoustic_vrz3d_backward_bs,
            acoustic_vrz3d_backward_ckpt,
            acoustic_vrz3d_backward_recursive_ckpt,
        )

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            base_nvar=3,
            pml_nvar=6,
            last_two_nvar=2,
            last_two_storage_nvar=1,
            checkpoint_nvar=8,
            boundary_tangent_pad=self.so // 2,
            boundary_save_nvar=1,
        )
