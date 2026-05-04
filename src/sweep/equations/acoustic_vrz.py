from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from .utils import to_backend
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
    model_b = vp / z
    kappa = z * vp
    dbdx = grad_op(model_b, h, axis=-1, kernels=grad_kernels)
    dbdz = grad_op(model_b, h, axis=-2, kernels=grad_kernels)

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
    model_b = vp / z
    kappa = z * vp
    dbdx = grad_op(model_b, h, axis=-1, kernels=grad_kernels)
    dbdy = grad_op(model_b, h, axis=-2, kernels=grad_kernels)
    dbdz = grad_op(model_b, h, axis=-3, kernels=grad_kernels)

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
    """
    Parameter order: vp, rx, rz

    Wavefields: (h1, h2)

    Reference: 10.3997/2214-4609.202010332
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

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=2):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        super().__init__(spatial_order, device, backend, other_kernels=True)
        super().init_laplace(ltype='1dsep', backend=backend)
        self.grad_kernels = {-2: self.gkernel_z, -1: self.gkernel_x}

    @property
    def models(self):
        return [spec.name for spec in self.MODEL_SPECS]

    @property
    def wavefields(self):
        return [spec.name for spec in self.FIELD_SPECS]

    @property
    def field_specs(self):
        return list(self.FIELD_SPECS)

    def func(self, *args, **kwargs):
        dh = args[9]
        hz, hx = self._spacings_2d(dh)
        lap_u_now_z, lap_u_now_x = self.laplace1d_sep(args[0], self.laplace_kernels, hz, hx)
        return step_cpml(*args, lap_u_now_x, lap_u_now_z, self.b, self.gradient, self.grad_kernels)

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
        )


class AcousticVRZ3D(SecondOrderEquation):
    """
    3D acoustic VRZ formulation.

    Parameter order: vp, z

    Wavefields: (h1, h2, psix, psiy, psiz, zetax, zetay, zetaz)

    Reference: 10.3997/2214-4609.202010332
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

    def __init__(self, spatial_order=4, device='cpu', backend='torch', dim=3):
        super().__init__(spatial_order, device, backend, dim=dim)
        super().init_laplace(ltype='3dsep', backend=backend)
        if backend == 'torch':
            self.grad_kernels = {
                -3: to_backend(_gradient_kernel3d(spatial_order, -3), backend=backend, device=device),
                -2: to_backend(_gradient_kernel3d(spatial_order, -2), backend=backend, device=device),
                -1: to_backend(_gradient_kernel3d(spatial_order, -1), backend=backend, device=device),
            }
        else:
            self.grad_kernels = None

    @property
    def models(self):
        return [spec.name for spec in self.MODEL_SPECS]

    @property
    def wavefields(self):
        return [spec.name for spec in self.FIELD_SPECS]

    @property
    def field_specs(self):
        return list(self.FIELD_SPECS)

    def func(self, *args, **kwargs):
        dh = args[11]
        hz, hy, hx = self._spacings_3d(dh)
        lap_z, lap_y, lap_x = self.laplace3d_sep(args[0], self.laplace_kernels, hz, hy, hx)
        return step_cpml_3d(*args, lap_x, lap_y, lap_z, self.b, self.gradient, self.grad_kernels)

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
        )
