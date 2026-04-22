from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec

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


class AcousticVRZ(SecondOrderEquation):
    """
    Parameter order: vp, rx, rz

    Wavefields: (h1, h2)

    Reference: 10.3997/2214-4609.202010332
    """
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
        return ['vp', 'z']

    @property
    def wavefields(self):
        return  ['h1', 'h2', 'psix', 'psiz', 'zetax', 'zetaz']

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
        )
