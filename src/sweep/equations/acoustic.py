from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from .utils import zero_top_halo_fields


def step_cpml(
        u_now, u_pre, psix, psiz, zetax, zetaz, 
        vp, dt, h, b, 
        lap_x, lap_z, 
        pml,
        grad_op,
        grad_kernels=None,
        ):

    az, bz, dbzdz, ax, bx, dbxdx = pml

    w_sum = 0.
    
    # Use fixed stencil convolutions when available; this is much cheaper than torch.gradient.
    dudz = grad_op(u_now, h, -2, kernels=grad_kernels)
    dudx = grad_op(u_now, h, -1, kernels=grad_kernels)
    
    # Z direction
    tmpz = ((1+bz)*lap_z + dbzdz * dudz) + grad_op(az*psiz, h, -2, kernels=grad_kernels)
    w_sum += (1+bz) * tmpz + az * zetaz

    psiyn = bz * dudz + az * psiz
    zetaz = bz * tmpz + az * zetaz

    # X direction
    tmpx = ((1+bx)*lap_x + dbxdx * dudx) + grad_op(ax*psix, h, -1, kernels=grad_kernels)
    w_sum += (1+bx) * tmpx + ax * zetax
    psixn = bx * dudx + ax * psix
    zetax = bx * tmpx + ax * zetax

    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * w_sum

    return u_next, u_now, psixn, psiyn, zetax, zetaz

class Acoustic(SecondOrderEquation):
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="Acoustic P-wave velocity model.", unit="m/s"),
    )
    FIELD_SPECS = (
        FieldSpec("h1", aliases=("pressure", "p"), description="Primary acoustic pressure-like wavefield.", supports_source=True, supports_receiver=True),
        FieldSpec("h2", aliases=("pressure_prev",), description="Previous-step pressure-like wavefield.", internal=True),
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
        use_fast_grad_kernels = backend == 'torch' and 'cuda' in str(device)
        super().__init__(spatial_order, device, backend, dim=dim, other_kernels=use_fast_grad_kernels)
        super().init_laplace(ltype='1dsep', backend=backend)
        self.grad_kernels = None
        if use_fast_grad_kernels:
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

    def func(self, wavefields, models, dt, h, b, **kwargs):
        u_now = wavefields[0]
        (vp,) = models
        hz, hx = self._spacings_2d(h)
        lap_u_now_z, lap_u_now_x = self.laplace1d_sep(u_now, self.laplace_kernels, hz, hx)
        out = step_cpml(*wavefields, vp, dt, h, b, lap_u_now_x, lap_u_now_z, self.b, self.gradient, self.grad_kernels)
        if getattr(self, "free_surface", False):
            out = zero_top_halo_fields(out, self.so // 2, axis=-2)
        return out

    def _C(self, ):
        # CUDA IMPLEMENTATION
        import torch
        from sweep._C import (
            acoustic2d_forward,
            acoustic2d_backward,
            acoustic2d_backward_bs,
            acoustic2d_backward_ckpt,
            acoustic2d_backward_recursive_ckpt,
        )
        return (
            acoustic2d_forward,
            acoustic2d_backward,
            acoustic2d_backward_bs,
            acoustic2d_backward_ckpt,
            acoustic2d_backward_recursive_ckpt,
        )

    def _C_rtm(self):
        import torch
        from sweep._C import acoustic2d_rtm

        return acoustic2d_rtm

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            base_nvar=3,
            pml_nvar=4,
            last_two_nvar=2,
            last_two_storage_nvar=1,
            checkpoint_nvar=6,
        )
