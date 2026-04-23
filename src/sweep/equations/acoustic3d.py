from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec

def step_cpml(
        u_now, u_pre, psix, psiy, psiz, zetax, zetay, zetaz, 
        vp, dt, h, b, 
        lap_x, lap_y, lap_z, 
        pml,
        grad_op
        ):

    az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx = pml

    w_sum = 0.

    dudz = grad_op(u_now, h, -3)
    dudy = grad_op(u_now, h, -2)
    dudx = grad_op(u_now, h, -1)

    # Z direction
    tmpz = ((1+bz)*lap_z + dbzdz * dudz) + grad_op(az*psiz, h, -3)
    w_sum += (1+bz) * tmpz + az * zetaz

    psizn = bz * dudz + az * psiz
    zetaz = bz * tmpz + az * zetaz

    # Y direction
    tmpy = ((1+by)*lap_y + dbydy * dudy) + grad_op(ay*psiy, h, -2)
    w_sum += (1+by) * tmpy + ay * zetay
    psiyn = by * dudy + ay * psiy
    zetay = by * tmpy + ay * zetay

    # X direction
    tmpx = ((1+bx)*lap_x + dbxdx * dudx) + grad_op(ax*psix, h, -1)
    w_sum += (1+bx) * tmpx + ax * zetax
    psixn = bx * dudx + ax * psix
    zetax = bx * tmpx + ax * zetax

    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * w_sum

    return u_next, u_now, psixn, psiyn, psizn, zetax, zetay, zetaz

class Acoustic3D(SecondOrderEquation):
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="3D acoustic P-wave velocity model.", unit="m/s"),
    )
    FIELD_SPECS = (
        FieldSpec("h1", aliases=("pressure", "p"), description="Primary 3D acoustic pressure-like wavefield.", supports_source=True, supports_receiver=True),
        FieldSpec("h2", aliases=("pressure_prev",), description="Previous-step pressure-like wavefield.", internal=True),
        FieldSpec("psix", description="CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiy", description="CPML memory variable for the y-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiz", description="CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("zetax", description="CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetay", description="CPML auxiliary wavefield for the y-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetaz", description="CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
    )

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=3):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        super().__init__(spatial_order, device, backend, dim=dim)
        self._wavefields = []
        super().init_laplace(ltype='3dsep', backend=backend)

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
        dh = args[10]
        hz, hy, hx = self._spacings_3d(dh)
        lap_z, lap_y, lap_x = self.laplace3d_sep(args[0], self.laplace_kernels, hz, hy, hx)
        return step_cpml(*args, lap_x, lap_y, lap_z, self.b, self.gradient)

    def _C(self, ):
        import torch
        from sweep._C import (
            acoustic3d_forward,
            acoustic3d_backward,
            acoustic3d_backward_bs,
            acoustic3d_backward_ckpt,
            acoustic3d_backward_recursive_ckpt,
        )
        return (
            acoustic3d_forward,
            acoustic3d_backward,
            acoustic3d_backward_bs,
            acoustic3d_backward_ckpt,
            acoustic3d_backward_recursive_ckpt,
        )

    def _C_rtm(self):
        import torch
        from sweep._C import acoustic3d_rtm

        return acoustic3d_rtm

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            base_nvar=3,
            pml_nvar=6,
            last_two_nvar=2,
            last_two_storage_nvar=1,
            checkpoint_nvar=8,
        )
