from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec

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
    # Z direction
    tmpz = ((1+bz)*lap_uz + dbzdz * dudz) + grad_op(az*psiz, h, -2)
    w_sum += (1+bz) * tmpz + az * zetaz

    psiyn = bz * dudz + az * psiz
    zetaz = bz * tmpz + az * zetaz

    # X direction
    tmpx = ((1+bx)*lap_ux + dbxdx * dudx) + grad_op(ax*psix, h, -1)
    w_sum += (1+bx) * tmpx + ax * zetax
    psixn = bx * dudx + ax * psix
    zetax = bx * tmpx + ax * zetax

    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * w_sum

    # Scatter wavefield
    w_sum_s = 0.
    # Z direction
    tmpsz = ((1+bz)*lap_suz + dbzdz * dsudz) + grad_op(az*spsiz, h, -2)
    w_sum_s += (1+bz) * tmpsz + az * szetaz
    spsiyn = bz * dsudz + az * spsiz
    szetaz = bz * tmpsz + az * szetaz   
    # X direction
    tmpx_s = ((1+bx)*lap_sux + dbxdx * dsudx) + grad_op(ax*spsix, h, -1)
    w_sum_s += (1+bx) * tmpx_s + ax * szetax
    spsixn = bx * dsudx + ax * spsix
    szetax = bx * tmpx_s + ax * szetax
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

class AcousticLSRTM(SecondOrderEquation):
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

    def __init__(self, spatial_order=4, device='cpu', backend='torch'):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        super().__init__(spatial_order, device, backend)
        super().init_laplace(ltype='1dsep')
    
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
        dh = args[15]
        hz, hx = self._spacings_2d(dh)
        lap_uz, lap_ux = self.laplace1d_sep(args[0], self.laplace_kernels, hz, hx)
        lap_suz, lap_sux = self.laplace1d_sep(args[6], self.laplace_kernels, hz, hx)
        return step(*args, lap_ux, lap_uz, lap_sux, lap_suz, self.b, self.gradient)

    def _C(self):
        from sweep._C import (
            acoustic_lsrtm2d_forward,
            acoustic_lsrtm2d_backward,
            acoustic_lsrtm2d_backward_bs,
            acoustic_lsrtm2d_backward_ckpt,
            acoustic_lsrtm2d_backward_recursive_ckpt,
        )

        return (
            acoustic_lsrtm2d_forward,
            acoustic_lsrtm2d_backward,
            acoustic_lsrtm2d_backward_bs,
            acoustic_lsrtm2d_backward_ckpt,
            acoustic_lsrtm2d_backward_recursive_ckpt,
        )

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            base_nvar=6,
            pml_nvar=8,
            last_two_nvar=2,
            last_two_storage_nvar=1,
            checkpoint_nvar=6,
        )
    
