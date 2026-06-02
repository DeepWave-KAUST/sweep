from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from .utils import zero_top_halo_fields

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
        super().init_laplace(ltype='1dsep')
    
    @property
    def default_source_fields(self):
        return ["h1"]

    @property
    def default_receiver_fields(self):
        return ["sh1"]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        hz, hx = self._spacings_2d(h)
        lap_uz, lap_ux = self.laplace1d_sep(wavefields[0], self.laplace_kernels, hz, hx)
        lap_suz, lap_sux = self.laplace1d_sep(wavefields[6], self.laplace_kernels, hz, hx)
        out = step(*wavefields, *models, dt, h, b, lap_ux, lap_uz, lap_sux, lap_suz, self.b, self.gradient)
        if getattr(self, "free_surface", False):
            out = zero_top_halo_fields(out, self.so // 2, axis=-2)
        return out

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
            boundary_save_nvar=1,
        )
    
