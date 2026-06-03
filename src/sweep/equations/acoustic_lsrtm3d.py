from .base import SecondOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from .utils import zero_top_halo_fields


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

    tmpz = ((1 + bz) * lap_z + dbzdz * dudz) + grad_op(az * psiz, h, -3)
    w_sum += (1 + bz) * tmpz + az * zetaz
    psizn = bz * dudz + az * psiz
    zetaz = bz * tmpz + az * zetaz

    tmpy = ((1 + by) * lap_y + dbydy * dudy) + grad_op(ay * psiy, h, -2)
    w_sum += (1 + by) * tmpy + ay * zetay
    psiyn = by * dudy + ay * psiy
    zetay = by * tmpy + ay * zetay

    tmpx = ((1 + bx) * lap_x + dbxdx * dudx) + grad_op(ax * psix, h, -1)
    w_sum += (1 + bx) * tmpx + ax * zetax
    psixn = bx * dudx + ax * psix
    zetax = bx * tmpx + ax * zetax

    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * w_sum

    sw_sum = 0.0

    stmpz = ((1 + bz) * lap_sz + dbzdz * dsudz) + grad_op(az * spsiz, h, -3)
    sw_sum += (1 + bz) * stmpz + az * szetaz
    spsizn = bz * dsudz + az * spsiz
    szetaz = bz * stmpz + az * szetaz

    stmpy = ((1 + by) * lap_sy + dbydy * dsudy) + grad_op(ay * spsiy, h, -2)
    sw_sum += (1 + by) * stmpy + ay * szetay
    spsiyn = by * dsudy + ay * spsiy
    szetay = by * stmpy + ay * szetay

    stmpx = ((1 + bx) * lap_sx + dbxdx * dsudx) + grad_op(ax * spsix, h, -1)
    sw_sum += (1 + bx) * stmpx + ax * szetax
    spsixn = bx * dsudx + ax * spsix
    szetax = bx * stmpx + ax * szetax

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
        super().init_laplace(ltype="3dsep")

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

    def _C(self):
        from sweep._C import (
            acoustic_lsrtm3d_forward,
            acoustic_lsrtm3d_backward,
            acoustic_lsrtm3d_backward_bs,
            acoustic_lsrtm3d_backward_ckpt,
            acoustic_lsrtm3d_backward_recursive_ckpt,
        )

        return (
            acoustic_lsrtm3d_forward,
            acoustic_lsrtm3d_backward,
            acoustic_lsrtm3d_backward_bs,
            acoustic_lsrtm3d_backward_ckpt,
            acoustic_lsrtm3d_backward_recursive_ckpt,
        )

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            base_nvar=6,
            pml_nvar=12,
            last_two_nvar=2,
            last_two_storage_nvar=1,
            checkpoint_nvar=8,
            boundary_save_nvar=1,
        )
