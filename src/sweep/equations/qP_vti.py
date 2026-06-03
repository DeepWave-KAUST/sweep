from .base import SecondOrderEquation
from .fields import FieldSpec, ModelSpec


def step_cpml(
    u_now,
    u_pre,
    psix,
    psiz,
    zetax,
    zetaz,
    vp,
    epsilon,
    delta,
    dt,
    h,
    b,
    pml,
    lap_x,
    lap_z,
    grad_op,
    grad_kernels=None,
):

    az, bz, dbzdz, ax, bx, dbxdx = pml

    dudz = grad_op(u_now, h, -2, kernels=grad_kernels)
    dudx = grad_op(u_now, h, -1, kernels=grad_kernels)

    # Use the split-field PML corrected first derivatives in the anisotropy term.
    dpdz = (1 + bz) * dudz + az * psiz
    dpdx = (1 + bx) * dudx + ax * psix

    tmpz = ((1 + bz) * lap_z + dbzdz * dudz) + grad_op(az * psiz, h, -2, kernels=grad_kernels)
    tmpx = ((1 + bx) * lap_x + dbxdx * dudx) + grad_op(ax * psix, h, -1, kernels=grad_kernels)

    lap_z_pml = (1 + bz) * tmpz + az * zetaz
    lap_x_pml = (1 + bx) * tmpx + ax * zetax

    # 10.1190/geo2022-0292.1 EQ(19) from 10.1190/geo2014-0242.1
    numerator = -2 * (epsilon - delta) * dpdx**2 * dpdz**2
    denominator = (1 + 2 * epsilon) * dpdx**4 + dpdz**4 + 2 * (1 + delta) * dpdx**2 * dpdz**2
    sk = numerator * ((denominator + 1e-26) ** -1)

    vp2dt2 = vp**2 * dt**2
    u_next = 2 * u_now - u_pre + vp2dt2 * (((1 + 2 * epsilon) + sk) * lap_x_pml + (1 + sk) * lap_z_pml)

    psixn = bx * dudx + ax * psix
    psizn = bz * dudz + az * psiz
    zetaxn = bx * tmpx + ax * zetax
    zetazn = bz * tmpz + az * zetaz

    return u_next, u_now, psixn, psizn, zetaxn, zetazn


class AcousticVTI(SecondOrderEquation):
    """Second-order 2-D pseudo-acoustic VTI wave equation (Liang 2022).

    Single-field pseudo-acoustic VTI formulation derived from the
    dispersion relation. The pressure-like field ``h1`` is driven by
    PML-corrected Laplacian terms with anisotropy-dependent weights
    that depend on the local propagation direction of the wavefront
    (computed from the spatial gradients of ``h1`` itself). This
    avoids the auxiliary ``f`` field needed by the Alkhalifah / eta
    family while still suppressing the shear-mode artifact that
    plagues the original Alkhalifah pseudo-acoustic.

    Source / receiver caveat: ``source_type=['h1']`` is the default;
    typical Ricker injection on ``h1`` works well at modest grid
    spacings. With strongly anisotropic media (large ``epsilon``) at
    ``dh=(5, 5)`` and ``dt=1 ms``, the scheme can go unstable — use
    coarser z spacing or smaller ``dt``. Also exposed as
    :class:`AcousticAniso(method='liang', symmetry='vti')`.

    Reference: Liang K. et al. 2022, 10.1190/geo2022-0292.1; underlying
    pseudo-acoustic derivation: 10.1190/geo2014-0242.1.

    
    """
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="VTI acoustic reference velocity.", unit="m/s"),
        ModelSpec("epsilon", description="Thomsen epsilon parameter."),
        ModelSpec("delta", description="Thomsen delta parameter."),
    )
    FIELD_SPECS = (
        FieldSpec("h1", aliases=("pressure", "p"), description="Primary acoustic-VTI pressure-like wavefield.", supports_source=True, supports_receiver=True),
        FieldSpec("h2", aliases=("pressure_prev",), description="Previous-step pressure-like wavefield.", internal=True),
        FieldSpec("psix", description="CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiz", description="CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("zetax", description="CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetaz", description="CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmlr"

    def __init__(self, spatial_order=4, device="cpu", backend="torch", dim=2):
        """Build the 2-D pseudo-acoustic VTI equation operator.

        Args:
            spatial_order: FD accuracy order of the spatial Laplacian and the auxiliary first-derivative kernels used by the anisotropy direction term — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding). Must be an even integer (``2, 4, 6, 8,
                10, …``). This equation has **no compiled `impl='c'`
                path; use `impl='eager'`** (the default). Defaults to 4.
            device: Device for the operator's static kernels. Use
                ``'cuda'`` / a ``torch.device`` for GPU eager runs.
                Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or
                ``'jax'``. Defaults to ``'torch'``.
            dim: Stored dimensionality. Always ``2`` for this class.
                Defaults to 2.
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
        hz, hx = self._spacings_2d(h)
        nabla_z, nabla_x = self.separable_d2_2d(wavefields[0], self.laplace_kernels, hz, hx)
        return step_cpml(*wavefields, *models, dt, h, b, self.b, nabla_x, nabla_z, self.gradient, self.grad_kernels, **kwargs)
