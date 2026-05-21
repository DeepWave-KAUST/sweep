from .base import SecondOrderEquation
from .fields import FieldSpec, ModelSpec


def step_cpml(
    u_now,
    u_pre,
    f_now,
    f_pre,
    psix,
    psiz,
    zetax,
    zetaz,
    vv,
    v,
    eta,
    dt,
    h,
    b,
    pml,
    lap_x,
    lap_z,
    dpdx2dz2,
    grad_op,
    grad_kernels=None,
):

    az, bz, dbzdz, ax, bx, dbxdx = pml

    dudz = grad_op(u_now, h, -2, kernels=grad_kernels)
    dudx = grad_op(u_now, h, -1, kernels=grad_kernels)

    tmpz = ((1 + bz) * lap_z + dbzdz * dudz) + grad_op(az * psiz, h, -2, kernels=grad_kernels)
    tmpx = ((1 + bx) * lap_x + dbxdx * dudx) + grad_op(ax * psix, h, -1, kernels=grad_kernels)

    lap_z_pml = (1 + bz) * tmpz + az * zetaz
    lap_x_pml = (1 + bx) * tmpx + ax * zetax

    u_next = (
        2 * u_now
        - u_pre
        + (1 + 2 * eta) * v**2 * lap_x_pml * dt**2
        + vv**2 * lap_z_pml * dt**2
        - 2 * eta * vv**2 * v**2 * dpdx2dz2 * dt**2
    )
    f_next = 2 * f_now - f_pre + dt**2 * u_now

    psixn = bx * dudx + ax * psix
    psizn = bz * dudz + az * psiz
    zetaxn = bx * tmpx + ax * zetax
    zetazn = bz * tmpz + az * zetaz

    return u_next, u_now, f_next, f_now, psixn, psizn, zetaxn, zetazn


class AcousticTariq(SecondOrderEquation):
    """Second-order 2-D pseudo-acoustic VTI/TTI wave equation (Alkhalifah eta).

    Two-field pseudo-acoustic formulation: the primary pressure-like
    field ``h1`` is coupled to an auxiliary integrated field ``f1``
    through the anellipticity parameter ``eta``. The split into
    ``(h1, f1)`` is what suppresses the shear-mode artifact that the
    original Alkhalifah pseudo-acoustic exhibits. The model parameter
    ``v`` here is the *NMO velocity* (``v = vp · sqrt(1 + 2 delta)``),
    **not** the horizontal velocity; ``vv`` is the squared vertical
    velocity. Source must include ``h1`` (default
    ``source_type=['h1']`` is set by the propagator).

    Reference: Alkhalifah T. 2000, *An acoustic wave equation for
    anisotropic media*, 10.1190/1.1444815.

    !!! info "Models (constructor input order)"

        - ``vv``: Squared vertical velocity-like parameter for the Tariq qP formulation.
        - ``v`` (m/s) (aliases: ``velocity``): Velocity-like parameter for the Tariq qP formulation.
        - ``eta``: Anellipticity parameter for the Tariq qP formulation.

    !!! info "Wavefields"

        - ``h1`` (aliases: ``pressure``, ``p``): Primary acoustic-Tariq qP pressure-like wavefield; default source and receiver.
        - ``h2`` (aliases: ``pressure_prev``): Previous-step pressure-like wavefield (internal).
        - ``f1``: Auxiliary wavefield integrating the primary pressure (Tariq qP) (internal).
        - ``f2``: Previous-step auxiliary wavefield (Tariq qP) (internal).
        - ``psix``: CPML memory variable for the x-derivative term (internal).
        - ``psiz``: CPML memory variable for the z-derivative term (internal).
        - ``zetax``: CPML auxiliary wavefield for the x-direction update (internal).
        - ``zetaz``: CPML auxiliary wavefield for the z-direction update (internal).

    !!! info "Defaults"

        - ``source_type``: ``['h1']``
        - ``receiver_type``: ``['h1']``
        - ``pml_type``: ``'cpmlr'``
    """
    MODEL_SPECS = (
        ModelSpec("vv", description="Squared vertical velocity-like parameter for the Tariq qP formulation."),
        ModelSpec("v", aliases=("velocity",), description="Velocity-like parameter for the Tariq qP formulation.", unit="m/s"),
        ModelSpec("eta", description="Anellipticity parameter for the Tariq qP formulation."),
    )
    FIELD_SPECS = (
        FieldSpec("h1", aliases=("pressure", "p"), description="Primary acoustic-Tariq qP pressure-like wavefield.", supports_source=True, supports_receiver=True),
        FieldSpec("h2", aliases=("pressure_prev",), description="Previous-step pressure-like wavefield.", internal=True),
        FieldSpec("f1", description="Auxiliary wavefield integrating the primary pressure (Tariq qP).", internal=True),
        FieldSpec("f2", description="Previous-step auxiliary wavefield (Tariq qP).", internal=True),
        FieldSpec("psix", description="CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiz", description="CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("zetax", description="CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetaz", description="CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmlr"

    def __init__(self, spatial_order=4, device="cpu", backend="torch", dim=2):
        """Build the 2-D Alkhalifah-eta pseudo-acoustic equation operator.

        Args:
            spatial_order: FD accuracy order of the spatial Laplacian and the auxiliary first-derivative kernels — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding). Must be an even integer
                (``2, 4, 6, 8, 10, …``). This equation has **no
                compiled `impl='c'` path; use `impl='eager'`** (the
                default). Defaults to 4.
            device: Device for the operator's static kernels. Use
                ``'cuda'`` / a ``torch.device`` for GPU eager runs.
                Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or
                ``'jax'``. Defaults to ``'torch'``.
            dim: Stored dimensionality. Always ``2`` for this class.
                Defaults to 2.
        """
        super().__init__(spatial_order, device, backend, other_kernels=True)
        super().init_laplace(ltype="1dsep", backend=backend)
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

    @property
    def default_source_fields(self):
        return ["h1"]

    @property
    def default_receiver_fields(self):
        return ["h1"]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        hz, hx = self._spacings_2d(h)
        nabla_z, nabla_x = self.laplace1d_sep(wavefields[0], self.laplace_kernels, hz, hx)
        _, dpdx2 = self.laplace1d_sep(wavefields[2], self.laplace_kernels, hz, hx)
        dpdx2dz2, _ = self.laplace1d_sep(dpdx2, self.laplace_kernels, hz, hx)
        return step_cpml(*wavefields, *models, dt, h, b, self.b, nabla_x, nabla_z, dpdx2dz2, self.gradient, self.grad_kernels, **kwargs)
