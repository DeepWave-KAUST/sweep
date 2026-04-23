from .base import SecondOrderEquation
from .fields import ModelSpec


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
    """Parameter order: vv, v, eta.

       Wavefields: (h1, h2, f1, f2, psix, psiz, zetax, zetaz)

       Reference: Alkhalifah Tariq, 10.1190/1.1444815
    """
    MODEL_SPECS = (
        ModelSpec("vv", description="Squared vertical velocity-like parameter for the Tariq qP formulation."),
        ModelSpec("v", aliases=("velocity",), description="Velocity-like parameter for the Tariq qP formulation.", unit="m/s"),
        ModelSpec("eta", description="Anellipticity parameter for the Tariq qP formulation."),
    )

    def __init__(self, spatial_order=4, device="cpu", backend="torch", dim=2):
        super().__init__(spatial_order, device, backend, other_kernels=True)
        super().init_laplace(ltype="1dsep", backend=backend)
        self.grad_kernels = {-2: self.gkernel_z, -1: self.gkernel_x}

    @property
    def models(self):
        return [spec.name for spec in self.MODEL_SPECS]

    @property
    def wavefields(self):
        return ["h1", "h2", "f1", "f2", "psix", "psiz", "zetax", "zetaz"]

    def func(self, *args, **kwargs):
        hz, hx = self._spacings_2d(args[12])
        nabla_z, nabla_x = self.laplace1d_sep(args[0], self.laplace_kernels, hz, hx)
        _, dpdx2 = self.laplace1d_sep(args[2], self.laplace_kernels, hz, hx)
        dpdx2dz2, _ = self.laplace1d_sep(dpdx2, self.laplace_kernels, hz, hx)
        return step_cpml(*args, self.b, nabla_x, nabla_z, dpdx2dz2, self.gradient, self.grad_kernels, **kwargs)
