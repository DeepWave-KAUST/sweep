from .base import SecondOrderEquation


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
    """Parameter order: vp, epsilon, delta.

       Wavefields: (h1, h2, psix, psiz, zetax, zetaz)

       Reference: Liang K., et.al, 10.1190/geo2022-0292.1
    """

    def __init__(self, spatial_order=4, device="cpu", backend="torch", dim=2):
        super().__init__(spatial_order, device, backend, other_kernels=True)
        super().init_laplace(ltype="1dsep", backend=backend)
        self.grad_kernels = {-2: self.gkernel_z, -1: self.gkernel_x}

    @property
    def models(self):
        return ["vp", "epsilon", "delta"]

    @property
    def wavefields(self):
        return ["h1", "h2", "psix", "psiz", "zetax", "zetaz"]

    def func(self, *args, **kwargs):
        hz, hx = self._spacings_2d(args[10])
        nabla_z, nabla_x = self.laplace1d_sep(args[0], self.laplace_kernels, hz, hx)
        return step_cpml(*args, self.b, nabla_x, nabla_z, self.gradient, self.grad_kernels, **kwargs)
