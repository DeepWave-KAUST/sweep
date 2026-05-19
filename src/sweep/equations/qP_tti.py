import torch
try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None

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
    theta,
    dt,
    h,
    b,
    pml,
    lap_x,
    lap_z,
    grad_op,
    op=None,
    grad_kernels=None,
):

    az, bz, dbzdz, ax, bx, dbxdx = pml

    dudz = grad_op(u_now, h, -2, kernels=grad_kernels)
    dudx = grad_op(u_now, h, -1, kernels=grad_kernels)

    dpdz = (1 + bz) * dudz + az * psiz
    dpdx = (1 + bx) * dudx + ax * psix

    tmpz = ((1 + bz) * lap_z + dbzdz * dudz) + grad_op(az * psiz, h, -2, kernels=grad_kernels)
    tmpx = ((1 + bx) * lap_x + dbxdx * dudx) + grad_op(ax * psix, h, -1, kernels=grad_kernels)

    lap_z_pml = (1 + bz) * tmpz + az * zetaz
    lap_x_pml = (1 + bx) * tmpx + ax * zetax

    # Approximate the mixed term from the PML-adjusted first derivative so the
    # anisotropic coupling follows the same boundary treatment as the Laplacian.
    dpdx_dz = grad_op(dpdx, h, -2, kernels=grad_kernels)

    theta = op.deg2rad(theta)
    sin0 = op.sin(theta)
    cos0 = op.cos(theta)
    sin20 = op.sin(2 * theta)

    numerator = -2 * (epsilon - delta) * (dpdx * cos0 - dpdz * sin0) ** 2 * (dpdx * sin0 + dpdz * cos0) ** 2
    denominator = (
        (1 + 2 * epsilon) * (dpdx * cos0 - dpdz * sin0) ** 4
        + (dpdx * sin0 + dpdz * cos0) ** 4
        + 2 * (1 + delta) * (dpdx * cos0 - dpdz * sin0) ** 2 * (dpdx * sin0 + dpdz * cos0) ** 2
    )
    sk = numerator * ((denominator + 1e-26) ** -1)

    vp2dt2 = vp**2 * dt**2
    u_next = 2 * u_now - u_pre + (
        vp2dt2 * ((1 + 2 * epsilon) * cos0**2 + sin0**2 + sk) * lap_x_pml
        + vp2dt2 * ((1 + 2 * epsilon) * sin0**2 + cos0**2 + sk) * lap_z_pml
        - 2 * epsilon * vp2dt2 * sin20 * dpdx_dz
    )

    psixn = bx * dudx + ax * psix
    psizn = bz * dudz + az * psiz
    zetaxn = bx * tmpx + ax * zetax
    zetazn = bz * tmpz + az * zetaz

    return u_next, u_now, psixn, psizn, zetaxn, zetazn


class AcousticTTI(SecondOrderEquation):
    """Parameter order: vp, epsilon, delta, theta.

       Wavefields: (h1, h2, psix, psiz, zetax, zetaz)

       Reference: Liang K., et.al, 10.1190/geo2022-0292.1
    """
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="TTI acoustic reference velocity.", unit="m/s"),
        ModelSpec("epsilon", description="Thomsen epsilon parameter."),
        ModelSpec("delta", description="Thomsen delta parameter."),
        ModelSpec("theta", description="Tilt angle parameter.", unit="rad"),
    )
    FIELD_SPECS = (
        FieldSpec("h1", aliases=("pressure", "p"), description="Primary acoustic-TTI pressure-like wavefield.", supports_source=True, supports_receiver=True),
        FieldSpec("h2", aliases=("pressure_prev",), description="Previous-step pressure-like wavefield.", internal=True),
        FieldSpec("psix", description="CPML memory variable for the x-derivative term.", internal=True, boundary_related=True),
        FieldSpec("psiz", description="CPML memory variable for the z-derivative term.", internal=True, boundary_related=True),
        FieldSpec("zetax", description="CPML auxiliary wavefield for the x-direction update.", internal=True, boundary_related=True),
        FieldSpec("zetaz", description="CPML auxiliary wavefield for the z-direction update.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmlr"

    def __init__(self, spatial_order=4, device="cpu", backend="torch", dim=2):
        super().__init__(spatial_order, device, backend, other_kernels=True)
        super().init_laplace(ltype="1dsep", backend=backend)
        self.grad_kernels = {-2: self.gkernel_z, -1: self.gkernel_x}
        if backend == "jax" and jnp is None:
            raise ImportError("AcousticTTI with backend='jax' requires jax to be installed.")
        self.op = {"torch": torch, "jax": jnp}[backend]

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
        return step_cpml(*wavefields, *models, dt, h, b, self.b, nabla_x, nabla_z, self.gradient, self.op, self.grad_kernels, **kwargs)
