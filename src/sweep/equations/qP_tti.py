import torch
try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None

from .base import SecondOrderEquation
from .fields import FieldSpec, ModelSpec
from .qP_vti import (adjoint_mode_step, check_qp_stability, check_s_gradient, qp_cuda_layout,
                     refuse_boundary_saving, s_gradient_aux, s_operator, stop_gradient)
from ._registry import register_equation


def s_tilted(px, pz, epsilon, delta, cos0, sin0):
    """S_n of a gradient pair in the frame tilted by theta (eq. A-5)."""
    numerator = -2 * (epsilon - delta) * (px * cos0 - pz * sin0) ** 2 * (px * sin0 + pz * cos0) ** 2
    denominator = (
        (1 + 2 * epsilon) * (px * cos0 - pz * sin0) ** 4
        + (px * sin0 + pz * cos0) ** 4
        + 2 * (1 + delta) * (px * cos0 - pz * sin0) ** 2 * (px * sin0 + pz * cos0) ** 2
    )
    # Division, not ``* (den + 1e-26) ** -1``: see qP_vti.s_operator.
    return numerator / (denominator + 1e-26)


def rhs_cpml(
    u_now,
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
    s_frozen=False,
    s_override=None,
):
    """The update of one TTI step and the next CPML memories (see
    ``qP_vti.rhs_cpml``, also for ``s_frozen`` / ``s_override``)."""

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

    # theta is the tilt angle in RADIANS — matching this equation's
    # ModelSpec(unit="rad") and every other TTI equation (e.g. elastic_tti uses
    # cos(theta) directly). Do not deg2rad it here: that double-converted a
    # radian input (30° = 0.5236 rad → 0.0091 rad ≈ 0.5°), collapsing the tilt
    # to almost nothing.
    sin0 = op.sin(theta)
    cos0 = op.cos(theta)
    sin20 = op.sin(2 * theta)

    if s_override is not None:
        sk = s_override
    else:
        px, pz = (stop_gradient(dpdx), stop_gradient(dpdz)) if s_frozen else (dpdx, dpdz)
        sk = s_tilted(px, pz, epsilon, delta, cos0, sin0)

    vp2dt2 = vp**2 * dt**2
    rhs = (
        vp2dt2 * ((1 + 2 * epsilon) * cos0**2 + sin0**2 + sk) * lap_x_pml
        + vp2dt2 * ((1 + 2 * epsilon) * sin0**2 + cos0**2 + sk) * lap_z_pml
        - 2 * epsilon * vp2dt2 * sin20 * dpdx_dz
    )

    psixn = bx * dudx + ax * psix
    psizn = bz * dudz + az * psiz
    zetaxn = bx * tmpx + ax * zetax
    zetazn = bz * tmpz + az * zetaz

    return rhs, (psixn, psizn, zetaxn, zetazn)


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
    s_frozen=False,
):
    rhs, aux = rhs_cpml(u_now, psix, psiz, zetax, zetaz, vp, epsilon, delta, theta, dt, h, b, pml,
                        lap_x, lap_z, grad_op, op, grad_kernels, s_frozen)
    u_next = 2 * u_now - u_pre + rhs
    return (u_next, u_now, *aux)


@register_equation(aliases=('AcousticTTILiang',))
class AcousticTTI(SecondOrderEquation):
    """Second-order 2-D pure-qP TTI wave equation (Liang et al. 2023, appendix A).

    Tilted-symmetry-axis extension of :class:`AcousticVTI`. The
    pressure-like field ``h1`` is updated with anisotropy weights that
    are computed in a tilted frame defined by ``theta``; the Laplacian
    along the symmetry axis is rotated accordingly, and an extra
    ``∂²p/∂x∂z`` mixed-term carries the off-diagonal anisotropy.
    A single-field pure qP equation -- no auxiliary ``f`` field. The paper gives
    the TTI operator in the wavenumber domain (eq. A-7); here S_n takes the
    wavefield gradient rotated into the symmetry-axis frame (eq. A-3). Also
    exposed as :class:`AcousticAniso(method='liang', symmetry='tti')`.

    Reference: Liang K. et al. 2023, 10.1190/geo2022-0292.1.

    ``s_gradient`` as :class:`AcousticVTI` (``'adjoint'``, the default,
    propagates the adjoint wavefield with its own S_n; ``'frozen'`` holds S_n
    at the forward field; ``'exact'`` differentiates through its direction).

    
    """

    # The isotropic image-method free surface is WRONG for an anisotropic
    # medium (see EquationBase.supports_free_surface); fail loud instead.
    supports_free_surface = False

    # As AcousticVTI: eager boundary saving zeroes the CPML while it rebuilds
    # the field, and the nonlinear qP adjoint reads that field in the
    # absorbing layer (about 2% off measured).  impl='c' keeps the frame.
    supports_boundary_saving_eager = False

    C_NAME = "qp_tti2d"
    C_HAS_RECURSIVE_CKPT = False   # no checkpoint backward (see AcousticVTI)
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

    def __init__(self, spatial_order=4, device="cpu", backend="torch", dim=2, s_gradient="adjoint"):
        """Build the 2-D pure-qP TTI equation operator.

        Args:
            spatial_order: FD accuracy order of the spatial Laplacian and the auxiliary first-derivative kernels used by the rotated-anisotropy term — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding). Must be an even integer (``2, 4, 6, 8, 10, …``).
                ``impl='c'`` (CUDA) runs the same scheme with full or
                boundary-saving gradients (no checkpointing). Defaults to 4.
            device: Device for the operator's static kernels. Use
                ``'cuda'`` / a ``torch.device`` for GPU eager runs.
                Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or
                ``'jax'``. Defaults to ``'torch'``. Requires the
                matching array library to be importable.
            dim: Stored dimensionality. Always ``2`` for this class.
                Defaults to 2.
            s_gradient: ``'adjoint'``, ``'frozen'`` or ``'exact'`` (see
                :class:`AcousticVTI`). Defaults to ``'adjoint'``.
        """
        super().__init__(spatial_order, device, backend)
        super().init_separable_laplace()
        super().init_grad_kernels()
        self.s_gradient = check_s_gradient(s_gradient, backend)
        if backend == "jax" and jnp is None:
            raise ImportError("AcousticTTI with backend='jax' requires jax to be installed.")
        self.op = {"torch": torch, "jax": jnp}[backend]

    @property
    def default_source_fields(self):
        return ["h1"]

    @property
    def default_receiver_fields(self):
        return ["h1"]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        hz, hx = self._spacings_2d(h)
        if self.s_gradient == "adjoint":
            rhs, aux = adjoint_mode_step(*self._adjoint_fns(dt, h, b), wavefields, models, h, self.b,
                                         self.gradient, self.grad_kernels)
            return (2 * wavefields[0] - wavefields[1] + rhs, wavefields[0], *aux)
        nabla_z, nabla_x = self.separable_d2_2d(wavefields[0], self.laplace_kernels, hz, hx)
        return step_cpml(*wavefields, *models, dt, h, b, self.b, nabla_x, nabla_z, self.gradient, self.op, self.grad_kernels,
                         s_frozen=self.s_gradient == "frozen", **kwargs)

    def _adjoint_fns(self, dt, h, b):
        """(rhs_fn, s_fn, lap_fn) of ``qP_vti.adjoint_mode_step``."""
        hz, hx = self._spacings_2d(h)

        def lap_fn(u):
            return self.separable_d2_2d(u, self.laplace_kernels, hz, hx)

        def rhs_fn(u, psix, psiz, zetax, zetaz, models, s_frozen, s_override):
            lap_z, lap_x = lap_fn(u)
            return rhs_cpml(u, psix, psiz, zetax, zetaz, *models, dt, h, b, self.b, lap_x, lap_z,
                            self.gradient, self.op, self.grad_kernels, s_frozen=s_frozen,
                            s_override=s_override)

        def s_fn(field, models):
            _, epsilon, delta, theta = models
            return s_tilted(self.gradient(field, h, -1, kernels=self.grad_kernels),
                            self.gradient(field, h, -2, kernels=self.grad_kernels),
                            epsilon, delta, self.op.cos(theta), self.op.sin(theta))

        return rhs_fn, s_fn, lap_fn

    @property
    def cuda_layout(self):
        return qp_cuda_layout()

    def c_eq_aux(self, prop):
        return s_gradient_aux(self)

    def check_stability(self, models, dt, spacing):
        """Warn once when ``dt`` breaks the scheme's stability condition
        (:func:`~sweep.equations.qP_vti.qp_stability_ratio`); ``PropTorch`` calls it before every forward."""
        check_qp_stability(self, models, dt, spacing)

    def c_refuse_grads(self, wavelet_grad, model_grad_mask, boundary_saving=False, boundary_dtype=None,
                       checkpoint=False):
        refuse_boundary_saving(self, boundary_saving, boundary_dtype)
