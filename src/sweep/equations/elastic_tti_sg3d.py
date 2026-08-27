"""3D three-component elastic TTI wave equation on an axis-aligned staggered grid."""

from __future__ import annotations

from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec
from .elastic_tti import ElasticTTI
from .fields import FieldSpec, ModelSpec


# Full 21-component Voigt upper triangle produced by the 3-D Bond rotation.
# Order matters: the CUDA backend unpacks prepared models in this order.
STIFFNESS_KEYS_3D = (
    "C11", "C12", "C13", "C14", "C15", "C16",
    "C22", "C23", "C24", "C25", "C26",
    "C33", "C34", "C35", "C36",
    "C44", "C45", "C46",
    "C55", "C56",
    "C66",
)

# (row, col) Voigt indices for each stiffness key, used when reading the
# rotated 3x3x3x3 tensor back into the 21 unique entries.
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
_KEY_TO_VOIGT = {
    key: (int(key[1]) - 1, int(key[2]) - 1) for key in STIFFNESS_KEYS_3D
}


def step(
    vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
    m_vxx, m_vxy, m_vxz,
    m_vyx, m_vyy, m_vyz,
    m_vzx, m_vzy, m_vzz,
    m_sxxx, m_szzz,
    m_sxyx, m_sxyy,
    m_sxzx, m_sxzz,
    m_syyy,
    m_syzy, m_syzz,
    rho,
    C11, C12, C13, C14, C15, C16,
    C22, C23, C24, C25, C26,
    C33, C34, C35, C36,
    C44, C45, C46,
    C55, C56,
    C66,
    dt, h, b, pd,
    pml=None,
):
    """One axis-aligned staggered-grid 3-D elastic-TTI step.

    Same Virieux SSG stencil and CPML staging as the isotropic 3-D
    :func:`sweep.equations.elastic3d.step`; only the constitutive update
    changes — the six stress components couple to all six Voigt strain
    rates through the full 21-entry rotated stiffness. Mixed-location
    stiffness couplings are used directly without interpolation (the
    no-interpolation SG convention shared with :class:`ElasticTTISG`).
    """

    del h
    pml = pml if pml is not None else ()
    if len(pml) != 12:
        raise ValueError("ElasticTTISG3D requires pml_type='cpmls', which provides twelve staggered CPML profiles.")
    az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh = pml

    dsxx_dx = pd.x_forward(sxx)
    dsxy_dy = pd.y_backward(sxy)
    dsxz_dz = pd.z_backward(sxz)

    dsxy_dx = pd.x_backward(sxy)
    dsyy_dy = pd.y_forward(syy)
    dsyz_dz = pd.z_backward(syz)

    dsxz_dx = pd.x_backward(sxz)
    dsyz_dy = pd.y_backward(syz)
    dszz_dz = pd.z_forward(szz)

    m_szzz = azh * m_szzz + bzh * dszz_dz
    dszz_dz = dszz_dz + m_szzz
    m_sxzx = ax * m_sxzx + bx * dsxz_dx
    dsxz_dx = dsxz_dx + m_sxzx

    m_sxzz = az * m_sxzz + bz * dsxz_dz
    dsxz_dz = dsxz_dz + m_sxzz
    m_sxxx = axh * m_sxxx + bxh * dsxx_dx
    dsxx_dx = dsxx_dx + m_sxxx

    m_sxyy = ay * m_sxyy + by * dsxy_dy
    dsxy_dy = dsxy_dy + m_sxyy

    m_sxyx = ax * m_sxyx + bx * dsxy_dx
    dsxy_dx = dsxy_dx + m_sxyx

    m_syyy = ayh * m_syyy + byh * dsyy_dy
    dsyy_dy = dsyy_dy + m_syyy
    m_syzz = az * m_syzz + bz * dsyz_dz
    dsyz_dz = dsyz_dz + m_syzz

    m_syzy = ay * m_syzy + by * dsyz_dy
    dsyz_dy = dsyz_dy + m_syzy

    vx = vx + dt / rho * (dsxx_dx + dsxy_dy + dsxz_dz)
    vy = vy + dt / rho * (dsxy_dx + dsyy_dy + dsyz_dz)
    vz = vz + dt / rho * (dsxz_dx + dsyz_dy + dszz_dz)

    dvx_dx = pd.x_backward(vx)
    dvx_dy = pd.y_forward(vx)
    dvx_dz = pd.z_forward(vx)

    dvy_dx = pd.x_forward(vy)
    dvy_dy = pd.y_backward(vy)
    dvy_dz = pd.z_forward(vy)

    dvz_dx = pd.x_forward(vz)
    dvz_dy = pd.y_forward(vz)
    dvz_dz = pd.z_backward(vz)

    m_vzz = az * m_vzz + bz * dvz_dz
    dvz_dz = dvz_dz + m_vzz
    m_vyy = ay * m_vyy + by * dvy_dy
    dvy_dy = dvy_dy + m_vyy
    m_vxx = ax * m_vxx + bx * dvx_dx
    dvx_dx = dvx_dx + m_vxx
    m_vxz = azh * m_vxz + bzh * dvx_dz
    dvx_dz = dvx_dz + m_vxz
    m_vzx = axh * m_vzx + bxh * dvz_dx
    dvz_dx = dvz_dx + m_vzx

    m_vxy = ayh * m_vxy + byh * dvx_dy
    dvx_dy = dvx_dy + m_vxy
    m_vyx = axh * m_vyx + bxh * dvy_dx
    dvy_dx = dvy_dx + m_vyx
    m_vyz = azh * m_vyz + bzh * dvy_dz
    dvy_dz = dvy_dz + m_vyz
    m_vzy = ayh * m_vzy + byh * dvz_dy
    dvz_dy = dvz_dy + m_vzy

    # Voigt strain rates (engineering shears).
    e1 = dvx_dx
    e2 = dvy_dy
    e3 = dvz_dz
    e4 = dvy_dz + dvz_dy
    e5 = dvx_dz + dvz_dx
    e6 = dvx_dy + dvy_dx

    sxx = sxx + dt * (C11 * e1 + C12 * e2 + C13 * e3 + C14 * e4 + C15 * e5 + C16 * e6)
    syy = syy + dt * (C12 * e1 + C22 * e2 + C23 * e3 + C24 * e4 + C25 * e5 + C26 * e6)
    szz = szz + dt * (C13 * e1 + C23 * e2 + C33 * e3 + C34 * e4 + C35 * e5 + C36 * e6)
    syz = syz + dt * (C14 * e1 + C24 * e2 + C34 * e3 + C44 * e4 + C45 * e5 + C46 * e6)
    sxz = sxz + dt * (C15 * e1 + C25 * e2 + C35 * e3 + C45 * e4 + C55 * e5 + C56 * e6)
    sxy = sxy + dt * (C16 * e1 + C26 * e2 + C36 * e3 + C46 * e4 + C56 * e5 + C66 * e6)

    return (
        vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
        m_vxx, m_vxy, m_vxz,
        m_vyx, m_vyy, m_vyz,
        m_vzx, m_vzy, m_vzz,
        m_sxxx, m_szzz,
        m_sxyx, m_sxyy,
        m_sxzx, m_sxzz,
        m_syyy,
        m_syzy, m_syzz,
    )


class ElasticTTISG3D(FirstOrderEquation):
    """First-order 3-D three-component elastic TTI wave equation (axis-aligned SG).

    Velocity-stress formulation for a tilted transversely isotropic 3-D
    medium: the 8 raw model parameters (Thomsen-style VTI constants plus
    tilt ``theta`` and azimuth ``phi``) are reduced to the full 21
    independent stiffness components ``C11…C66`` via the 3-D Bond
    rotation, and the six stresses couple to all six Voigt strain rates.
    Derivatives use the Virieux staggered grid shared with
    :class:`~sweep.equations.elastic3d.Elastic`; mixed-location
    stiffness couplings are used directly without interpolation, making
    this the 3-D companion of :class:`ElasticTTISG`. CPML follows the
    staggered-grid ``cpmls`` convention (12 profiles).

    With isotropic parameters (``epsilon=delta=gamma=theta=phi=0``) the
    stiffness collapses to ``lambda``/``mu`` and the update is
    numerically identical to the isotropic 3-D elastic equation.


    """

    # The isotropic image-method free surface is WRONG for an anisotropic
    # medium (see EquationBase.supports_free_surface); fail loud instead.
    supports_free_surface = False

    prepare_models_for_c = True
    default_pml_type = "cpmls"

    MODEL_SPECS = (
        ModelSpec("vp0", description="VTI-frame vertical P velocity.", unit="m/s"),
        ModelSpec("vs0", description="VTI-frame vertical S velocity.", unit="m/s"),
        ModelSpec("rho", aliases=("density",), description="Density.", unit="kg/m^3"),
        ModelSpec("epsilon", description="Thomsen epsilon."),
        ModelSpec("delta", description="Thomsen delta."),
        ModelSpec("gamma", description="Thomsen gamma."),
        ModelSpec("theta", description="Tilt angle.", unit="rad"),
        ModelSpec("phi", description="Azimuth angle.", unit="rad"),
    )

    FIELD_SPECS = (
        FieldSpec("vx", aliases=("velocity_x",), description="Particle velocity in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("vy", aliases=("velocity_y",), description="Particle velocity in the y direction.", supports_source=True, supports_receiver=True),
        FieldSpec("vz", aliases=("velocity_z",), description="Particle velocity in the z direction.", supports_source=True, supports_receiver=True),
        FieldSpec("sxx", aliases=("stress_xx",), description="Normal stress in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("syy", aliases=("stress_yy",), description="Normal stress in the y direction.", supports_source=True, supports_receiver=True),
        FieldSpec("szz", aliases=("stress_zz",), description="Normal stress in the z direction.", supports_source=True, supports_receiver=True),
        FieldSpec("sxy", aliases=("stress_xy", "shear_xy"), description="Shear stress component.", supports_source=True),
        FieldSpec("sxz", aliases=("stress_xz", "shear_xz"), description="Shear stress component.", supports_source=True),
        FieldSpec("syz", aliases=("stress_yz", "shear_yz"), description="Shear stress component.", supports_source=True),
        FieldSpec("m_vxx", description="CPML memory variable for dvx/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vxy", description="CPML memory variable for dvx/dy.", internal=True, boundary_related=True),
        FieldSpec("m_vxz", description="CPML memory variable for dvx/dz.", internal=True, boundary_related=True),
        FieldSpec("m_vyx", description="CPML memory variable for dvy/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vyy", description="CPML memory variable for dvy/dy.", internal=True, boundary_related=True),
        FieldSpec("m_vyz", description="CPML memory variable for dvy/dz.", internal=True, boundary_related=True),
        FieldSpec("m_vzx", description="CPML memory variable for dvz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vzy", description="CPML memory variable for dvz/dy.", internal=True, boundary_related=True),
        FieldSpec("m_vzz", description="CPML memory variable for dvz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_sxxx", description="CPML memory variable for dsxx/dx.", internal=True, boundary_related=True),
        FieldSpec("m_szzz", description="CPML memory variable for dszz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_sxyx", description="CPML memory variable for dsxy/dx.", internal=True, boundary_related=True),
        FieldSpec("m_sxyy", description="CPML memory variable for dsxy/dy.", internal=True, boundary_related=True),
        FieldSpec("m_sxzx", description="CPML memory variable for dsxz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_sxzz", description="CPML memory variable for dsxz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_syyy", description="CPML memory variable for dsyy/dy.", internal=True, boundary_related=True),
        FieldSpec("m_syzy", description="CPML memory variable for dsyz/dy.", internal=True, boundary_related=True),
        FieldSpec("m_syzz", description="CPML memory variable for dsyz/dz.", internal=True, boundary_related=True),
    )

    def __init__(self, spatial_order=4, device="cpu", backend="torch"):
        """Build the 3-D-3C elastic TTI equation operator (axis-aligned SG).

        Args:
            spatial_order: FD accuracy order of the staggered first-derivative operator — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding). Must
                be an even integer (``2, 4, 6, 8, 10, …``).
                **Performance note (`impl='c'` on CUDA):** the compiled
                kernels ship template specialisations only for
                ``spatial_order ∈ {2, 4, 6, 8}``. Above 8 the dispatcher
                drops to a generic runtime path (``order = -1`` in
                ``src/sweep/csrc/cuda/equations/elastic_tti_sg3d/forward.cu``)
                which uses more registers and runs noticeably slower.
                The PyTorch eager path is unaffected. Defaults to 4.
            device: Device for the operator's static gradient kernels.
                Use ``'cuda'`` / a ``torch.device`` for GPU runs so the
                propagator can follow without a host↔device copy.
                Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or
                ``'jax'``. When you later want ``impl='c'``, leave this
                on ``'torch'``. Defaults to ``'torch'``.
        """
        super().__init__(spatial_order, device, backend, ndim=3)

    @property
    def default_source_fields(self):
        return ["sxx", "syy", "szz"]

    @property
    def default_receiver_fields(self):
        return ["vx", "vy", "vz"]

    def prepare_models(self, models):
        vp0, vs0, rho, epsilon, delta, gamma, theta, phi = models
        C33_0 = rho * vp0**2
        C44_0 = rho * vs0**2
        C11_0 = C33_0 * (1.0 + 2.0 * epsilon)
        C66_0 = C44_0 * (1.0 + 2.0 * gamma)
        C13_0 = ElasticTTI._compute_C13_from_delta(C33_0, C44_0, delta)
        stiffness = self._bond_rotate_3d(C11_0, C13_0, C33_0, C44_0, C66_0, theta, phi)
        return [rho] + [stiffness[key] for key in STIFFNESS_KEYS_3D]

    @staticmethod
    def _bond_rotate_3d(C11_0, C13_0, C33_0, C44_0, C66_0, theta, phi):
        if hasattr(theta, "new_zeros"):
            return ElasticTTISG3D._bond_rotate_3d_torch(C11_0, C13_0, C33_0, C44_0, C66_0, theta, phi)
        return ElasticTTISG3D._bond_rotate_3d_jax(C11_0, C13_0, C33_0, C44_0, C66_0, theta, phi)

    @staticmethod
    def _bond_matrix_torch(R):
        """Voigt-space Bond stress-transformation matrix (…, 6, 6) of a
        rotation R (…, 3, 3); rows/cols ordered xx, yy, zz, yz, xz, xy.

        Equivalent to the rank-4 rotation ``R R R R : C`` but the autograd
        graph only carries (…, 6, 6) intermediates — the einsum form
        materialises (…, 3^8) tensors in its backward, which is ~16 GB on a
        padded 3-D grid and OOMs the CUDA prepare_models path.
        """
        import torch

        def r(i, j):
            return R[..., i, j]

        rows = []
        for i in range(3):
            rows.append(torch.stack((
                r(i, 0) ** 2, r(i, 1) ** 2, r(i, 2) ** 2,
                2.0 * r(i, 1) * r(i, 2),
                2.0 * r(i, 0) * r(i, 2),
                2.0 * r(i, 0) * r(i, 1),
            ), dim=-1))
        for a, b in ((1, 2), (0, 2), (0, 1)):
            rows.append(torch.stack((
                r(a, 0) * r(b, 0), r(a, 1) * r(b, 1), r(a, 2) * r(b, 2),
                r(a, 1) * r(b, 2) + r(a, 2) * r(b, 1),
                r(a, 0) * r(b, 2) + r(a, 2) * r(b, 0),
                r(a, 0) * r(b, 1) + r(a, 1) * r(b, 0),
            ), dim=-1))
        return torch.stack(rows, dim=-2)

    @staticmethod
    def _bond_rotate_3d_torch(C11_0, C13_0, C33_0, C44_0, C66_0, theta, phi):
        import torch

        C_vti = ElasticTTI._vti_voigt_torch(C11_0, C13_0, C33_0, C44_0, C66_0)
        R = ElasticTTI._bond_rotation_matrix_torch(theta, phi)
        M = ElasticTTISG3D._bond_matrix_torch(R)
        C_rot = M @ C_vti @ M.transpose(-1, -2)

        mask = torch.abs(theta) < 1e-7
        zero = torch.zeros_like(C11_0)
        C12_0 = C11_0 - 2.0 * C66_0
        vti_entries = {
            "C11": C11_0,
            "C12": C12_0,
            "C13": C13_0,
            "C22": C11_0,
            "C23": C13_0,
            "C33": C33_0,
            "C44": C44_0,
            "C55": C44_0,
            "C66": C66_0,
        }
        out = {}
        for key in STIFFNESS_KEYS_3D:
            i, j = _KEY_TO_VOIGT[key]
            value = C_rot[..., i, j]
            out[key] = torch.where(mask, vti_entries.get(key, zero), value)
        return out

    @staticmethod
    def _bond_rotate_3d_jax(C11_0, C13_0, C33_0, C44_0, C66_0, theta, phi):
        import jax.numpy as jnp

        C12_0 = C11_0 - 2.0 * C66_0
        rows = (
            (C11_0, C12_0, C13_0, 0.0, 0.0, 0.0),
            (C12_0, C11_0, C13_0, 0.0, 0.0, 0.0),
            (C13_0, C13_0, C33_0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, C44_0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, C44_0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, C66_0),
        )
        C_vti = jnp.stack([jnp.stack([jnp.zeros_like(C11_0) + v for v in row], axis=-1) for row in rows], axis=-2)
        R = ElasticTTI._bond_rotation_matrix_jax(theta, phi)

        def r(i, j):
            return R[..., i, j]

        m_rows = []
        for i in range(3):
            m_rows.append(jnp.stack((
                r(i, 0) ** 2, r(i, 1) ** 2, r(i, 2) ** 2,
                2.0 * r(i, 1) * r(i, 2),
                2.0 * r(i, 0) * r(i, 2),
                2.0 * r(i, 0) * r(i, 1),
            ), axis=-1))
        for a, b in ((1, 2), (0, 2), (0, 1)):
            m_rows.append(jnp.stack((
                r(a, 0) * r(b, 0), r(a, 1) * r(b, 1), r(a, 2) * r(b, 2),
                r(a, 1) * r(b, 2) + r(a, 2) * r(b, 1),
                r(a, 0) * r(b, 2) + r(a, 2) * r(b, 0),
                r(a, 0) * r(b, 1) + r(a, 1) * r(b, 0),
            ), axis=-1))
        M = jnp.stack(m_rows, axis=-2)
        C_rot = M @ C_vti @ jnp.swapaxes(M, -1, -2)

        mask = jnp.abs(theta) < 1e-7
        zero = jnp.zeros_like(C11_0)
        vti_entries = {
            "C11": C11_0,
            "C12": C12_0,
            "C13": C13_0,
            "C22": C11_0,
            "C23": C13_0,
            "C33": C33_0,
            "C44": C44_0,
            "C55": C44_0,
            "C66": C66_0,
        }
        out = {}
        for key in STIFFNESS_KEYS_3D:
            i, j = _KEY_TO_VOIGT[key]
            value = C_rot[..., i, j]
            out[key] = jnp.where(mask, vti_entries.get(key, zero), value)
        return out

    def func(self, wavefields, models, dt, h, b, **kwargs):
        if len(models) == len(self.MODEL_SPECS):
            models = self.prepare_models(models)
        elif len(models) != 1 + len(STIFFNESS_KEYS_3D):
            raise ValueError(
                "ElasticTTISG3D.func expected 8 raw models or "
                f"{1 + len(STIFFNESS_KEYS_3D)} prepared models, got {len(models)}."
            )
        return step(
            *wavefields,
            *models,
            dt,
            h,
            b,
            pd=self.pd,
            pml=self.b,
            **kwargs,
        )

    def _C(self):
        import sweep._C as _C

        return (
            _C.elastic_tti_sg3d_forward,
            _C.elastic_tti_sg3d_backward,
            _C.elastic_tti_sg3d_backward_bs,
            _C.elastic_tti_sg3d_backward_ckpt,
            None,
        )

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            base_nvar=9,
            pml_nvar=27,
            last_two_nvar=1,
            last_two_storage_nvar=9,
            backward_workspace_nvar=18,
        )
