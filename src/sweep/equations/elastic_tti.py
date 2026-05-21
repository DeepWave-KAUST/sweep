"""2D three-component elastic TTI wave equation on a rotated staggered grid."""

from __future__ import annotations

from .base import FirstOrderEquation
from .fields import FieldSpec, ModelSpec
from ._free_surface import top_free_surface_cell_derivative, top_free_surface_derivative
from .utils import to_backend
from sweep.operators.rsg import RSGDerivative


STIFFNESS_KEYS = (
    "C11",
    "C13",
    "C14",
    "C15",
    "C16",
    "C33",
    "C34",
    "C35",
    "C36",
    "C44",
    "C45",
    "C46",
    "C55",
    "C56",
    "C66",
)


def step(
    vx,
    vy,
    vz,
    sxx,
    szz,
    syz,
    sxz,
    sxy,
    m_vxx,
    m_vxz,
    m_vyx,
    m_vyz,
    m_vzx,
    m_vzz,
    m_txxx,
    m_txzz,
    m_txyx,
    m_tyzz,
    m_txzx,
    m_tzzz,
    rho,
    C11,
    C13,
    C14,
    C15,
    C16,
    C33,
    C34,
    C35,
    C36,
    C44,
    C45,
    C46,
    C55,
    C56,
    C66,
    dt,
    h,
    b,
    rsg,
    pml=None,
    free_surface=False,
):
    """One RSG elastic-TTI step.

    The update follows the markdown spec ordering: stress derivatives update
    the three particle velocities first, then velocity derivatives update the
    five stress components. The RSG operator supplies integer-to-cell and
    cell-to-integer derivatives; no interpolation between staggered locations
    is used.
    """

    del h
    pml = b if pml is None else pml
    if pml is None or len(pml) != 6:
        raise ValueError("ElasticTTI requires pml_type='cpmlr', which provides six CPML profiles.")
    az, bz, _dbzdz, ax, bx, _dbxdx = pml
    top_halo = rsg.L

    if free_surface:
        # RSG Lx/Lz are diagonal 2D stencils, so both derivatives touch top
        # ghost cells. Stresses are cell-centered: the free surface lies
        # between the first interior stress row and its ghost image.
        dsxx_dx = top_free_surface_cell_derivative(sxx, rsg.lx_bwd, top_halo, odd=False, axis=-2)
        dsxz_dz = top_free_surface_cell_derivative(sxz, rsg.lz_bwd, top_halo, odd=True, axis=-2)
        dsxy_dx = top_free_surface_cell_derivative(sxy, rsg.lx_bwd, top_halo, odd=False, axis=-2)
        dsyz_dz = top_free_surface_cell_derivative(syz, rsg.lz_bwd, top_halo, odd=True, axis=-2)
        dsxz_dx = top_free_surface_cell_derivative(sxz, rsg.lx_bwd, top_halo, odd=True, axis=-2)
        dszz_dz = top_free_surface_cell_derivative(szz, rsg.lz_bwd, top_halo, odd=True, axis=-2)
    else:
        dsxx_dx = rsg.lx_bwd(sxx)
        dsxz_dz = rsg.lz_bwd(sxz)
        dsxy_dx = rsg.lx_bwd(sxy)
        dsyz_dz = rsg.lz_bwd(syz)
        dsxz_dx = rsg.lx_bwd(sxz)
        dszz_dz = rsg.lz_bwd(szz)

    m_txxx = ax * m_txxx + bx * dsxx_dx
    m_txzz = az * m_txzz + bz * dsxz_dz
    m_txyx = ax * m_txyx + bx * dsxy_dx
    m_tyzz = az * m_tyzz + bz * dsyz_dz
    m_txzx = ax * m_txzx + bx * dsxz_dx
    m_tzzz = az * m_tzzz + bz * dszz_dz

    dsxx_dx = dsxx_dx + m_txxx
    dsxz_dz = dsxz_dz + m_txzz
    dsxy_dx = dsxy_dx + m_txyx
    dsyz_dz = dsyz_dz + m_tyzz
    dsxz_dx = dsxz_dx + m_txzx
    dszz_dz = dszz_dz + m_tzzz

    vx = vx + (dt / rho) * (dsxx_dx + dsxz_dz)
    vy = vy + (dt / rho) * (dsxy_dx + dsyz_dz)
    vz = vz + (dt / rho) * (dsxz_dx + dszz_dz)

    if free_surface:
        # Velocities are sampled on the surface node: tangential components
        # mirror evenly, and the normal component mirrors oddly.
        dvx_dx = top_free_surface_derivative(vx, rsg.lx_fwd, top_halo, odd=False, axis=-2)
        dvy_dx = top_free_surface_derivative(vy, rsg.lx_fwd, top_halo, odd=False, axis=-2)
        dvz_dx = top_free_surface_derivative(vz, rsg.lx_fwd, top_halo, odd=True, axis=-2)
        dvz_dz = top_free_surface_derivative(vz, rsg.lz_fwd, top_halo, odd=True, axis=-2)
        dvx_dz = top_free_surface_derivative(vx, rsg.lz_fwd, top_halo, odd=False, axis=-2)
        dvy_dz = top_free_surface_derivative(vy, rsg.lz_fwd, top_halo, odd=False, axis=-2)
    else:
        dvx_dx = rsg.lx_fwd(vx)
        dvy_dx = rsg.lx_fwd(vy)
        dvz_dx = rsg.lx_fwd(vz)
        dvz_dz = rsg.lz_fwd(vz)
        dvx_dz = rsg.lz_fwd(vx)
        dvy_dz = rsg.lz_fwd(vy)

    m_vxx = ax * m_vxx + bx * dvx_dx
    m_vxz = az * m_vxz + bz * dvx_dz
    m_vyx = ax * m_vyx + bx * dvy_dx
    m_vyz = az * m_vyz + bz * dvy_dz
    m_vzx = ax * m_vzx + bx * dvz_dx
    m_vzz = az * m_vzz + bz * dvz_dz

    dvx_dx = dvx_dx + m_vxx
    dvx_dz = dvx_dz + m_vxz
    dvy_dx = dvy_dx + m_vyx
    dvy_dz = dvy_dz + m_vyz
    dvz_dx = dvz_dx + m_vzx
    dvz_dz = dvz_dz + m_vzz
    shear_xz = dvz_dx + dvx_dz

    sxx = sxx + dt * (C11 * dvx_dx + C16 * dvy_dx + C15 * shear_xz + C14 * dvy_dz + C13 * dvz_dz)
    szz = szz + dt * (C13 * dvx_dx + C36 * dvy_dx + C35 * shear_xz + C34 * dvy_dz + C33 * dvz_dz)
    syz = syz + dt * (C14 * dvx_dx + C46 * dvy_dx + C45 * shear_xz + C44 * dvy_dz + C34 * dvz_dz)
    sxz = sxz + dt * (C15 * dvx_dx + C56 * dvy_dx + C55 * shear_xz + C45 * dvy_dz + C35 * dvz_dz)
    sxy = sxy + dt * (C16 * dvx_dx + C66 * dvy_dx + C56 * shear_xz + C46 * dvy_dz + C36 * dvz_dz)

    return (
        vx,
        vy,
        vz,
        sxx,
        szz,
        syz,
        sxz,
        sxy,
        m_vxx,
        m_vxz,
        m_vyx,
        m_vyz,
        m_vzx,
        m_vzz,
        m_txxx,
        m_txzz,
        m_txyx,
        m_tyzz,
        m_txzx,
        m_tzzz,
    )


class ElasticTTI(FirstOrderEquation):
    """First-order 2-D three-component elastic TTI wave equation (RSG).

    Velocity-stress formulation for a fully anisotropic 2-D three-component
    medium (8 stiffness parameters reduced to the 15 independent
    components ``C11…C66`` via Bond rotation from the Thomsen-style
    VTI parameters). Derivatives are taken on a rotated staggered grid
    (RSG: cell-corner ↔ cell-centre with diagonal stencils) so no
    interpolation is required between staggered locations; CPML is the
    recursive variant ``cpmlr`` (six profiles).

    Reference: Bond-rotated VTI stiffness with RSG operators (Saenger /
    Bohlen rotated staggered grid).

    !!! info "Models (constructor input order)"

        - ``vp0`` (m/s): VTI-frame vertical P velocity.
        - ``vs0`` (m/s): VTI-frame vertical S velocity.
        - ``rho`` (kg/m^3) (aliases: ``density``): Density.
        - ``epsilon``: Thomsen epsilon.
        - ``delta``: Thomsen delta.
        - ``gamma``: Thomsen gamma.
        - ``theta`` (rad): Tilt angle.
        - ``phi`` (rad): Azimuth angle.

    !!! info "Wavefields"

        - ``vx`` (aliases: ``velocity_x``): Particle velocity in x; default receiver.
        - ``vy`` (aliases: ``velocity_y``): Particle velocity in y.
        - ``vz`` (aliases: ``velocity_z``): Particle velocity in z; default receiver.
        - ``sxx`` (aliases: ``stress_xx``): Normal stress xx; default source.
        - ``szz`` (aliases: ``stress_zz``): Normal stress zz; default source.
        - ``syz`` (aliases: ``stress_yz``): Shear stress yz.
        - ``sxz`` (aliases: ``stress_xz``): Shear stress xz.
        - ``sxy`` (aliases: ``stress_xy``): Shear stress xy.
        - ``m_vxx``: CPML memory for dvx/dx (internal).
        - ``m_vxz``: CPML memory for dvx/dz (internal).
        - ``m_vyx``: CPML memory for dvy/dx (internal).
        - ``m_vyz``: CPML memory for dvy/dz (internal).
        - ``m_vzx``: CPML memory for dvz/dx (internal).
        - ``m_vzz``: CPML memory for dvz/dz (internal).
        - ``m_txxx``: CPML memory for dsxx/dx (internal).
        - ``m_txzz``: CPML memory for dsxz/dz (internal).
        - ``m_txyx``: CPML memory for dsxy/dx (internal).
        - ``m_tyzz``: CPML memory for dsyz/dz (internal).
        - ``m_txzx``: CPML memory for dsxz/dx (internal).
        - ``m_tzzz``: CPML memory for dszz/dz (internal).

    !!! info "Defaults"

        - ``source_type``: ``['sxx', 'szz']``
        - ``receiver_type``: ``['vx', 'vz']``
        - ``pml_type``: ``'cpmlr'``
    """

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
        FieldSpec("vx", aliases=("velocity_x",), description="Particle velocity in x.", supports_source=True, supports_receiver=True),
        FieldSpec("vy", aliases=("velocity_y",), description="Particle velocity in y.", supports_source=True, supports_receiver=True),
        FieldSpec("vz", aliases=("velocity_z",), description="Particle velocity in z.", supports_source=True, supports_receiver=True),
        FieldSpec("sxx", aliases=("stress_xx",), description="Normal stress xx.", supports_source=True, supports_receiver=True),
        FieldSpec("szz", aliases=("stress_zz",), description="Normal stress zz.", supports_source=True, supports_receiver=True),
        FieldSpec("syz", aliases=("stress_yz",), description="Shear stress yz.", supports_source=True),
        FieldSpec("sxz", aliases=("stress_xz",), description="Shear stress xz.", supports_source=True),
        FieldSpec("sxy", aliases=("stress_xy",), description="Shear stress xy.", supports_source=True),
        FieldSpec("m_vxx", description="CPML memory for dvx/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vxz", description="CPML memory for dvx/dz.", internal=True, boundary_related=True),
        FieldSpec("m_vyx", description="CPML memory for dvy/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vyz", description="CPML memory for dvy/dz.", internal=True, boundary_related=True),
        FieldSpec("m_vzx", description="CPML memory for dvz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vzz", description="CPML memory for dvz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_txxx", description="CPML memory for dsxx/dx.", internal=True, boundary_related=True),
        FieldSpec("m_txzz", description="CPML memory for dsxz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_txyx", description="CPML memory for dsxy/dx.", internal=True, boundary_related=True),
        FieldSpec("m_tyzz", description="CPML memory for dsyz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_txzx", description="CPML memory for dsxz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_tzzz", description="CPML memory for dszz/dz.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmlr"

    def __init__(self, spatial_order=8, device="cpu", backend="torch"):
        """Build the 2-D-3C elastic TTI equation operator (RSG variant).

        Args:
            spatial_order: FD accuracy order of the rotated staggered-grid (RSG) first derivative — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding). Must be an even integer (``2, 4, 6, 8, 10,
                …``). Higher orders reduce grid dispersion at the cost
                of more compute per step and a wider PML halo.
                This equation has **no compiled `impl='c'` path; use
                `impl='eager'`** (the SG variant :class:`ElasticTTISG`
                does have a CUDA kernel). Defaults to 8.
            device: Device for the operator's static RSG kernels. Use
                ``'cuda'`` / a ``torch.device`` for GPU eager runs.
                Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or
                ``'jax'``. Defaults to ``'torch'``.
        """
        super().__init__(spatial_order, device, backend, ndim=2)
        self.rsg = RSGDerivative(spatial_order, device, backend, ndim=2)
        self.rsg.to_backend(to_backend)
        self.pd = self.rsg

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
        return ["sxx", "szz"]

    @property
    def default_receiver_fields(self):
        return ["vx", "vz"]

    def prepare_models(self, models):
        vp0, vs0, rho, epsilon, delta, gamma, theta, phi = models
        C33_0 = rho * vp0**2
        C44_0 = rho * vs0**2
        C11_0 = C33_0 * (1.0 + 2.0 * epsilon)
        C66_0 = C44_0 * (1.0 + 2.0 * gamma)
        C13_0 = self._compute_C13_from_delta(C33_0, C44_0, delta)
        stiffness = self._bond_rotate(C11_0, C13_0, C33_0, C44_0, C66_0, theta, phi)
        return [rho] + [stiffness[key] for key in STIFFNESS_KEYS]

    @staticmethod
    def _compute_C13_from_delta(C33, C44, delta):
        value = 2.0 * delta * C33 * (C33 - C44) + (C33 - C44) ** 2
        if hasattr(value, "clamp_min"):
            value = value.clamp_min(0.0)
        else:
            import jax.numpy as jnp

            value = jnp.maximum(value, 0.0)
        return value**0.5 - C44

    @staticmethod
    def _bond_rotate(C11_0, C13_0, C33_0, C44_0, C66_0, theta, phi):
        if hasattr(theta, "new_zeros"):
            return ElasticTTI._bond_rotate_torch(C11_0, C13_0, C33_0, C44_0, C66_0, theta, phi)
        return ElasticTTI._bond_rotate_jax(C11_0, C13_0, C33_0, C44_0, C66_0, theta, phi)

    @staticmethod
    def _bond_rotation_matrix_torch(theta, phi):
        import torch

        ct, st = torch.cos(theta), torch.sin(theta)
        cp, sp = torch.cos(phi), torch.sin(phi)
        zero = torch.zeros_like(theta)
        row0 = torch.stack((cp * ct, -sp, cp * st), dim=-1)
        row1 = torch.stack((sp * ct, cp, sp * st), dim=-1)
        row2 = torch.stack((-st, zero, ct), dim=-1)
        return torch.stack((row0, row1, row2), dim=-2)

    @staticmethod
    def _bond_rotation_matrix_jax(theta, phi):
        import jax.numpy as jnp

        ct, st = jnp.cos(theta), jnp.sin(theta)
        cp, sp = jnp.cos(phi), jnp.sin(phi)
        zero = jnp.zeros_like(theta)
        row0 = jnp.stack((cp * ct, -sp, cp * st), axis=-1)
        row1 = jnp.stack((sp * ct, cp, sp * st), axis=-1)
        row2 = jnp.stack((-st, zero, ct), axis=-1)
        return jnp.stack((row0, row1, row2), axis=-2)

    @staticmethod
    def _vti_voigt_torch(C11_0, C13_0, C33_0, C44_0, C66_0):
        import torch

        shape = C11_0.shape + (6, 6)
        C = torch.zeros(shape, dtype=C11_0.dtype, device=C11_0.device)
        C12_0 = C11_0 - 2.0 * C66_0
        C[..., 0, 0] = C11_0
        C[..., 1, 1] = C11_0
        C[..., 0, 1] = C12_0
        C[..., 1, 0] = C12_0
        C[..., 0, 2] = C13_0
        C[..., 2, 0] = C13_0
        C[..., 1, 2] = C13_0
        C[..., 2, 1] = C13_0
        C[..., 2, 2] = C33_0
        C[..., 3, 3] = C44_0
        C[..., 4, 4] = C44_0
        C[..., 5, 5] = C66_0
        return C

    @staticmethod
    def _voigt_to_tensor_torch(C):
        import torch

        pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        out = torch.zeros(C.shape[:-2] + (3, 3, 3, 3), dtype=C.dtype, device=C.device)
        for i, (a, b) in enumerate(pairs):
            for j, (c, d) in enumerate(pairs):
                value = C[..., i, j]
                out[..., a, b, c, d] = value
                out[..., b, a, c, d] = value
                out[..., a, b, d, c] = value
                out[..., b, a, d, c] = value
        return out

    @staticmethod
    def _tensor_to_entries_torch(C):
        pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        wanted = {
            "C11": (0, 0),
            "C13": (0, 2),
            "C14": (0, 3),
            "C15": (0, 4),
            "C16": (0, 5),
            "C33": (2, 2),
            "C34": (2, 3),
            "C35": (2, 4),
            "C36": (2, 5),
            "C44": (3, 3),
            "C45": (3, 4),
            "C46": (3, 5),
            "C55": (4, 4),
            "C56": (4, 5),
            "C66": (5, 5),
        }
        return {
            key: C[..., pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1]]
            for key, (i, j) in wanted.items()
        }

    @staticmethod
    def _bond_rotate_torch(C11_0, C13_0, C33_0, C44_0, C66_0, theta, phi):
        import torch

        C_vti = ElasticTTI._vti_voigt_torch(C11_0, C13_0, C33_0, C44_0, C66_0)
        C_tensor = ElasticTTI._voigt_to_tensor_torch(C_vti)
        R = ElasticTTI._bond_rotation_matrix_torch(theta, phi)
        C_rot = torch.einsum("...ip,...jq,...kr,...ls,...pqrs->...ijkl", R, R, R, R, C_tensor)
        entries = ElasticTTI._tensor_to_entries_torch(C_rot)
        mask = torch.abs(theta) < 1e-7
        zero = torch.zeros_like(C11_0)
        vti_entries = {
            "C11": C11_0,
            "C13": C13_0,
            "C33": C33_0,
            "C44": C44_0,
            "C55": C44_0,
            "C66": C66_0,
        }
        return {
            key: torch.where(mask, vti_entries.get(key, zero), value)
            for key, value in entries.items()
        }

    @staticmethod
    def _bond_rotate_jax(C11_0, C13_0, C33_0, C44_0, C66_0, theta, phi):
        import jax.numpy as jnp

        pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
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
        C_tensor = jnp.zeros(C_vti.shape[:-2] + (3, 3, 3, 3), dtype=C_vti.dtype)
        for i, (a, b) in enumerate(pairs):
            for j, (c, d) in enumerate(pairs):
                value = C_vti[..., i, j]
                C_tensor = C_tensor.at[..., a, b, c, d].set(value)
                C_tensor = C_tensor.at[..., b, a, c, d].set(value)
                C_tensor = C_tensor.at[..., a, b, d, c].set(value)
                C_tensor = C_tensor.at[..., b, a, d, c].set(value)
        R = ElasticTTI._bond_rotation_matrix_jax(theta, phi)
        C_rot = jnp.einsum("...ip,...jq,...kr,...ls,...pqrs->...ijkl", R, R, R, R, C_tensor)
        wanted = {
            "C11": (0, 0),
            "C13": (0, 2),
            "C14": (0, 3),
            "C15": (0, 4),
            "C16": (0, 5),
            "C33": (2, 2),
            "C34": (2, 3),
            "C35": (2, 4),
            "C36": (2, 5),
            "C44": (3, 3),
            "C45": (3, 4),
            "C46": (3, 5),
            "C55": (4, 4),
            "C56": (4, 5),
            "C66": (5, 5),
        }
        mask = jnp.abs(theta) < 1e-7
        zero = jnp.zeros_like(C11_0)
        vti_entries = {
            "C11": C11_0,
            "C13": C13_0,
            "C33": C33_0,
            "C44": C44_0,
            "C55": C44_0,
            "C66": C66_0,
        }
        return {
            key: jnp.where(
                mask,
                vti_entries.get(key, zero),
                C_rot[..., pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1]],
            )
            for key, (i, j) in wanted.items()
        }

    def func(self, wavefields, models, dt, h, b, **kwargs):
        if len(models) == len(self.MODEL_SPECS):
            models = self.prepare_models(models)
        elif len(models) != 1 + len(STIFFNESS_KEYS):
            raise ValueError(
                "ElasticTTI.func expected 8 raw models or "
                f"{1 + len(STIFFNESS_KEYS)} prepared models, got {len(models)}."
            )
        return step(
            *wavefields,
            *models,
            dt,
            h,
            b,
            rsg=self.rsg,
            pml=self.b,
            free_surface=getattr(self, "free_surface", False),
            **kwargs,
        )
