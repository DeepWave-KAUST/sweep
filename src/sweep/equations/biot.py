"""2-D Biot poroelastic waves: first-order velocity-stress on the Virieux staggered grid.

Unknowns: solid particle velocity ``v``, Darcy flux ``q = phi (v_f - v)`` (fluid velocity
relative to the solid, per unit bulk area), total stress ``sigma`` and pore pressure ``p``.
The low-frequency Biot equations (Biot 1962; Carcione, *Wave Fields in Real Media*, ch. 7)::

    rho  dv/dt + rho_f dq/dt         = div sigma
    rho_f dv/dt + m dq/dt + b q      = -grad p
    d sigma/dt = (H - 2 mu) div v I + mu (grad v + grad v^T) + C div q I
    d p/dt     = -C div v - M div q

``H = lambda_u + 2 mu`` is the undrained P-wave modulus, ``C = alpha M`` (Biot coefficient
``alpha`` times Biot modulus ``M``), ``m = T rho_f / phi`` the fluid's effective inertia
(tortuosity ``T``) and ``b = eta / kappa`` the viscous coupling.  Written with the relative
flux, the equations hold in heterogeneous media with no porosity-gradient terms.

The viscous term is stiff: its relaxation rate ``gamma = b / (m - rho_f^2 / rho)`` is of order
1e5-1e6 1/s for ordinary rocks, far above any seismic time step.  With the stresses frozen
over a step (as the leapfrog does) the flux obeys ``dq/dt = -gamma q + G``, which is
integrated exactly::

    q^{n+1/2} = exp(-gamma dt) q^{n-1/2} + (1 - exp(-gamma dt)) / gamma * G
    v^{n+1/2} = v^{n-1/2} + (dt div sigma - rho_f (q^{n+1/2} - q^{n-1/2})) / rho

so any permeability is stable, the Darcy limit is exact, and ``b -> inf`` (no relative flow)
reduces to elastic waves in the undrained medium (``Elastic`` with ``vp = sqrt(H / rho)``,
``vs = sqrt(mu / rho)``).  The time step is limited by the fast P wave of the inviscid medium,
see :func:`biot_fast_p_velocity`.

Discretisation follows ``Elastic``: ``vx, qx`` share the ``vx`` node and ``vz, qz`` the
``vz`` node, ``p`` sits with ``sxx, szz``; one CPML memory field per first derivative
(``cpmls``).  Material arrays are used at every node without averaging, as in ``Elastic``.
Eager only; no free surface yet.

References: M. A. Biot (1962), J. Appl. Phys. 33(4), 1482-1498; N. Dai, A. Vafidis &
E. R. Kanasewich (1995), Geophysics 60(2), 327-340; J. M. Carcione & G. Quiroga-Goode (1995),
J. Comput. Acoust. 3(4), 261-280.
"""

from __future__ import annotations

import math

import torch

from ._cpml import cpml_update
from ._registry import register_equation
from .base import FirstOrderEquation
from .fields import FieldSpec, ModelSpec

__all__ = ["Biot", "biot_from_rock", "biot_from_biot_constants", "biot_fast_p_velocity",
           "biot_wave_velocities"]


def biot_from_rock(k_s, rho_s, k_f, rho_f, eta, k_d, mu, phi, kappa, tortuosity=1.0):
    """Biot's coefficients ``[H, mu, C, M, rho, rho_f, m, b]`` from rock and fluid properties.

    ``k_s``, ``rho_s``: grain bulk modulus (Pa) and density; ``k_f``, ``rho_f``, ``eta``: fluid
    bulk modulus, density and viscosity (Pa s); ``k_d``, ``mu``: drained (dry-frame) bulk and
    shear moduli; ``phi`` porosity; ``kappa`` permeability (m^2); ``tortuosity`` >= 1.
    Arguments may be tensors (gradients flow through) or floats.  Uses Gassmann's
    ``alpha = 1 - k_d / k_s`` and ``1/M = (alpha - phi)/k_s + phi/k_f``.
    """
    alpha = 1.0 - k_d / k_s
    biot_m = 1.0 / ((alpha - phi) / k_s + phi / k_f)
    h = k_d + 4.0 / 3.0 * mu + alpha * alpha * biot_m
    rho = (1.0 - phi) * rho_s + phi * rho_f
    m = tortuosity * rho_f / phi
    b = eta / kappa
    return [h, mu, alpha * biot_m, biot_m, rho, rho_f, m, b]


def biot_from_biot_constants(P, Q, R, N, rho11, rho12, rho22, b, phi):
    """Coefficients ``[H, mu, C, M, rho, rho_f, m, b]`` from Biot's ``P, Q, R, N``, ``rho_ij`` and
    damping ``b`` in the absolute-fluid-velocity form (e.g. Dai et al. 1995, Table 1).

    ``R = phi^2 M``, ``Q = phi (alpha - phi) M``, ``P = K_d + 4N/3 + (alpha - phi)^2 M``,
    ``rho12 + rho22 = phi rho_f``, ``rho22 = phi T rho_f``; that damping multiplies
    ``v_f - v = q / phi``, so it is ``b / phi^2`` here.  In homogeneous media the two forms give
    the same solid velocity and total stress; ``phi`` only rescales ``q`` and ``p``.
    """
    biot_m = R / phi**2
    c = (Q + R) / phi
    h = P + 2.0 * Q + R
    rho = rho11 + 2.0 * rho12 + rho22
    rho_f = (rho12 + rho22) / phi
    m = rho22 / phi**2
    return [h, N, c, biot_m, rho, rho_f, m, b / phi**2]


def biot_wave_velocities(models):
    """High-frequency (inviscid) fast-P, slow-P and S velocities of ``[H, mu, C, M, rho, rho_f, m, b]``.

    Roots of ``(rho m - rho_f^2) V^4 - (H m + M rho - 2 C rho_f) V^2 + (H M - C^2) = 0`` and
    ``V_S^2 = mu / (rho - rho_f^2 / m)``.
    """
    h, mu, c, biot_m, rho, rho_f, m = models[:7]
    a = rho * m - rho_f**2
    bq = h * m + biot_m * rho - 2.0 * c * rho_f
    cq = h * biot_m - c**2
    disc = (bq**2 - 4.0 * a * cq) ** 0.5
    vfast = ((bq + disc) / (2.0 * a)) ** 0.5
    vslow = ((bq - disc) / (2.0 * a)) ** 0.5
    vs = (mu / (rho - rho_f**2 / m)) ** 0.5
    return vfast, vslow, vs


def biot_fast_p_velocity(models):
    """Largest fast-P velocity of the models; sets the explicit time step like ``vp`` does."""
    vfast = biot_wave_velocities(models)[0]
    return float(vfast.max()) if isinstance(vfast, torch.Tensor) else float(vfast)


@register_equation()
class Biot(FirstOrderEquation):
    """First-order 2-D Biot poroelastic wave equation on a staggered grid.

    Velocity-stress-pressure form with the Darcy flux as the fluid unknown (Biot 1962), the
    staggered scheme of Virieux (1986) as in ``Elastic``, and an exact exponential update of the
    stiff viscous coupling, so the time step is set by the inviscid fast P wave whatever the
    permeability.  Build the eight coefficient arrays from rock properties with
    :func:`biot_from_rock`.  Sources are typically an explosion in the total stress
    (``['sxx', 'szz']``) or a fluid-pressure source (``['p']``); receivers read ``vx``, ``vz``,
    ``p`` or the flux.
    """

    MODEL_SPECS = (
        ModelSpec("H", description="Undrained P-wave modulus lambda_u + 2 mu.", unit="Pa"),
        ModelSpec("mu", aliases=("shear_modulus",), description="Shear modulus of the frame.", unit="Pa"),
        ModelSpec("C", description="Coupling modulus alpha M (Biot coefficient times Biot modulus).", unit="Pa"),
        ModelSpec("M", description="Biot modulus.", unit="Pa"),
        ModelSpec("rho", aliases=("density",), description="Bulk density (1 - phi) rho_s + phi rho_f.", unit="kg/m^3"),
        ModelSpec("rho_f", aliases=("fluid_density",), description="Pore-fluid density.", unit="kg/m^3"),
        ModelSpec("m", description="Effective fluid inertia T rho_f / phi (tortuosity T).", unit="kg/m^3"),
        ModelSpec("b", description="Viscous coupling eta / kappa (fluid viscosity / permeability).", unit="Pa s/m^2"),
    )
    FIELD_SPECS = (
        FieldSpec("vx", aliases=("velocity_x",), description="Solid particle velocity, x.", supports_source=True, supports_receiver=True),
        FieldSpec("vz", aliases=("velocity_z",), description="Solid particle velocity, z.", supports_source=True, supports_receiver=True),
        FieldSpec("qx", aliases=("darcy_flux_x",), description="Darcy flux (fluid velocity relative to the solid), x.", supports_source=True, supports_receiver=True),
        FieldSpec("qz", aliases=("darcy_flux_z",), description="Darcy flux (fluid velocity relative to the solid), z.", supports_source=True, supports_receiver=True),
        FieldSpec("sxx", aliases=("stress_xx",), description="Total normal stress, x.", supports_source=True, supports_receiver=True),
        FieldSpec("szz", aliases=("stress_zz",), description="Total normal stress, z.", supports_source=True, supports_receiver=True),
        FieldSpec("sxz", aliases=("stress_xz", "shear_xz"), description="Total shear stress.", supports_source=True, supports_receiver=True),
        FieldSpec("p", aliases=("pressure", "pore_pressure"), description="Pore-fluid pressure.", supports_source=True, supports_receiver=True),
        FieldSpec("m_sxx_x", description="CPML memory for dsxx/dx.", internal=True, boundary_related=True, cpml_axis="x"),
        FieldSpec("m_sxz_z", description="CPML memory for dsxz/dz.", internal=True, boundary_related=True, cpml_axis="z"),
        FieldSpec("m_sxz_x", description="CPML memory for dsxz/dx.", internal=True, boundary_related=True, cpml_axis="x"),
        FieldSpec("m_szz_z", description="CPML memory for dszz/dz.", internal=True, boundary_related=True, cpml_axis="z"),
        FieldSpec("m_p_x", description="CPML memory for dp/dx.", internal=True, boundary_related=True, cpml_axis="x"),
        FieldSpec("m_p_z", description="CPML memory for dp/dz.", internal=True, boundary_related=True, cpml_axis="z"),
        FieldSpec("m_vx_x", description="CPML memory for dvx/dx.", internal=True, boundary_related=True, cpml_axis="x"),
        FieldSpec("m_vz_z", description="CPML memory for dvz/dz.", internal=True, boundary_related=True, cpml_axis="z"),
        FieldSpec("m_vx_z", description="CPML memory for dvx/dz.", internal=True, boundary_related=True, cpml_axis="z"),
        FieldSpec("m_vz_x", description="CPML memory for dvz/dx.", internal=True, boundary_related=True, cpml_axis="x"),
        FieldSpec("m_qx_x", description="CPML memory for dqx/dx.", internal=True, boundary_related=True, cpml_axis="x"),
        FieldSpec("m_qz_z", description="CPML memory for dqz/dz.", internal=True, boundary_related=True, cpml_axis="z"),
    )

    default_pml_type = "cpmls"
    supports_batched_models = True

    def __init__(self, spatial_order=4, device="cpu", backend="torch"):
        """Build the Biot operator.

        Args:
            spatial_order: FD accuracy order of the staggered first derivative (even).
            device: device of the static derivative kernels.
            backend: ``'torch'`` (the only backend for this equation).
        """
        if backend != "torch":
            raise ValueError(f"Biot runs on the torch backend only, got backend={backend!r}")
        super().__init__(spatial_order, device, backend)

    @property
    def default_source_fields(self):
        return ["sxx", "szz"]

    @property
    def default_receiver_fields(self):
        return ["vx", "vz"]

    @staticmethod
    def _derived(models):
        """``models`` plus ``(lam_u, rho_f / rho, m - rho_f^2 / rho, gamma)``; idempotent."""
        if len(models) == 12:
            return tuple(models)
        if len(models) != 8:
            raise ValueError(f"Biot models must be the 8 coefficients [H, mu, C, M, rho, rho_f, m, b], got {len(models)}")
        h, mu, c, biot_m, rho, rho_f, m, b = models
        ratio = rho_f / rho
        m_t = m - rho_f * ratio
        return (h, mu, c, biot_m, rho, rho_f, m, b, h - 2.0 * mu, ratio, m_t, b / m_t)

    def prepare_models(self, models):
        return list(self._derived(models))

    def _check_boundaries(self):
        fs_faces = getattr(self, "fs_faces", None)
        if getattr(self, "free_surface", False) or (fs_faces is not None and any(fs_faces)):
            raise NotImplementedError("Biot: free surfaces are not implemented yet; run with free_surface=False")
        if getattr(self, "_apm_air_mask_runtime", None) is not None or getattr(self, "_topo_rows_runtime", None) is not None:
            raise NotImplementedError("Biot: topography is not implemented")

    def func(self, wavefields, models, dt, h, b, **kwargs):
        self._check_boundaries()
        (vx, vz, qx, qz, sxx, szz, sxz, p,
         m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z, m_p_x, m_p_z,
         m_vx_x, m_vz_z, m_vx_z, m_vz_x, m_qx_x, m_qz_z) = wavefields
        (hmod, mu, c, biot_m, rho, rho_f, m, bvis,
         lam_u, ratio, m_t, gamma) = self._derived(models)
        pd = self.pd
        az, bz, azh, bzh, ax, bx, axh, bxh = self.b
        hz, hx = pd._spacing["z"], pd._spacing["x"]

        # ---- velocities and fluxes (stresses and pressure frozen at t^n) ----------------
        sxx_x, m_sxx_x = cpml_update(pd.x_forward(sxx), m_sxx_x, axh, bxh, (sxx, pd.kxf, hx))
        sxz_z, m_sxz_z = cpml_update(pd.z_backward(sxz), m_sxz_z, az, bz, (sxz, pd.kzb, hz))
        sxz_x, m_sxz_x = cpml_update(pd.x_backward(sxz), m_sxz_x, ax, bx, (sxz, pd.kxb, hx))
        szz_z, m_szz_z = cpml_update(pd.z_forward(szz), m_szz_z, azh, bzh, (szz, pd.kzf, hz))
        p_x, m_p_x = cpml_update(pd.x_forward(p), m_p_x, axh, bxh, (p, pd.kxf, hx))
        p_z, m_p_z = cpml_update(pd.z_forward(p), m_p_z, azh, bzh, (p, pd.kzf, hz))

        decay = torch.exp(-gamma * dt)
        # (1 - exp(-gamma dt)) / gamma, -> dt as gamma -> 0
        safe = torch.where(gamma * dt > 1e-12, gamma, torch.ones_like(gamma))
        weight = torch.where(gamma * dt > 1e-12, -torch.expm1(-gamma * dt) / safe, dt * torch.ones_like(gamma))

        div_sx = sxx_x + sxz_z
        div_sz = sxz_x + szz_z
        qx_new = decay * qx - weight * (p_x + ratio * div_sx) / m_t
        qz_new = decay * qz - weight * (p_z + ratio * div_sz) / m_t
        vx = vx + (dt * div_sx - rho_f * (qx_new - qx)) / rho
        vz = vz + (dt * div_sz - rho_f * (qz_new - qz)) / rho
        qx, qz = qx_new, qz_new

        # ---- stresses and pressure -------------------------------------------------------
        vx_x, m_vx_x = cpml_update(pd.x_backward(vx), m_vx_x, ax, bx, (vx, pd.kxb, hx))
        vz_z, m_vz_z = cpml_update(pd.z_backward(vz), m_vz_z, az, bz, (vz, pd.kzb, hz))
        qx_x, m_qx_x = cpml_update(pd.x_backward(qx), m_qx_x, ax, bx, (qx, pd.kxb, hx))
        qz_z, m_qz_z = cpml_update(pd.z_backward(qz), m_qz_z, az, bz, (qz, pd.kzb, hz))
        div_v = vx_x + vz_z
        div_q = qx_x + qz_z
        sxx = sxx + dt * (hmod * vx_x + lam_u * vz_z + c * div_q)
        szz = szz + dt * (lam_u * vx_x + hmod * vz_z + c * div_q)
        p = p - dt * (c * div_v + biot_m * div_q)
        vx_z, m_vx_z = cpml_update(pd.z_forward(vx), m_vx_z, azh, bzh, (vx, pd.kzf, hz))
        vz_x, m_vz_x = cpml_update(pd.x_forward(vz), m_vz_x, axh, bxh, (vz, pd.kxf, hx))
        sxz = sxz + dt * mu * (vx_z + vz_x)

        return (vx, vz, qx, qz, sxx, szz, sxz, p,
                m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z, m_p_x, m_p_z,
                m_vx_x, m_vz_z, m_vx_z, m_vz_x, m_qx_x, m_qz_z)
