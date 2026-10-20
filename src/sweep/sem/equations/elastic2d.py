"""Element-local SEM operators for 2-D isotropic elastic (P-SV).

Strong form
-----------
::

    ρ ∂²u/∂t² = ∇ · σ + f          u = (u_x, u_z)
    σ_xx = (λ + 2μ) u_x,x + λ u_z,z
    σ_zz = λ u_x,x + (λ + 2μ) u_z,z
    σ_xz = μ (u_x,z + u_z,x)

with the isotropic constitutive ``λ = ρ(v_p² − 2 v_s²)``,
``μ = ρ v_s²``.

Weak form
---------
::

    ∫ ρ v · ü dΩ  +  ∫ σ(u) : ε(v) dΩ  =  ∫ v · f dΩ  +  ∮ v · (σ·n) dΓ

Traction-free top surface (``σ · n = 0``) is a *natural* BC — the
boundary integral drops automatically. No projection / image method.

Conventions
-----------
- ``α`` ∈ {0..N}: η (z-down) index.
- ``β`` ∈ {0..N}: ξ (x) index.
- All velocity / density / displacement fields shape ``(E, ngll, ngll)``.

Stiffness derivation summary
----------------------------
After expanding ``σ(u) : ε(v)`` and collapsing the element Jacobian
``|J| = dx·dz/4`` with the inverse-Jacobian factors ``(2/dx)`` and
``(2/dz)`` from the chain rule, the per-element fluxes simplify to:

::

    F_xx[α,β]      = w · [(λ+2μ) · J_x · du_x/dξ  +  λ          · du_z/dη]
    F_zz[α,β]      = w · [λ          · du_x/dξ  +  (λ+2μ) · J_z · du_z/dη]
    F_xz_to_x[α,β] = w · μ · [J_z · du_x/dη  +  du_z/dξ]
    F_xz_to_z[α,β] = w · μ · [du_x/dη       +  J_x · du_z/dξ]

with ``J_x = dz/dx``, ``J_z = dx/dz``. The element residuals then follow
the exact same two einsums used by the acoustic stiffness:

::

    R_e^x[α,β] = Σ_δ D[δ,β] · F_xx[α,δ]    + Σ_γ D[γ,α] · F_xz_to_x[γ,β]
    R_e^z[α,β] = Σ_δ D[δ,β] · F_xz_to_z[α,δ] + Σ_γ D[γ,α] · F_zz[γ,β]

Why there is no C-PML here
--------------------------
This flat-Quad4 equation models no absorbing boundary and ``abcn_elements`` is
refused at construction. Implementing one would duplicate physics that already
exists: :class:`ElasticSEM2DDeformed` on an axis-aligned Quad9 mesh solves the
identical equations (agreement to float64 round-off, measured ~1e-15) AND
carries the SPECFEM2D-ported displacement C-PML. Use that; this class remains
for the cheap flat-mesh fast path where a reflecting box is acceptable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from ..fields import DerivedInitState, FieldSpec, check_element_arrays
from ..mesh.quad4 import Quad4Mesh


@dataclass
class ElasticSEM2D(DerivedInitState):
    """Cached SEM operators for 2-D isotropic elastic on a :class:`Quad4Mesh`."""

    # ---- Field specification: the positional state layout (see fields.py) ---- #
    field_specs: ClassVar[list] = [
        FieldSpec("u_x", supports_source=True, supports_receiver=True,
                  seis_key="seis_x", snap_key="snapshots_x"),
        FieldSpec("u_z", supports_source=True, supports_receiver=True,
                  seis_key="seis_z", snap_key="snapshots_z"),
        FieldSpec("u_x_prev", internal=True),
        FieldSpec("u_z_prev", internal=True),
    ]

    # FD-aligned receiver components: particle velocity from the displacement trace: v = u_dot.
    derived_receiver_types: ClassVar[dict] = {"vx": ("u_x", "velocity"), "vz": ("u_z", "velocity")}

    mesh: Quad4Mesh
    vp_e: torch.Tensor                # (E, ngll, ngll)
    vs_e: torch.Tensor
    rho_e: torch.Tensor

    mass: torch.Tensor                # (n_global,) — Σ_e w_α w_β |J| ρ
    _inv_M: torch.Tensor              # (n_global,) — 1 / mass

    # Six pre-baked per-element stiffness coefficients
    _D: torch.Tensor                  # (ngll, ngll) Lagrange derivative matrix
    _w_lam2mu_Jx: torch.Tensor        # F_xx ← du_x/dξ
    _w_lam:       torch.Tensor        # F_xx ← du_z/dη   AND   F_zz ← du_x/dξ
    _w_lam2mu_Jz: torch.Tensor        # F_zz ← du_z/dη
    _w_mu_Jz:     torch.Tensor        # F_xz_to_x ← du_x/dη
    _w_mu:        torch.Tensor        # F_xz_to_x ← du_z/dξ AND F_xz_to_z ← du_x/dη
    _w_mu_Jx:     torch.Tensor        # F_xz_to_z ← du_z/dξ

    @classmethod
    def build(
        cls,
        mesh: Quad4Mesh,
        vp_e: torch.Tensor,
        vs_e: torch.Tensor,
        rho_e: torch.Tensor,
    ) -> "ElasticSEM2D":
        check_element_arrays(mesh, vp_e=vp_e, vs_e=vs_e, rho_e=rho_e)
        device, dtype = vp_e.device, vp_e.dtype
        w_outer = mesh.w_ref[:, None] * mesh.w_ref[None, :]      # (ngll, ngll)

        mu  = rho_e * vs_e * vs_e
        lam = rho_e * (vp_e * vp_e - 2.0 * vs_e * vs_e)
        lam2mu = lam + 2.0 * mu

        # Mass (lumped): M[g] = Σ_e ρ · w_α w_β · |J|
        m_e = w_outer * rho_e * mesh.jacobian_det
        mass = torch.zeros(mesh.n_global, dtype=dtype, device=device)
        mass.index_add_(0, mesh.ibool.reshape(-1), m_e.reshape(-1))

        # Stiffness flux prefactors (see module docstring).
        Jx = mesh.dz_e / mesh.dx_e
        Jz = mesh.dx_e / mesh.dz_e
        w_lam2mu_Jx = w_outer * lam2mu * Jx
        w_lam       = w_outer * lam
        w_lam2mu_Jz = w_outer * lam2mu * Jz
        w_mu_Jz     = w_outer * mu * Jz
        w_mu        = w_outer * mu
        w_mu_Jx     = w_outer * mu * Jx

        return cls(
            mesh=mesh,
            vp_e=vp_e, vs_e=vs_e, rho_e=rho_e,
            mass=mass,
            _inv_M=1.0 / mass,
            _D=mesh.deriv,
            _w_lam2mu_Jx=w_lam2mu_Jx,
            _w_lam=w_lam,
            _w_lam2mu_Jz=w_lam2mu_Jz,
            _w_mu_Jz=w_mu_Jz,
            _w_mu=w_mu,
            _w_mu_Jx=w_mu_Jx,
        )

    # ------------------------------------------------------------------ #
    # Stiffness K · (u_x, u_z) → (K_x u, K_z u)
    # ------------------------------------------------------------------ #
    def stiffness_apply(
        self,
        u_x_global: torch.Tensor,
        u_z_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_x_e = u_x_global[self.mesh.ibool]
        u_z_e = u_z_global[self.mesh.ibool]

        d_ux_dxi  = torch.einsum("bg,eag->eab", self._D, u_x_e)
        d_ux_deta = torch.einsum("ag,egb->eab", self._D, u_x_e)
        d_uz_dxi  = torch.einsum("bg,eag->eab", self._D, u_z_e)
        d_uz_deta = torch.einsum("ag,egb->eab", self._D, u_z_e)

        # See module docstring for the prefactor table.
        F_xx      = self._w_lam2mu_Jx * d_ux_dxi  + self._w_lam       * d_uz_deta
        F_zz      = self._w_lam       * d_ux_dxi  + self._w_lam2mu_Jz * d_uz_deta
        F_xz_to_x = self._w_mu_Jz     * d_ux_deta + self._w_mu        * d_uz_dxi
        F_xz_to_z = self._w_mu        * d_ux_deta + self._w_mu_Jx     * d_uz_dxi

        R_x_e = (torch.einsum("db,ead->eab", self._D, F_xx)
                 + torch.einsum("ga,egb->eab", self._D, F_xz_to_x))
        R_z_e = (torch.einsum("db,ead->eab", self._D, F_xz_to_z)
                 + torch.einsum("ga,egb->eab", self._D, F_zz))

        out_x = torch.zeros_like(u_x_global)
        out_z = torch.zeros_like(u_z_global)
        out_x.index_add_(0, self.mesh.ibool.reshape(-1), R_x_e.reshape(-1))
        out_z.index_add_(0, self.mesh.ibool.reshape(-1), R_z_e.reshape(-1))
        return out_x, out_z

    # ================================================================== #
    # SEMPropagator interface: positional init_state + step (Newmark β=0, γ=1/2).
    # State layout = (u_x, u_z, u_x_prev, u_z_prev) per field_specs.
    # ================================================================== #
    def step(self, fields: tuple, src, dt: float) -> tuple:
        u_x_now, u_z_now, u_x_prev, u_z_prev = fields
        dt2, inv_M = dt * dt, self._inv_M
        b_x, b_z = src[0], src[1]                   # source -> fields 0, 1
        Kux, Kuz = self.stiffness_apply(u_x_now, u_z_now)
        u_x_next = 2.0 * u_x_now - u_x_prev + dt2 * inv_M * (-Kux + b_x)
        u_z_next = 2.0 * u_z_now - u_z_prev + dt2 * inv_M * (-Kuz + b_z)
        return (u_x_next, u_z_next, u_x_now, u_z_now)
