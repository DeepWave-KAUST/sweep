"""Variable-density acoustic SEM in SPECFEM's **potential** normalization (Quad9).

    (1/κ) ∂²χ/∂t² = ∇·((1/ρ) ∇χ) + s        κ = ρ c²

with displacement ``u = (1/ρ) ∇χ`` and pressure ``p = −∂²χ/∂t²``. Weak form::

    ∫ (1/κ) v χ̈ dΩ  +  ∫ (1/ρ) ∇v · ∇χ dΩ  =  ∫ v s dΩ   (+ surface terms)

so ``M = ∫ κ⁻¹ φφ`` (lumped diagonal) and ``K = ∫ ρ⁻¹ ∇φ·∇φ``.

Why a second acoustic class (vs :class:`AcousticSEM2DDeformed`)
---------------------------------------------------------------
1. **Variable density.** ``AcousticSEM2DDeformed`` carries only ``vp`` — it
   cannot express a density contrast at all, and a pure density jump (same c,
   different ρ) is invisible to it even though it reflects with
   ``R = (ρ₂c₂−ρ₁c₁)/(ρ₂c₂+ρ₁c₁)``.
2. **It is the normalization the fluid–solid coupling needs.** SPECFEM's
   coupling surface terms carry *no explicit ρ* **only** under these operators.
   Grafting them onto a constant-density normalization is wrong by a factor of
   ρ — and a homogeneous-water test cannot catch it, because a constant ρ
   quietly rescales everything.

For constant (κ, ρ) this reduces to ``χ̈ = c²∇²χ`` and reproduces
``AcousticSEM2DDeformed``'s homogeneous evolution to roundoff (gate 1a).

CPML (opt-in ``cpml=``): the Komatitsch–Martin double-convolution scheme of
:class:`AcousticSEM2DDeformed`, with the flux ``q = (1/ρ)∇χ`` riding inside the
directional operators. In the potential form the CPML operator and the bare
operator agree EXACTLY for variable ρ (both are ``∇·((1/ρ)∇χ)``) — the
velocity-form's known c²-in-flux vs c²∇²u discrepancy has no analogue here.
The node-level ``κ`` factor (``χ̈ = κ_node · w_sum``) equals ``M_κ⁻¹`` exactly
for element-wise-constant κ around a node; sources/coupling keep ``M_κ⁻¹``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from .._assembly import assemble, phys_grads
from ..cpml.acoustic import CPMLProfile
from ..mesh.boundaries import free_surface_node_idx
from ..fields import DerivedInitState, DirichletNodes, FieldSpec
from ..mesh.domains import DomainMask, masked_inv_mass
from ..mesh.nodes import node_field
from ..mesh.quad9 import Quad9Mesh


@dataclass
class AcousticPotentialSEM2D(DerivedInitState, DirichletNodes):
    """SPECFEM-normalized variable-density acoustic (potential χ) on Quad9."""

    field_specs: ClassVar[list] = [
        FieldSpec("chi", supports_source=True, supports_receiver=True,
                  seis_key="seis_chi", snap_key="snap_chi"),
        FieldSpec("chi_prev", internal=True),
        FieldSpec("psi_x", internal=True, boundary_related=True),
        FieldSpec("psi_z", internal=True, boundary_related=True),
        FieldSpec("zeta_x", internal=True, boundary_related=True),
        FieldSpec("zeta_z", internal=True, boundary_related=True),
    ]

    # FD-aligned receiver components: SPECFEM potential normalization: p = -chi_ddot (module docstring).
    derived_receiver_types: ClassVar[dict] = {"p": ("chi", "pressure")}

    mesh: Quad9Mesh
    vp_e: torch.Tensor            # (E, ngll, ngll)
    rho_e: torch.Tensor
    mass: torch.Tensor            # (n_global,) — Σ_e w_jac · κ⁻¹  (device/dtype ref)
    _inv_M: torch.Tensor
    dirichlet: tuple[str, ...]
    dirichlet_idx: torch.Tensor
    cpml: Optional[CPMLProfile]   # opt-in absorbing profile (None = bare)

    domain: Optional[DomainMask]  # None = the whole mesh is fluid
    _mask_e: torch.Tensor         # (E, 1, 1) 0/1 — acoustic elements
    _alive: torch.Tensor          # (n_global,) bool — nodes carrying a live χ

    kappa_node: torch.Tensor      # (n_global,) — κ scattered from FLUID elements
    mass_geo: torch.Tensor        # (n_global,) — geometric ∫φφ (CPML projections)
    _inv_M_geo: torch.Tensor

    _D: torch.Tensor
    _inv_rho: torch.Tensor        # (E, ngll, ngll) — ρ⁻¹ at GLL nodes
    _w_J: torch.Tensor
    _ixx: torch.Tensor
    _ixz: torch.Tensor
    _iex: torch.Tensor
    _iez: torch.Tensor

    @classmethod
    def build(cls, mesh: Quad9Mesh, vp_e: torch.Tensor, rho_e: torch.Tensor, *,
              dirichlet: tuple[str, ...] = ("top",),
              domain: Optional[DomainMask] = None,
              cpml: Optional[CPMLProfile] = None) -> "AcousticPotentialSEM2D":
        """``domain`` restricts the fluid to its acoustic elements: the operators
        are assembled over those only, and χ on nodes outside them is a dead DOF
        pinned to exactly 0 (``inv_M = 0``). ``None`` = the whole mesh is fluid.

        ``cpml`` enables the double-convolution absorbing bands (potential
        flavor — the flux ``(1/ρ)∇χ`` rides inside the directional operators).
        """
        shp = (mesh.n_elements, mesh.ngll, mesh.ngll)
        for name, t in (("vp_e", vp_e), ("rho_e", rho_e)):
            if tuple(t.shape) != shp:
                raise ValueError(f"{name} shape {tuple(t.shape)} != {shp}")
        device, dtype = vp_e.device, vp_e.dtype
        # domain + cpml compose: the directional operators are element-masked
        # (see _cpml_grads), kappa_node is fluid-scattered, and psi/zeta stay
        # exactly zero on solid-interior nodes (their flux input is masked out).
        # The corner-PML-on-interface rule is enforced by the COUPLED build.
        if domain is None:
            mask_e = torch.ones((mesh.n_elements, 1, 1), dtype=dtype, device=device)
            alive = torch.ones(mesh.n_global, dtype=torch.bool, device=device)
        else:
            mask_e = domain.elem_weight("acoustic", vp_e)
            alive = domain.node_in_acoustic
        inv_kappa = 1.0 / (rho_e * vp_e * vp_e)            # κ = ρ c²
        mass = torch.zeros(mesh.n_global, dtype=dtype, device=device)
        mass.index_add_(0, mesh.ibool.reshape(-1),
                        (mesh.w_jac * inv_kappa * mask_e).reshape(-1))
        mass_geo = torch.zeros(mesh.n_global, dtype=dtype, device=device)
        mass_geo.index_add_(0, mesh.ibool.reshape(-1),
                            (mesh.w_jac * mask_e).reshape(-1))
        # κ at nodes, read from one FLUID copy: at an interface node the
        # fluid side's κ must win (a copy in a solid element would pick an
        # arbitrary side), and one copy keeps the gradient from being
        # multiplied by the share count (see mesh/nodes.py).
        kappa_e = rho_e * vp_e * vp_e
        sel = (mask_e.expand(shp) > 0).reshape(-1)
        kappa_node = node_field(mesh.ibool, mesh.n_global, kappa_e, sel)
        return cls(
            mesh=mesh, vp_e=vp_e, rho_e=rho_e, mass=mass,
            _inv_M=masked_inv_mass(mass, alive),
            dirichlet=tuple(dirichlet),
            dirichlet_idx=free_surface_node_idx(mesh, dirichlet, device),
            cpml=cpml, domain=domain, _mask_e=mask_e, _alive=alive,
            kappa_node=kappa_node, mass_geo=mass_geo,
            _inv_M_geo=masked_inv_mass(mass_geo, alive),
            _D=mesh.deriv, _inv_rho=1.0 / rho_e, _w_J=mesh.w_jac,
            _ixx=mesh.inv_J[..., 0, 0], _ixz=mesh.inv_J[..., 0, 1],
            _iex=mesh.inv_J[..., 1, 0], _iez=mesh.inv_J[..., 1, 1],
        )

    # ------------------------------------------------------------------ #
    def _phys_grad(self, chi):
        """∂χ/∂x, ∂χ/∂z at every GLL node (inv_J chain rule)."""
        return phys_grads(self._D, self.mesh.ibool, self._ixx, self._ixz,
                          self._iex, self._iez, chi)

    def displacement(self, chi):
        """Fluid displacement ``u = (1/ρ) ∇χ``, element-local ``(E,ngll,ngll)``.

        This is the quantity the fluid–solid interface matches to ``u_s·n``.
        """
        dx, dz = self._phys_grad(chi)
        return self._inv_rho * dx, self._inv_rho * dz

    def stiffness_apply(self, chi: torch.Tensor) -> torch.Tensor:
        """``(K χ)_global = Σ_{e ∈ fluid} ∫ ρ⁻¹ ∇v · ∇χ dΩ`` assembled to global
        nodes. Domain masking is just the ``* _mask_e`` on the element-local
        integrand before the scatter — non-fluid elements contribute nothing."""
        dx, dz = self._phys_grad(chi)
        q_x = self._inv_rho * dx
        q_z = self._inv_rho * dz
        F_beta = self._w_J * (q_x * self._ixx + q_z * self._ixz)
        F_alpha = self._w_J * (q_x * self._iex + q_z * self._iez)
        r_e = (torch.einsum("db,ead->eab", self._D, F_beta)
               + torch.einsum("ga,egb->eab", self._D, F_alpha))
        return assemble(r_e * self._mask_e, self.mesh.ibool, chi)

    # ------------------------------------------------------------------ #
    # CPML: Komatitsch–Martin double convolution, potential flavor. Identical
    # tmp/ψ/ζ algebra to AcousticSEM2DDeformed.step_cpml; the ONLY change is
    # that the directional quantities carry the flux q = (1/ρ)∇χ (so ψ is a
    # flux memory), the projections use the GEOMETRIC mass, and the outer
    # factor is κ_node (χ̈ = κ·∇·q) instead of vp².
    # ------------------------------------------------------------------ #
    def _cpml_grads(self, chi_global, apsi_x, apsi_z):
        """``(G_x χ, G_z χ, K_x χ, K_z χ, ∂(a·ψ_x)/∂x, ∂(a·ψ_z)/∂z)`` with
        ``G_d χ = ∫ v q_d`` and ``K_d χ = ∫ ∂_d v q_d`` on the flux
        ``q = (1/ρ)∇χ``; one gather + batched einsums + one batched scatter."""
        ibool = self.mesh.ibool
        ixx, ixz, iex, iez, wJ = self._ixx, self._ixz, self._iex, self._iez, self._w_J
        F = torch.stack([chi_global, apsi_x, apsi_z], dim=-1)      # (n_global,3)
        F_e = F[ibool]                                             # (E,ngll,ngll,3)
        dF_dxi = torch.einsum("bg,eagk->eabk", self._D, F_e)
        dF_deta = torch.einsum("ag,egbk->eabk", self._D, F_e)
        q_x = self._inv_rho * (ixx * dF_dxi[..., 0] + iex * dF_deta[..., 0])
        q_z = self._inv_rho * (ixz * dF_dxi[..., 0] + iez * dF_deta[..., 0])
        gx_e = wJ * q_x
        gz_e = wJ * q_z
        kx_e = (torch.einsum("db,ead->eab", self._D, wJ * ixx * q_x)
                + torch.einsum("ga,egb->eab", self._D, wJ * iex * q_x))
        kz_e = (torch.einsum("db,ead->eab", self._D, wJ * ixz * q_z)
                + torch.einsum("ga,egb->eab", self._D, wJ * iez * q_z))
        gax_e = wJ * (ixx * dF_dxi[..., 1] + iex * dF_deta[..., 1])
        gaz_e = wJ * (ixz * dF_dxi[..., 2] + iez * dF_deta[..., 2])
        stk = torch.stack([gx_e, gz_e, kx_e, kz_e, gax_e, gaz_e], dim=-1)
        return assemble(stk * self._mask_e.unsqueeze(-1), ibool, chi_global)

    def step_cpml(self, chi_now, psi_x, psi_z, zeta_x, zeta_z):
        cp = self.cpml
        inv_Mg = self._inv_M_geo
        gx, gz, kx, kz, gax, gaz = self._cpml_grads(chi_now, cp.a_x * psi_x,
                                                    cp.a_z * psi_z)
        qx = gx * inv_Mg
        qz = gz * inv_Mg
        div_x = -kx * inv_Mg
        div_z = -kz * inv_Mg
        grad_apsi_x = gax * inv_Mg
        grad_apsi_z = gaz * inv_Mg

        one_pbx = 1.0 + cp.b_x
        one_pbz = 1.0 + cp.b_z
        tmp_x = one_pbx * div_x + cp.dbx_dx * qx + grad_apsi_x
        tmp_z = one_pbz * div_z + cp.dbz_dz * qz + grad_apsi_z
        w_sum = (one_pbx * tmp_x + cp.a_x * zeta_x
                 + one_pbz * tmp_z + cp.a_z * zeta_z)

        psi_x_new = cp.b_x * qx + cp.a_x * psi_x
        psi_z_new = cp.b_z * qz + cp.a_z * psi_z
        zeta_x_new = cp.b_x * tmp_x + cp.a_x * zeta_x
        zeta_z_new = cp.b_z * tmp_z + cp.a_z * zeta_z
        return w_sum, psi_x_new, psi_z_new, zeta_x_new, zeta_z_new

    # ------------------------------------------------------------------ #
    def step(self, fields: tuple, src, dt: float) -> tuple:
        chi_now, chi_prev, psi_x, psi_z, zeta_x, zeta_z = fields
        dt2 = dt * dt
        if self.cpml is not None:
            w_sum, psi_x, psi_z, zeta_x, zeta_z = self.step_cpml(
                chi_now, psi_x, psi_z, zeta_x, zeta_z)
            chi_next = (2.0 * chi_now - chi_prev
                        + dt2 * self.kappa_node * w_sum
                        + dt2 * self._inv_M * src[0])
            chi_next, psi_x, psi_z, zeta_x, zeta_z = self._apply_dirichlet(
                chi_next, psi_x, psi_z, zeta_x, zeta_z)
        else:
            K = self.stiffness_apply(chi_now)
            chi_next = (2.0 * chi_now - chi_prev
                        + dt2 * self._inv_M * (-K + src[0]))
            (chi_next,) = self._apply_dirichlet(chi_next)
        return (chi_next, chi_now, psi_x, psi_z, zeta_x, zeta_z)

    # ================================================================== #
    # Boundary-saving hooks (swap2nd reverse). The interior leapfrog is
    # exactly time-reversible (memories are zero where the CPML coefficients
    # vanish, and the Dirichlet nodes are zero at every level); the CPML band
    # is ring-saved / restored by the BS framework (cpml_node_masks).
    # ================================================================== #
    # Ring-save chi AND the four CPML memories on the full band, so the
    # reverse reconstruction restores the exact step input everywhere and the
    # BS gradient equals the full-tape gradient (the memories are exactly zero
    # outside the band).
    bs_ring_save: ClassVar[list] = [0, 2, 3, 4, 5]
    bs_ring_restore: ClassVar[list] = [(0, 0, 0), (1, 0, -1), (2, 1, -1),
                                       (3, 2, -1), (4, 3, -1), (5, 4, -1)]
    # Each memory field is exactly zero off ITS axis's support (psi_x' =
    # b_x * q + a_x * psi_x from zero), so its ring lives on band & (b_x != 0)
    # rather than on the whole band -- see boundary_saving.band_support_idx.
    bs_ring_support: ClassVar[dict] = {2: "b_x", 3: "b_z", 4: "b_x", 5: "b_z"}
    # (The C-PML memory adjoint is threaded by the framework default; it used to
    # be dropped, which cost cos 0.99972 against the full tape on Marmousi at
    # 6.3 s -- see boundary_saving.grad_idx.)

    def reverse_step(self, fields, src, dt):
        chi_ip1, chi_i = fields[0], fields[1]
        z = torch.zeros_like(chi_i)
        chi_im1 = self.step((chi_i, chi_ip1, z, z, z, z), src, dt)[0]
        return (chi_i, chi_im1, z, z, z, z)

    def bs_model_tensors(self) -> dict:
        """Every model-dependent tensor ``step`` reads: the kappa-mass inverse,
        the stiffness density and the node kappa (the geometric mass is pure
        mesh)."""
        return {"inv_M": self._inv_M, "inv_rho": self._inv_rho,
                "kappa": self.kappa_node}

    def bs_recompute(self, fields, src, dt, leaves):
        saved = (self._inv_M, self._inv_rho, self.kappa_node)
        self._inv_M, self._inv_rho = leaves["inv_M"], leaves["inv_rho"]
        self.kappa_node = leaves["kappa"]
        try:
            return self.step(fields, src, dt)
        finally:
            self._inv_M, self._inv_rho, self.kappa_node = saved

