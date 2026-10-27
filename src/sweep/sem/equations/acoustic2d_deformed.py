"""Constant-density acoustic SEM on a deformed Quad9 mesh.

Scalar analogue of :class:`ElasticSEM2DDeformed`: solves

    ∂²u/∂t² = c²(x) ∇²u + s(x, t)

as ``ü = c² M⁻¹ (b − K u)`` with the c²-free stiffness ``K = ∫ ∇v · ∇u dΩ``
and c² applied per global node (see :mod:`.acoustic2d` for why c² stays outside
the integral), on a Q2 (9-node) deformed mesh, with the per-GLL Jacobian chain
rule

    ∂/∂x = (∂ξ/∂x) ∂/∂ξ + (∂η/∂x) ∂/∂η
    ∂/∂z = (∂ξ/∂z) ∂/∂ξ + (∂η/∂z) ∂/∂η .

On a flat (axis-aligned) Quad9 mesh this reduces to
:class:`AcousticSEM2D` on the equivalent Quad4 mesh.

Boundaries: pressure-release Dirichlet free surface (``dirichlet=("top",)``,
acoustic ``u = 0``), applied on the conforming (curved) top edge. Outgoing
waves: **CPML** (``cpml`` set) — the same Komatitsch-Martin double-convolution
form as :class:`AcousticSEM2D`, but the directional operators use the per-GLL
``inv_J`` chain rule (they reduce to the axis-aligned operators in the
absorbing band, where the side/bottom elements are rectangular even under
topography). With ``cpml=None`` the domain is a bare reflecting box.

Index convention (matches :mod:`sweep.sem.basis.quad9`):
    inv_J[..., 0, 0] = ∂ξ/∂x       inv_J[..., 0, 1] = ∂ξ/∂z
    inv_J[..., 1, 0] = ∂η/∂x       inv_J[..., 1, 1] = ∂η/∂z
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from .._assembly import assemble, phys_grads
from ..cpml.acoustic import CPMLProfile
from ..mesh.boundaries import free_surface_node_idx
from ..mesh.nodes import node_field
from ..fields import DerivedInitState, DirichletNodes, FieldSpec, check_element_arrays
from ..mesh.quad9 import Quad9Mesh, boundary_element_edges


@dataclass
class AcousticSEM2DDeformed(DerivedInitState, DirichletNodes):
    """Cached factors + ops for 2-D acoustic SEM on a :class:`Quad9Mesh`."""

    # ---- Field specification: the positional state layout (see fields.py) ---- #
    field_specs: ClassVar[list] = [
        FieldSpec("u", supports_source=True, supports_receiver=True,
                  seis_key="seismograms", snap_key="snapshots"),
        FieldSpec("u_prev", internal=True),
        FieldSpec("psi_x", internal=True, boundary_related=True),
        FieldSpec("psi_z", internal=True, boundary_related=True),
        FieldSpec("zeta_x", internal=True, boundary_related=True),
        FieldSpec("zeta_z", internal=True, boundary_related=True),
    ]

    # FD-aligned receiver components: same scalar-wave convention as AcousticSEM2D.
    derived_receiver_types: ClassVar[dict] = {"p": ("u", None)}

    mesh: Quad9Mesh
    vp_e: torch.Tensor               # (E, ngll, ngll)
    vp_squared_node: torch.Tensor    # (n_global,) — c² scattered to globals
    mass: torch.Tensor               # (n_global,) — Σ_e w_α w_β |J|
    _inv_M: torch.Tensor             # (n_global,) — 1 / mass
    dirichlet: tuple[str, ...]
    dirichlet_idx: torch.Tensor
    cpml: Optional[CPMLProfile]

    _D: torch.Tensor                 # (ngll, ngll) — Lagrange derivative matrix
    _w_J: torch.Tensor               # (E, ngll, ngll) — w_α w_β |J|
    _ixx: torch.Tensor               # ∂ξ/∂x
    _ixz: torch.Tensor               # ∂ξ/∂z
    _iex: torch.Tensor               # ∂η/∂x
    _iez: torch.Tensor               # ∂η/∂z

    @classmethod
    def build(
        cls,
        mesh: Quad9Mesh,
        vp_e: torch.Tensor,
        *,
        dirichlet: tuple[str, ...] = ("top",),
        cpml: Optional[CPMLProfile] = None,
    ) -> "AcousticSEM2DDeformed":
        check_element_arrays(mesh, vp_e=vp_e)
        device, dtype = vp_e.device, vp_e.dtype

        mass = torch.zeros(mesh.n_global, dtype=dtype, device=device)
        mass.index_add_(0, mesh.ibool.reshape(-1), mesh.w_jac.reshape(-1))

        vp_squared_node = node_field(mesh.ibool, mesh.n_global, vp_e * vp_e)

        return cls(
            mesh=mesh, vp_e=vp_e, vp_squared_node=vp_squared_node, mass=mass,
            _inv_M=1.0 / mass,
            dirichlet=tuple(dirichlet),
            dirichlet_idx=free_surface_node_idx(mesh, dirichlet, device),
            cpml=cpml,
            _D=mesh.deriv, _w_J=mesh.w_jac,
            _ixx=mesh.inv_J[..., 0, 0], _ixz=mesh.inv_J[..., 0, 1],
            _iex=mesh.inv_J[..., 1, 0], _iez=mesh.inv_J[..., 1, 1],
        )

    # ------------------------------------------------------------------ #
    # Symmetric stiffness K · u = ∫ ∇v · ∇u dΩ  (c²-free) — bare path + unit tests
    # ------------------------------------------------------------------ #
    def _phys_grad(self, u_global):
        """∂u/∂x, ∂u/∂z at every GLL node (inv_J chain rule)."""
        du_dx, du_dz = phys_grads(self._D, self.mesh.ibool, self._ixx,
                                  self._ixz, self._iex, self._iez, u_global)
        return du_dx, du_dz

    def stiffness_apply(self, u_global: torch.Tensor) -> torch.Tensor:
        du_dx, du_dz = self._phys_grad(u_global)
        F_beta  = self._w_J * (du_dx * self._ixx + du_dz * self._ixz)
        F_alpha = self._w_J * (du_dx * self._iex + du_dz * self._iez)
        r_e = (torch.einsum("db,ead->eab", self._D, F_beta)
               + torch.einsum("ga,egb->eab", self._D, F_alpha))
        return assemble(r_e, self.mesh.ibool, u_global)

    # ------------------------------------------------------------------ #
    # CPML directional operators (no c²; deformed inv_J). In the rectangular
    # absorbing band the cross terms (ixz, iex) vanish and these match the
    # Quad4 operators.
    # ------------------------------------------------------------------ #
    def _cpml_grads(self, u_global, apsi_x, apsi_z):
        """All six CPML directional quantities — ``(G_x u, G_z u, K_x u, K_z u,
        ∂(a·ψ_x)/∂x, ∂(a·ψ_z)/∂z)`` — in one fused pass.  ``u`` and the two
        ``a·ψ`` fields are gathered together once,
        their ξ/η reference derivatives taken in two batched einsums, and the six
        element-local results assembled with a single batched scatter.
        Bit-identical; ``torch.vmap``-safe."""
        ibool = self.mesh.ibool
        ixx, ixz, iex, iez, wJ = self._ixx, self._ixz, self._iex, self._iez, self._w_J
        F = torch.stack([u_global, apsi_x, apsi_z], dim=-1)        # (n_global,3)
        F_e = F[ibool]                                            # (E,ngll,ngll,3)
        dF_dxi  = torch.einsum("bg,eagk->eabk", self._D, F_e)
        dF_deta = torch.einsum("ag,egbk->eabk", self._D, F_e)
        du_dx = ixx * dF_dxi[..., 0] + iex * dF_deta[..., 0]
        du_dz = ixz * dF_dxi[..., 0] + iez * dF_deta[..., 0]
        gx_e = wJ * du_dx
        gz_e = wJ * du_dz
        kx_e = (torch.einsum("db,ead->eab", self._D, wJ * ixx * du_dx)
                + torch.einsum("ga,egb->eab", self._D, wJ * iex * du_dx))
        kz_e = (torch.einsum("db,ead->eab", self._D, wJ * ixz * du_dz)
                + torch.einsum("ga,egb->eab", self._D, wJ * iez * du_dz))
        gax_e = wJ * (ixx * dF_dxi[..., 1] + iex * dF_deta[..., 1])
        gaz_e = wJ * (ixz * dF_dxi[..., 2] + iez * dF_deta[..., 2])
        stk = torch.stack([gx_e, gz_e, kx_e, kz_e, gax_e, gaz_e], dim=-1)
        return assemble(stk, ibool, u_global)

    def step_cpml(self, u_now, psi_x, psi_z, zeta_x, zeta_z):
        """Komatitsch-Martin double-convolution residual + ψ/ζ recursion
        (identical to :meth:`AcousticSEM2D.step_cpml`; deformed operators)."""
        cp = self.cpml
        if cp is None:
            raise RuntimeError("CPML not configured on this AcousticSEM2DDeformed")
        inv_M = self._inv_M

        gx, gz, kx, kz, gax, gaz = self._cpml_grads(u_now, cp.a_x * psi_x,
                                                    cp.a_z * psi_z)
        dudx  =  gx * inv_M
        dudz  =  gz * inv_M
        lap_x = -kx * inv_M
        lap_z = -kz * inv_M
        grad_apsi_x = gax * inv_M
        grad_apsi_z = gaz * inv_M

        one_pbx = 1.0 + cp.b_x
        one_pbz = 1.0 + cp.b_z
        tmp_x = one_pbx * lap_x + cp.dbx_dx * dudx + grad_apsi_x
        tmp_z = one_pbz * lap_z + cp.dbz_dz * dudz + grad_apsi_z
        w_sum = (one_pbx * tmp_x + cp.a_x * zeta_x
                 + one_pbz * tmp_z + cp.a_z * zeta_z)

        psi_x_new = cp.b_x * dudx + cp.a_x * psi_x
        psi_z_new = cp.b_z * dudz + cp.a_z * psi_z
        zeta_x_new = cp.b_x * tmp_x + cp.a_x * zeta_x
        zeta_z_new = cp.b_z * tmp_z + cp.a_z * zeta_z
        return w_sum, psi_x_new, psi_z_new, zeta_x_new, zeta_z_new

    # ================================================================== #
    # SEMPropagator interface: positional init_state + step (scalar Newmark).
    # State layout = (u, u_prev, psi_x, psi_z, zeta_x, zeta_z) per field_specs.
    # ================================================================== #
    def step(self, fields: tuple, src, dt: float) -> tuple:
        u_now, u_prev, psi_x, psi_z, zeta_x, zeta_z = fields
        dt2, inv_M = dt * dt, self._inv_M
        b = src[0]                                  # source -> field 0 (u)
        if self.cpml is not None:
            w_sum, psi_x, psi_z, zeta_x, zeta_z = self.step_cpml(
                u_now, psi_x, psi_z, zeta_x, zeta_z)
            u_next = (2.0 * u_now - u_prev
                      + dt2 * self.vp_squared_node * w_sum + dt2 * inv_M * b)
            u_next, psi_x, psi_z, zeta_x, zeta_z = self._apply_dirichlet(
                u_next, psi_x, psi_z, zeta_x, zeta_z)
        else:
            # c² per node, as on the CPML path (where w_sum = -K u / M)
            Ku = self.stiffness_apply(u_now)
            u_next = (2.0 * u_now - u_prev
                      + dt2 * self.vp_squared_node * (-Ku * inv_M) + dt2 * inv_M * b)
            (u_next,) = self._apply_dirichlet(u_next)
        return (u_next, u_now, psi_x, psi_z, zeta_x, zeta_z)

    # ================================================================== #
    # Boundary-saving hooks (positional; swap2nd reverse). Indices into the
    # field_specs layout: ring-save field 0 (u); reverse re-runs the step with
    # the two time levels swapped + CPML memory zeroed.
    # ================================================================== #
    bs_ring_save: ClassVar[list] = [0]                       # save u's ring
    bs_ring_restore: ClassVar[list] = [(0, 0, 0), (1, 0, -1)]  # (field, slot, offset)

    def reverse_step(self, fields, src, dt):
        u_ip1, u_i = fields[0], fields[1]
        z = torch.zeros_like(u_i)
        u_im1 = self.step((u_i, u_ip1, z, z, z, z), src, dt)[0]
        return (u_i, u_im1, z, z, z, z)

    def bs_model_tensors(self) -> dict:
        return {"vp2": self.vp_squared_node}

    def bs_recompute(self, fields, src, dt, leaves):
        saved = self.vp_squared_node
        self.vp_squared_node = leaves["vp2"]
        try:
            return self.step(fields, src, dt)
        finally:
            self.vp_squared_node = saved
