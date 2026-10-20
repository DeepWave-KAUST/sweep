"""Element-local SEM operators for 2-D constant-density acoustic.

Strong form    :  ∂²u/∂t²  =  c²(x) ∇²u  +  s(x, t)
Weak form      :  ∫ v ü dΩ  +  ∫ c² ∇v · ∇u dΩ  =  ∫ v s dΩ
Discretized    :  M ü  +  K u  =  b      (M diagonal after GLL collocation;
                                          K applied element-locally)

Conventions
-----------
- Local index ``α`` ∈ {0..N} is the η (z, downward) index;
  ``β`` ∈ {0..N} is the ξ (x) index.
- Reference Jacobian factors are baked in once: ``Jx = dz/dx``, ``Jz = dx/dz``.
- Velocity ``vp_e`` is sampled at GLL nodes: shape (E, N+1, N+1).

Boundaries
----------
- Free surface: ``dirichlet`` (pressure-release) node set, zeroed each step.
- Outgoing waves: **CPML** (``cpml`` set) — the double-convolution unsplit form
  (Komatitsch & Martin 2007), matching ``sweep.equations.acoustic.step_cpml``.
  With ``cpml=None`` the domain is a bare reflecting box (used by the operator
  unit tests). There is no paraxial / Stacey absorbing path.

Two stiffness forms (deliberate, not redundant)
-----------------------------------------------
- :meth:`stiffness_apply` bakes c² inside the element integral → the *symmetric*
  variational operator ``∇·(c²∇)`` (energy-conserving). Used by the bare path.
- The CPML path (:meth:`_cpml_grads`) uses c²-free directional operators so vp²
  can be factored out and applied per node alongside the per-node CPML
  coefficients (``c²∇²``, non-symmetric — acceptable in/near the absorbing layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from .._assembly import assemble
from ..cpml.acoustic import CPMLProfile
from ..mesh.boundaries import free_surface_node_idx
from ..fields import DerivedInitState, DirichletNodes, FieldSpec, check_element_arrays
from ..mesh.quad4 import Quad4Mesh, boundary_element_edges


@dataclass
class AcousticSEM2D(DerivedInitState, DirichletNodes):
    """Cached factors + ops for 2-D acoustic SEM on a :class:`Quad4Mesh`."""

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

    # FD-aligned receiver components: scalar wave equation: the field IS the pressure-like wavefield (FD's
    # Acoustic aliases its h1 to 'p' on the same equation), so 'p' is a name
    # alias, not a transform.
    derived_receiver_types: ClassVar[dict] = {"p": ("u", None)}

    mesh: Quad4Mesh
    vp_e: torch.Tensor            # (E, ngll, ngll) — c at GLL nodes
    vp_squared_node: torch.Tensor # (n_global,) — c² projected to global nodes
    mass: torch.Tensor            # (n_global,)  — Σ w_α w_β |J|
    _inv_M: torch.Tensor          # (n_global,)  — 1 / mass
    dirichlet: tuple[str, ...]
    dirichlet_idx: torch.Tensor
    cpml: Optional[CPMLProfile]

    _c2_jx: torch.Tensor          # (E, ngll, ngll) — c² · w_α w_β · (dz/dx)
    _c2_jz: torch.Tensor          # (E, ngll, ngll) — c² · w_α w_β · (dx/dz)
    _D: torch.Tensor              # (ngll, ngll) — Lagrange derivative matrix
    _w_jx_noc2: torch.Tensor      # CPML path: w_α w_β · (dz/dx)   (no c²)
    _w_jz_noc2: torch.Tensor      # CPML path: w_α w_β · (dx/dz)
    _w_jac: torch.Tensor          # CPML path: w_α w_β · |J|
    _inv_half_dx: float           # 2 / dx_e
    _inv_half_dz: float           # 2 / dz_e

    @classmethod
    def build(
        cls,
        mesh: Quad4Mesh,
        vp_e: torch.Tensor,
        *,
        dirichlet: tuple[str, ...] = ("top",),
        cpml: Optional[CPMLProfile] = None,
    ) -> "AcousticSEM2D":
        check_element_arrays(mesh, vp_e=vp_e)
        device, dtype = vp_e.device, vp_e.dtype
        w_outer = mesh.w_ref[:, None] * mesh.w_ref[None, :]      # (ngll, ngll)

        # Mass: M[g] = Σ_e w_α w_β |J|
        m_e = w_outer.expand(mesh.n_elements, -1, -1) * mesh.jacobian_det
        mass = torch.zeros(mesh.n_global, dtype=dtype, device=device)
        mass.index_add_(0, mesh.ibool.reshape(-1), m_e.reshape(-1))

        # Per-element c² and Jacobian weights for the two stiffness forms.
        c2 = vp_e * vp_e
        Jx = mesh.dz_e / mesh.dx_e
        Jz = mesh.dx_e / mesh.dz_e
        c2_jx = c2 * w_outer * Jx
        c2_jz = c2 * w_outer * Jz
        w_jx_noc2 = (w_outer * Jx).expand(mesh.n_elements, -1, -1).contiguous()
        w_jz_noc2 = (w_outer * Jz).expand(mesh.n_elements, -1, -1).contiguous()
        w_jac = (w_outer * mesh.jacobian_det).expand(
            mesh.n_elements, -1, -1).contiguous()

        # Per-global-node c² via first-touch scatter (last write wins; exact for
        # continuous c², one adjacent value at discontinuities).
        vp_squared_node = torch.zeros(mesh.n_global, dtype=dtype, device=device)
        vp_squared_node[mesh.ibool] = c2

        return cls(
            mesh=mesh,
            vp_e=vp_e,
            vp_squared_node=vp_squared_node,
            mass=mass,
            _inv_M=1.0 / mass,
            dirichlet=tuple(dirichlet),
            dirichlet_idx=free_surface_node_idx(mesh, dirichlet, device),
            cpml=cpml,
            _c2_jx=c2_jx,
            _c2_jz=c2_jz,
            _D=mesh.deriv,
            _w_jx_noc2=w_jx_noc2,
            _w_jz_noc2=w_jz_noc2,
            _w_jac=w_jac,
            _inv_half_dx=2.0 / mesh.dx_e,
            _inv_half_dz=2.0 / mesh.dz_e,
        )

    # ------------------------------------------------------------------ #
    # Symmetric stiffness K · u  (c² baked in) — bare path + unit tests.
    # ------------------------------------------------------------------ #
    def stiffness_apply(self, u_global: torch.Tensor) -> torch.Tensor:
        """``(K u)_global = Σ_e ∫ c² ∇v · ∇u dΩ`` assembled to global nodes."""
        u_e = u_global[self.mesh.ibool]
        du_dxi  = torch.einsum("bg,eag->eab", self._D, u_e)
        du_deta = torch.einsum("ag,egb->eab", self._D, u_e)
        r_e = (
            torch.einsum("db,ead->eab", self._D, self._c2_jx * du_dxi)
          + torch.einsum("ga,egb->eab", self._D, self._c2_jz * du_deta)
        )
        return assemble(r_e, self.mesh.ibool, u_global)

    # ------------------------------------------------------------------ #
    # CPML building blocks: directional Galerkin operators without c²
    # (so vp² can be factored out and applied per node).
    # ------------------------------------------------------------------ #
    def _cpml_grads(self, u_global, apsi_x, apsi_z):
        """All six CPML directional quantities — ``(G_x u, G_z u, K_x u, K_z u,
        ∂(a·ψ_x)/∂x, ∂(a·ψ_z)/∂z)`` — in one fused pass:

        * stack ``u`` and the two ``a·ψ`` memory fields and take their ξ/η
          reference derivatives together (1 gather + 2 batched first-derivative
          einsums, vs 3 gathers + 4 in the per-operator form), then
        * assemble all six element-local fields to global nodes with a **single**
          batched scatter instead of six ``index_add`` calls.

        Bit-identical to the per-operator ``_Gx/_Gz/_Kx/_Kz`` calls, and
        ``torch.vmap``-safe (the scatter target is derived from the batched
        ``u_global`` and the add is out-of-place)."""
        ibool = self.mesh.ibool
        F = torch.stack([u_global, apsi_x, apsi_z], dim=-1)        # (n_global, 3)
        F_e = F[ibool]                                            # (E,ngll,ngll,3)
        dF_dxi  = torch.einsum("bg,eagk->eabk", self._D, F_e)
        dF_deta = torch.einsum("ag,egbk->eabk", self._D, F_e)
        du_dxi, du_deta = dF_dxi[..., 0], dF_deta[..., 0]
        gx_e  = self._w_jac * (du_dxi * self._inv_half_dx)
        gz_e  = self._w_jac * (du_deta * self._inv_half_dz)
        kx_e  = torch.einsum("db,ead->eab", self._D, self._w_jx_noc2 * du_dxi)
        kz_e  = torch.einsum("ga,egb->eab", self._D, self._w_jz_noc2 * du_deta)
        gax_e = self._w_jac * (dF_dxi[..., 1] * self._inv_half_dx)
        gaz_e = self._w_jac * (dF_deta[..., 2] * self._inv_half_dz)
        stk = torch.stack([gx_e, gz_e, kx_e, kz_e, gax_e, gaz_e], dim=-1)
        return assemble(stk, ibool, u_global)                     # six (n_global,)

    # ------------------------------------------------------------------ #
    # Double-convolution CPML residual + ψ/ζ recursion (sweep-FD parity).
    # ------------------------------------------------------------------ #
    def step_cpml(self, u_now, psi_x, psi_z, zeta_x, zeta_z):
        """Returns ``(w_sum, psi_x', psi_z', zeta_x', zeta_z')`` where ``w_sum``
        is the node-wise quantity multiplied by ``dt²·vp²`` in the leap-frog:

            u_next = 2 u_now − u_pre + dt² · vp²(x) · w_sum + source/M

        Per direction (x; z analogous), Komatitsch & Martin (2007):

            dudx  ≈ ∂u/∂x ;  lap_x ≈ ∂²u/∂x²        (lumped Galerkin, no c²)
            tmp_x  = (1+b)·lap_x + db/dx·dudx + ∂(a·ψ)/∂x
            w_sum += (1+b)·tmp_x + a·ζ
            ψ_x'   = b·dudx + a·ψ_x ;  ζ_x' = b·tmp_x + a·ζ_x
        """
        cp = self.cpml
        if cp is None:
            raise RuntimeError("CPML not configured on this AcousticSEM2D")
        inv_M = self._inv_M                       # precomputed (was 1.0 / self.mass)

        # One fused pass over u and the two a·ψ memory fields (1 gather +
        # 2 batched first-derivative einsums + 1 batched scatter).
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
    # SEMPropagator interface: positional init_state + step (Newmark).
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
            Ku = self.stiffness_apply(u_now)
            u_next = 2.0 * u_now - u_prev + dt2 * inv_M * (-Ku + b)
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
