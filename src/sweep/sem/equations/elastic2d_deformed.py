"""Element-local SEM operators for 2-D elastic on a Quad9 deformed mesh.

Mathematically identical to ``ElasticSEM2D`` on a :class:`Quad4Mesh`:

    ρ ∂²u/∂t² = ∇ · σ + f
    σ        = C : ε(u)

The only structural difference is the mesh geometry: Quad9 elements have
per-GLL Jacobian ``J(ξ, η)`` and its inverse, so every spatial-derivative
chain rule gains cross terms

    ∂/∂x = (∂ξ/∂x) ∂/∂ξ + (∂η/∂x) ∂/∂η
    ∂/∂z = (∂ξ/∂z) ∂/∂ξ + (∂η/∂z) ∂/∂η .

Boundaries
----------
- Traction-free top surface is a *natural* BC (the boundary integral drops).
- Outgoing waves: **SPECFEM2D-style C-PML** (``cpml`` set) — the full 2nd-order
  displacement form with (a) asymmetric strain stretching, (b) a non-symmetric
  stress divergence, and (c) the acceleration/mass damping contribution.  This
  last LHS term is what makes the 2nd-order displacement CPML stable; see
  :mod:`sweep.sem.cpml.elastic`.  With ``cpml=None`` the domain is a bare
  reflecting box.

Index convention (matches :mod:`sweep.sem.basis.quad9`):
    inv_J[..., 0, 0] = ∂ξ/∂x       inv_J[..., 0, 1] = ∂ξ/∂z
    inv_J[..., 1, 0] = ∂η/∂x       inv_J[..., 1, 1] = ∂η/∂z
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from .._assembly import assemble, contract, phys_grads
from ..cpml.elastic import ElasticCPMLProfile, build_elastic_cpml
from ..fields import DerivedInitState, FieldSpec, check_element_arrays
from ..mesh.domains import DomainMask, masked_inv_mass
from ..mesh.quad9 import Quad9Mesh


@dataclass
class ElasticSEM2DDeformed(DerivedInitState):
    """Elastic SEM on a deformed Quad9 mesh."""

    # ---- Field specification: the positional state layout (see fields.py) ---- #
    # Memory layout: 4 displacement levels + 4 strain-memory + 2 displ-memory.
    # The 6 memory fields each carry the two SPECFEM convolution branches in a
    # trailing length-2 axis; they stay zero when ``cpml is None``.
    field_specs: ClassVar[list] = [
        FieldSpec("u_x", supports_source=True, supports_receiver=True,
                  seis_key="seis_x", snap_key="snapshots_x"),
        FieldSpec("u_z", supports_source=True, supports_receiver=True,
                  seis_key="seis_z", snap_key="snapshots_z"),
        FieldSpec("u_x_prev", internal=True),
        FieldSpec("u_z_prev", internal=True),
        FieldSpec("r_dux_dx", internal=True, boundary_related=True, layout="element2"),
        FieldSpec("r_dux_dz", internal=True, boundary_related=True, layout="element2"),
        FieldSpec("r_duz_dx", internal=True, boundary_related=True, layout="element2"),
        FieldSpec("r_duz_dz", internal=True, boundary_related=True, layout="element2"),
        FieldSpec("r_displ_x", internal=True, boundary_related=True, layout="element2"),
        FieldSpec("r_displ_z", internal=True, boundary_related=True, layout="element2"),
    ]

    # FD-aligned receiver components: particle velocity from the displacement trace: v = u_dot.
    derived_receiver_types: ClassVar[dict] = {"vx": ("u_x", "velocity"), "vz": ("u_z", "velocity")}

    mesh: Quad9Mesh
    vp_e: torch.Tensor                # (E, ngll, ngll)
    vs_e: torch.Tensor
    rho_e: torch.Tensor

    mass: torch.Tensor                # (n_global,) — Σ_e ρ · w_α w_β · |J(ξ, η)|
    _inv_M: torch.Tensor              # (n_global,) — 1 / mass
    cpml: Optional[ElasticCPMLProfile]  # SPECFEM-style elastic CPML, opt-in

    # Element-local Lamé and Jacobian-weight bundles precomputed once.
    _D: torch.Tensor                  # (ngll, ngll)
    _lam: torch.Tensor                # (E, ngll, ngll)
    _mu: torch.Tensor
    _lam2mu: torch.Tensor
    _w_J: torch.Tensor                # w_α w_β · |J|
    _w_rho_J: torch.Tensor            # w_α w_β · |J| · ρ  (element mass, CPML accel)
    _ixx: torch.Tensor                # inv_J[..., 0, 0]  ≡  ∂ξ/∂x
    _ixz: torch.Tensor                # ∂ξ/∂z
    _iex: torch.Tensor                # ∂η/∂x
    _iez: torch.Tensor                # ∂η/∂z

    # Domain restriction (fluid–solid coupling): assemble over the solid elements
    # only; **u** on nodes outside them is a dead DOF pinned to exactly 0.
    # ``None`` = the whole mesh is solid (the standalone behaviour).
    domain: Optional[DomainMask] = None
    _mask_e: Optional[torch.Tensor] = None    # (E, 1, 1) 0/1
    _alive: Optional[torch.Tensor] = None     # (n_global,) bool

    @classmethod
    def build(
        cls,
        mesh: Quad9Mesh,
        vp_e: torch.Tensor,
        vs_e: torch.Tensor,
        rho_e: torch.Tensor,
        *,
        cpml: Optional[ElasticCPMLProfile] = None,
        domain: Optional[DomainMask] = None,
    ) -> "ElasticSEM2DDeformed":
        """``domain`` restricts the solid to its elastic elements (fluid–solid
        coupling): the operators are assembled over those only and **u** outside
        them is a dead DOF pinned to exactly 0. ``None`` = the whole mesh."""
        check_element_arrays(mesh, vp_e=vp_e, vs_e=vs_e, rho_e=rho_e)
        device, dtype = vp_e.device, vp_e.dtype

        mu  = rho_e * vs_e * vs_e
        lam = rho_e * (vp_e * vp_e - 2.0 * vs_e * vs_e)
        lam2mu = lam + 2.0 * mu

        if domain is None:
            mask_e = torch.ones((mesh.n_elements, 1, 1), dtype=dtype, device=device)
            alive = torch.ones(mesh.n_global, dtype=torch.bool, device=device)
        else:
            if cpml is not None:
                # freeze every CPML memory recursion on the fluid elements —
                # see cpml.elastic.mask_elastic_profile. (The corner-PML-on-
                # interface restriction is enforced by the COUPLED equation's
                # build, which knows the interface.)
                from ..cpml.elastic import mask_elastic_profile
                cpml = mask_elastic_profile(cpml, domain.is_elastic)
            mask_e = domain.elem_weight("elastic", vp_e)
            alive = domain.node_in_elastic

        # Mass: M[g] = Σ_{e ∈ solid} ρ · w_α w_β · |J(ξ, η)| — per-GLL Jacobian.
        m_e = mesh.w_jac * rho_e * mask_e
        mass = torch.zeros(mesh.n_global, dtype=dtype, device=device)
        mass.index_add_(0, mesh.ibool.reshape(-1), m_e.reshape(-1))

        return cls(
            mesh=mesh, vp_e=vp_e, vs_e=vs_e, rho_e=rho_e,
            mass=mass, _inv_M=masked_inv_mass(mass, alive), cpml=cpml,
            domain=domain, _mask_e=mask_e, _alive=alive,
            _D=mesh.deriv,
            _lam=lam, _mu=mu, _lam2mu=lam2mu,
            _w_J=mesh.w_jac, _w_rho_J=m_e,
            _ixx=mesh.inv_J[..., 0, 0], _ixz=mesh.inv_J[..., 0, 1],
            _iex=mesh.inv_J[..., 1, 0], _iez=mesh.inv_J[..., 1, 1],
        )

    # ------------------------------------------------------------------ #
    # Bare (no-CPML) stiffness K · (u_x, u_z) → (R_x, R_z)
    # ------------------------------------------------------------------ #
    def stiffness_apply(
        self,
        u_x_global: torch.Tensor,
        u_z_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_ux_dx, d_ux_dz, d_uz_dx, d_uz_dz = self._element_derivatives(
            u_x_global, u_z_global)

        sigma_xx = self._lam2mu * d_ux_dx + self._lam   * d_uz_dz
        sigma_zz = self._lam   * d_ux_dx + self._lam2mu * d_uz_dz
        sigma_xz = self._mu * (d_ux_dz + d_uz_dx)
        # symmetric: σ_zx == σ_xz
        return self._assemble_stress(sigma_xx, sigma_zz, sigma_xz, sigma_xz,
                                     u_x_global)

    def _element_derivatives(self, u_x_global, u_z_global):
        """Element-local physical derivatives ∂u/∂x, ∂u/∂z at every GLL."""
        return tuple(phys_grads(self._D, self.mesh.ibool, self._ixx, self._ixz,
                                self._iex, self._iez, u_x_global, u_z_global))

    def _assemble_stress(self, sigma_xx, sigma_zz, sigma_xz, sigma_zx, like_x):
        """Weak-form flux ∫ ∇v : σ + assembly (possibly non-symmetric stress).

        ``accel_x`` is the divergence of the row ``[σ_xx (x-flux), σ_zx (z-flux)]``
        and ``accel_z`` of ``[σ_xz (x-flux), σ_zz (z-flux)]`` — matching the
        SPECFEM2D PML convention; for symmetric σ pass ``σ_zx = σ_xz``.
        """
        w_J = self._w_J
        F_x_beta  = w_J * (sigma_xx * self._ixx + sigma_zx * self._ixz)
        F_x_alpha = w_J * (sigma_xx * self._iex + sigma_zx * self._iez)
        F_z_beta  = w_J * (sigma_xz * self._ixx + sigma_zz * self._ixz)
        F_z_alpha = w_J * (sigma_xz * self._iex + sigma_zz * self._iez)
        # batched D^T contraction (2 einsums vs 4) + single scatter for R_x,R_z.
        F_beta  = torch.stack([F_x_beta, F_z_beta], dim=-1)        # (E,ngll,ngll,2)
        F_alpha = torch.stack([F_x_alpha, F_z_alpha], dim=-1)
        R_e = (contract("db,eadk->eabk", self._D, F_beta)
               + contract("ga,egbk->eabk", self._D, F_alpha))  # (E,ngll,ngll,2)
        if self._mask_e is not None:            # domain restriction (see build)
            R_e = R_e * self._mask_e.unsqueeze(-1)
        return assemble(R_e, self.mesh.ibool, like_x)

    # ================================================================== #
    # SPECFEM2D-style 2nd-order elastic CPML step.
    # ================================================================== #
    @staticmethod
    def _conv_update(r, c1, c2, d_now, d_old):
        """Two-branch recursive-convolution memory update.

        ``r`` is ``(E, ngll, ngll, 2)``; ``c1, c2`` are ``(coef0, coef1, coef2)``
        for the two branches.  Returns the updated ``(E, ngll, ngll, 2)`` stack.
        """
        z0, z1, z2 = c1
        y0, y1, y2 = c2
        r1 = z0 * r[..., 0] + z1 * d_now + z2 * d_old
        r2 = y0 * r[..., 1] + y1 * d_now + y2 * d_old
        return torch.stack([r1, r2], dim=-1)

    def _step_cpml(self, u_x_now, u_z_now, u_x_prev, u_z_prev,
                   r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz,
                   r_displ_x, r_displ_z, b_x, b_z, dt):
        cp = self.cpml
        dt2, inv_M, ib = dt * dt, self._inv_M, self.mesh.ibool

        # 1) strains now & at the previous step (element-local, physical).
        #    Gather all four displacement fields once and take their ξ/η
        #    reference derivatives in two batched einsums (vs 4 gathers + 8
        #    einsums); the raw gathered values are reused for the acceleration
        #    term in step 6 (no second gather).
        ixx, ixz, iex, iez = self._ixx, self._ixz, self._iex, self._iez
        U_e = torch.stack([u_x_now, u_z_now, u_x_prev, u_z_prev], dim=-1)[ib]
        dU_dxi  = contract("bg,eagk->eabk", self._D, U_e)   # (E,ngll,ngll,4)
        dU_deta = contract("ag,egbk->eabk", self._D, U_e)
        dU_dx = ixx.unsqueeze(-1) * dU_dxi + iex.unsqueeze(-1) * dU_deta
        dU_dz = ixz.unsqueeze(-1) * dU_dxi + iez.unsqueeze(-1) * dU_deta
        dux_dx,   duz_dx   = dU_dx[..., 0], dU_dx[..., 1]
        dux_dz,   duz_dz   = dU_dz[..., 0], dU_dz[..., 1]
        dux_dx_o, duz_dx_o = dU_dx[..., 2], dU_dx[..., 3]
        dux_dz_o, duz_dz_o = dU_dz[..., 2], dU_dz[..., 3]
        ux_e, uz_e, ux_e_o, uz_e_o = (U_e[..., 0], U_e[..., 1],
                                      U_e[..., 2], U_e[..., 3])

        # 2) strain-memory recursion. x-derivatives use the zx coefficient set,
        #    z-derivatives use the xz set (SPECFEM pml_compute_memory_variables).
        r_dux_dx = self._conv_update(r_dux_dx, cp.zx1, cp.zx2, dux_dx, dux_dx_o)
        r_duz_dx = self._conv_update(r_duz_dx, cp.zx1, cp.zx2, duz_dx, duz_dx_o)
        r_dux_dz = self._conv_update(r_dux_dz, cp.xz1, cp.xz2, dux_dz, dux_dz_o)
        r_duz_dz = self._conv_update(r_duz_dz, cp.xz1, cp.xz2, duz_dz, duz_dz_o)

        # 3) asymmetrically stretched derivatives  d̃ = S0·d + S1·r1 + S2·r2.
        dux_dx_s = cp.Sx0 * dux_dx + cp.Sx1 * r_dux_dx[..., 0] + cp.Sx2 * r_dux_dx[..., 1]
        duz_dx_s = cp.Sx0 * duz_dx + cp.Sx1 * r_duz_dx[..., 0] + cp.Sx2 * r_duz_dx[..., 1]
        dux_dz_s = cp.Sz0 * dux_dz + cp.Sz1 * r_dux_dz[..., 0] + cp.Sz2 * r_dux_dz[..., 1]
        duz_dz_s = cp.Sz0 * duz_dz + cp.Sz1 * r_duz_dz[..., 0] + cp.Sz2 * r_duz_dz[..., 1]

        # 4) non-symmetric PML stress (raw = unstretched gradient).
        sig_xx = self._lam2mu * dux_dx_s + self._lam * duz_dz
        sig_zz = self._lam2mu * duz_dz_s + self._lam * dux_dx
        sig_zx = self._mu * (duz_dx   + dux_dz_s)     # -> accel_x
        sig_xz = self._mu * (dux_dz   + duz_dx_s)     # -> accel_z

        # 5) non-symmetric weak-form divergence.
        R_x, R_z = self._assemble_stress(sig_xx, sig_zz, sig_xz, sig_zx, u_x_now)

        # 6) acceleration / mass contribution (element-local, then assembled).
        #    ux_e, uz_e, ux_e_o, uz_e_o were gathered once in step 1.
        r_displ_x = self._conv_update(r_displ_x, cp.l1, cp.l2, ux_e, ux_e_o)
        r_displ_z = self._conv_update(r_displ_z, cp.l1, cp.l2, uz_e, uz_e_o)
        v_x_e = (ux_e - ux_e_o) / dt
        v_z_e = (uz_e - uz_e_o) / dt
        wrhoJ = self._w_rho_J
        aPML_x_e = wrhoJ * (cp.La1 * v_x_e + cp.La2 * ux_e
                            + cp.La3 * r_displ_x[..., 0] + cp.La4 * r_displ_x[..., 1])
        aPML_z_e = wrhoJ * (cp.La1 * v_z_e + cp.La2 * uz_e
                            + cp.La3 * r_displ_z[..., 0] + cp.La4 * r_displ_z[..., 1])
        aPML_x, aPML_z = assemble(
            torch.stack([aPML_x_e, aPML_z_e], dim=-1), ib, u_x_now)

        # 7) leap-frog update (κ = 1 ⇒ raw mass; subtract stiffness + PML accel).
        u_x_next = 2.0 * u_x_now - u_x_prev + dt2 * inv_M * (-R_x - aPML_x + b_x)
        u_z_next = 2.0 * u_z_now - u_z_prev + dt2 * inv_M * (-R_z - aPML_z + b_z)
        return (u_x_next, u_z_next,
                r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz, r_displ_x, r_displ_z)

    # ================================================================== #
    # SEMPropagator interface: positional init_state + step.
    # State layout = (u_x, u_z, u_x_prev, u_z_prev, r_dux_dx, r_dux_dz,
    #                 r_duz_dx, r_duz_dz, r_displ_x, r_displ_z) per field_specs.
    # ================================================================== #
    def step(self, fields: tuple, src, dt: float) -> tuple:
        (u_x_now, u_z_now, u_x_prev, u_z_prev,
         r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz, r_displ_x, r_displ_z) = fields
        b_x, b_z = src[0], src[1]                   # source -> fields 0, 1

        if self.cpml is not None:
            (u_x_next, u_z_next, r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz,
             r_displ_x, r_displ_z) = self._step_cpml(
                u_x_now, u_z_now, u_x_prev, u_z_prev,
                r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz, r_displ_x, r_displ_z,
                b_x, b_z, dt)
            # Rigid-wall PML termination: pin u=0 on the outer PML boundary.
            # Without it the α→0 outer-edge convolution memory is an undamped
            # integrator (coef0→1) times a ~d_max stretch coefficient and the
            # layer blows up (see cpml.elastic.outer_pml_edge_nodes). Static
            # index_fill is autograd-safe (zeros the adjoint at the wall too).
            oi = self.cpml.outer_edge_idx
            u_x_next = u_x_next.index_fill(0, oi, 0.0)
            u_z_next = u_z_next.index_fill(0, oi, 0.0)
        else:
            dt2, inv_M = dt * dt, self._inv_M
            Kux, Kuz = self.stiffness_apply(u_x_now, u_z_now)
            u_x_next = 2.0 * u_x_now - u_x_prev + dt2 * inv_M * (-Kux + b_x)
            u_z_next = 2.0 * u_z_now - u_z_prev + dt2 * inv_M * (-Kuz + b_z)

        return (u_x_next, u_z_next, u_x_now, u_z_now,
                r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz, r_displ_x, r_displ_z)

    # ================================================================== #
    # Boundary-saving hooks (swap2nd reverse). The interior 2nd-order
    # displacement leapfrog is exactly time-reversible, so only the two
    # displacement components are ring-saved on the full CPML band; the six
    # CPML memory fields (4-9) are element-local (E, ngll, ngll, 2), exactly
    # zero outside the band, and the reverse march simply zeroes them.
    #
    # The band memories ARE ring-saved on the band elements. They look
    # negligible on short records (the band is barely excited, so zeroing them
    # costs <1e-8), but once surface waves load the PML the band carries a
    # large share of the gradient (measured: 22% of |g| at nt=1200) and the
    # zeroed memories corrupt the adjoint entering the INTERIOR too — interior
    # cos fell to 0.988, and re-anchoring the reverse march does not help
    # because the error is systematic per step, not accumulated roundoff.
    # The memory recursion cannot be inverted in reverse instead: r_old =
    # (r_new - z1*d - z2*d_old)/z0 with z0 < 1 amplifies exponentially.
    # ================================================================== #
    # The C-PML builder this equation's ``cpml=`` argument expects.
    # Declared on the class rather than in a lookup table inside the
    # propagator: the profile dataclasses are NOT interchangeable, and
    # the equation is what knows which one it takes.
    cpml_builder: ClassVar[object] = staticmethod(build_elastic_cpml)

    bs_ring_save: ClassVar[list] = [0, 1]
    bs_ring_restore: ClassVar[list] = [(0, 0, 0), (1, 1, 0), (2, 0, -1), (3, 1, -1)]
    # Thread the adjoint through the CPML memories too (the ring reconstructs
    # them exactly): their recursion carries real adjoint once the band is loaded.
    bs_grad_fields: ClassVar[list] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    bs_elem_ring_save: ClassVar[list] = [4, 5, 6, 7, 8, 9]
    bs_elem_ring_restore: ClassVar[list] = [(4, 0, 0), (5, 1, 0), (6, 2, 0),
                                            (7, 3, 0), (8, 4, 0), (9, 5, 0)]

    def reverse_step(self, fields, src, dt):
        ux1, uz1, ux0, uz0 = fields[0], fields[1], fields[2], fields[3]
        ze = tuple(torch.zeros_like(m) for m in fields[4:])
        out = self.step((ux0, uz0, ux1, uz1, *ze), src, dt)
        return (ux0, uz0, out[0], out[1], *ze)

    def bs_model_tensors(self) -> dict:
        """Model-dependent leaves ``step`` reads: the Lamé bundles (stiffness),
        the mass inverse (leap-frog) and the CPML element mass ``w_rho_J`` (the
        acceleration/mass PML term — carries the ρ gradient in the band)."""
        return {"lam": self._lam, "mu": self._mu, "lam2mu": self._lam2mu,
                "inv_M": self._inv_M, "w_rho_J": self._w_rho_J}

    def bs_recompute(self, fields, src, dt, leaves):
        saved = (self._lam, self._mu, self._lam2mu, self._inv_M, self._w_rho_J)
        self._lam, self._mu, self._lam2mu = (leaves["lam"], leaves["mu"],
                                             leaves["lam2mu"])
        self._inv_M, self._w_rho_J = leaves["inv_M"], leaves["w_rho_J"]
        try:
            return self.step(fields, src, dt)
        finally:
            (self._lam, self._mu, self._lam2mu,
             self._inv_M, self._w_rho_J) = saved
