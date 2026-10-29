"""1st-order velocity-stress elastic SEM on a Quad9 deformed mesh.

Solves the elastodynamic equations as a first-order hyperbolic system:

    ρ ∂v_i/∂t   = ∂σ_ij/∂x_j + f_i        (momentum)
    ∂σ_ij/∂t  = C_ijkl ∂v_k/∂x_l         (constitutive)

with isotropic ``C_ijkl = λ δ_ij δ_kl + μ (δ_ik δ_jl + δ_il δ_jk)``.

Storage convention
------------------
* Velocity components ``v_x``, ``v_z`` are continuous-Galerkin global
  fields, shape ``(n_global,)``.
* Stress components ``σ_xx``, ``σ_zz``, ``σ_xz`` are element-local
  (DG-style), shape ``(E, ngll, ngll)``. This lets material jumps live
  at element boundaries; for continuous models the values agree across
  the shared face up to roundoff.
* CPML memory variables are element-local with the same shape as σ
  (8 fields total, see :class:`SimpleElasticCPMLProfile`).

Operators
---------
* :meth:`velocity_gradients` — strong-form ∂v/∂x and ∂v/∂z at each GLL,
  element-local. Used in the σ update.
* :meth:`assemble_stress_divergence` — weak-form ``-∫ ∇φ · σ dV``
  assembled to global. Used in the v update; the negative sign matches
  the displacement form's ``-K · u`` term so the caller can write
  ``M_ρ · ∂v/∂t = -R + b`` (consistent with ``M_ρ · ü = -K + b``).

Free-surface BC
---------------
* ``free_top=False`` (default): natural BC — ``σ · n = 0`` enforced
  weakly by dropping the boundary-traction surface integral in the
  v-update assembly.
* ``free_top=True``: essential BC — ``σ_nn = σ_nt = 0`` imposed strongly
  at the top GLL row in the local (tangent, normal) frame:
    1. Precompute unit tangent ``t = (∂x/∂ξ, ∂z/∂ξ) / |·|`` along the
       η=−1 edge of each top element; outward normal ``n = (t_z, −t_x)``.
    2. After the σ update, project to the local frame:
          ``σ_tt = σ_xx t_x² + 2 σ_xz t_x t_z + σ_zz t_z²``
    3. Reconstruct under ``σ_nn = σ_nt = 0``:
          ``σ_xx ← σ_tt t_x²,  σ_zz ← σ_tt t_z²,  σ_xz ← σ_tt t_x t_z``.
  For a flat surface (t = (1, 0)) this is the pointwise ``σ_zz=σ_xz=0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from .._assembly import assemble, contract, phys_grads
from ..cpml.elastic_vs import SimpleElasticCPMLProfile, build_simple_elastic_cpml
from ..fields import DerivedInitState, FieldSpec, check_element_arrays
from ..mesh.quad9 import Quad9Mesh, boundary_element_edges


@dataclass
class ElasticVSSEM2DDeformed(DerivedInitState):
    """1st-order velocity-stress elastic SEM on a deformed Quad9 mesh."""

    # ---- Field specification: the positional state layout (see fields.py) ---- #
    # (v_x, v_z, sigma_xx, sigma_zz, sigma_xz, then the 8 CPML memory variables —
    #  always present; they stay zero when cpml is None.)
    field_specs: ClassVar[list] = [
        FieldSpec("v_x", supports_source=True, supports_receiver=True,
                  seis_key="seis_vx", snap_key="snapshots_vx"),
        FieldSpec("v_z", supports_source=True, supports_receiver=True,
                  seis_key="seis_vz", snap_key="snapshots_vz"),
        FieldSpec("sigma_xx", internal=True, layout="element"),
        FieldSpec("sigma_zz", internal=True, layout="element"),
        FieldSpec("sigma_xz", internal=True, layout="element"),
        FieldSpec("m_sxx_x", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_sxz_z", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_sxz_x", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_szz_z", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_vxx", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_vzz", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_vxz", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_vzx", internal=True, boundary_related=True, layout="element"),
    ]

    # FD-aligned receiver components: velocity-stress evolves velocities natively: FD-name aliases only.
    derived_receiver_types: ClassVar[dict] = {"vx": ("v_x", None), "vz": ("v_z", None)}

    mesh: Quad9Mesh
    vp_e: torch.Tensor                 # (E, ngll, ngll)
    vs_e: torch.Tensor
    rho_e: torch.Tensor

    # ρ-weighted lumped mass at global v-nodes:
    #   mass_rho[g] = Σ_e ρ_e · w_α w_β · |J(ξ,η)|
    # Same as the 2nd-order class's ``mass``; appears in the v update.
    mass_rho: torch.Tensor             # (n_global,)
    _inv_M: torch.Tensor               # (n_global,) — 1 / mass_rho

    cpml: Optional[SimpleElasticCPMLProfile]
    free_top: bool                     # False = natural BC; True = essential
                                       # BC (σ·n=0 in local coords at top row)
    top_elem_ids: torch.Tensor         # (nex,) — element ids of the top row
    _tx_top: torch.Tensor              # (n_top, ngll) — unit-tangent x at η=-1
    _tz_top: torch.Tensor              # (n_top, ngll) — unit-tangent z at η=-1

    # Precomputed element-local fields (same set as 2nd-order class).
    _D: torch.Tensor                   # (ngll, ngll)
    _lam: torch.Tensor                 # (E, ngll, ngll)
    _mu: torch.Tensor
    _lam2mu: torch.Tensor
    _w_J: torch.Tensor                 # w_α w_β · |J|
    _ixx: torch.Tensor                 # inv_J[..., 0, 0] = ∂ξ/∂x
    _ixz: torch.Tensor                 # ∂ξ/∂z
    _iex: torch.Tensor                 # ∂η/∂x
    _iez: torch.Tensor                 # ∂η/∂z

    @classmethod
    def build(
        cls,
        mesh: Quad9Mesh,
        vp_e: torch.Tensor,
        vs_e: torch.Tensor,
        rho_e: torch.Tensor,
        *,
        free_top: bool = False,
        cpml: Optional[SimpleElasticCPMLProfile] = None,
    ) -> "ElasticVSSEM2DDeformed":
        """Build the equation operators.

        ``free_top=False`` (default) → natural free-surface BC, weakly
        imposed by dropping the boundary integral. ``free_top=True`` →
        strong essential BC (``σ·n=0`` in local coords at the top row).
        """
        check_element_arrays(mesh, vp_e=vp_e, vs_e=vs_e, rho_e=rho_e)
        device, dtype = vp_e.device, vp_e.dtype

        mu     = rho_e * vs_e * vs_e
        lam    = rho_e * (vp_e * vp_e - 2.0 * vs_e * vs_e)
        lam2mu = lam + 2.0 * mu

        m_e = mesh.w_jac * rho_e
        mass_rho = torch.zeros(mesh.n_global, dtype=dtype, device=device)
        mass_rho.index_add_(0, mesh.ibool.reshape(-1), m_e.reshape(-1))

        edges = boundary_element_edges(mesh)
        top_elem_ids = edges.top.to(torch.long).to(device)

        ixx = mesh.inv_J[..., 0, 0]
        ixz = mesh.inv_J[..., 0, 1]
        iex = mesh.inv_J[..., 1, 0]
        iez = mesh.inv_J[..., 1, 1]

        # Unit tangent along the η = −1 (top) edge of every top element.
        # J = inv(inv_J) gives ``∂x/∂ξ = iez · det_J`` and
        # ``∂z/∂ξ = -iex · det_J`` (cofactor formula for 2×2 inverse).
        # Normalise to unit length.
        iex_top = iex[top_elem_ids, 0, :]                    # (n_top, ngll)
        iez_top = iez[top_elem_ids, 0, :]
        det_J_top = mesh.det_J[top_elem_ids, 0, :]
        tx_raw =  iez_top * det_J_top
        tz_raw = -iex_top * det_J_top
        t_norm = torch.sqrt(tx_raw ** 2 + tz_raw ** 2).clamp(min=1e-30)
        tx_top = tx_raw / t_norm
        tz_top = tz_raw / t_norm

        return cls(
            mesh=mesh, vp_e=vp_e, vs_e=vs_e, rho_e=rho_e,
            mass_rho=mass_rho, _inv_M=1.0 / mass_rho, cpml=cpml,
            free_top=free_top, top_elem_ids=top_elem_ids,
            _tx_top=tx_top, _tz_top=tz_top,
            _D=mesh.deriv,
            _lam=lam, _mu=mu, _lam2mu=lam2mu,
            _w_J=mesh.w_jac,
            _ixx=ixx, _ixz=ixz, _iex=iex, _iez=iez,
        )

    # ------------------------------------------------------------------ #
    # Strong-form velocity gradients (element-local) — for the σ update.
    # ------------------------------------------------------------------ #
    def velocity_gradients(
        self,
        v_x_global: torch.Tensor,
        v_z_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ∂v/∂x, ∂v/∂z at every element-local GLL node.

        Returns ``(d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz)``, each shape
        ``(E, ngll, ngll)``.
        """
        # ∂v/∂x, ∂v/∂z for both components: one batched gather + 2 einsums.
        d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz = phys_grads(
            self._D, self.mesh.ibool, self._ixx, self._ixz, self._iex,
            self._iez, v_x_global, v_z_global)
        return d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz

    # ------------------------------------------------------------------ #
    # Weak-form stress divergence (assembled) — for the v update.
    # ------------------------------------------------------------------ #
    def assemble_stress_divergence(
        self,
        sigma_xx_xpart: torch.Tensor,   # σ_xx ⊗ ∂φ/∂x-part (with CPML stretching)
        sigma_xz_zpart: torch.Tensor,   # σ_xz ⊗ ∂φ/∂z-part of v_x equation
        sigma_xz_xpart: torch.Tensor,   # σ_xz ⊗ ∂φ/∂x-part of v_z equation
        sigma_zz_zpart: torch.Tensor,   # σ_zz ⊗ ∂φ/∂z-part
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble ``R_i = ∫ ∂φ/∂x_j · σ_ij dV`` at global GLL nodes.

        The caller supplies four element-local σ-arrays, one per
        (test-function-axis × stress-component) pairing. Without CPML
        these are all identical to the bare σ field; with CPML they are
        the CPML-stretched fields ``σ + m_σ`` for the appropriate axis.

        For the v_x equation:
            ``R_x = ∫ (∂φ/∂x σ_xx + ∂φ/∂z σ_xz) dV``
        for the v_z equation:
            ``R_z = ∫ (∂φ/∂x σ_xz + ∂φ/∂z σ_zz) dV``.

        The chain rule ``∂φ/∂x = ixx ∂φ/∂ξ + iex ∂φ/∂η`` (and similar
        for z) is folded into the per-direction-flux contraction. The
        returned tensors have shape ``(n_global,)``.
        """
        w_J = self._w_J
        F_x_beta  = w_J * (sigma_xx_xpart * self._ixx
                           + sigma_xz_zpart * self._ixz)
        F_x_alpha = w_J * (sigma_xx_xpart * self._iex
                           + sigma_xz_zpart * self._iez)
        F_z_beta  = w_J * (sigma_xz_xpart * self._ixx
                           + sigma_zz_zpart * self._ixz)
        F_z_alpha = w_J * (sigma_xz_xpart * self._iex
                           + sigma_zz_zpart * self._iez)

        # Stack the x/z β- and α-fluxes and do the D^T contraction batched
        # (2 einsums vs 4), then assemble R_x and R_z in one scatter.
        F_beta  = torch.stack([F_x_beta, F_z_beta], dim=-1)        # (E,ngll,ngll,2)
        F_alpha = torch.stack([F_x_alpha, F_z_alpha], dim=-1)
        R_e = (contract("db,eadk->eabk", self._D, F_beta)
               + contract("ga,egbk->eabk", self._D, F_alpha))  # (E,ngll,ngll,2)
        like = sigma_xx_xpart.new_zeros(self.mesh.n_global)
        return assemble(R_e, self.mesh.ibool, like)

    # ------------------------------------------------------------------ #
    # Free surface: traction-free in local coords at top GLL nodes.
    # ------------------------------------------------------------------ #
    def apply_free_top(
        self,
        sigma_xx: torch.Tensor,
        sigma_zz: torch.Tensor,
        sigma_xz: torch.Tensor,
    ):
        """In-place local-coord rotation: ``σ_nn = σ_nt = 0`` at the top GLL.

        Uses the precomputed unit tangent ``(t_x, t_z)`` at each top GLL
        node. The only stress component that survives the BC is the
        in-plane tangential ``σ_tt``; we project current σ onto that
        component and rotate back to global x, z. For a flat horizontal
        surface (``t = (1, 0)``) this reduces to the pointwise
        ``σ_zz = σ_xz = 0`` BC with ``σ_xx`` unchanged. For curved
        topography the slight tilt is correctly accounted for.
        """
        if not self.free_top:
            return
        ids = self.top_elem_ids
        tx = self._tx_top                          # (n_top, ngll)
        tz = self._tz_top
        sxx = sigma_xx[ids, 0, :]
        szz = sigma_zz[ids, 0, :]
        sxz = sigma_xz[ids, 0, :]
        # Project old σ to local tangential component σ_tt:
        s_tt = sxx * tx * tx + 2.0 * sxz * tx * tz + szz * tz * tz
        # Reconstruct global σ assuming σ_nn = σ_nt = 0:
        sigma_xx[ids, 0, :] = s_tt * tx * tx
        sigma_zz[ids, 0, :] = s_tt * tz * tz
        sigma_xz[ids, 0, :] = s_tt * tx * tz

    # ------------------------------------------------------------------ #
    # Zero CPML memory variables at the top GLL row.
    # ------------------------------------------------------------------ #
    def taper_cpml_at_top(
        self,
        m_sxx_x: torch.Tensor, m_sxz_z: torch.Tensor,
        m_sxz_x: torch.Tensor, m_szz_z: torch.Tensor,
        m_vxx:   torch.Tensor, m_vzz:   torch.Tensor,
        m_vxz:   torch.Tensor, m_vzx:   torch.Tensor,
    ):
        """Zero CPML memory variables at the top surface GLL row only
        (α=0); deeper rows retain their full CPML memory. No-op unless
        ``free_top`` is set.
        """
        if not self.free_top:
            return
        ids = self.top_elem_ids
        for m in (m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z,
                  m_vxx, m_vzz, m_vxz, m_vzx):
            m[ids, 0, :] = 0.0

    # ================================================================== #
    # SEMPropagator interface: init_state + step (staggered leapfrog).
    # ================================================================== #
    def _stress_increment(self, d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz, dt):
        """Constitutive update ``dσ = dt · C : ∇v`` (per element, per GLL node).

        Isotropic Hooke here; anisotropic subclasses override this single
        method (the layout-agnostic constitutive seam) and inherit the entire
        velocity-stress time loop, CPML, free surface and gradient machinery
        unchanged. ``∇v`` already carries any CPML stretching from the caller.
        """
        inc_xx = dt * (self._lam2mu * d_vx_dx + self._lam * d_vz_dz)
        inc_zz = dt * (self._lam * d_vx_dx + self._lam2mu * d_vz_dz)
        inc_xz = dt * self._mu * (d_vx_dz + d_vz_dx)
        return inc_xx, inc_zz, inc_xz

    def step(self, fields: tuple, src, dt: float) -> tuple:
        """One staggered-leapfrog step (v at n+1/2, σ at n+1)."""
        (v_x, v_z, sigma_xx, sigma_zz, sigma_xz,
         m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z,
         m_vxx, m_vzz, m_vxz, m_vzx) = fields
        cp = self.cpml
        use_cpml = cp is not None
        inv_M = self._inv_M
        b_x, b_z = src[0], src[1]

        # ----- velocity update from current σ (with σ-field CPML) -----
        if use_cpml:
            m_sxx_x = cp.a_x * m_sxx_x + cp.b_x * sigma_xx
            m_sxz_z = cp.a_z * m_sxz_z + cp.b_z * sigma_xz
            m_sxz_x = cp.a_x * m_sxz_x + cp.b_x * sigma_xz
            m_szz_z = cp.a_z * m_szz_z + cp.b_z * sigma_zz
            sxx_xpart = sigma_xx + m_sxx_x
            sxz_zpart = sigma_xz + m_sxz_z
            sxz_xpart = sigma_xz + m_sxz_x
            szz_zpart = sigma_zz + m_szz_z
        else:
            sxx_xpart = sigma_xx
            sxz_zpart = sigma_xz
            sxz_xpart = sigma_xz
            szz_zpart = sigma_zz

        R_x, R_z = self.assemble_stress_divergence(
            sxx_xpart, sxz_zpart, sxz_xpart, szz_zpart)
        v_x = v_x + dt * inv_M * (-R_x + b_x)
        v_z = v_z + dt * inv_M * (-R_z + b_z)

        # ----- stress update from new v (with v-gradient CPML) -----
        d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz = self.velocity_gradients(v_x, v_z)
        if use_cpml:
            m_vxx = cp.a_x * m_vxx + cp.b_x * d_vx_dx
            m_vzz = cp.a_z * m_vzz + cp.b_z * d_vz_dz
            m_vxz = cp.a_z * m_vxz + cp.b_z * d_vx_dz
            m_vzx = cp.a_x * m_vzx + cp.b_x * d_vz_dx
            d_vx_dx = d_vx_dx + m_vxx
            d_vz_dz = d_vz_dz + m_vzz
            d_vx_dz = d_vx_dz + m_vxz
            d_vz_dx = d_vz_dx + m_vzx

        inc_xx, inc_zz, inc_xz = self._stress_increment(
            d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz, dt)
        sigma_xx = sigma_xx + inc_xx
        sigma_zz = sigma_zz + inc_zz
        sigma_xz = sigma_xz + inc_xz

        # ----- free surface BC + corner taper -----
        self.apply_free_top(sigma_xx, sigma_zz, sigma_xz)
        if use_cpml:
            self.taper_cpml_at_top(m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z,
                                   m_vxx, m_vzz, m_vxz, m_vzx)

        return (v_x, v_z, sigma_xx, sigma_zz, sigma_xz,
                m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z,
                m_vxx, m_vzz, m_vxz, m_vzx)

    # ================================================================== #
    # Boundary-saving hooks (see sweep.sem.boundary_saving).
    # The reverse driver is 'substep': undo the leapfrog sub-steps in
    # reversed order at −dt with the CPML memory zeroed. Exact in the
    # lossless interior; assumes natural free-surface BC (free_top=False).
    # ================================================================== #
    # Global-node wavefields whose interior interface ring is saved/restored
    # each step (σ is reconstructed exactly from the restored v, so it needs
    # no ring of its own). Each restored at the same step index, offset 0.
    # Positional field indices (see field_specs): ring-save v_x (0), v_z (1);
    # σ (2,3,4) reconstructs exactly from the restored v so it needs no ring.
    # The C-PML builder this equation's ``cpml=`` argument expects.
    # Declared on the class rather than in a lookup table inside the
    # propagator: the profile dataclasses are NOT interchangeable, and
    # the equation is what knows which one it takes.
    cpml_builder: ClassVar[object] = staticmethod(build_simple_elastic_cpml)

    bs_ring_save: ClassVar[list] = [0, 1]
    bs_ring_restore: ClassVar[list] = [(0, 0, 0), (1, 1, 0)]   # (field, slot, offset)
    bs_model_attrs: ClassVar[tuple] = ("_lam", "_mu", "_lam2mu")
    # The BS grad_idx (boundary_saving.py) = fields with not boundary_related,
    # i.e. v (0,1) AND stress (2,3,4) — stress carries the per-step adjoint
    # (σ feeds v via the divergence); CPML memory (5..12) is boundary_related.

    def reverse_step(self, fields, src, dt):
        """Reconstruct the step's input state from its output (CPML zeroed)."""
        v_x, v_z, sigma_xx, sigma_zz, sigma_xz = fields[:5]
        inv_M = self._inv_M
        b_x, b_z = src[0], src[1]
        # undo the stress update: σ_old = σ_new − dt·C:∇(v_new)   (no CPML stretch)
        d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz = self.velocity_gradients(v_x, v_z)
        inc_xx, inc_zz, inc_xz = self._stress_increment(
            d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz, dt)
        sigma_xx = sigma_xx - inc_xx
        sigma_zz = sigma_zz - inc_zz
        sigma_xz = sigma_xz - inc_xz
        # undo the velocity update: v_old = v_new − dt·inv_M·(−R(σ_old) + b)
        R_x, R_z = self.assemble_stress_divergence(
            sigma_xx, sigma_xz, sigma_xz, sigma_zz)
        v_x = v_x - dt * inv_M * (-R_x + b_x)
        v_z = v_z - dt * inv_M * (-R_z + b_z)
        z = torch.zeros_like(sigma_xx)
        return (v_x, v_z, sigma_xx, sigma_zz, sigma_xz,
                z, z, z, z, z, z, z, z)            # CPML memory zeroed

    def bs_model_tensors(self) -> dict:
        """Model-dependent tensors to differentiate (graph tensors connected to
        the user's vp/vs/rho). ``mass_rho`` feeds the velocity update via inv_M;
        the others feed the constitutive stress update."""
        d = {a: getattr(self, a) for a in self.bs_model_attrs}
        d["mass_rho"] = self.mass_rho
        return d

    def bs_recompute(self, fields, src, dt, leaves):
        """Re-run one forward step with the model tensors replaced by ``leaves``
        (for the per-step VJP): substitute the constitutive attrs + inv_M (from
        ``mass_rho``) on ``self``, run, then restore."""
        saved = {a: getattr(self, a) for a in self.bs_model_attrs}
        saved_inv = self._inv_M
        for a in self.bs_model_attrs:
            setattr(self, a, leaves[a])
        self._inv_M = 1.0 / leaves["mass_rho"]
        try:
            return self.step(fields, src, dt)
        finally:
            for a, v in saved.items():
                setattr(self, a, v)
            self._inv_M = saved_inv
