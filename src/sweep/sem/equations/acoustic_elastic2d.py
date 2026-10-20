"""Coupled acoustic–elastic (fluid–solid) SEM on ONE deformed Quad9 mesh.

Fluid: SPECFEM potential form ``(1/κ)χ̈ = ∇·((1/ρ)∇χ)`` (see
:mod:`~sweep.sem.equations.acoustic_potential2d`).
Solid: 2nd-order displacement ``ρü = ∇·σ`` (:class:`ElasticSEM2DDeformed`).
Both are 3-level leapfrog — SPECFEM's Newmark (β=0, γ=½) eliminates to exactly
``x^{n+1} = 2x^n − x^{n−1} + Δt² a^n``, which is our update, so there is no
predictor to hoist and no sub-cycling.

**One class, both domains — deliberately.** The framework wraps *one* equation:
the propagator calls ``step(state, src, dt)`` once on one state tuple, and
boundary saving reads one ``field_specs`` / one ``eq.cpml`` / monkeypatches one
``eq.step``. Two equations + a coupling operator would mean rewriting both. It
also keeps the whole forward step inside **one traced function**, so autograd
produces the **exact discrete transpose** of the coupling automatically — that
is why, unlike SPECFEM (which flips the domain order and uses *different*
coupling formulas for ``SIMULATION_TYPE==3``), we never hand-code an adjoint
coupling at all. The sub-operators are composed, not re-implemented.

Ordering inside ``step`` — **do not reorder**
---------------------------------------------
``S^ac`` enters the fluid residual **before** the mass inversion; the solid then
reads ``χ̈`` **after** it, at the **same** time level. Reading a pre-inversion
residual, or last step's ``χ̈``, silently drops the coupling to 1st order — a
*plausible* wavefield with wrong amplitudes. At a triple point (Γ meeting the
free surface) ``χ̈`` must be zeroed on the Dirichlet nodes **before** ``S^el``:
the free surface wins over the coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from ..coupling import FluidSolidInterface
from ..fields import DerivedInitState, FieldSpec
from ..mesh.domains import DomainMask
from ..mesh.quad9 import Quad9Mesh
from .acoustic_potential2d import AcousticPotentialSEM2D
from .elastic2d_deformed import ElasticSEM2DDeformed


@dataclass
class AcousticElasticSEM2D(DerivedInitState):
    """Fluid–solid coupled SEM. State: ``(χ, χ_prev, u_x, u_z, u_x_prev, u_z_prev)``."""

    field_specs: ClassVar[list] = [
        # χ must stay at index 0 — the source-kind dispatch keys on it; indices
        # 0-5 are frozen (receivers, BS ring specs and callers rely on them).
        FieldSpec("chi", supports_source=True, supports_receiver=True,
                  seis_key="seis_chi", snap_key="snap_chi"),
        FieldSpec("chi_prev", internal=True),
        FieldSpec("u_x", supports_source=True, supports_receiver=True,
                  seis_key="seis_x", snap_key="snap_ux"),
        FieldSpec("u_z", supports_source=True, supports_receiver=True,
                  seis_key="seis_z", snap_key="snap_uz"),
        FieldSpec("u_x_prev", internal=True),
        FieldSpec("u_z_prev", internal=True),
        # acoustic CPML memory, (n_global,) — stay zero when profiles are None
        FieldSpec("psi_x", internal=True, boundary_related=True),
        FieldSpec("psi_z", internal=True, boundary_related=True),
        FieldSpec("zeta_x", internal=True, boundary_related=True),
        FieldSpec("zeta_z", internal=True, boundary_related=True),
        # elastic CPML memory, element-local (E, ngll, ngll, 2)
        FieldSpec("r_dux_dx", internal=True, boundary_related=True, layout="element2"),
        FieldSpec("r_dux_dz", internal=True, boundary_related=True, layout="element2"),
        FieldSpec("r_duz_dx", internal=True, boundary_related=True, layout="element2"),
        FieldSpec("r_duz_dz", internal=True, boundary_related=True, layout="element2"),
        FieldSpec("r_displ_x", internal=True, boundary_related=True, layout="element2"),
        FieldSpec("r_displ_z", internal=True, boundary_related=True, layout="element2"),
    ]

    # FD-aligned receiver components: hydrophone p = -chi_ddot in the fluid; OBS vx/vz = u_dot in the solid.
    derived_receiver_types: ClassVar[dict] = {"p": ("chi", "pressure"), "vx": ("u_x", "velocity"),
        "vz": ("u_z", "velocity")}

    mesh: Quad9Mesh
    ac: AcousticPotentialSEM2D
    el: ElasticSEM2DDeformed
    ifc: FluidSolidInterface
    domain: DomainMask
    mass: torch.Tensor            # device/dtype ref for the framework
    cpml: Optional[object] = None      # (ac_profile, el_profile) or None

    # The two absorbing profiles this equation's build() takes, keyed by their
    # keyword. PropSEM reads this to construct BOTH from one abcn_elements --
    # without it the fluid-solid form was refused at construction ("takes
    # keyword argument(s) ['cpml_ac', 'cpml_el'] that PropSEM does not
    # supply") even though it has a perfectly good absorbing layer.
    cpml_builders = None               # set after the class body (needs imports)

    @classmethod
    def build(cls, mesh: Quad9Mesh, vp_e, vs_e, rho_e, *,
              dirichlet: tuple[str, ...] = ("top",),
              domain: Optional[DomainMask] = None,
              cpml_ac=None, cpml_el=None) -> "AcousticElasticSEM2D":
        """``domain`` defaults to SPECFEM's rule: an element is acoustic iff
        vs==0. ``cpml_ac`` (acoustic :class:`CPMLProfile`) and ``cpml_el``
        (:class:`ElasticCPMLProfile`) enable the absorbing bands — both or
        neither. The interface may cross an X-only or Z-only band (plain
        coupling term there, SPECFEM-style); a corner region touching the
        interface is rejected at build time."""
        if (cpml_ac is None) != (cpml_el is None):
            raise ValueError("pass both cpml_ac and cpml_el, or neither")
        if domain is None:
            domain = DomainMask.from_vs(mesh, vs_e)
        ac = AcousticPotentialSEM2D.build(mesh, vp_e, rho_e,
                                          dirichlet=dirichlet, domain=domain,
                                          cpml=cpml_ac)
        el = ElasticSEM2DDeformed.build(mesh, vp_e, vs_e, rho_e,
                                        cpml=cpml_el, domain=domain)
        ifc = FluidSolidInterface.detect(mesh, domain)
        if cpml_ac is not None:
            from ..coupling import assert_no_corner_pml_on_interface
            assert_no_corner_pml_on_interface(ifc, cpml_ac)
        return cls(mesh=mesh, ac=ac, el=el, ifc=ifc, domain=domain,
                   mass=ac.mass,
                   cpml=None if cpml_ac is None else (cpml_ac, cpml_el))

    # ------------------------------------------------------------------ #
    def accelerations(self, chi, u_x, u_z, b_chi, b_ux, b_uz):
        """``(χ̈, ü_x, ü_z)`` at the current level — the whole coupling lives here.

        The order below is load-bearing; see the module docstring.
        """
        # 1. fluid residual INCLUDING the coupling, BEFORE the mass inversion.
        #    S^ac = ∮ (u_s·n) φ dΓ  reads the solid displacement at THIS level.
        R_ac = (-self.ac.stiffness_apply(chi) + b_chi
                + self.ifc.apply_ac(u_x, u_z, chi))
        # 2. χ̈ = M_ac⁻¹ R_ac — the POST-inversion value the solid must read.
        chi_dd = self.ac._inv_M * R_ac
        # 3. triple point: the free surface wins over the coupling.
        (chi_dd,) = self.ac._apply_dirichlet(chi_dd)
        # 4-5. p = −χ̈ ; S^el = ∮ p n φ dΓ into the solid residual, SAME level.
        s_x, s_z = self.ifc.apply_el(-chi_dd, chi)
        k_x, k_z = self.el.stiffness_apply(u_x, u_z)
        # 6. ü = M_el⁻¹ R_el
        return (chi_dd,
                self.el._inv_M * (-k_x + b_ux + s_x),
                self.el._inv_M * (-k_z + b_uz + s_z))

    def step(self, fields: tuple, src, dt: float) -> tuple:
        (chi, chi_prev, u_x, u_z, u_x_prev, u_z_prev,
         psi_x, psi_z, zeta_x, zeta_z,
         r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz, r_displ_x, r_displ_z) = fields
        b_chi, b_ux, b_uz = src[0], src[1], src[2]
        dt2 = dt * dt

        if self.cpml is None:
            chi_dd, ux_dd, uz_dd = self.accelerations(
                chi, u_x, u_z, b_chi, b_ux, b_uz)
            chi_next = 2.0 * chi - chi_prev + dt2 * chi_dd
            ux_next = 2.0 * u_x - u_x_prev + dt2 * ux_dd
            uz_next = 2.0 * u_z - u_z_prev + dt2 * uz_dd
            (chi_next,) = self.ac._apply_dirichlet(chi_next)
            return (chi_next, chi, ux_next, uz_next, u_x, u_z,
                    psi_x, psi_z, zeta_x, zeta_z,
                    r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz,
                    r_displ_x, r_displ_z)

        # ---- CPML on both sides; ordering identical to `accelerations` ---- #
        # 1. fluid: stretched stiffness (w_sum) + PLAIN coupling S^ac into the
        #    pre-inversion residual (SPECFEM-style — the plain term inside an
        #    X-only band costs absorption at Γ, not stability).
        w_sum, psi_x, psi_z, zeta_x, zeta_z = self.ac.step_cpml(
            chi, psi_x, psi_z, zeta_x, zeta_z)
        s_ac = self.ifc.apply_ac(u_x, u_z, chi)
        # 2. χ̈ POST-inversion, same level; κ rides the stretched part, M_κ⁻¹
        #    the surface/source terms (exactly the bare path's split).
        chi_dd = (self.ac.kappa_node * w_sum
                  + self.ac._inv_M * (b_chi + s_ac))
        # 3. triple point: the free surface wins over the coupling.
        (chi_dd,) = self.ac._apply_dirichlet(chi_dd)
        # 4. p = −χ̈ → solid surface force, injected through the elastic CPML
        #    step's body-force slot (it adds b at the final update).
        s_x, s_z = self.ifc.apply_el(-chi_dd, chi)
        (ux_next, uz_next, r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz,
         r_displ_x, r_displ_z) = self.el._step_cpml(
            u_x, u_z, u_x_prev, u_z_prev,
            r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz, r_displ_x, r_displ_z,
            b_ux + s_x, b_uz + s_z, dt)
        # 4b. rigid-wall PML termination on the solid outer edge (the coupled
        #     step calls _step_cpml directly, so it applies the pin itself;
        #     see cpml.elastic.outer_pml_edge_nodes). The fluid CPML is
        #     self-damped and its outer nodes carry dead solid DOFs anyway.
        oi = self.el.cpml.outer_edge_idx
        ux_next = ux_next.index_fill(0, oi, 0.0)
        uz_next = uz_next.index_fill(0, oi, 0.0)
        # 5. fluid update + free surface (χ and its memories).
        chi_next = 2.0 * chi - chi_prev + dt2 * chi_dd
        chi_next, psi_x, psi_z, zeta_x, zeta_z = self.ac._apply_dirichlet(
            chi_next, psi_x, psi_z, zeta_x, zeta_z)
        return (chi_next, chi, ux_next, uz_next, u_x, u_z,
                psi_x, psi_z, zeta_x, zeta_z,
                r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz,
                r_displ_x, r_displ_z)

    # ================================================================== #
    # Boundary-saving hooks — swap2nd on BOTH domains at once.
    #
    # With no CPML (this class, until the interface-CPML stage) the coupled
    # step is EXACTLY time-reversible: both domains are 3-level leapfrog and
    # all forces are evaluated at the "now" level, so swapping (now, prev) in
    # both domains and re-running ``step`` reconstructs the previous state
    # bit-for-bit — including the coupling (S^ac reads u at the correct
    # unswapped slot, S^el reads the recomputed χ̈ at the same level) and the
    # Dirichlet nodes (χ is 0 at every level there, and χ̈ is zeroed before
    # S^el, so re-zeroing is exact). The interface ring is therefore EMPTY and
    # BS memory is O(1) in nt. The ring specs below only become active when a
    # CPML profile appears (cpml_node_masks keys off ``eq.cpml``).
    # ================================================================== #
    bs_ring_save: ClassVar[list] = [0, 2, 3]
    bs_ring_restore: ClassVar[list] = [(0, 0, 0), (1, 0, -1), (2, 1, 0),
                                       (3, 2, 0), (4, 1, -1), (5, 2, -1)]
    # The COUPLED reverse march accumulates error the acoustic and elastic ones
    # do not, so BS re-anchors on a full-state checkpoint every K steps (memory
    # nt/K frames). Re-measured on the bare box (max over the ringed fields,
    # error normalised by the record's PEAK field, not the instantaneous one):
    #
    #   nt=200   no anchor 2.4e-05   anchor 9.8e-11
    #   nt=600   no anchor 1.8e-05   anchor 4.2e-11
    #
    # i.e. the march grows for ~200 steps and then PLATEAUS, and the anchors buy
    # 4-5 orders. An earlier note here claimed ~×1.15/step compounding "and
    # eventually overflows"; that does not reproduce (~×1.009/step, no overflow
    # at nt=600) and is the signature of normalising by the instantaneous
    # ||truth||, which vanishes at the end of the reverse march. The anchors are
    # worth keeping on the measurement above; the compounding rate was not.
    bs_frame_checkpoint_every: ClassVar[int] = 50

    def _zero_memories(self, like):
        ng = self.mesh.ngll
        z = lambda: torch.zeros_like(like)
        r = lambda: like.new_zeros((self.mesh.n_elements, ng, ng, 2))
        return (z(), z(), z(), z(), r(), r(), r(), r(), r(), r())

    def reverse_step(self, fields, src, dt):
        chi_n, chi_p, ux_n, uz_n, ux_p, uz_p = fields[:6]    # levels (i+1, i)
        Z = self._zero_memories(chi_n)
        out = self.step((chi_p, chi_n, ux_p, uz_p, ux_n, uz_n, *Z), src, dt)
        # out = (χ^{i-1}, χ^i, u^{i-1}, …, u^i); reorder to the step-i input.
        # Memory fields are reconstructed as zeros — exact in the interior
        # (their coefficients vanish there); the bands are zeroed + ring-
        # restored / frame-re-anchored by the BS framework.
        return (out[1], out[0], out[4], out[5], out[2], out[3], *Z)

    def bs_model_tensors(self) -> dict:
        """Every model-dependent tensor ``step`` reads (complete by
        construction for the BARE path: stiffness + mass on each side; the
        interface operator is pure geometry). The CPML path reads MORE model
        tensors (kappa_node, the geometric mass, the rho-mass weight of the
        elastic aPML) — wiring those leaves + the dual-profile node masks is
        the next sub-stage, so BS with CPML fails fast instead of silently
        producing a wrong gradient."""
        if self.cpml is not None:
            raise NotImplementedError(
                "boundary saving for the coupled equation WITH interface CPML "
                "is not wired yet (dual-profile cpml_node_masks + extra model "
                "leaves); run full-tape/checkpointed autograd instead.")
        return {"ac_inv_M": self.ac._inv_M, "ac_inv_rho": self.ac._inv_rho,
                "el_inv_M": self.el._inv_M, "el_lam": self.el._lam,
                "el_mu": self.el._mu, "el_lam2mu": self.el._lam2mu}

    def bs_recompute(self, fields, src, dt, leaves):
        ac, el = self.ac, self.el
        saved = (ac._inv_M, ac._inv_rho, el._inv_M, el._lam, el._mu, el._lam2mu)
        ac._inv_M, ac._inv_rho = leaves["ac_inv_M"], leaves["ac_inv_rho"]
        el._inv_M = leaves["el_inv_M"]
        el._lam, el._mu, el._lam2mu = (leaves["el_lam"], leaves["el_mu"],
                                       leaves["el_lam2mu"])
        try:
            return self.step(fields, src, dt)
        finally:
            (ac._inv_M, ac._inv_rho, el._inv_M,
             el._lam, el._mu, el._lam2mu) = saved

    # ------------------------------------------------------------------ #
    def energy(self, fields, dt: float) -> torch.Tensor:
        """Total (fluid + solid) energy — the stage-4a conservation diagnostic.

        ``½ χ̇ᵀM_ac χ̇ + ½ χᵀK_ac χ + ½ u̇ᵀM_el u̇ + ½ uᵀK_el u`` with the
        velocities from the leapfrog levels. A wrong-sign coupling turns Γ into
        an energy *source* and this diverges exponentially.
        """
        chi, chi_prev, u_x, u_z, u_x_prev, u_z_prev = fields[:6]
        chi_d = (chi - chi_prev) / dt
        ux_d, uz_d = (u_x - u_x_prev) / dt, (u_z - u_z_prev) / dt
        k_x, k_z = self.el.stiffness_apply(u_x, u_z)
        return (0.5 * (self.ac.mass * chi_d * chi_d).sum()
                + 0.5 * (chi * self.ac.stiffness_apply(chi)).sum()
                + 0.5 * (self.el.mass * (ux_d * ux_d + uz_d * uz_d)).sum()
                + 0.5 * (u_x * k_x + u_z * k_z).sum())


# Assigned outside the class body: the builders live in modules this one
# already imports for the profiles themselves.
from ..cpml.acoustic import build_cpml as _build_cpml_ac
from ..cpml.elastic import build_elastic_cpml as _build_cpml_el

AcousticElasticSEM2D.cpml_builders = {"cpml_ac": _build_cpml_ac,
                                      "cpml_el": _build_cpml_el}

# How PropSEM injects this equation's default source. The source fields are
# (chi, u_x, u_z) and the low-level stamp components are zipped onto them
# FROM FIELD 0, so a 2-component force payload would land on (chi, u_x) --
# silently wrong. A scalar into chi (a marine airgun in the water column) is
# the one injection that maps correctly, and it is also what this equation
# is for. Anything else must go through SEMPropagator with explicit
# source_kinds/source_payloads.
AcousticElasticSEM2D.default_source_type = ["chi"]
AcousticElasticSEM2D.propsem_source = ("scalar", (1.0,))
