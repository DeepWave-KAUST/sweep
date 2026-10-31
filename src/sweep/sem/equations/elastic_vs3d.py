"""3-D 1st-order velocity-stress isotropic elastic SEM on an axis-aligned Hex8
mesh — the 3-D analogue of :mod:`sweep.sem.equations.elastic_vs_deformed`.

First-order hyperbolic system::

    ρ ∂v_i/∂t = ∂σ_ij/∂x_j + f_i              (momentum)
    ∂σ_ij/∂t = λ δ_ij (∇·v) + μ(∂_i v_j + ∂_j v_i)   (isotropic constitutive)

Storage
-------
* Velocity ``v_x, v_y, v_z`` — continuous-Galerkin global fields ``(n_global,)``.
* Stress ``σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz`` — element-local ``(E,g,g,g)``.
* CPML memory — element-local; 9 for the stress divergence (one per (stress,
  derivative-direction) pairing in the v update) + 9 for the velocity gradients
  (one per (velocity-component, derivative-direction) in the σ update).

Why velocity-stress (vs the 2nd-order displacement form): the FD-style
single-recursion CPML ``m' = a·m + b·∂`` is long-nt stable here, whereas the
2nd-order displacement CPML needs the full SPECFEM accel/mass damping machinery
to avoid blow-up. This mirrors the 2-D choice (``ElasticVSSEM2DDeformed`` is the
production absorbing path).

Free surface = natural BC: the boundary-traction integral is simply dropped from
the v-update assembly (the locked-in 2-D standard; no essential-BC variant here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from .._assembly import assemble_each, contract
from ..cpml.band import active_elements, element_split
from ..cpml.elastic_vs3d import (SimpleElasticCPML3DProfile,
                                 build_simple_elastic_cpml_3d)
from ..fields import DerivedInitState, FieldSpec, check_element_arrays
from ..mesh.hex8 import Hex8Mesh


@dataclass
class ElasticVSSEM3D(DerivedInitState):
    """1st-order velocity-stress elastic SEM on an axis-aligned Hex8 mesh."""

    # Positional state layout (see fields.py). 3 v + 6 σ + 9 σ-div memory
    # + 9 v-grad memory = 27 fields. The CPML memory lives on the C-PML band
    # elements only (``band``: zero everywhere else); none when cpml is None.
    field_specs: ClassVar[list] = [
        FieldSpec("v_x", supports_source=True, supports_receiver=True,
                  seis_key="seis_vx", snap_key="snapshots_vx"),
        FieldSpec("v_y", supports_source=True, supports_receiver=True,
                  seis_key="seis_vy", snap_key="snapshots_vy"),
        FieldSpec("v_z", supports_source=True, supports_receiver=True,
                  seis_key="seis_vz", snap_key="snapshots_vz"),
        FieldSpec("sigma_xx", internal=True, layout="element"),
        FieldSpec("sigma_yy", internal=True, layout="element"),
        FieldSpec("sigma_zz", internal=True, layout="element"),
        FieldSpec("sigma_xy", internal=True, layout="element"),
        FieldSpec("sigma_xz", internal=True, layout="element"),
        FieldSpec("sigma_yz", internal=True, layout="element"),
        # stress-divergence CPML memory (9)
        FieldSpec("m_sxx_x", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_sxy_y", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_sxz_z", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_sxy_x", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_syy_y", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_syz_z", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_sxz_x", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_syz_y", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_szz_z", internal=True, boundary_related=True, layout="band"),
        # velocity-gradient CPML memory (9)
        FieldSpec("m_vx_x", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_vy_y", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_vz_z", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_vy_x", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_vx_y", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_vz_x", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_vx_z", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_vz_y", internal=True, boundary_related=True, layout="band"),
        FieldSpec("m_vy_z", internal=True, boundary_related=True, layout="band"),
    ]

    # FD-aligned receiver components: velocity-stress evolves velocities natively: FD-name aliases only.
    derived_receiver_types: ClassVar[dict] = {"vx": ("v_x", None), "vy": ("v_y", None), "vz": ("v_z", None)}

    mesh: Hex8Mesh
    vp_e: torch.Tensor
    vs_e: torch.Tensor
    rho_e: torch.Tensor

    mass_rho: torch.Tensor             # (n_global,) — Σ_e ρ w₃ |J|
    _inv_M: torch.Tensor
    cpml: Optional[SimpleElasticCPML3DProfile]

    _D: torch.Tensor                   # (ngll, ngll)
    _lam: torch.Tensor                 # (E, g, g, g)
    _mu: torch.Tensor
    _lam2mu: torch.Tensor
    _w3: torch.Tensor                  # (g, g, g) GLL weight product

    # The C-PML builder this equation's ``cpml=`` argument expects -- same
    # declaration the 2-D equations carry, so PropSEM's routing is uniform.
    cpml_builder = staticmethod(build_simple_elastic_cpml_3d)
    # divergence weights w₃·|J|·(2/d_j)
    _wHx: torch.Tensor
    _wHy: torch.Tensor
    _wHz: torch.Tensor
    _inv_half_dx: float
    _inv_half_dy: float
    _inv_half_dz: float

    # The C-PML band split (with a ``cpml``; see sweep.sem.cpml.band): the
    # elements a C-PML coefficient reaches, ascending -- the rows of the 18 band
    # memory fields -- and the interior ones; the a/b coefficients on the band,
    # and each part's divergence geometry (see _part_geometry).
    band_elements: Optional[torch.Tensor] = None
    interior_elements: Optional[torch.Tensor] = None
    _band: Optional[dict] = None
    _parts: Optional[dict] = None

    @classmethod
    def build(
        cls,
        mesh: Hex8Mesh,
        vp_e: torch.Tensor,
        vs_e: torch.Tensor,
        rho_e: torch.Tensor,
        *,
        cpml: Optional[SimpleElasticCPML3DProfile] = None,
        band_split: Optional[bool] = None,
    ) -> "ElasticVSSEM3D":
        """``band_split`` runs the C-PML terms on the absorbing band elements only
        (``None``: when the mesh is large enough to pay; see
        :mod:`sweep.sem.cpml.band`)."""
        check_element_arrays(mesh, vp_e=vp_e, vs_e=vs_e, rho_e=rho_e)
        device, dtype = vp_e.device, vp_e.dtype
        w = mesh.w_ref
        w3 = w[:, None, None] * w[None, :, None] * w[None, None, :]

        mu = rho_e * vs_e * vs_e
        lam = rho_e * (vp_e * vp_e - 2.0 * vs_e * vs_e)
        lam2mu = lam + 2.0 * mu

        m_e = w3 * rho_e * mesh.jacobian_det
        mass_rho = torch.zeros(mesh.n_global, dtype=dtype, device=device)
        mass_rho.index_add_(0, mesh.ibool.reshape(-1), m_e.reshape(-1))

        jac = mesh.jacobian_det
        wHx = w3 * (jac * 2.0 / mesh.dx_e)
        wHy = w3 * (jac * 2.0 / mesh.dy_e)
        wHz = w3 * (jac * 2.0 / mesh.dz_e)
        return cls(
            mesh=mesh, vp_e=vp_e, vs_e=vs_e, rho_e=rho_e,
            mass_rho=mass_rho, _inv_M=1.0 / mass_rho, cpml=cpml,
            _D=mesh.deriv, _lam=lam, _mu=mu, _lam2mu=lam2mu, _w3=w3,
            _wHx=wHx, _wHy=wHy, _wHz=wHz,
            _inv_half_dx=2.0 / mesh.dx_e,
            _inv_half_dy=2.0 / mesh.dy_e,
            _inv_half_dz=2.0 / mesh.dz_e,
        )._with_band_split(band_split)

    # ------------------------------------------------------------------ #
    # The C-PML band split (sweep.sem.cpml.band).
    # ------------------------------------------------------------------ #
    def _with_band_split(self, split=None):
        """Set the band split fields from ``self.cpml`` (none without one).
        Unsplit, the band is every element and ``_parts`` is ``None``."""
        cp = self.cpml
        if cp is None:
            return self
        names = ("a_x", "b_x", "a_y", "b_y", "a_z", "b_z")
        band, inner = element_split(active_elements(
            [(getattr(cp, n), 0.0) for n in names], self.mesh.n_elements), split)
        self.band_elements, self.interior_elements = band, inner
        if inner.numel() == 0:
            self._band = {n: getattr(cp, n) for n in names}
            self._parts = None
            return self
        self._band = {n: getattr(cp, n)[band] for n in names}
        self._parts = {"band": self._part_geometry(band), "inner": self._part_geometry(inner)}
        return self

    def _part_geometry(self, rows):
        """What :meth:`assemble_stress_divergence` needs on the elements ``rows``:
        their node ids (the weights are per-node constants on this mesh)."""
        return {"ibool": self.mesh.ibool[rows]}

    # ------------------------------------------------------------------ #
    # Strong-form velocity gradients (element-local) — for the σ update.
    # Returns the nine ∂_k v_l in a fixed order.
    # ------------------------------------------------------------------ #
    def velocity_gradients(self, vx_g, vy_g, vz_g):
        # Separate one-wide reference-derivative einsums (one velocity component
        # at a time) rather than a 3-field stacked einsum per direction — ~1.15x
        # faster on the GPU (a narrow einsum beats a fat one here; see the
        # acoustic3d._cpml_grads note). Return order matches _stress_increment.
        D = self._D
        ib = self.mesh.ibool
        vx_e, vy_e, vz_e = vx_g[ib], vy_g[ib], vz_g[ib]
        dx, dy, dz = self._inv_half_dx, self._inv_half_dy, self._inv_half_dz
        ex = lambda F: contract("xX,ezyX->ezyx", D, F)
        ey = lambda F: contract("yY,ezYx->ezyx", D, F)
        ez = lambda F: contract("zZ,eZyx->ezyx", D, F)
        return (ex(vx_e) * dx, ey(vy_e) * dy, ez(vz_e) * dz,   # gvx_x, gvy_y, gvz_z
                ex(vy_e) * dx, ey(vx_e) * dy, ex(vz_e) * dx,   # gvy_x, gvx_y, gvz_x
                ez(vx_e) * dz, ey(vz_e) * dy, ez(vy_e) * dz)   # gvx_z, gvz_y, gvy_z

    # ------------------------------------------------------------------ #
    # Weak-form stress divergence (assembled to global) — for the v update.
    # R_i = ∫ ∂φ/∂x_j · σ_ij dV ; arguments are the (CPML-stretched) σ fields
    # paired with their derivative direction.
    # ------------------------------------------------------------------ #
    def assemble_stress_divergence(self, sxx_x, sxy_y, sxz_z,
                                   sxy_x, syy_y, syz_z,
                                   sxz_x, syz_y, szz_z, part=None):
        """``part`` (``"band"`` / ``"inner"``): the stresses are on those
        elements only (the band split), and so is the assembly."""
        D = self._D
        dtx = lambda F: contract("Xx,ezyX->ezyx", D, F)
        dty = lambda F: contract("Yy,ezYx->ezyx", D, F)
        dtz = lambda F: contract("Zz,eZyx->ezyx", D, F)
        wHx, wHy, wHz = self._wHx, self._wHy, self._wHz
        R_x = dtx(wHx * sxx_x) + dty(wHy * sxy_y) + dtz(wHz * sxz_z)
        R_y = dtx(wHx * sxy_x) + dty(wHy * syy_y) + dtz(wHz * syz_z)
        R_z = dtx(wHx * sxz_x) + dty(wHy * syz_y) + dtz(wHz * szz_z)
        like = sxx_x.new_zeros(self.mesh.n_global)
        ibool = self.mesh.ibool if part is None else self._parts[part]["ibool"]
        return assemble_each([R_x, R_y, R_z], ibool, like)

    def _stress_increment(self, g, dt, rows=None):
        """``dσ = dt·C:∇v`` (isotropic). ``g`` = the 9-tuple from
        :meth:`velocity_gradients` (already CPML-stretched by the caller), or
        on the elements ``rows`` only -- the band memories' share, which the
        step adds to the bare increment (this is linear in ``g``)."""
        gvx_x, gvy_y, gvz_z, gvy_x, gvx_y, gvz_x, gvx_z, gvz_y, gvy_z = g
        l, m, l2 = self._lam, self._mu, self._lam2mu
        if rows is not None:
            l, m, l2 = l[rows], m[rows], l2[rows]
        inc_xx = dt * (l2 * gvx_x + l * (gvy_y + gvz_z))
        inc_yy = dt * (l2 * gvy_y + l * (gvx_x + gvz_z))
        inc_zz = dt * (l2 * gvz_z + l * (gvx_x + gvy_y))
        inc_xy = dt * m * (gvy_x + gvx_y)
        inc_xz = dt * m * (gvz_x + gvx_z)
        inc_yz = dt * m * (gvz_y + gvy_z)
        return inc_xx, inc_yy, inc_zz, inc_xy, inc_xz, inc_yz

    # ================================================================== #
    # SEMPropagator interface (staggered leapfrog).
    # ================================================================== #
    def step(self, fields: tuple, src, dt: float) -> tuple:
        (vx, vy, vz,
         sxx, syy, szz, sxy, sxz, syz,
         m_sxx_x, m_sxy_y, m_sxz_z, m_sxy_x, m_syy_y, m_syz_z,
         m_sxz_x, m_syz_y, m_szz_z,
         m_vx_x, m_vy_y, m_vz_z, m_vy_x, m_vx_y, m_vz_x,
         m_vx_z, m_vz_y, m_vy_z) = fields
        cp = self.cpml
        use_cpml = cp is not None
        inv_M = self._inv_M
        b_x, b_y, b_z = src[0], src[1], src[2]

        # ----- velocity update from current σ (σ-field CPML) -----
        # The memories live on the band rows (sweep.sem.cpml.band): the band
        # elements take the stretched stress, the interior the bare one.
        if use_cpml:
            cb, eb, ei = self._band, self.band_elements, self.interior_elements
            split = self._parts is not None
            pick = (lambda t: t[eb]) if split else (lambda t: t)
            ax, bx, ay, by, az, bz = (cb[n] for n in ("a_x", "b_x", "a_y", "b_y", "a_z", "b_z"))
            # m = a*m + b*s fused via addcmul (mul + fused-multiply-add = 2 kernels
            # vs 3) — ~1.08x on the full elastic step, the recurrence block alone
            # ~1.25x; memory-bound elementwise dominates the 27-field step.
            ac = torch.addcmul
            sb = [pick(t) for t in (sxx, syy, szz, sxy, sxz, syz)]
            sxx_b, syy_b, szz_b, sxy_b, sxz_b, syz_b = sb
            m_sxx_x = ac(ax * m_sxx_x, bx, sxx_b)
            m_sxy_y = ac(ay * m_sxy_y, by, sxy_b)
            m_sxz_z = ac(az * m_sxz_z, bz, sxz_b)
            m_sxy_x = ac(ax * m_sxy_x, bx, sxy_b)
            m_syy_y = ac(ay * m_syy_y, by, syy_b)
            m_syz_z = ac(az * m_syz_z, bz, syz_b)
            m_sxz_x = ac(ax * m_sxz_x, bx, sxz_b)
            m_syz_y = ac(ay * m_syz_y, by, syz_b)
            m_szz_z = ac(az * m_szz_z, bz, szz_b)
            R_b = self.assemble_stress_divergence(
                sxx_b + m_sxx_x, sxy_b + m_sxy_y, sxz_b + m_sxz_z,
                sxy_b + m_sxy_x, syy_b + m_syy_y, syz_b + m_syz_z,
                sxz_b + m_sxz_x, syz_b + m_syz_y, szz_b + m_szz_z,
                part="band" if split else None)
            if split:
                si = [t[ei] for t in (sxx, syy, szz, sxy, sxz, syz)]
                sxx_i, syy_i, szz_i, sxy_i, sxz_i, syz_i = si
                R_i = self.assemble_stress_divergence(
                    sxx_i, sxy_i, sxz_i, sxy_i, syy_i, syz_i, sxz_i, syz_i, szz_i,
                    part="inner")
                R_x, R_y, R_z = (a + b for a, b in zip(R_i, R_b))
            else:
                R_x, R_y, R_z = R_b
        else:
            R_x, R_y, R_z = self.assemble_stress_divergence(
                sxx, sxy, sxz, sxy, syy, syz, sxz, syz, szz)
        vx = vx + dt * inv_M * (-R_x + b_x)
        vy = vy + dt * inv_M * (-R_y + b_y)
        vz = vz + dt * inv_M * (-R_z + b_z)

        # ----- stress update from new v (velocity-gradient CPML) -----
        # Split: the bare increment on every element, plus -- the update being
        # linear in ∇v -- the memories' share on the band rows. Unsplit: the
        # stretched gradients everywhere, as before the split.
        g = self.velocity_gradients(vx, vy, vz)
        if use_cpml:
            gvx_x, gvy_y, gvz_z, gvy_x, gvx_y, gvz_x, gvx_z, gvz_y, gvy_z = (pick(t) for t in g)
            m_vx_x = ac(ax * m_vx_x, bx, gvx_x)
            m_vy_y = ac(ay * m_vy_y, by, gvy_y)
            m_vz_z = ac(az * m_vz_z, bz, gvz_z)
            m_vy_x = ac(ax * m_vy_x, bx, gvy_x)
            m_vx_y = ac(ay * m_vx_y, by, gvx_y)
            m_vz_x = ac(ax * m_vz_x, bx, gvz_x)
            m_vx_z = ac(az * m_vx_z, bz, gvx_z)
            m_vz_y = ac(ay * m_vz_y, by, gvz_y)
            m_vy_z = ac(az * m_vy_z, bz, gvy_z)
            mv = (m_vx_x, m_vy_y, m_vz_z, m_vy_x, m_vx_y, m_vz_x, m_vx_z, m_vz_y, m_vy_z)
            if not split:
                g = tuple(a + b for a, b in zip(g, mv))
        inc = self._stress_increment(g, dt)
        sxx = sxx + inc[0]; syy = syy + inc[1]; szz = szz + inc[2]
        sxy = sxy + inc[3]; sxz = sxz + inc[4]; syz = syz + inc[5]
        if use_cpml and split:
            c = self._stress_increment(mv, dt, rows=eb)
            sxx = sxx.index_add(0, eb, c[0]); syy = syy.index_add(0, eb, c[1])
            szz = szz.index_add(0, eb, c[2]); sxy = sxy.index_add(0, eb, c[3])
            sxz = sxz.index_add(0, eb, c[4]); syz = syz.index_add(0, eb, c[5])

        return (vx, vy, vz,
                sxx, syy, szz, sxy, sxz, syz,
                m_sxx_x, m_sxy_y, m_sxz_z, m_sxy_x, m_syy_y, m_syz_z,
                m_sxz_x, m_syz_y, m_szz_z,
                m_vx_x, m_vy_y, m_vz_z, m_vy_x, m_vx_y, m_vz_x,
                m_vx_z, m_vz_y, m_vy_z)

    # ================================================================== #
    # Boundary-saving hooks ('substep' reverse — undo the leapfrog at −dt
    # with CPML memory zeroed; exact in the lossless interior, natural BC).
    # ================================================================== #
    bs_ring_save: ClassVar[list] = [0, 1, 2]
    bs_ring_restore: ClassVar[list] = [(0, 0, 0), (1, 1, 0), (2, 2, 0)]
    bs_model_attrs: ClassVar[tuple] = ("_lam", "_mu", "_lam2mu")

    def reverse_step(self, fields, src, dt):
        vx, vy, vz = fields[0], fields[1], fields[2]
        sxx, syy, szz, sxy, sxz, syz = fields[3:9]
        inv_M = self._inv_M
        b_x, b_y, b_z = src[0], src[1], src[2]
        # undo σ update (no CPML stretch): σ_old = σ_new − dt·C:∇(v_new)
        g = self.velocity_gradients(vx, vy, vz)
        inc = self._stress_increment(g, dt)
        sxx = sxx - inc[0]; syy = syy - inc[1]; szz = szz - inc[2]
        sxy = sxy - inc[3]; sxz = sxz - inc[4]; syz = syz - inc[5]
        # undo v update: v_old = v_new − dt·inv_M·(−R(σ_old) + b)
        R_x, R_y, R_z = self.assemble_stress_divergence(
            sxx, sxy, sxz, sxy, syy, syz, sxz, syz, szz)
        vx = vx - dt * inv_M * (-R_x + b_x)
        vy = vy - dt * inv_M * (-R_y + b_y)
        vz = vz - dt * inv_M * (-R_z + b_z)
        z = torch.zeros_like(fields[9])            # the band memory layout
        return (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
                *(z for _ in range(18)))

    def bs_model_tensors(self) -> dict:
        d = {a: getattr(self, a) for a in self.bs_model_attrs}
        d["mass_rho"] = self.mass_rho
        return d

    def bs_recompute(self, fields, src, dt, leaves):
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
