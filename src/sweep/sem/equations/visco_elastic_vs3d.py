"""1st-order velocity-stress visco-elastic SEM in 3-D (GSLS, nearly constant Q).

:class:`ElasticVSSEM3D` (and :class:`ElasticVSSEM3DDeformed`) with the rheology of
:class:`~sweep.sem.equations.visco_elastic_vs_deformed.ViscoElasticVSSEM2DDeformed`, the FD
:class:`sweep.equations.ViscoElastic`'s and SPECFEM's: ``L`` standard linear solids in
parallel whose relaxation times ``tau_sigma_l`` are fixed by the band ``f_band``, each with the
strength ``tau_l(Q)`` of the least-squares constant-Q fit. At every element GLL node, with the
unrelaxed P and shear moduli ``pi_U``, ``mu_U``, the relaxed ``pi_R``, ``mu_R``,
``tp_l = tau_l(Qp)``, ``ts_l = tau_l(Qs)`` and ``theta = div v``::

    dsxx/dt = pi_U theta - 2 mu_U (dvy/dy + dvz/dz) + sum_l r_xx_l
    dsxy/dt = mu_U (dvx/dy + dvy/dx)               + sum_l r_xy_l
    dr_xx_l/dt = -(r_xx_l + pi_R tp_l theta - 2 mu_R ts_l (dvy/dy + dvz/dz)) / tau_sigma_l
    dr_xy_l/dt = -(r_xy_l + mu_R ts_l (dvx/dy + dvy/dx)) / tau_sigma_l

(``yy``, ``zz``, ``xz``, ``yz`` alike), ``pi_U = pi_R (1 + sum_l tp_l)``, ``mu_U = mu_R (1 +
sum_l ts_l)``. Six element-local memories per mechanism advance with the trapezoidal rule
(stable for any ``dt / tau_sigma_l``) and the stress takes their time average, as in 2-D. The
free surface stays the natural one, and in the C-PML the memories see the stretched strain
rate, as the stress does. The C-PML terms run on every element (no band split): the split
would need the memories' share of the stretched strain rate on the band rows.

Models ``vp, vs, rho, Qp, Qs``, the 2-D class's ``f_ref``, ``f_band``, ``n_sls``, ``n_fit`` and
``velocities``, and all five models differentiable; element arrays of ``Q = inf`` give the
elastic step bit for bit. The explicit stress update runs at the unrelaxed velocities: take
``dt`` from those (:meth:`ViscoElasticVSSEM3D.unrelaxed_velocities`). Eager only; no boundary
saving, since the attenuation is dissipative.
"""

from __future__ import annotations

from typing import ClassVar, Optional

import torch

from .._assembly import assemble_each
from ..cpml.elastic_vs3d import SimpleElasticCPML3DProfile
from ..fields import FieldSpec, check_element_arrays
from ..mesh.hex27 import Hex27Mesh
from ..mesh.hex8 import Hex8Mesh
from .elastic_vs3d import ElasticVSSEM3D
from .elastic_vs3d_deformed import ElasticVSSEM3DDeformed
from .visco_elastic_vs_deformed import gsls_setup, gsls_weights

_COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")


def gsls_memory_specs3(n_sls):
    """The ``6 n_sls`` element-local memories, ``r_xx_l ... r_yz_l`` per mechanism."""
    return [FieldSpec(f"r_{c}_{l}", internal=True, layout="element")
            for l in range(n_sls) for c in _COMPONENTS]


def gsls_memories3(mem, p_tau, m_tau, weights, g):
    """The 3-D memories one step on: ``mem`` is ``(r_xx_l, ..., r_yz_l)`` per mechanism,
    ``p_tau`` / ``m_tau`` the relaxed P / shear moduli times the strengths and ``g`` the nine
    velocity gradients in :meth:`ElasticVSSEM3D.velocity_gradients`' order. Returns the new
    memories and the sums ``r+ + r`` per stress component, whose ``dt / 2`` multiple the
    stress takes (their time average)."""
    a_l, c_l = weights
    vx_x, vy_y, vz_z, vy_x, vx_y, vz_x, vx_z, vz_y, vy_z = g
    theta = vx_x + vy_y + vz_z
    shear = (vx_y + vy_x, vx_z + vz_x, vy_z + vz_y)
    new_mem = []
    sums = [0.0] * 6
    for l in range(len(p_tau)):
        r = mem[6 * l:6 * l + 6]
        tp, ts = p_tau[l] * theta, m_tau[l]
        forcing = (tp - 2.0 * ts * (vy_y + vz_z), tp - 2.0 * ts * (vx_x + vz_z),
                   tp - 2.0 * ts * (vx_x + vy_y), ts * shear[0], ts * shear[1], ts * shear[2])
        for k in range(6):
            n = a_l[l] * r[k] - c_l[l] * forcing[k]
            sums[k] = sums[k] + (n + r[k])
            new_mem.append(n)
    return new_mem, sums


class ViscoElasticVSSEM3D(ElasticVSSEM3D):
    """Visco-elastic (GSLS) velocity-stress SEM on Hex8: the elastic state, then ``6 L``
    memories ``r_xx_l ... r_yz_l``. See the module docstring."""

    # the elastic class whose build gives the geometry, the mass and the C-PML
    _elastic: ClassVar[type] = ElasticVSSEM3D
    # PropSEM refuses memory=BoundarySaving() with this reason
    bs_refusal: ClassVar[str] = (
        "ViscoElasticVSSEM3D has no boundary saving: the attenuation is dissipative, so "
        "stepping back amplifies round-off and the reverse reconstruction does not exist. "
        "Use memory=Full() or Ckpt(chunks=...).")

    @classmethod
    def build(
        cls,
        mesh: Hex8Mesh,
        vp_e: torch.Tensor,
        vs_e: torch.Tensor,
        rho_e: torch.Tensor,
        Qp_e: torch.Tensor,
        Qs_e: torch.Tensor,
        *,
        cpml: Optional[SimpleElasticCPML3DProfile] = None,
        f_ref: float = 10.0,
        f_band: Optional[tuple] = None,
        n_sls: int = 3,
        n_fit: int = 200,
        velocities: str = "phase",
    ) -> "ViscoElasticVSSEM3D":
        """Build the equation: the arguments of
        :meth:`~sweep.sem.equations.visco_elastic_vs_deformed.ViscoElasticVSSEM2DDeformed.build`
        on a 3-D mesh."""
        return cls._build(mesh, vp_e, vs_e, rho_e, Qp_e, Qs_e, cpml=cpml, f_ref=f_ref,
                          f_band=f_band, n_sls=n_sls, n_fit=n_fit, velocities=velocities)

    @classmethod
    def _build(cls, mesh, vp_e, vs_e, rho_e, Qp_e, Qs_e, *, cpml, f_ref, f_band, n_sls, n_fit,
               velocities):
        check_element_arrays(mesh, Qp_e=Qp_e, Qs_e=Qs_e)
        g = gsls_setup(Qp_e, Qs_e, f_ref, f_band, n_sls, cls.__name__, n_fit=n_fit,
                       velocities=velocities)
        (sU_p, sR_p), (sU_s, sR_s) = g.p, g.s
        # The elastic build on the unrelaxed velocities: mass, geometry, the C-PML and the
        # unrelaxed Lame set (bit for bit the elastic one at Q = inf).
        eq = cls._elastic.build(mesh, vp_e * sU_p.sqrt(), vs_e * sU_s.sqrt(), rho_e, cpml=cpml,
                                band_split=False)
        eq.__class__ = cls
        pi_R = rho_e * vp_e * vp_e * sR_p
        mu_R = rho_e * vs_e * vs_e * sR_s
        eq.Qp_e, eq.Qs_e = Qp_e, Qs_e
        eq.f_ref, eq.f_band, eq.n_sls, eq.q_min = g.f_ref, g.f_band, g.n_sls, g.q_min
        eq.n_fit, eq.velocities = int(n_fit), velocities
        eq._tau_sigma = g.tau_sigma
        eq._p_tau = tuple(pi_R * t for t in g.tp)
        eq._m_tau = tuple(mu_R * t for t in g.ts)
        eq.field_specs = ElasticVSSEM3D.field_specs + gsls_memory_specs3(g.n_sls)
        return eq

    def unrelaxed_velocities(self):
        """``(vp_U, vs_U)``, ``(E, ngll, ngll, ngll)``: the velocities the explicit stress
        update runs at -- take ``dt`` from the CFL condition on ``vp_U``."""
        return self.vp_e, self.vs_e

    @property
    def reverse_step(self):
        # no reverse reconstruction (see bs_refusal): hasattr() is False, and a direct
        # run_boundary_saved refuses the equation
        raise AttributeError(self.bs_refusal)

    def step(self, fields: tuple, src, dt: float) -> tuple:
        """One staggered-leapfrog step (v at n+1/2, σ and the memories at n+1)."""
        n = len(ElasticVSSEM3D.field_specs)
        (vx, vy, vz,
         sxx, syy, szz, sxy, sxz, syz,
         m_sxx_x, m_sxy_y, m_sxz_z, m_sxy_x, m_syy_y, m_syz_z,
         m_sxz_x, m_syz_y, m_szz_z,
         m_vx_x, m_vy_y, m_vz_z, m_vy_x, m_vx_y, m_vz_x,
         m_vx_z, m_vz_y, m_vy_z) = fields[:n]
        mem = fields[n:]
        cp = self.cpml
        use_cpml = cp is not None
        inv_M = self._inv_M
        b_x, b_y, b_z = src[0], src[1], src[2]

        # ----- velocity update from the current σ (σ-derivative CPML, every element) -----
        # the parent's, unsplit: the recursion on the assembled divergence split by
        # derivative direction, its stretch added once per node
        R_x, R_y, R_z = self.assemble_stress_divergence(
            sxx, sxy, sxz, sxy, syy, syz, sxz, syz, szz)
        if use_cpml:
            cb, ib = self._band, self.mesh.ibool
            ax, bx, ay, by, az, bz = (cb[k] for k in ("a_x", "b_x", "a_y", "b_y", "a_z", "b_z"))
            ac = torch.addcmul
            P = torch.stack(self._divergence_parts(
                [sxx, sxy, sxz, sxy, syy, syz, sxz, syz, szz], False), -1)
            P = P.new_zeros(self.mesh.n_global, 9).index_add(
                0, ib.reshape(-1), P.reshape(-1, 9))[ib]
            m_sxx_x = ac(ax * m_sxx_x, bx, P[..., 0])
            m_sxy_y = ac(ay * m_sxy_y, by, P[..., 1])
            m_sxz_z = ac(az * m_sxz_z, bz, P[..., 2])
            m_sxy_x = ac(ax * m_sxy_x, bx, P[..., 3])
            m_syy_y = ac(ay * m_syy_y, by, P[..., 4])
            m_syz_z = ac(az * m_syz_z, bz, P[..., 5])
            m_sxz_x = ac(ax * m_sxz_x, bx, P[..., 6])
            m_syz_y = ac(ay * m_syz_y, by, P[..., 7])
            m_szz_z = ac(az * m_szz_z, bz, P[..., 8])
            f = cb["first"]
            Q = assemble_each([f * (m_sxx_x + m_sxy_y + m_sxz_z),
                               f * (m_sxy_x + m_syy_y + m_syz_z),
                               f * (m_sxz_x + m_syz_y + m_szz_z)], ib, R_x)
            R_x, R_y, R_z = R_x + Q[0], R_y + Q[1], R_z + Q[2]
        vx = vx + dt * inv_M * (-R_x + b_x)
        vy = vy + dt * inv_M * (-R_y + b_y)
        vz = vz + dt * inv_M * (-R_z + b_z)

        # ----- stress and memories from the new v (stretched gradients in the C-PML) -----
        g = self.velocity_gradients(vx, vy, vz)
        if use_cpml:
            gvx_x, gvy_y, gvz_z, gvy_x, gvx_y, gvz_x, gvx_z, gvz_y, gvy_z = g
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
            g = tuple(a + b for a, b in zip(g, mv))
        inc = self._stress_increment(g, dt)
        new_mem, sums = gsls_memories3(mem, self._p_tau, self._m_tau,
                                       gsls_weights(self._tau_sigma, dt), g)
        half_dt = 0.5 * dt
        sxx = sxx + inc[0] + half_dt * sums[0]
        syy = syy + inc[1] + half_dt * sums[1]
        szz = szz + inc[2] + half_dt * sums[2]
        sxy = sxy + inc[3] + half_dt * sums[3]
        sxz = sxz + inc[4] + half_dt * sums[4]
        syz = syz + inc[5] + half_dt * sums[5]

        return (vx, vy, vz,
                sxx, syy, szz, sxy, sxz, syz,
                m_sxx_x, m_sxy_y, m_sxz_z, m_sxy_x, m_syy_y, m_syz_z,
                m_sxz_x, m_syz_y, m_szz_z,
                m_vx_x, m_vy_y, m_vz_z, m_vy_x, m_vx_y, m_vz_x,
                m_vx_z, m_vz_y, m_vy_z, *new_mem)


class ViscoElasticVSSEM3DDeformed(ViscoElasticVSSEM3D, ElasticVSSEM3DDeformed):
    """:class:`ViscoElasticVSSEM3D` on a deformed Hex27 mesh (a topography): its step on the
    geometry of :class:`ElasticVSSEM3DDeformed`."""

    _elastic: ClassVar[type] = ElasticVSSEM3DDeformed
    bs_refusal: ClassVar[str] = ViscoElasticVSSEM3D.bs_refusal.replace(
        "ViscoElasticVSSEM3D", "ViscoElasticVSSEM3DDeformed")

    @classmethod
    def build(
        cls,
        mesh: Hex27Mesh,
        vp_e: torch.Tensor,
        vs_e: torch.Tensor,
        rho_e: torch.Tensor,
        Qp_e: torch.Tensor,
        Qs_e: torch.Tensor,
        *,
        cpml: Optional[SimpleElasticCPML3DProfile] = None,
        f_ref: float = 10.0,
        f_band: Optional[tuple] = None,
        n_sls: int = 3,
        n_fit: int = 200,
        velocities: str = "phase",
    ) -> "ViscoElasticVSSEM3DDeformed":
        """Build the equation: :meth:`ViscoElasticVSSEM3D.build` on a Hex27 mesh."""
        return cls._build(mesh, vp_e, vs_e, rho_e, Qp_e, Qs_e, cpml=cpml, f_ref=f_ref,
                          f_band=f_band, n_sls=n_sls, n_fit=n_fit, velocities=velocities)
