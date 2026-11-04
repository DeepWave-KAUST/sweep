"""Biot poroelastic SEM with an attenuating frame: the GSLS on the drained moduli.

:class:`~sweep.sem.equations.biot_vs_deformed.BiotVSSEM2DDeformed` whose frame -- the
drained P modulus ``pi_d = lambda_d + 2 mu = H - C^2 / M`` and the shear modulus ``mu`` -- is
the generalized standard linear solid of
:class:`~sweep.sem.equations.visco_elastic_vs_deformed.ViscoElasticVSSEM2DDeformed`: ``Qp``
attenuates ``pi_d`` and ``Qs`` attenuates ``mu``, each fitted constant over ``f_band`` with
``n_sls`` mechanisms. ``pi_d`` and ``mu`` are the moduli at ``f_ref`` in the sense of that
class's velocities (``rho v^2`` of the phase velocity there); the pore fluid's share of
``H``, ``C^2 / M``, does not relax, and the viscous drag (with ``viscous_q``) is unchanged.
The memories are element-local, as the total stress is, and enter its frame part only
(the effective stress ``sigma + alpha p I``).

In a dry solid (``M = 0``) the class is ViscoElasticVSSEM2DDeformed to round-off, which is how
it is checked against SPECFEM2D; element arrays of ``Q = inf`` give BiotVSSEM2DDeformed bit for
bit. ``impl='c'`` runs it on the 2-D Biot kernels with the memories in the stress kernel;
no boundary saving on either backend (the drag and the frame are dissipative).
"""

from __future__ import annotations

from typing import Optional

import torch

from ..cpml.elastic_vs import SimpleElasticCPMLProfile
from ..fields import check_element_arrays
from ..mesh.quad9 import Quad9Mesh
from .biot_vs_deformed import BiotVSSEM2DDeformed
from .visco_elastic_vs_deformed import (gsls_memories, gsls_memory_specs, gsls_setup,
                                        gsls_weights)


class ViscoBiotVSSEM2DDeformed(BiotVSSEM2DDeformed):
    """Biot SEM with a GSLS frame: the Biot state, then ``3 L`` frame memories
    ``r_xx_l, r_zz_l, r_xz_l``. See the module docstring."""

    @classmethod
    def build(
        cls,
        mesh: Quad9Mesh,
        H_e: torch.Tensor,
        mu_e: torch.Tensor,
        C_e: torch.Tensor,
        M_e: torch.Tensor,
        rho_e: torch.Tensor,
        rho_f_e: torch.Tensor,
        m_e: torch.Tensor,
        b_e: torch.Tensor,
        Qp_e: torch.Tensor,
        Qs_e: torch.Tensor,
        *,
        dirichlet: tuple = (),
        cpml: Optional[SimpleElasticCPMLProfile] = None,
        viscous_q: Optional[tuple] = None,
        f_ref: float = 10.0,
        f_band: Optional[tuple] = None,
        n_sls: int = 3,
    ) -> "ViscoBiotVSSEM2DDeformed":
        """Build from the eight Biot coefficients (``H`` and ``mu`` at ``f_ref``, Hz) and the
        frame's quality factors ``Qp_e`` (drained P modulus) and ``Qs_e`` (shear), fitted
        constant over ``f_band`` (default: a factor 12 around ``f_ref``, see
        :func:`~sweep.equations.visco_elastic.gsls_band`) with
        ``n_sls`` mechanisms; ``dirichlet``, ``cpml`` and ``viscous_q`` as on Biot."""
        check_element_arrays(mesh, Qp_e=Qp_e, Qs_e=Qs_e)
        g = gsls_setup(Qp_e, Qs_e, f_ref, f_band, n_sls, cls.__name__)
        (sU_p, sR_p), (sU_s, sR_s) = g.p, g.s
        pore = M_e > 0
        fluid_share = torch.where(pore, C_e * C_e / torch.where(pore, M_e, torch.ones_like(M_e)),
                                  torch.zeros_like(M_e))
        pi_d = H_e - fluid_share
        # Biot on the unrelaxed frame, H's fluid share unchanged (bit for bit at Q = inf)
        eq = BiotVSSEM2DDeformed.build(
            mesh, H_e + pi_d * (sU_p - 1.0), mu_e * sU_s, C_e, M_e, rho_e, rho_f_e, m_e, b_e,
            dirichlet=dirichlet, cpml=cpml, viscous_q=viscous_q)
        eq.__class__ = cls
        eq.Qp_e, eq.Qs_e = Qp_e, Qs_e
        eq.f_ref, eq.f_band, eq.n_sls, eq.q_min = g.f_ref, g.f_band, g.n_sls, g.q_min
        eq._tau_sigma = g.tau_sigma
        eq._p_tau = tuple(pi_d * sR_p * t for t in g.tp)
        eq._m_tau = tuple(mu_e * sR_s * t for t in g.ts)
        eq.field_specs = BiotVSSEM2DDeformed.field_specs + gsls_memory_specs(g.n_sls)
        return eq

    def _frame_stress(self, s_xx, s_zz, s_xz, grads, div_v, dt, extra):
        """The unrelaxed frame increment, plus the time average of the memories."""
        s_xx, s_zz, s_xz, _ = super()._frame_stress(s_xx, s_zz, s_xz, grads, div_v, dt, ())
        d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz = grads
        new_mem, sum_xx, sum_zz, sum_xz = gsls_memories(
            extra, self._p_tau, self._m_tau, gsls_weights(self._tau_sigma, dt),
            d_vx_dx, d_vz_dz, d_vx_dz + d_vz_dx)
        half_dt = 0.5 * dt
        return (s_xx + half_dt * sum_xx, s_zz + half_dt * sum_zz, s_xz + half_dt * sum_xz,
                tuple(new_mem))
