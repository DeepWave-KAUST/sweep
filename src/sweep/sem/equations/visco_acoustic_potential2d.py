"""Visco-acoustic potential SEM on Quad9: :class:`AcousticPotentialSEM2D` with SPECFEM2D's
``ATTENUATION_VISCOACOUSTIC`` (its weak form and Newmark memory update).

``L`` standard linear solids relax the bulk modulus, their relaxation times fixed over
``f_band`` and their strengths ``t_l`` fitted for a constant ``Qkappa`` there
(:class:`sweep.equations.visco_elastic.GSLSFit`, as for the visco-elastic class). Per node,
with ``d = M_geo^-1 (-K chi)`` (the stiffness, with ``1 / rho``, over the geometric mass)::

    e_l^n  = A_l^2 e_l^(n-1) + B_l (d^n + A_l d^(n-1)),   A_l = exp(-dt / (2 tau_sigma_l)),
                                                         B_l = -t_l (1 - A_l) / (1 + sum t)
    chi''^n = M_kappa^-1 (-K chi^n + M_geo sum_l e_l^n + src)

-- SPECFEM2D's ``update_memory_var_acous_weak_form`` (``phi_l tau_sigma_l (1 - A_l)`` is
``B_l``) and ``compute_forces_viscoacoustic``. ``M_kappa`` holds the unrelaxed modulus
``kappa_U``: the memories bring the low-frequency one down to ``kappa_U / (1 + sum t)``. The
models are ``vp, rho, Qkappa``; ``vp`` is the phase velocity at ``f_ref`` by default, and
``velocities`` takes the visco-elastic class's other readings, ``"real_modulus"`` and
``"unrelaxed"`` (``vp`` the infinite-frequency velocity, which is what SPECFEM2D's input is
without ``READ_VELOCITIES_AT_f0`` -- with it, its acoustic mass matrix is not rebuilt).
``fit="nonlinear"`` fits the relaxation times with the strengths for each ``Qkappa``
(SPECFEM2D's ``USE_SOLVOPT``, :class:`sweep.equations.gsls_nonlinear.GSLSNonlinearFit`): each
node then has its own ``tau_sigma_l``, read from the same element copy as its strengths.

The memories and ``d`` are node fields, as SPECFEM2D's are. In the C-PML band ``d`` is the
stretched stiffness the step already has. ``Qkappa = inf`` gives ``t_l = 0`` and the
potential step. Eager only; no boundary saving, since the attenuation is dissipative.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar, Optional

import torch

from ..cpml.acoustic import CPMLProfile
from ..fields import FieldSpec, check_element_arrays
from ..mesh.domains import DomainMask
from ..mesh.nodes import node_field
from ..mesh.quad9 import Quad9Mesh
from .acoustic_potential2d import AcousticPotentialSEM2D
from .visco_elastic2d_deformed import _decay
from .visco_elastic_vs_deformed import gsls_setup


@dataclass
class ViscoAcousticPotentialSEM2D(AcousticPotentialSEM2D):
    """Visco-acoustic (GSLS bulk attenuation) potential SEM: the potential state, then the
    ``L`` node memories ``e_l`` and the last step's ``d``. See the module docstring."""

    # PropSEM refuses memory=BoundarySaving() with this reason
    bs_refusal: ClassVar[str] = (
        "ViscoAcousticPotentialSEM2D has no boundary saving: the attenuation is dissipative, "
        "so stepping back amplifies round-off and the reverse reconstruction does not exist. "
        "Use memory=Full() or Ckpt(chunks=...).")

    Qk_e: Optional[torch.Tensor] = None
    tau_sigma: Optional[tuple] = ()   # the linear fit's fixed relaxation times (None: fitted)
    _tau_node: tuple = ()          # the relaxation times at the nodes: floats, or tensors
    _t_node: tuple = ()            # the strengths t_l at the nodes
    _t_sum: Optional[torch.Tensor] = None
    f_ref: float = 10.0
    f_band: tuple = ()
    n_sls: int = 3
    n_fit: Optional[int] = 200
    velocities: str = "phase"
    fit: object = "linear"

    @classmethod
    def build(cls, mesh: Quad9Mesh, vp_e, rho_e, Qk_e, *,
              dirichlet: tuple[str, ...] = ("top",),
              domain: Optional[DomainMask] = None,
              cpml: Optional[CPMLProfile] = None,
              f_ref: float = 10.0,
              f_band: Optional[tuple] = None,
              n_sls: int = 3,
              n_fit: Optional[int] = 200,
              velocities: str = "phase",
              fit="linear") -> "ViscoAcousticPotentialSEM2D":
        """Build the equation: ``vp_e`` read at ``f_ref`` as ``velocities`` says, ``rho_e``,
        and ``Qk_e`` the quality factor of the bulk modulus, fitted constant over ``f_band``
        (default: a factor 12 around ``f_ref``) with ``n_sls`` mechanisms by least squares at
        ``n_fit`` log-spaced frequencies (SPECFEM2D's: ``2 n_sls - 1``, and ``4 n_sls`` for
        ``fit="nonlinear"``, which ``None`` takes)."""
        check_element_arrays(mesh, Qk_e=Qk_e)
        if n_fit is None:
            n_fit = 2 * int(n_sls) - 1 if fit == "linear" else 4 * int(n_sls)
        g = gsls_setup(Qk_e, Qk_e, f_ref, f_band, n_sls, cls.__name__, n_fit=n_fit,
                       velocities=velocities, fit=fit)
        s_U, _ = g.p
        base = AcousticPotentialSEM2D.build(mesh, vp_e * s_U.sqrt(), rho_e,
                                            dirichlet=dirichlet, domain=domain, cpml=cpml)
        eq = cls(**{f.name: getattr(base, f.name) for f in fields(base) if f.init})
        sel = (eq._mask_e.expand(vp_e.shape) > 0).reshape(-1)
        at_nodes = lambda t: (node_field(mesh.ibool, mesh.n_global, t, sel)
                              if torch.is_tensor(t) else t)
        t_node = tuple(at_nodes(t) for t in g.tp)
        eq.Qk_e, eq.tau_sigma, eq._t_node = Qk_e, g.tau_sigma, t_node
        eq._tau_node = tuple(at_nodes(t) for t in g.tau_p)
        eq._t_sum = sum(t_node)
        eq.f_ref, eq.f_band, eq.n_sls = g.f_ref, g.f_band, g.n_sls
        eq.n_fit, eq.velocities, eq.fit = int(n_fit), velocities, fit
        eq.field_specs = (AcousticPotentialSEM2D.field_specs
                          + [FieldSpec(f"e_{l}", internal=True) for l in range(g.n_sls)]
                          + [FieldSpec("d_prev", internal=True)])
        return eq

    def step(self, fields: tuple, src, dt: float) -> tuple:
        chi_now, chi_prev, psi_x, psi_z, zeta_x, zeta_z, *mem = fields
        e, d_prev = mem[:-1], mem[-1]
        dt2 = dt * dt
        if self.cpml is not None:
            w_sum, psi_x, psi_z, zeta_x, zeta_z = self.step_cpml(
                chi_now, psi_x, psi_z, zeta_x, zeta_z)
            d = w_sum
        else:
            K = self.stiffness_apply(chi_now)
            d = -K * self._inv_M_geo
        new_e = []
        for l, ts in enumerate(self._tau_node):
            a = _decay(dt, ts)
            b = -self._t_node[l] * (1.0 - a) / (1.0 + self._t_sum)
            new_e.append(a * a * e[l] + b * (d + a * d_prev))
        load = self.mass_geo * sum(new_e) + src[0]
        if self.cpml is not None:
            chi_next = (2.0 * chi_now - chi_prev + dt2 * self.kappa_node * w_sum
                        + dt2 * self._inv_M * load)
            chi_next, psi_x, psi_z, zeta_x, zeta_z = self._apply_dirichlet(
                chi_next, psi_x, psi_z, zeta_x, zeta_z)
        else:
            chi_next = 2.0 * chi_now - chi_prev + dt2 * self._inv_M * (-K + load)
            (chi_next,) = self._apply_dirichlet(chi_next)
        return (chi_next, chi_now, psi_x, psi_z, zeta_x, zeta_z, *new_e, d)
