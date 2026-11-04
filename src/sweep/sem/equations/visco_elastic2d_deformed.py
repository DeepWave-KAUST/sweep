"""2nd-order displacement visco-elastic SEM on Quad9: :class:`ElasticSEM2DDeformed` with
SPECFEM2D's ``ATTENUATION_VISCOELASTIC`` as its Newmark path runs it.

``L`` standard linear solids relax the bulk modulus ``kappa = lambda + mu`` (2-D) and the shear
modulus, with relaxation times fixed over ``f_band`` and strengths ``w_l`` fitted for constant
``Qkappa`` and ``Qmu`` there (:class:`sweep.equations.visco_elastic.GSLSFit`). At every element
GLL node, from the current and the last step's strains ``theta = dux/dx + duz/dz``,
``dev = dux/dx - theta / 2`` and ``sh = dux/dz + duz/dx``::

    e1_l  = A_l^2 e1_l  + B1_l (theta + A_l theta_old),   A_l = exp(-dt / (2 tau_sigma_l)),
    e11_l = A_l^2 e11_l + B2_l (dev + A_l dev_old),         B_l = -w_l (1 - A_l) / (1 + sum w)
    e13_l = A_l^2 e13_l + B2_l (sh + A_l sh_old)

and the stress gains ``(lambda + mu)_U sum e1 + 2 mu_U sum e11`` (xx), ``... - 2 mu_U sum e11``
(zz) and ``mu_U sum e13`` (xz): SPECFEM2D's ``compute_attenuation_viscoelastic`` (its
``phi_l tau_sigma_l (1 - A_l)`` is ``B_l``) and ``compute_forces_viscoelastic``. The models are
SPECFEM2D's, ``vp, vs, rho, Qkappa, Qmu``, and so are the defaults: the band
``f_ref / 10 .. 10 f_ref``, the fit at ``2 n_sls - 1`` frequencies, and ``vp``, ``vs`` read as
the unrelaxed velocities (``velocities="unrelaxed"``; ``"real_modulus"`` is its
``READ_VELOCITIES_AT_f0``, which sets the real part of the bulk and of the shear modulus at
``f_ref``). ``Q = inf`` gives ``w_l = 0`` and the elastic step.

The absorbing layer runs without attenuation: the strengths are zero on its elements (those
with an interior node the C-PML damps; an edge node on the layer's inner boundary can carry a
round-off damping), so their memories stay zero. SPECFEM2D skips the memory update in its PML
elements too, and runs its visco-elastic C-PML there instead. Eager only; no boundary saving,
since the attenuation is dissipative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import ClassVar, Optional

import torch

from ..cpml.elastic import ElasticCPMLProfile
from ..fields import FieldSpec, check_element_arrays
from ..mesh.domains import DomainMask
from ..mesh.quad9 import Quad9Mesh
from .elastic2d_deformed import ElasticSEM2DDeformed
from .visco_elastic_vs_deformed import gsls_setup


@dataclass
class ViscoElasticSEM2DDeformed(ElasticSEM2DDeformed):
    """Visco-elastic displacement SEM (SPECFEM2D's attenuation): the elastic state, then the
    element memories ``e1_l, e11_l, e13_l`` and the last step's strains. See the module
    docstring."""

    # PropSEM refuses memory=BoundarySaving() with this reason
    bs_refusal: ClassVar[str] = (
        "ViscoElasticSEM2DDeformed has no boundary saving: the attenuation is dissipative, so "
        "stepping back amplifies round-off and the reverse reconstruction does not exist. "
        "Use memory=Full() or Ckpt(chunks=...).")

    Qk_e: Optional[torch.Tensor] = None
    Qmu_e: Optional[torch.Tensor] = None
    tau_sigma: tuple = ()
    _w1: tuple = ()                # bulk strengths, (E, ngll, ngll) each, zero in the PML
    _w2: tuple = ()                # shear strengths
    _kappa_U: Optional[torch.Tensor] = None  # (lambda + mu)_U, the elastic step's
    _mu_U: Optional[torch.Tensor] = None
    f_ref: float = 10.0
    f_band: tuple = ()
    n_sls: int = 3
    n_fit: int = 5
    velocities: str = "unrelaxed"

    @classmethod
    def build(cls, mesh: Quad9Mesh, vp_e, vs_e, rho_e, Qk_e, Qmu_e, *,
              cpml: Optional[ElasticCPMLProfile] = None,
              domain: Optional[DomainMask] = None,
              band_split: Optional[bool] = None,
              f_ref: float = 10.0,
              f_band: Optional[tuple] = None,
              n_sls: int = 3,
              n_fit: Optional[int] = None,
              velocities: str = "unrelaxed") -> "ViscoElasticSEM2DDeformed":
        """Build the equation: ``vp_e``, ``vs_e`` read at ``f_ref`` as ``velocities`` says
        (``"unrelaxed"`` or ``"real_modulus"``), ``rho_e``, and the bulk and shear quality
        factors ``Qk_e``, ``Qmu_e``, fitted constant over ``f_band`` (default
        ``f_ref / 10 .. 10 f_ref``) with ``n_sls`` mechanisms at ``n_fit`` frequencies
        (default ``2 n_sls - 1``)."""
        if velocities not in ("unrelaxed", "real_modulus"):
            raise ValueError(
                f"velocities must be 'unrelaxed' or 'real_modulus' (SPECFEM2D's readings, "
                f"applied to the bulk and the shear modulus each), got {velocities!r}")
        check_element_arrays(mesh, Qk_e=Qk_e, Qmu_e=Qmu_e)
        f_band = (f_ref / 10.0, 10.0 * f_ref) if f_band is None else f_band
        n_fit = 2 * int(n_sls) - 1 if n_fit is None else int(n_fit)
        g = gsls_setup(Qk_e, Qmu_e, f_ref, f_band, n_sls, cls.__name__, n_fit=n_fit,
                       velocities=velocities)
        # the unrelaxed velocities, as factors on vp and vs (exactly 1 where Q = inf)
        mu = rho_e * vs_e * vs_e
        kappa = rho_e * vp_e * vp_e - mu                    # 2-D: kappa = lambda + mu
        s_p = (kappa * g.p[0] + mu * g.s[0]) / (kappa + mu)
        base = ElasticSEM2DDeformed.build(mesh, vp_e * s_p.sqrt(), vs_e * g.s[0].sqrt(), rho_e,
                                          cpml=cpml, domain=domain, band_split=band_split)
        eq = cls(**{f.name: getattr(base, f.name) for f in fields(base) if f.init})
        w1, w2 = tuple(g.tp), tuple(g.ts)
        if cpml is not None:
            E = mesh.n_elements
            pml_e = cpml.node_in_pml[:, 1:-1, 1:-1].reshape(E, -1).any(1)
            off = (~pml_e).to(vp_e.dtype).view(E, 1, 1)
            w1, w2 = tuple(w * off for w in w1), tuple(w * off for w in w2)
        eq.Qk_e, eq.Qmu_e, eq.tau_sigma, eq._w1, eq._w2 = Qk_e, Qmu_e, g.tau_sigma, w1, w2
        eq._kappa_U, eq._mu_U = base._lam + base._mu, base._mu
        eq.f_ref, eq.f_band, eq.n_sls, eq.n_fit, eq.velocities = (
            g.f_ref, g.f_band, g.n_sls, n_fit, velocities)
        eq.field_specs = (ElasticSEM2DDeformed.field_specs
                          + [FieldSpec(f"{c}_{l}", internal=True, layout="element")
                             for l in range(g.n_sls) for c in ("e1", "e11", "e13")]
                          + [FieldSpec(n, internal=True, layout="element")
                             for n in ("theta_old", "dev_old", "sh_old")])
        return eq

    def step(self, fields: tuple, src, dt: float) -> tuple:
        n_base = len(ElasticSEM2DDeformed.field_specs)
        base, mem = fields[:n_base], fields[n_base:]
        e, (theta_old, dev_old, sh_old) = mem[:-3], mem[-3:]
        dux_dx, dux_dz, duz_dx, duz_dz = self._element_derivatives(base[0], base[1])
        theta = dux_dx + duz_dz
        dev = dux_dx - 0.5 * theta
        sh = dux_dz + duz_dx
        s1, s2 = 1.0 + sum(self._w1), 1.0 + sum(self._w2)
        new_e = []
        for l, ts in enumerate(self.tau_sigma):
            a = math.exp(-0.5 * dt / ts)
            b1 = -self._w1[l] * (1.0 - a) / s1
            b2 = -self._w2[l] * (1.0 - a) / s2
            e1, e11, e13 = e[3 * l:3 * l + 3]
            new_e += [a * a * e1 + b1 * (theta + a * theta_old),
                      a * a * e11 + b2 * (dev + a * dev_old),
                      a * a * e13 + b2 * (sh + a * sh_old)]
        bulk = self._kappa_U * sum(new_e[0::3])
        dvt = 2.0 * self._mu_U * sum(new_e[1::3])
        sig_xz = self._mu_U * sum(new_e[2::3])
        R_x, R_z = self._assemble_stress(bulk + dvt, bulk - dvt, sig_xz, sig_xz, base[0])
        nxt = list(super().step(base, src, dt))
        dt2 = dt * dt
        nxt[0] = nxt[0] - dt2 * self._inv_M * R_x
        nxt[1] = nxt[1] - dt2 * self._inv_M * R_z
        return (*nxt, *new_e, theta, dev, sh)
