"""2nd-order displacement visco-elastic SEM on Quad9: :class:`ElasticSEM2DDeformed` with
SPECFEM2D's ``ATTENUATION_VISCOELASTIC`` as its Newmark path runs it.

``L`` standard linear solids relax the bulk modulus ``kappa = lambda + mu`` (2-D) and the shear
modulus, with strengths ``w_l`` fitted for constant ``Qkappa`` and ``Qmu`` over ``f_band``. At
every element GLL node, from the current and the last step's strains ``theta = dux/dx +
duz/dz``, ``dev = dux/dx - theta / 2`` and ``sh = dux/dz + duz/dx``::

    e1_l  = A_l^2 e1_l  + B1_l (theta + A_l theta_old),   A_l = exp(-dt / (2 tau_sigma_l)),
    e11_l = A_l^2 e11_l + B2_l (dev + A_l dev_old),         B_l = -w_l (1 - A_l) / (1 + sum w)
    e13_l = A_l^2 e13_l + B2_l (sh + A_l sh_old)

and the stress gains ``(lambda + mu)_U sum e1 + 2 mu_U sum e11`` (xx), ``... - 2 mu_U sum e11``
(zz) and ``mu_U sum e13`` (xz): SPECFEM2D's ``compute_attenuation_viscoelastic`` (its
``phi_l tau_sigma_l (1 - A_l)`` is ``B_l``) and ``compute_forces_viscoelastic``. The models are
SPECFEM2D's, ``vp, vs, rho, Qkappa, Qmu``, and so are the defaults: the band
``f_ref / 10 .. 10 f_ref``, and ``vp``, ``vs`` read as the unrelaxed velocities
(``velocities="unrelaxed"``; ``"real_modulus"`` is its ``READ_VELOCITIES_AT_f0``, which sets the
real part of the bulk and of the shear modulus at ``f_ref``). ``Q = inf`` gives ``w_l = 0`` and
the elastic step.

The fit (``fit``) is SPECFEM2D's either way: ``"linear"`` its default, the relaxation times
fixed, log-spaced over the band, and the strengths the linear least squares at ``2 n_sls - 1``
frequencies (:class:`sweep.equations.visco_elastic.GSLSFit`); ``"nonlinear"`` its
``USE_SOLVOPT``, relaxation times and strengths fitted together for each Q at ``4 n_sls``
frequencies (:class:`sweep.equations.gsls_nonlinear.GSLSNonlinearFit`) -- a far flatter Q(f)
for few mechanisms or a low Q, with each node's own relaxation times, ``tau_kappa`` from
``Qkappa`` and ``tau_mu`` from ``Qmu``.

The absorbing layer attenuates too, as SPECFEM2D's visco-elastic C-PML does: the C-PML stress
there is ``(kappa + mu)(w) (s_z / s_x) dux/dx + (kappa - mu)(w) duz/dz`` and so on, the moduli
the complex ``M(w)``. The memories take the stretched strains in place of the strains: those
of ``sigma_xx`` (``dux/dx`` stretched), of ``sigma_zz`` (``duz/dz`` stretched) and of the two
shear stresses each their own, so on the band elements every memory has a second copy (``e1z_l,
e11z_l, e13z_l`` and their last strains, for ``sigma_zz`` and ``sigma_xz``). SPECFEM2D expands
the product of the stretching and the relaxation in partial fractions, one recursion per pole;
the recursions here run one after the other, the relaxation on the stretched strain, which is
the same operator to second order in ``dt`` and has no pole to coincide (SPECFEM2D's
coefficients blow up where a relaxation frequency meets a C-PML ``alpha`` or ``beta``).
Without attenuation in the layer the medium's modulus would jump at its edge, by the
dispersion of the band (a few percent at ``Q = 30``), and the layer would reflect that.
Eager only; no boundary saving, since the attenuation is dissipative.
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


def _decay(dt, tau):
    """``A = exp(-dt / (2 tau))``: a float for a fixed relaxation time, a tensor for
    per-node ones."""
    return torch.exp(-0.5 * dt / tau) if torch.is_tensor(tau) else math.exp(-0.5 * dt / tau)


@dataclass
class ViscoElasticSEM2DDeformed(ElasticSEM2DDeformed):
    """Visco-elastic displacement SEM (SPECFEM2D's attenuation): the elastic state, then the
    element memories ``e1_l, e11_l, e13_l`` and the last step's strains, and with a C-PML the
    band's second copies of them. See the module docstring."""

    # PropSEM refuses memory=BoundarySaving() with this reason
    bs_refusal: ClassVar[str] = (
        "ViscoElasticSEM2DDeformed has no boundary saving: the attenuation is dissipative, so "
        "stepping back amplifies round-off and the reverse reconstruction does not exist. "
        "Use memory=Full() or Ckpt(chunks=...).")

    Qk_e: Optional[torch.Tensor] = None
    Qmu_e: Optional[torch.Tensor] = None
    tau_sigma: Optional[tuple] = ()   # the linear fit's fixed relaxation times (None: fitted)
    _tau_k: tuple = ()             # bulk relaxation times: floats, or (E, ngll, ngll) each
    _tau_mu: tuple = ()            # shear relaxation times
    _w1: tuple = ()                # bulk strengths, (E, ngll, ngll) each
    _w2: tuple = ()                # shear strengths
    _kappa_U: Optional[torch.Tensor] = None  # (lambda + mu)_U, the elastic step's
    _mu_U: Optional[torch.Tensor] = None
    f_ref: float = 10.0
    f_band: tuple = ()
    n_sls: int = 3
    n_fit: int = 5
    velocities: str = "unrelaxed"
    fit: object = "linear"

    @classmethod
    def build(cls, mesh: Quad9Mesh, vp_e, vs_e, rho_e, Qk_e, Qmu_e, *,
              cpml: Optional[ElasticCPMLProfile] = None,
              domain: Optional[DomainMask] = None,
              band_split: Optional[bool] = None,
              f_ref: float = 10.0,
              f_band: Optional[tuple] = None,
              n_sls: int = 3,
              n_fit: Optional[int] = None,
              velocities: str = "unrelaxed",
              fit="linear") -> "ViscoElasticSEM2DDeformed":
        """Build the equation: ``vp_e``, ``vs_e`` read at ``f_ref`` as ``velocities`` says
        (``"unrelaxed"`` or ``"real_modulus"``), ``rho_e``, and the bulk and shear quality
        factors ``Qk_e``, ``Qmu_e``, fitted constant over ``f_band`` (default
        ``f_ref / 10 .. 10 f_ref``) with ``n_sls`` mechanisms at ``n_fit`` frequencies by
        ``fit`` (``"linear"``: default ``2 n_sls - 1`` frequencies; ``"nonlinear"``:
        ``4 n_sls``)."""
        if velocities not in ("unrelaxed", "real_modulus"):
            raise ValueError(
                f"velocities must be 'unrelaxed' or 'real_modulus' (SPECFEM2D's readings, "
                f"applied to the bulk and the shear modulus each), got {velocities!r}")
        check_element_arrays(mesh, Qk_e=Qk_e, Qmu_e=Qmu_e)
        f_band = (f_ref / 10.0, 10.0 * f_ref) if f_band is None else f_band
        if n_fit is None:
            n_fit = 2 * int(n_sls) - 1 if fit == "linear" else 4 * int(n_sls)
        n_fit = int(n_fit)
        g = gsls_setup(Qk_e, Qmu_e, f_ref, f_band, n_sls, cls.__name__, n_fit=n_fit,
                       velocities=velocities, fit=fit)
        # the unrelaxed velocities, as factors on vp and vs (exactly 1 where Q = inf)
        mu = rho_e * vs_e * vs_e
        kappa = rho_e * vp_e * vp_e - mu                    # 2-D: kappa = lambda + mu
        s_p = (kappa * g.p[0] + mu * g.s[0]) / (kappa + mu)
        base = ElasticSEM2DDeformed.build(mesh, vp_e * s_p.sqrt(), vs_e * g.s[0].sqrt(), rho_e,
                                          cpml=cpml, domain=domain, band_split=band_split)
        eq = cls(**{f.name: getattr(base, f.name) for f in fields(base) if f.init})
        eq.Qk_e, eq.Qmu_e, eq.tau_sigma = Qk_e, Qmu_e, g.tau_sigma
        eq._tau_k, eq._tau_mu, eq._w1, eq._w2 = g.tau_p, g.tau_s, tuple(g.tp), tuple(g.ts)
        eq._kappa_U, eq._mu_U = base._lam + base._mu, base._mu
        eq.f_ref, eq.f_band, eq.n_sls, eq.n_fit, eq.velocities, eq.fit = (
            g.f_ref, g.f_band, g.n_sls, n_fit, velocities, fit)
        mems = lambda layout, z="": (
            [FieldSpec(f"{c}{z}_{l}", internal=True, layout=layout)
             for l in range(g.n_sls) for c in ("e1", "e11", "e13")]
            + [FieldSpec(f"{n}{z}_old", internal=True, layout=layout)
               for n in ("theta", "dev", "sh")])
        eq.field_specs = (ElasticSEM2DDeformed.field_specs + mems("element")
                          + (mems("band", "z") if cpml is not None else []))
        return eq

    def _memories(self, e, drivers, olds, dt, rows=None):
        """The memories ``e`` (``e1_l, e11_l, e13_l`` per mechanism) one step on, driven by
        ``drivers = (theta, dev, sh)`` now and ``olds`` a step back, on the element ``rows``
        (``None``: all)."""
        pick = (lambda t: t) if rows is None else (
            lambda t: t[rows] if torch.is_tensor(t) else t)
        s1, s2 = pick(1.0 + sum(self._w1)), pick(1.0 + sum(self._w2))
        (theta, dev, sh), (theta_o, dev_o, sh_o) = drivers, olds
        new_e = []
        for l in range(self.n_sls):
            a1, a2 = _decay(dt, pick(self._tau_k[l])), _decay(dt, pick(self._tau_mu[l]))
            b1 = -pick(self._w1[l]) * (1.0 - a1) / s1
            b2 = -pick(self._w2[l]) * (1.0 - a2) / s2
            e1, e11, e13 = e[3 * l:3 * l + 3]
            new_e += [a1 * a1 * e1 + b1 * (theta + a1 * theta_o),
                      a2 * a2 * e11 + b2 * (dev + a2 * dev_o),
                      a2 * a2 * e13 + b2 * (sh + a2 * sh_o)]
        return new_e

    def step(self, fields: tuple, src, dt: float) -> tuple:
        n_base, L = len(ElasticSEM2DDeformed.field_specs), self.n_sls
        base, mem = fields[:n_base], fields[n_base:]
        e, olds = mem[:3 * L], mem[3 * L:3 * L + 3]
        nxt = list(super().step(base, src, dt))
        dux_dx, dux_dz, duz_dx, duz_dz = self._element_derivatives(base[0], base[1])
        theta = dux_dx + duz_dz
        dev = dux_dx - 0.5 * theta
        sh = dux_dz + duz_dx
        if self.cpml is None:
            new_e = self._memories(e, (theta, dev, sh), olds, dt)
            bulk = self._kappa_U * sum(new_e[0::3])
            dvt = 2.0 * self._mu_U * sum(new_e[1::3])
            sig_xz = self._mu_U * sum(new_e[2::3])
            R_x, R_z = self._assemble_stress(bulk + dvt, bulk - dvt, sig_xz, sig_xz, base[0])
            band_out = []
        else:
            # the C-PML's stretched strains on the band, from the memories the elastic step
            # has just advanced (as its _band_residual forms them)
            eb, cp = self.band_elements, self._band
            r_dux_dx, r_dux_dz, r_duz_dx, r_duz_dz = nxt[4:8]
            stretch = lambda d, r, k: (cp[k + "0"] * d[eb] + cp[k + "1"] * r[..., 0]
                                       + cp[k + "2"] * r[..., 1])
            g_xx = stretch(dux_dx, r_dux_dx, "Sx")
            g_zx = stretch(duz_dx, r_duz_dx, "Sx")
            g_xz = stretch(dux_dz, r_dux_dz, "Sz")
            g_zz = stretch(duz_dz, r_duz_dz, "Sz")
            # sigma_xx's and sigma_zx's drivers (sigma_zx -> accel_x) on every element
            theta = theta.index_copy(0, eb, g_xx + duz_dz[eb])
            dev = dev.index_copy(0, eb, 0.5 * (g_xx - duz_dz[eb]))
            sh = sh.index_copy(0, eb, duz_dx[eb] + g_xz)
            # sigma_zz's and sigma_xz's (-> accel_z) on the band
            drv_z = (dux_dx[eb] + g_zz, 0.5 * (dux_dx[eb] - g_zz), dux_dz[eb] + g_zx)
            band_e, band_olds = mem[3 * L + 3:6 * L + 3], mem[6 * L + 3:]
            new_e = self._memories(e, (theta, dev, sh), olds, dt)
            new_ez = self._memories(band_e, drv_z, band_olds, dt, rows=eb)
            k_U, m_U = self._kappa_U, self._mu_U
            s1, s11, s13 = (sum(new_e[i::3]) for i in range(3))
            z1, z11, z13 = (sum(new_ez[i::3]) for i in range(3))
            sig_xx = k_U * s1 + 2.0 * m_U * s11
            sig_zx = m_U * s13
            sig_zz = (k_U * s1 - 2.0 * m_U * s11).index_copy(
                0, eb, k_U[eb] * z1 - 2.0 * m_U[eb] * z11)
            sig_xz = sig_zx.index_copy(0, eb, m_U[eb] * z13)
            R_x, R_z = self._assemble_stress(sig_xx, sig_zz, sig_xz, sig_zx, base[0])
            band_out = [*new_ez, *drv_z]
        dt2 = dt * dt
        nxt[0] = nxt[0] - dt2 * self._inv_M * R_x
        nxt[1] = nxt[1] - dt2 * self._inv_M * R_z
        if self.cpml is not None:
            # the layer's memories load its outer edge too: pin it again, as the elastic step
            # does (an undamped outer-edge convolution otherwise grows without bound)
            oi = self.cpml.outer_edge_idx
            nxt[0], nxt[1] = nxt[0].index_fill(0, oi, 0.0), nxt[1].index_fill(0, oi, 0.0)
        return (*nxt, *new_e, theta, dev, sh, *band_out)
