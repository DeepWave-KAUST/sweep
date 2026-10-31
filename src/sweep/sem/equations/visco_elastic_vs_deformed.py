"""1st-order velocity-stress visco-elastic SEM on a Quad9 deformed mesh (GSLS, nearly constant Q).

:class:`ElasticVSSEM2DDeformed` with the rheology of the FD
:class:`sweep.equations.ViscoElastic`, which is SPECFEM2D's (and so SeisFlows'): ``L``
standard linear solids in parallel whose relaxation times ``tau_sigma_l`` are fixed by the
band ``f_band``, each with the strength ``tau_l(Q)`` of the least-squares constant-Q fit
(:class:`sweep.equations.visco_elastic.GSLSFit`). At every element GLL node, with the
unrelaxed P and shear moduli ``pi_U``, ``mu_U``, the relaxed ``pi_R``, ``mu_R``,
``tp_l = tau_l(Qp)``, ``ts_l = tau_l(Qs)`` and ``theta = dvx/dx + dvz/dz``::

    dsxx/dt = pi_U theta - 2 mu_U dvz/dz + sum_l r_xx_l
    dszz/dt = pi_U theta - 2 mu_U dvx/dx + sum_l r_zz_l
    dsxz/dt = mu_U (dvx/dz + dvz/dx)    + sum_l r_xz_l
    dr_xx_l/dt = -(r_xx_l + pi_R tp_l theta - 2 mu_R ts_l dvz/dz) / tau_sigma_l

(``r_zz_l`` and ``r_xz_l`` alike), ``pi_U = pi_R (1 + sum_l tp_l)``, ``mu_U = mu_R (1 + sum_l
ts_l)``. The memories are element-local, as the stress is, and advance with the
trapezoidal rule (stable for any ``dt / tau_sigma_l``); the stress takes their time average
-- the FD update, at every GLL node. The free surface stays the natural one: the memories
are part of the stress whose traction integral the weak form drops. In the C-PML they see
the stretched strain rate, as the stress does.

Models ``vp, vs, rho, Qp, Qs``: ``vp`` / ``vs`` are the phase velocities at ``f_ref``
(SPECFEM's ``READ_VELOCITIES_AT_f0``), turned into the relaxed moduli through the GSLS
complex modulus there, and all five are differentiable. Element arrays of ``Q = inf`` give
``tau_l = 0`` and :class:`ElasticVSSEM2DDeformed`'s step bit for bit. ``Q`` must exceed the
fit's ``q_min`` (about 1.8 for 3 mechanisms over the default band).

SPECFEM2D's conventions are options: ``velocities="real_modulus"`` takes ``rho vp^2`` and
``rho vs^2`` as the real parts of the moduli at ``f_ref``, as its ``READ_VELOCITIES_AT_f0``
does, which leaves the phase velocity higher than ``vp`` by about ``3 / (8 Q^2)``; and
``n_fit = 2 n_sls - 1`` samples the constant-Q least squares at its frequencies (log-spaced
over ``f_band`` with the ends; the default 200 fits the band throughout). ``velocities=
"unrelaxed"`` takes ``vp``, ``vs`` as the infinite-frequency velocities, which is how SPECFEM2D
reads them without ``READ_VELOCITIES_AT_f0``.

The explicit stress update runs at the unrelaxed velocities, above ``vp`` / ``vs`` for a
finite Q (~8% at Q = 10 for the default band of a factor 12): take ``dt`` from those
(:meth:`ViscoElasticVSSEM2DDeformed.unrelaxed_velocities`). The C-PML terms run on every
element: the band split of :class:`ElasticVSSEM2DDeformed` would need the memories' share of
the stretched strain rate on the band rows. Eager only; no boundary saving, since the
attenuation is dissipative.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar, Optional

import numpy as np
import torch

from sweep.equations.visco_elastic import (GSLSFit, MAX_SLS, gsls_band, gsls_moduli_factors,
                                           gsls_relaxation_times)

from ..cpml.elastic_vs import SimpleElasticCPMLProfile
from ..fields import FieldSpec, check_element_arrays
from ..mesh.quad9 import Quad9Mesh
from .elastic_vs_deformed import ElasticVSSEM2DDeformed


def gsls_memory_specs(n_sls):
    """The ``3 n_sls`` element-local memories, ``r_xx_l, r_zz_l, r_xz_l`` per mechanism."""
    return [FieldSpec(f"r_{c}_{l}", internal=True, layout="element")
            for l in range(n_sls) for c in ("xx", "zz", "xz")]


def gsls_real_modulus_factors(taus, tau_sigma, f_ref):
    """``(s_U, s_R)`` mapping ``rho v^2`` to the unrelaxed and relaxed moduli when ``rho v^2``
    is the real part of the modulus at ``f_ref``, as SPECFEM2D's ``READ_VELOCITIES_AT_f0``
    takes it (``shift_velocities_from_f0``), instead of the phase velocity's square
    (:func:`~sweep.equations.visco_elastic.gsls_moduli_factors`)."""
    wt = 2.0 * np.pi * f_ref * np.asarray(tau_sigma, dtype=np.float64)
    ca = wt ** 2 / (1.0 + wt ** 2)
    s_R = 1.0 / (1.0 + sum(float(ca[l]) * t for l, t in enumerate(taus)))
    return s_R * (1.0 + sum(taus)), s_R


def gsls_unrelaxed_factors(taus, tau_sigma, f_ref):
    """``(s_U, s_R)`` when ``v`` is the unrelaxed (infinite-frequency) velocity, as SPECFEM2D
    reads its input without ``READ_VELOCITIES_AT_f0``: ``s_U = 1``, ``s_R = 1 / (1 + sum t)``."""
    total = 1.0 + sum(taus)
    return total / total, 1.0 / total


VELOCITY_CONVENTIONS = {"phase": gsls_moduli_factors, "real_modulus": gsls_real_modulus_factors,
                        "unrelaxed": gsls_unrelaxed_factors}


def gsls_setup(Qp_e, Qs_e, f_ref, f_band, n_sls, who, *, n_fit=200, velocities="phase"):
    """The GSLS of ``n_sls`` mechanisms over ``f_band`` (default: a factor 12 around
    ``f_ref``, see :func:`~sweep.equations.visco_elastic.gsls_band`) for the quality
    factors ``Qp_e``, ``Qs_e``, the constant-Q least squares sampled at ``n_fit``
    frequencies: a namespace of the band, the relaxation times, the fit's ``q_min`` (``Q``
    below it is refused, naming ``who``), the strengths ``tp``, ``ts`` and the factors
    ``(s_U, s_R)`` of ``VELOCITY_CONVENTIONS[velocities]`` for each."""
    if not 1 <= int(n_sls) <= MAX_SLS:
        raise ValueError(f"n_sls must be in [1, {MAX_SLS}], got {n_sls}")
    if velocities not in VELOCITY_CONVENTIONS:
        raise ValueError(f"velocities must be one of {sorted(VELOCITY_CONVENTIONS)}, "
                         f"got {velocities!r}")
    f_ref, n_sls = float(f_ref), int(n_sls)
    if int(n_fit) < n_sls:
        raise ValueError(f"n_fit = {n_fit} frequencies cannot fit {n_sls} mechanisms")
    f_band = gsls_band(f_ref, f_band)
    tau_sigma = gsls_relaxation_times(n_sls, *f_band)
    fit = GSLSFit(tau_sigma, *f_band, n_freq=int(n_fit))
    q_min = fit.min_valid_q()
    q_lo = float(torch.minimum(Qp_e.detach().min(), Qs_e.detach().min()))
    if not q_lo > q_min:
        raise ValueError(
            f"{who}: min(Qp, Qs) = {q_lo:.3g} is below {q_min:.3g}, where the constant-Q fit "
            f"yields a negative relaxation strength (n_sls={n_sls}, f_band={f_band}). Bound Q "
            "from below.")
    tp, ts = fit.tau(Qp_e), fit.tau(Qs_e)
    factors = VELOCITY_CONVENTIONS[velocities]
    return SimpleNamespace(f_ref=f_ref, f_band=f_band, n_sls=n_sls, q_min=q_min,
                           tau_sigma=tuple(float(t) for t in tau_sigma), tp=tp, ts=ts,
                           p=factors(tp, tau_sigma, f_ref), s=factors(ts, tau_sigma, f_ref))


def gsls_weights(tau_sigma, dt):
    """Trapezoidal memory-update weights ``r+ = a_l r - c_l F``:
    ``a_l = (1 - dt/2ts) / (1 + dt/2ts)``, ``c_l = (dt/ts) / (1 + dt/2ts)``."""
    a_l, c_l = [], []
    for ts in tau_sigma:
        half = 0.5 * dt / ts
        a_l.append((1.0 - half) / (1.0 + half))
        c_l.append((dt / ts) / (1.0 + half))
    return a_l, c_l


def gsls_memories(mem, p_tau, m_tau, weights, d_vx_dx, d_vz_dz, exz):
    """The memories one step on (FD ``ViscoElastic``'s update): ``mem`` is ``(r_xx_l,
    r_zz_l, r_xz_l)`` per mechanism, ``p_tau`` / ``m_tau`` the relaxed P / shear moduli times
    the strengths. Returns the new memories and the sums ``r+ + r`` per stress component,
    whose ``dt / 2`` multiple the stress takes (their time average)."""
    a_l, c_l = weights
    theta = d_vx_dx + d_vz_dz
    new_mem = []
    sum_xx = sum_zz = sum_xz = 0.0
    for l in range(len(p_tau)):
        r_xx, r_zz, r_xz = mem[3 * l:3 * l + 3]
        tp, ts = p_tau[l] * theta, m_tau[l]
        n_xx = a_l[l] * r_xx - c_l[l] * (tp - 2.0 * ts * d_vz_dz)
        n_zz = a_l[l] * r_zz - c_l[l] * (tp - 2.0 * ts * d_vx_dx)
        n_xz = a_l[l] * r_xz - c_l[l] * (ts * exz)
        sum_xx = sum_xx + (n_xx + r_xx)
        sum_zz = sum_zz + (n_zz + r_zz)
        sum_xz = sum_xz + (n_xz + r_xz)
        new_mem += [n_xx, n_zz, n_xz]
    return new_mem, sum_xx, sum_zz, sum_xz


class ViscoElasticVSSEM2DDeformed(ElasticVSSEM2DDeformed):
    """Visco-elastic (GSLS) velocity-stress SEM: the elastic state, then ``3 L`` memories
    ``r_xx_l, r_zz_l, r_xz_l``. See the module docstring."""

    # PropSEM refuses memory=BoundarySaving() with this reason
    bs_refusal: ClassVar[str] = (
        "ViscoElasticVSSEM2DDeformed has no boundary saving: the attenuation is dissipative, "
        "so stepping back amplifies round-off and the reverse reconstruction does not exist. "
        "Use memory=Full() or Ckpt(chunks=...).")

    @classmethod
    def build(
        cls,
        mesh: Quad9Mesh,
        vp_e: torch.Tensor,
        vs_e: torch.Tensor,
        rho_e: torch.Tensor,
        Qp_e: torch.Tensor,
        Qs_e: torch.Tensor,
        *,
        free_top: bool = False,
        cpml: Optional[SimpleElasticCPMLProfile] = None,
        f_ref: float = 10.0,
        f_band: Optional[tuple] = None,
        n_sls: int = 3,
        n_fit: int = 200,
        velocities: str = "phase",
    ) -> "ViscoElasticVSSEM2DDeformed":
        """Build the equation. ``vp_e`` / ``vs_e`` are the phase velocities at ``f_ref``
        (Hz) -- or, with ``velocities="real_modulus"``, the square roots of the moduli's
        real parts there over ``rho_e``, SPECFEM2D's reading --, ``Qp_e`` / ``Qs_e`` the
        quality factors of the P and shear moduli, fitted constant over
        ``f_band = (f_min, f_max)`` (default: a factor 12 around ``f_ref``, see
        :func:`~sweep.equations.visco_elastic.gsls_band`) with ``n_sls`` mechanisms
        (SPECFEM's default 3; 1 to 4), by least squares at ``n_fit`` log-spaced
        frequencies (SPECFEM2D's: ``2 n_sls - 1``)."""
        check_element_arrays(mesh, Qp_e=Qp_e, Qs_e=Qs_e)
        g = gsls_setup(Qp_e, Qs_e, f_ref, f_band, n_sls, cls.__name__, n_fit=n_fit,
                       velocities=velocities)
        (sU_p, sR_p), (sU_s, sR_s) = g.p, g.s
        # The elastic build on the unrelaxed velocities: mass, geometry, the free surface,
        # the C-PML and the unrelaxed Lame set (bit for bit the elastic one at Q = inf).
        eq = ElasticVSSEM2DDeformed.build(
            mesh, vp_e * sU_p.sqrt(), vs_e * sU_s.sqrt(), rho_e, free_top=free_top, cpml=cpml,
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
        eq.field_specs = ElasticVSSEM2DDeformed.field_specs + gsls_memory_specs(g.n_sls)
        return eq

    def unrelaxed_velocities(self):
        """``(vp_U, vs_U)``, ``(E, ngll, ngll)``: the velocities the explicit stress update
        runs at -- take ``dt`` from the CFL condition on ``vp_U``."""
        return self.vp_e, self.vs_e

    @property
    def reverse_step(self):
        # no reverse reconstruction (see bs_refusal): hasattr() is False, and a direct
        # run_boundary_saved refuses the equation
        raise AttributeError(self.bs_refusal)

    def step(self, fields: tuple, src, dt: float) -> tuple:
        """One staggered-leapfrog step (v at n+1/2, σ and the memories at n+1)."""
        n = len(ElasticVSSEM2DDeformed.field_specs)
        (v_x, v_z, sigma_xx, sigma_zz, sigma_xz,
         m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z,
         m_vxx, m_vzz, m_vxz, m_vzx) = fields[:n]
        mem = fields[n:]
        cp = self.cpml
        use_cpml = cp is not None
        inv_M = self._inv_M
        b_x, b_z = src[0], src[1]

        # ----- velocity update from the current σ (σ-derivative CPML, every element) -----
        # the parent's, unsplit: the recursion on the assembled divergence split by
        # derivative direction, its stretch added once per node
        R_x, R_z = self.assemble_stress_divergence(
            sigma_xx, sigma_xz, sigma_xz, sigma_zz)
        if use_cpml:
            cb = self._band
            z = torch.zeros_like(sigma_xx)
            ib = cb["ibool"]
            # (x part of R_x, x part of R_z, z part of R_x, z part of R_z)
            P = torch.cat([self._divergence_e(sigma_xx, z, sigma_xz, z, cb),
                           self._divergence_e(z, sigma_xz, z, sigma_zz, cb)], -1)
            P = sigma_xx.new_zeros(self.mesh.n_global, 4).index_add(
                0, ib.reshape(-1), P.reshape(-1, 4))[ib]
            m_sxx_x = cb["a_x"] * m_sxx_x + cb["b_x"] * P[..., 0]
            m_sxz_x = cb["a_x"] * m_sxz_x + cb["b_x"] * P[..., 1]
            m_sxz_z = cb["a_z"] * m_sxz_z + cb["b_z"] * P[..., 2]
            m_szz_z = cb["a_z"] * m_szz_z + cb["b_z"] * P[..., 3]
            psi = torch.stack([m_sxx_x + m_sxz_z, m_sxz_x + m_szz_z], -1) * cb["first"]
            acc = sigma_xx.new_zeros(self.mesh.n_global, 2).index_add(
                0, ib.reshape(-1), psi.reshape(-1, 2))
            R_x = R_x + acc[:, 0]
            R_z = R_z + acc[:, 1]
        v_x = v_x + dt * inv_M * (-R_x + b_x)
        v_z = v_z + dt * inv_M * (-R_z + b_z)

        # ----- stress and memories from the new v (stretched gradients in the C-PML) -----
        d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz = self.velocity_gradients(v_x, v_z)
        if use_cpml:
            m_vxx = cb["a_x"] * m_vxx + cb["b_x"] * d_vx_dx
            m_vzz = cb["a_z"] * m_vzz + cb["b_z"] * d_vz_dz
            m_vxz = cb["a_z"] * m_vxz + cb["b_z"] * d_vx_dz
            m_vzx = cb["a_x"] * m_vzx + cb["b_x"] * d_vz_dx
            d_vx_dx = d_vx_dx + m_vxx
            d_vz_dz = d_vz_dz + m_vzz
            d_vx_dz = d_vx_dz + m_vxz
            d_vz_dx = d_vz_dx + m_vzx
        inc_xx, inc_zz, inc_xz = self._stress_increment(
            d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz, dt)
        new_mem, sum_xx, sum_zz, sum_xz = gsls_memories(
            mem, self._p_tau, self._m_tau, gsls_weights(self._tau_sigma, dt),
            d_vx_dx, d_vz_dz, d_vx_dz + d_vz_dx)
        half_dt = 0.5 * dt
        sigma_xx = sigma_xx + inc_xx + half_dt * sum_xx
        sigma_zz = sigma_zz + inc_zz + half_dt * sum_zz
        sigma_xz = sigma_xz + inc_xz + half_dt * sum_xz

        # ----- free surface BC + corner taper -----
        self.apply_free_top(sigma_xx, sigma_zz, sigma_xz)
        if use_cpml:
            (m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z,
             m_vxx, m_vzz, m_vxz, m_vzx) = self.taper_cpml_at_top(
                m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z, m_vxx, m_vzz, m_vxz, m_vzx)

        return (v_x, v_z, sigma_xx, sigma_zz, sigma_xz,
                m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z,
                m_vxx, m_vzz, m_vxz, m_vzx, *new_mem)
