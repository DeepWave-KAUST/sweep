"""Isotropic elastic SEM on a deformed Hex27 mesh — the 3-D analogue of
:class:`ElasticSEM2DDeformed`, and the deformed counterpart of
:class:`ElasticSEM3D`.

Weak form ``∫ ρ v·ü dΩ + ∫ σ(u):ε(v) dΩ = ∫ v·f dΩ`` on a Q2 (27-node) deformed
mesh, with the per-GLL 3×3 Jacobian chain rule

    ∂u_p/∂x_d = Σ_m inv_J[m, d] ∂u_p/∂ξ_m   (m ∈ {ξ, η, γ}, d ∈ {x, y, z}).

A traction-free surface (σ·n = 0) is a *natural* BC — the boundary integral
drops automatically, so the curved top face needs no projection or image
method. ``cpml`` is not modelled here (matching :class:`ElasticSEM3D`): the
domain is a bare reflecting box with natural free surfaces on every face.

On a flat (axis-aligned) Hex27 mesh this reduces to :class:`ElasticSEM3D` on
the equivalent Hex8 mesh -- same global node set, same physics, agreeing to
float64 round-off (measured 1.1e-15 relative on a 60-step run), NOT bit-for-bit:
the zero cross terms still enter the sums and re-associate them. Assert with a
tolerance, never ``torch.equal``. (Contrast ``ElasticVTISEM2DDeformed`` at
epsilon=delta=0, which IS bit-identical to the isotropic equation -- measured
exactly 0.0 -- because there the coefficients collapse before any extra
addition.)

inv_J index convention (see :mod:`sweep.sem.basis.hex27`):
    inv_J[..., m, d] = ∂ξ_m/∂x_d   with m,d ordered (ξ,η,γ) / (x,y,z).

Why there is no C-PML here, and what to use instead
---------------------------------------------------
This displacement-form equation models NO absorbing boundary, in either
backend; ``abcn_elements`` is refused at construction. Porting the 2-D
displacement C-PML (:mod:`sweep.sem.cpml.elastic`) to 3-D was assessed and
rejected rather than half-done: 3-D has SEVEN PML regions (2-D has three), the
XYZ triple corner needs a 3-pole partial-fraction split for which the 2-D
2-pole closed forms do not generalise, the corner pole-separation nudge (three
pairwise adjustments in 2-D) does not guarantee three mutually distinct alphas
plus six beta-alpha constraints simultaneously, and the failure mode of getting
any of it wrong is LATE-TIME DIVERGENCE, not an error. Estimated 700-950 lines
of new derivation with no reference implementation in this repository to check
against.

For 3-D elastic with an absorbing boundary use the velocity-stress family:
:class:`ElasticVSSEM3D` (flat) or :class:`ElasticVSSEM3DDeformed` (topography),
whose C-PML is implemented and gated by a big-domain-subtraction absorption
test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from .._assembly import assemble_each, contract
from ..fields import DerivedInitState, FieldSpec, check_element_arrays
from ..mesh.hex27 import Hex27Mesh


# reference-derivative + transpose einsums on (E, kz, ky, kx) tensors
def _dxi(D, f):   return contract("xX,ezyX->ezyx", D, f)   # ∂/∂ξ (x-axis)
def _deta(D, f):  return contract("yY,ezYx->ezyx", D, f)   # ∂/∂η (y-axis)
def _dgam(D, f):  return contract("zZ,eZyx->ezyx", D, f)   # ∂/∂γ (z-axis)
def _DTxi(D, F):  return contract("Xx,ezyX->ezyx", D, F)
def _DTeta(D, F): return contract("Yy,ezYx->ezyx", D, F)
def _DTgam(D, F): return contract("Zz,eZyx->ezyx", D, F)


@dataclass
class ElasticSEM3DDeformed(DerivedInitState):
    """Cached factors + ops for 3-D isotropic elastic SEM on a Hex27 mesh."""

    field_specs: ClassVar[list] = [
        FieldSpec("u_x", supports_source=True, supports_receiver=True,
                  seis_key="seis_x", snap_key="snapshots_x"),
        FieldSpec("u_y", supports_source=True, supports_receiver=True,
                  seis_key="seis_y", snap_key="snapshots_y"),
        FieldSpec("u_z", supports_source=True, supports_receiver=True,
                  seis_key="seis_z", snap_key="snapshots_z"),
        FieldSpec("u_x_prev", internal=True),
        FieldSpec("u_y_prev", internal=True),
        FieldSpec("u_z_prev", internal=True),
    ]

    # FD-aligned receiver components: particle velocity from the displacement trace: v = u_dot.
    derived_receiver_types: ClassVar[dict] = {"vx": ("u_x", "velocity"), "vy": ("u_y", "velocity"),
        "vz": ("u_z", "velocity")}

    mesh: Hex27Mesh
    vp_e: torch.Tensor
    vs_e: torch.Tensor
    rho_e: torch.Tensor

    mass: torch.Tensor                # (n_global,) — Σ_e ρ · w₃ |J|
    _inv_M: torch.Tensor

    _D: torch.Tensor
    _lam: torch.Tensor                # (E, ng, ng, ng)
    _mu: torch.Tensor
    _lam2mu: torch.Tensor
    _w_J: torch.Tensor                # w₃ |J|
    _I: tuple                         # 9 inv_J tensors (xi_x, xi_y, xi_z, eta_x, ...)

    @classmethod
    def build(cls, mesh: Hex27Mesh, vp_e, vs_e, rho_e) -> "ElasticSEM3DDeformed":
        check_element_arrays(mesh, vp_e=vp_e, vs_e=vs_e, rho_e=rho_e)
        device, dtype = vp_e.device, vp_e.dtype
        mu = rho_e * vs_e * vs_e
        lam = rho_e * (vp_e * vp_e - 2.0 * vs_e * vs_e)
        lam2mu = lam + 2.0 * mu
        m_e = mesh.w_jac * rho_e
        mass = torch.zeros(mesh.n_global, dtype=dtype, device=device)
        mass.index_add_(0, mesh.ibool.reshape(-1), m_e.reshape(-1))
        J = mesh.inv_J
        I = (J[..., 0, 0], J[..., 0, 1], J[..., 0, 2],
             J[..., 1, 0], J[..., 1, 1], J[..., 1, 2],
             J[..., 2, 0], J[..., 2, 1], J[..., 2, 2])
        return cls(
            mesh=mesh, vp_e=vp_e, vs_e=vs_e, rho_e=rho_e,
            mass=mass, _inv_M=1.0 / mass, _D=mesh.deriv,
            _lam=lam, _mu=mu, _lam2mu=lam2mu, _w_J=mesh.w_jac, _I=I)

    # ------------------------------------------------------------------ #
    def _phys_grad(self, dxi, deta, dgam):
        (Ixx, Ixy, Ixz, Iex, Iey, Iez, Igx, Igy, Igz) = self._I
        return (Ixx * dxi + Iex * deta + Igx * dgam,
                Ixy * dxi + Iey * deta + Igy * dgam,
                Ixz * dxi + Iez * deta + Igz * dgam)

    def _proj_ref(self, q_x, q_y, q_z):
        """w_J · inv_J · q  ->  (F_xi, F_eta, F_gam)."""
        (Ixx, Ixy, Ixz, Iex, Iey, Iez, Igx, Igy, Igz) = self._I
        wJ = self._w_J
        return (wJ * (q_x * Ixx + q_y * Ixy + q_z * Ixz),
                wJ * (q_x * Iex + q_y * Iey + q_z * Iez),
                wJ * (q_x * Igx + q_y * Igy + q_z * Igz))

    def stiffness_apply(self, ux_g, uy_g, uz_g):
        D, ib = self._D, self.mesh.ibool
        ux_e, uy_e, uz_e = ux_g[ib], uy_g[ib], uz_g[ib]
        # physical displacement gradients g_{p,d} = ∂u_p/∂x_d
        gxx, gxy, gxz = self._phys_grad(_dxi(D, ux_e), _deta(D, ux_e), _dgam(D, ux_e))
        gyx, gyy, gyz = self._phys_grad(_dxi(D, uy_e), _deta(D, uy_e), _dgam(D, uy_e))
        gzx, gzy, gzz = self._phys_grad(_dxi(D, uz_e), _deta(D, uz_e), _dgam(D, uz_e))
        lam, mu, l2m = self._lam, self._mu, self._lam2mu
        sxx = l2m * gxx + lam * (gyy + gzz)
        syy = l2m * gyy + lam * (gxx + gzz)
        szz = l2m * gzz + lam * (gxx + gyy)
        sxy = mu * (gxy + gyx)
        sxz = mu * (gxz + gzx)
        syz = mu * (gyz + gzy)
        # per-equation ref fluxes + D^T divergence
        Fx_xi, Fx_eta, Fx_gam = self._proj_ref(sxx, sxy, sxz)
        Fy_xi, Fy_eta, Fy_gam = self._proj_ref(sxy, syy, syz)
        Fz_xi, Fz_eta, Fz_gam = self._proj_ref(sxz, syz, szz)
        R_x = _DTxi(D, Fx_xi) + _DTeta(D, Fx_eta) + _DTgam(D, Fx_gam)
        R_y = _DTxi(D, Fy_xi) + _DTeta(D, Fy_eta) + _DTgam(D, Fy_gam)
        R_z = _DTxi(D, Fz_xi) + _DTeta(D, Fz_eta) + _DTgam(D, Fz_gam)
        return assemble_each([R_x, R_y, R_z], ib, ux_g)

    # ================================================================== #
    # SEMPropagator interface (Newmark β=0, γ=1/2 leapfrog); state layout
    # = (u_x, u_y, u_z, u_x_prev, u_y_prev, u_z_prev).
    # ================================================================== #
    def step(self, fields: tuple, src, dt: float) -> tuple:
        ux_now, uy_now, uz_now, ux_prev, uy_prev, uz_prev = fields
        dt2, inv_M = dt * dt, self._inv_M
        b_x, b_y, b_z = src[0], src[1], src[2]
        Kx, Ky, Kz = self.stiffness_apply(ux_now, uy_now, uz_now)
        ux_next = 2.0 * ux_now - ux_prev + dt2 * inv_M * (-Kx + b_x)
        uy_next = 2.0 * uy_now - uy_prev + dt2 * inv_M * (-Ky + b_y)
        uz_next = 2.0 * uz_now - uz_prev + dt2 * inv_M * (-Kz + b_z)
        return (ux_next, uy_next, uz_next, ux_now, uy_now, uz_now)

    # Boundary-saving hooks (bare 3-level leapfrog is time-reversible; identical
    # to ElasticSEM3D — the ring holds only the 3 displacement components).
    bs_ring_save: ClassVar[list] = [0, 1, 2]
    bs_ring_restore: ClassVar[list] = [(0, 0, 0), (1, 1, 0), (2, 2, 0),
                                       (3, 0, -1), (4, 1, -1), (5, 2, -1)]

    def reverse_step(self, fields, src, dt):
        ux1, uy1, uz1, ux0, uy0, uz0 = fields
        out = self.step((ux0, uy0, uz0, ux1, uy1, uz1), src, dt)
        return (ux0, uy0, uz0, out[0], out[1], out[2])

    def bs_model_tensors(self) -> dict:
        return {"lam": self._lam, "mu": self._mu, "lam2mu": self._lam2mu,
                "inv_M": self._inv_M}

    def bs_recompute(self, fields, src, dt, leaves):
        saved = (self._lam, self._mu, self._lam2mu, self._inv_M)
        self._lam, self._mu = leaves["lam"], leaves["mu"]
        self._lam2mu, self._inv_M = leaves["lam2mu"], leaves["inv_M"]
        try:
            return self.step(fields, src, dt)
        finally:
            self._lam, self._mu, self._lam2mu, self._inv_M = saved
