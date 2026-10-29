"""Element-local SEM operators for 3-D isotropic elastic.

The 3-D analogue of :mod:`sweep.sem.equations.elastic2d`. Three displacement
components ``u = (u_x, u_y, u_z)``; six independent stress components.

Strong form
-----------
::

    ρ ∂²u/∂t² = ∇ · σ + f
    σ_ij = λ (∇·u) δ_ij + μ (∂_i u_j + ∂_j u_i)

with ``λ = ρ(v_p² − 2 v_s²)``, ``μ = ρ v_s²``.

Weak form
---------
::

    ∫ ρ v · ü dΩ + ∫ σ(u) : ε(v) dΩ = ∫ v · f dΩ + ∮ v · (σ·n) dΓ

A traction-free surface (``σ·n = 0``) is a *natural* BC — the boundary integral
drops automatically (no projection / image method).

Stiffness assembly
------------------
The residual of component ``i`` is ``R_i = Σ_j ∫ σ_ij ∂_j v_i dΩ``. On the
axis-aligned hex the integral collapses to the **direction factors**
``G_ab = (2/d_a)(2/d_b)·|J|`` (``|J| = dx·dy·dz/8``); each term
``∫ C ∂_a u_p ∂_b v_q dΩ`` becomes ``D_b^T [ w₃ · C · G_ab · ∂_ξa u_p ]`` where
``∂_ξa`` is the GLL reference derivative along axis ``a`` (one ``einsum``). This
reduces bit-for-bit to the 2-D :class:`ElasticSEM2D` flux table when one
dimension is dropped (verified against that derivation).

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
from ..mesh.hex8 import Hex8Mesh


@dataclass
class ElasticSEM3D(DerivedInitState):
    """Cached SEM operators for 3-D isotropic elastic on a :class:`Hex8Mesh`."""

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

    mesh: Hex8Mesh
    vp_e: torch.Tensor
    vs_e: torch.Tensor
    rho_e: torch.Tensor

    mass: torch.Tensor                # (n_global,) — Σ_e w₃ |J| ρ
    _inv_M: torch.Tensor

    _D: torch.Tensor                  # (ngll, ngll) Lagrange derivative matrix
    _w3: torch.Tensor                 # (ngll, ngll, ngll) GLL weight product
    _lam: torch.Tensor                # (E, ngll, ngll, ngll)
    _mu: torch.Tensor
    _lam2mu: torch.Tensor
    # Direction factors G_ab = (2/d_a)(2/d_b)·|J|  (scalars, 6 distinct).
    _Gxx: float
    _Gyy: float
    _Gzz: float
    _Gxy: float
    _Gxz: float
    _Gyz: float

    @classmethod
    def build(
        cls,
        mesh: Hex8Mesh,
        vp_e: torch.Tensor,
        vs_e: torch.Tensor,
        rho_e: torch.Tensor,
    ) -> "ElasticSEM3D":
        check_element_arrays(mesh, vp_e=vp_e, vs_e=vs_e, rho_e=rho_e)
        device, dtype = vp_e.device, vp_e.dtype
        w = mesh.w_ref
        w3 = w[:, None, None] * w[None, :, None] * w[None, None, :]

        mu = rho_e * vs_e * vs_e
        lam = rho_e * (vp_e * vp_e - 2.0 * vs_e * vs_e)
        lam2mu = lam + 2.0 * mu

        # Mass (lumped): M[g] = Σ_e ρ · w₃ · |J|
        m_e = w3 * rho_e * mesh.jacobian_det
        mass = torch.zeros(mesh.n_global, dtype=dtype, device=device)
        mass.index_add_(0, mesh.ibool.reshape(-1), m_e.reshape(-1))

        jac = mesh.jacobian_det
        gf = lambda da, db: (2.0 / da) * (2.0 / db) * jac
        return cls(
            mesh=mesh, vp_e=vp_e, vs_e=vs_e, rho_e=rho_e,
            mass=mass, _inv_M=1.0 / mass,
            _D=mesh.deriv, _w3=w3,
            _lam=lam, _mu=mu, _lam2mu=lam2mu,
            _Gxx=gf(mesh.dx_e, mesh.dx_e),
            _Gyy=gf(mesh.dy_e, mesh.dy_e),
            _Gzz=gf(mesh.dz_e, mesh.dz_e),
            _Gxy=gf(mesh.dx_e, mesh.dy_e),
            _Gxz=gf(mesh.dx_e, mesh.dz_e),
            _Gyz=gf(mesh.dy_e, mesh.dz_e),
        )

    # ------------------------------------------------------------------ #
    # Stiffness K · (u_x, u_y, u_z) → (K_x u, K_y u, K_z u)
    # ------------------------------------------------------------------ #
    def stiffness_apply(self, ux_g, uy_g, uz_g):
        D = self._D
        ibool = self.mesh.ibool
        # Separate one-wide reference-derivative einsums (one displacement
        # component at a time) rather than a 3-field stacked einsum per direction
        # — ~1.2-1.35x faster on the GPU (a narrow einsum beats a fat one here;
        # see the acoustic3d._cpml_grads note). g_{p,a}: p=component, a=axis.
        ux_e, uy_e, uz_e = ux_g[ibool], uy_g[ibool], uz_g[ibool]
        ex = lambda F: contract("xX,ezyX->ezyx", D, F)
        ey = lambda F: contract("yY,ezYx->ezyx", D, F)
        ez = lambda F: contract("zZ,eZyx->ezyx", D, F)
        gxx, gyx, gzx = ex(ux_e), ex(uy_e), ex(uz_e)          # ∂_x of u_{x,y,z}
        gxy, gyy, gzy = ey(ux_e), ey(uy_e), ey(uz_e)          # ∂_y
        gxz, gyz, gzz = ez(ux_e), ez(uy_e), ez(uz_e)          # ∂_z

        w3, lam, mu, lam2mu = self._w3, self._lam, self._mu, self._lam2mu
        Gxx, Gyy, Gzz = self._Gxx, self._Gyy, self._Gzz
        Gxy, Gxz, Gyz = self._Gxy, self._Gxz, self._Gyz

        # Normal-stress fluxes (F_ii = w₃·|J|·(2/d_i)·σ_ii).
        F_xx = w3 * (lam2mu * Gxx * gxx + lam * Gxy * gyy + lam * Gxz * gzz)
        F_yy = w3 * (lam * Gxy * gxx + lam2mu * Gyy * gyy + lam * Gyz * gzz)
        F_zz = w3 * (lam * Gxz * gxx + lam * Gyz * gyy + lam2mu * Gzz * gzz)
        # Shear-stress fluxes (each σ_ij feeds both R_i via j and R_j via i).
        F_xy = w3 * mu * (Gyy * gxy + Gxy * gyx)   # σ_xy → R_x (∂_y)
        F_yx = w3 * mu * (Gxy * gxy + Gxx * gyx)   # σ_xy → R_y (∂_x)
        F_xz = w3 * mu * (Gzz * gxz + Gxz * gzx)   # σ_xz → R_x (∂_z)
        F_zx = w3 * mu * (Gxz * gxz + Gxx * gzx)   # σ_xz → R_z (∂_x)
        F_yz = w3 * mu * (Gzz * gyz + Gyz * gzy)   # σ_yz → R_y (∂_z)
        F_zy = w3 * mu * (Gyz * gyz + Gyy * gzy)   # σ_yz → R_z (∂_y)

        dtx = lambda F: contract("Xx,ezyX->ezyx", D, F)   # D_x^T
        dty = lambda F: contract("Yy,ezYx->ezyx", D, F)   # D_y^T
        dtz = lambda F: contract("Zz,eZyx->ezyx", D, F)   # D_z^T
        R_x = dtx(F_xx) + dty(F_xy) + dtz(F_xz)
        R_y = dtx(F_yx) + dty(F_yy) + dtz(F_yz)
        R_z = dtx(F_zx) + dty(F_zy) + dtz(F_zz)

        return assemble_each([R_x, R_y, R_z], ibool, ux_g)    # (Kx, Ky, Kz)

    # ================================================================== #
    # SEMPropagator interface (Newmark β=0, γ=1/2 leapfrog).
    # State layout = (u_x, u_y, u_z, u_x_prev, u_y_prev, u_z_prev).
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

    # ================================================================== #
    # Boundary-saving hooks (swap2nd reverse). Bare elastic: the 3-level
    # leapfrog with natural boundaries is exactly time-reversible, so the
    # ring is empty (no CPML) and BS memory is O(1) in nt.
    # ================================================================== #
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
