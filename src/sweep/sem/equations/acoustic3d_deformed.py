"""Constant-density acoustic SEM on a deformed Hex27 mesh — the 3-D analogue of
:class:`AcousticSEM2DDeformed`.

``ü = c² M⁻¹ (b − K u)`` with the c²-free stiffness ``K = ∫ ∇v · ∇u dΩ`` and c²
per global node (see :mod:`.acoustic2d`), on a Q2 (27-node) deformed mesh, with
the per-GLL 3×3 Jacobian chain rule

    ∂/∂x_d = Σ_m inv_J[m, d] ∂/∂ξ_m   (m ∈ {ξ, η, γ}, d ∈ {x, y, z}).

On a flat (axis-aligned) Hex27 mesh this reduces to :class:`AcousticSEM3D` on
the equivalent Hex8 mesh -- the same global node set (``ibool`` is identical)
and the same physics, agreeing to float64 round-off (measured 1.2e-15 relative
on an 80-step run), NOT bit-for-bit. The cross terms are exactly zero there but
still take part in the sums, so the additions re-associate and the last few ulp
differ. Assert with a tolerance, never ``torch.equal``.

Boundaries: pressure-release Dirichlet free surface (``dirichlet=("top",)``) on
the conforming (curved) top face. Outgoing waves: **CPML** (3-direction
Komatitsch-Martin double-convolution, deformed directional operators that reduce
to the axis-aligned ones in the rectangular absorbing band). ``cpml=None`` is a
bare reflecting box.

inv_J index convention (see :mod:`sweep.sem.basis.hex27`):
    inv_J[..., m, d] = ∂ξ_m/∂x_d   with m,d ordered (ξ,η,γ) / (x,y,z).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from .._assembly import assemble, assemble_each, contract
from ..cpml.acoustic3d import CPML3DProfile
from ..mesh.boundaries import free_surface_node_idx
from ..mesh.nodes import node_field
from ..fields import DerivedInitState, DirichletNodes, FieldSpec, check_element_arrays
from ..mesh.hex27 import Hex27Mesh
from ..mesh.hex8 import boundary_element_faces


# reference-derivative + transpose einsums on (E, kz, ky, kx) tensors
def _dxi(D, f):   return contract("xX,ezyX->ezyx", D, f)   # ∂/∂ξ (x-axis)
def _deta(D, f):  return contract("yY,ezYx->ezyx", D, f)   # ∂/∂η (y-axis)
def _dgam(D, f):  return contract("zZ,eZyx->ezyx", D, f)   # ∂/∂γ (z-axis)
def _DTxi(D, F):  return contract("Xx,ezyX->ezyx", D, F)
def _DTeta(D, F): return contract("Yy,ezYx->ezyx", D, F)
def _DTgam(D, F): return contract("Zz,eZyx->ezyx", D, F)


@dataclass
class AcousticSEM3DDeformed(DerivedInitState, DirichletNodes):
    """Cached factors + ops for 3-D acoustic SEM on a :class:`Hex27Mesh`."""

    field_specs: ClassVar[list] = [
        FieldSpec("u", supports_source=True, supports_receiver=True,
                  seis_key="seismograms", snap_key="snapshots"),
        FieldSpec("u_prev", internal=True),
        FieldSpec("psi_x", internal=True, boundary_related=True),
        FieldSpec("psi_y", internal=True, boundary_related=True),
        FieldSpec("psi_z", internal=True, boundary_related=True),
        FieldSpec("zeta_x", internal=True, boundary_related=True),
        FieldSpec("zeta_y", internal=True, boundary_related=True),
        FieldSpec("zeta_z", internal=True, boundary_related=True),
    ]

    # FD-aligned receiver components: same scalar-wave convention as AcousticSEM2D.
    derived_receiver_types: ClassVar[dict] = {"p": ("u", None)}

    mesh: Hex27Mesh
    vp_e: torch.Tensor
    vp_squared_node: torch.Tensor
    mass: torch.Tensor
    _inv_M: torch.Tensor
    dirichlet: tuple
    dirichlet_idx: torch.Tensor
    cpml: Optional[CPML3DProfile]

    _D: torch.Tensor
    _w_J: torch.Tensor                # w₃ |J|
    # inv_J components I[m, d] = ∂ξ_m/∂x_d
    _I: tuple                         # 9 tensors: (xi_x, xi_y, xi_z, eta_x, ...)

    @classmethod
    def build(cls, mesh: Hex27Mesh, vp_e, *, dirichlet=("top",), cpml=None):
        check_element_arrays(mesh, vp_e=vp_e)
        device, dtype = vp_e.device, vp_e.dtype
        mass = torch.zeros(mesh.n_global, dtype=dtype, device=device)
        mass.index_add_(0, mesh.ibool.reshape(-1), mesh.w_jac.reshape(-1))
        vp_squared_node = node_field(mesh.ibool, mesh.n_global, vp_e * vp_e)
        J = mesh.inv_J
        I = (J[..., 0, 0], J[..., 0, 1], J[..., 0, 2],
             J[..., 1, 0], J[..., 1, 1], J[..., 1, 2],
             J[..., 2, 0], J[..., 2, 1], J[..., 2, 2])
        return cls(
            mesh=mesh, vp_e=vp_e, vp_squared_node=vp_squared_node, mass=mass,
            _inv_M=1.0 / mass, dirichlet=tuple(dirichlet),
            dirichlet_idx=free_surface_node_idx(mesh, dirichlet, device),
            cpml=cpml, _D=mesh.deriv, _w_J=mesh.w_jac, _I=I)

    # ------------------------------------------------------------------ #
    # physical gradient via the 3×3 inv_J chain rule
    # ------------------------------------------------------------------ #
    def _phys_grad_from_ref(self, dxi, deta, dgam):
        (Ixx, Ixy, Ixz, Iex, Iey, Iez, Igx, Igy, Igz) = self._I
        du_dx = Ixx * dxi + Iex * deta + Igx * dgam
        du_dy = Ixy * dxi + Iey * deta + Igy * dgam
        du_dz = Ixz * dxi + Iez * deta + Igz * dgam
        return du_dx, du_dy, du_dz

    def stiffness_apply(self, u_global: torch.Tensor) -> torch.Tensor:
        """``(K u)_global = Σ_e ∫ ∇v · ∇u dΩ`` (c²-free) assembled to global nodes."""
        D = self._D
        u_e = u_global[self.mesh.ibool]
        du_dx, du_dy, du_dz = self._phys_grad_from_ref(
            _dxi(D, u_e), _deta(D, u_e), _dgam(D, u_e))
        F_xi, F_eta, F_gam = self._proj_ref(du_dx, du_dy, du_dz)
        r_e = _DTxi(D, F_xi) + _DTeta(D, F_eta) + _DTgam(D, F_gam)
        return assemble(r_e, self.mesh.ibool, u_global)

    # ------------------------------------------------------------------ #
    # CPML directional operators (no c²; deformed inv_J). In the rectangular
    # absorbing band the cross terms vanish and these match the Hex8 operators.
    # ------------------------------------------------------------------ #
    def _proj_ref(self, q_x, q_y, q_z):
        """w_J · inv_J · q  ->  (F_xi, F_eta, F_gam)."""
        (Ixx, Ixy, Ixz, Iex, Iey, Iez, Igx, Igy, Igz) = self._I
        wJ = self._w_J
        return (wJ * (q_x * Ixx + q_y * Ixy + q_z * Ixz),
                wJ * (q_x * Iex + q_y * Iey + q_z * Iez),
                wJ * (q_x * Igx + q_y * Igy + q_z * Igz))

    def _cpml_grads(self, u_global, apsi_x, apsi_y, apsi_z):
        # Separate one-wide reference-derivative einsums (one field at a time)
        # instead of a 4-field stacked einsum per direction — ~1.2x faster on the
        # GPU even though the deformed inv_J uses all 12 derivatives (a narrow
        # einsum beats a fat one here; see the flat acoustic3d._cpml_grads note).
        D = self._D
        ibool = self.mesh.ibool
        wJ = self._w_J
        ex = lambda F: contract("xX,ezyX->ezyx", D, F)
        ey = lambda F: contract("yY,ezYx->ezyx", D, F)
        ez = lambda F: contract("zZ,eZyx->ezyx", D, F)
        u_e = u_global[ibool]
        du_dx, du_dy, du_dz = self._phys_grad_from_ref(ex(u_e), ey(u_e), ez(u_e))
        gx_e = wJ * du_dx; gy_e = wJ * du_dy; gz_e = wJ * du_dz
        # K_d u = ∫ ∂v/∂x_d ∂u/∂x_d : project (q_d = du_dx_d e_d) to ref, D^T.
        z = torch.zeros_like(du_dx)
        Fx_xi, Fx_eta, Fx_gam = self._proj_ref(du_dx, z, z)
        Fy_xi, Fy_eta, Fy_gam = self._proj_ref(z, du_dy, z)
        Fz_xi, Fz_eta, Fz_gam = self._proj_ref(z, z, du_dz)
        kx_e = _DTxi(D, Fx_xi) + _DTeta(D, Fx_eta) + _DTgam(D, Fx_gam)
        ky_e = _DTxi(D, Fy_xi) + _DTeta(D, Fy_eta) + _DTgam(D, Fy_gam)
        kz_e = _DTxi(D, Fz_xi) + _DTeta(D, Fz_eta) + _DTgam(D, Fz_gam)
        # ∂(a·ψ_d)/∂x_d via chain rule on each ψ_d's own reference derivatives.
        (Ixx, Ixy, Ixz, Iex, Iey, Iez, Igx, Igy, Igz) = self._I
        apx_e, apy_e, apz_e = apsi_x[ibool], apsi_y[ibool], apsi_z[ibool]
        gax_e = wJ * (Ixx * ex(apx_e) + Iex * ey(apx_e) + Igx * ez(apx_e))
        gay_e = wJ * (Ixy * ex(apy_e) + Iey * ey(apy_e) + Igy * ez(apy_e))
        gaz_e = wJ * (Ixz * ex(apz_e) + Iez * ey(apz_e) + Igz * ez(apz_e))
        return assemble_each(
            [gx_e, gy_e, gz_e, kx_e, ky_e, kz_e, gax_e, gay_e, gaz_e],
            ibool, u_global)

    def step_cpml(self, u_now, psi_x, psi_y, psi_z, zeta_x, zeta_y, zeta_z):
        cp = self.cpml
        if cp is None:
            raise RuntimeError("CPML not configured on AcousticSEM3DDeformed")
        inv_M = self._inv_M
        gx, gy, gz, kx, ky, kz, gax, gay, gaz = self._cpml_grads(
            u_now, cp.a_x * psi_x, cp.a_y * psi_y, cp.a_z * psi_z)
        dudx = gx * inv_M; dudy = gy * inv_M; dudz = gz * inv_M
        lap_x = -kx * inv_M; lap_y = -ky * inv_M; lap_z = -kz * inv_M
        grad_apsi_x = gax * inv_M; grad_apsi_y = gay * inv_M; grad_apsi_z = gaz * inv_M
        one_pbx = 1.0 + cp.b_x; one_pby = 1.0 + cp.b_y; one_pbz = 1.0 + cp.b_z
        tmp_x = one_pbx * lap_x + cp.dbx_dx * dudx + grad_apsi_x
        tmp_y = one_pby * lap_y + cp.dby_dy * dudy + grad_apsi_y
        tmp_z = one_pbz * lap_z + cp.dbz_dz * dudz + grad_apsi_z
        w_sum = (one_pbx * tmp_x + cp.a_x * zeta_x
                 + one_pby * tmp_y + cp.a_y * zeta_y
                 + one_pbz * tmp_z + cp.a_z * zeta_z)
        psi_x_new = cp.b_x * dudx + cp.a_x * psi_x
        psi_y_new = cp.b_y * dudy + cp.a_y * psi_y
        psi_z_new = cp.b_z * dudz + cp.a_z * psi_z
        zeta_x_new = cp.b_x * tmp_x + cp.a_x * zeta_x
        zeta_y_new = cp.b_y * tmp_y + cp.a_y * zeta_y
        zeta_z_new = cp.b_z * tmp_z + cp.a_z * zeta_z
        return (w_sum, psi_x_new, psi_y_new, psi_z_new,
                zeta_x_new, zeta_y_new, zeta_z_new)

    # ================================================================== #
    # SEMPropagator interface (same layout as AcousticSEM3D).
    # ================================================================== #
    def step(self, fields: tuple, src, dt: float) -> tuple:
        u_now, u_prev, psi_x, psi_y, psi_z, zeta_x, zeta_y, zeta_z = fields
        dt2, inv_M = dt * dt, self._inv_M
        b = src[0]
        if self.cpml is not None:
            (w_sum, psi_x, psi_y, psi_z,
             zeta_x, zeta_y, zeta_z) = self.step_cpml(
                u_now, psi_x, psi_y, psi_z, zeta_x, zeta_y, zeta_z)
            u_next = (2.0 * u_now - u_prev
                      + dt2 * self.vp_squared_node * w_sum + dt2 * inv_M * b)
            (u_next, psi_x, psi_y, psi_z,
             zeta_x, zeta_y, zeta_z) = self._apply_dirichlet(
                u_next, psi_x, psi_y, psi_z, zeta_x, zeta_y, zeta_z)
        else:
            # c² per node, as on the CPML path (where w_sum = -K u / M)
            Ku = self.stiffness_apply(u_now)
            u_next = (2.0 * u_now - u_prev
                      + dt2 * self.vp_squared_node * (-Ku * inv_M) + dt2 * inv_M * b)
            (u_next,) = self._apply_dirichlet(u_next)
        return (u_next, u_now, psi_x, psi_y, psi_z, zeta_x, zeta_y, zeta_z)

    # Boundary-saving hooks (swap2nd; identical shape to AcousticSEM3D).
    bs_ring_save: ClassVar[list] = [0]
    bs_ring_restore: ClassVar[list] = [(0, 0, 0), (1, 0, -1)]

    def reverse_step(self, fields, src, dt):
        u_ip1, u_i = fields[0], fields[1]
        z = torch.zeros_like(u_i)
        u_im1 = self.step((u_i, u_ip1, z, z, z, z, z, z), src, dt)[0]
        return (u_i, u_im1, z, z, z, z, z, z)

    def bs_model_tensors(self) -> dict:
        return {"vp2": self.vp_squared_node}

    def bs_recompute(self, fields, src, dt, leaves):
        saved = self.vp_squared_node
        self.vp_squared_node = leaves["vp2"]
        try:
            return self.step(fields, src, dt)
        finally:
            self.vp_squared_node = saved
