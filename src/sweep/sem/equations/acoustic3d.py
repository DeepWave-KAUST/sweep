"""Element-local SEM operators for 3-D constant-density acoustic.

The 3-D analogue of :mod:`sweep.sem.equations.acoustic2d`. Same weak form, same
double-convolution CPML, one extra spatial direction.

Strong form    :  ∂²u/∂t²  =  c²(x) ∇²u  +  s(x, t)
Discretized    :  ü  =  c² M⁻¹ (b − K u)     (c² per global node, M diagonal,
                                          K = ∫ ∇v · ∇u dΩ c²-free; see
                                          :mod:`.acoustic2d` for why)

Conventions
-----------
- Element-local axes ``(kz, ky, kx)`` ↔ reference ``(ζ, η, ξ)`` ↔ physical
  ``(z, y, x)``; all fields shape ``(E, ngll, ngll, ngll)``.
- Geometric direction factors on the axis-aligned hex (constant per element):
  ``G_a = (2/d_a)² · |J|`` with ``|J| = dx·dy·dz/8`` — so the stiffness weight
  along ``a`` is ``w₃ · G_a`` (``w₃`` = GLL weight product). This is the direct
  generalisation of the 2-D ``Jx = dz/dx`` factor.

Boundaries
----------
- Free surface: ``dirichlet`` (pressure-release) node set, zeroed each step.
- Outgoing waves: **CPML** (3-direction double-convolution, Komatitsch & Martin
  2007), matching the 2-D path and ``sweep.equations``' FD CPML. ``cpml=None``
  is a bare reflecting box (operator unit tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from .._assembly import assemble, assemble_each
from ..cpml.acoustic3d import CPML3DProfile
from ..mesh.boundaries import free_surface_node_idx
from ..mesh.nodes import node_field
from ..fields import DerivedInitState, DirichletNodes, FieldSpec, check_element_arrays
from ..mesh.hex8 import Hex8Mesh, boundary_element_faces


@dataclass
class AcousticSEM3D(DerivedInitState, DirichletNodes):
    """Cached factors + ops for 3-D acoustic SEM on a :class:`Hex8Mesh`."""

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

    mesh: Hex8Mesh
    vp_e: torch.Tensor            # (E, ngll, ngll, ngll) — c at GLL nodes
    vp_squared_node: torch.Tensor # (n_global,) — c² projected to global nodes
    mass: torch.Tensor            # (n_global,)
    _inv_M: torch.Tensor          # (n_global,)
    dirichlet: tuple[str, ...]
    dirichlet_idx: torch.Tensor
    cpml: Optional[CPML3DProfile]

    _D: torch.Tensor              # (ngll, ngll) Lagrange derivative matrix
    _w_jx_noc2: torch.Tensor      # w₃ · G_x  (no c²)
    _w_jy_noc2: torch.Tensor
    _w_jz_noc2: torch.Tensor
    _w_jac: torch.Tensor          # CPML path: w₃ · |J|
    _inv_half_dx: float           # 2 / dx_e
    _inv_half_dy: float
    _inv_half_dz: float

    @classmethod
    def build(
        cls,
        mesh: Hex8Mesh,
        vp_e: torch.Tensor,
        *,
        dirichlet: tuple[str, ...] = ("top",),
        cpml: Optional[CPML3DProfile] = None,
    ) -> "AcousticSEM3D":
        check_element_arrays(mesh, vp_e=vp_e)
        device, dtype = vp_e.device, vp_e.dtype
        w = mesh.w_ref
        w3 = w[:, None, None] * w[None, :, None] * w[None, None, :]   # (ng,ng,ng)

        # Mass: M[g] = Σ_e w₃ |J|
        m_e = (w3 * mesh.jacobian_det).expand(mesh.n_elements, -1, -1, -1)
        mass = torch.zeros(mesh.n_global, dtype=dtype, device=device)
        mass.index_add_(0, mesh.ibool.reshape(-1), m_e.reshape(-1))

        # Direction factors G_a = (2/d_a)² · |J|.
        jac = mesh.jacobian_det
        Gx = (2.0 / mesh.dx_e) ** 2 * jac
        Gy = (2.0 / mesh.dy_e) ** 2 * jac
        Gz = (2.0 / mesh.dz_e) ** 2 * jac
        c2 = vp_e * vp_e
        ones_E = (lambda t: t.expand(mesh.n_elements, -1, -1, -1).contiguous())
        w_jx_noc2 = ones_E(w3 * Gx)
        w_jy_noc2 = ones_E(w3 * Gy)
        w_jz_noc2 = ones_E(w3 * Gz)
        w_jac = ones_E(w3 * jac)

        # Per-global-node c² from one element copy (see mesh/nodes.py).
        vp_squared_node = node_field(mesh.ibool, mesh.n_global, c2)

        return cls(
            mesh=mesh, vp_e=vp_e, vp_squared_node=vp_squared_node,
            mass=mass, _inv_M=1.0 / mass,
            dirichlet=tuple(dirichlet),
            dirichlet_idx=free_surface_node_idx(mesh, dirichlet, device),
            cpml=cpml,
            _D=mesh.deriv,
            _w_jx_noc2=w_jx_noc2, _w_jy_noc2=w_jy_noc2, _w_jz_noc2=w_jz_noc2,
            _w_jac=w_jac,
            _inv_half_dx=2.0 / mesh.dx_e,
            _inv_half_dy=2.0 / mesh.dy_e,
            _inv_half_dz=2.0 / mesh.dz_e,
        )

    # ------------------------------------------------------------------ #
    # Symmetric stiffness K · u  (c²-free) — bare path + unit tests.
    # ------------------------------------------------------------------ #
    def stiffness_apply(self, u_global: torch.Tensor) -> torch.Tensor:
        """``(K u)_global = Σ_e ∫ ∇v · ∇u dΩ`` assembled to global nodes."""
        D = self._D
        u_e = u_global[self.mesh.ibool]                       # (E,kz,ky,kx)
        du_dxi   = torch.einsum("xX,ezyX->ezyx", D, u_e)      # ∂/∂x (last axis)
        du_deta  = torch.einsum("yY,ezYx->ezyx", D, u_e)      # ∂/∂y
        du_dzeta = torch.einsum("zZ,eZyx->ezyx", D, u_e)      # ∂/∂z
        r_e = (
            torch.einsum("Xx,ezyX->ezyx", D, self._w_jx_noc2 * du_dxi)
            + torch.einsum("Yy,ezYx->ezyx", D, self._w_jy_noc2 * du_deta)
            + torch.einsum("Zz,eZyx->ezyx", D, self._w_jz_noc2 * du_dzeta)
        )
        return assemble(r_e, self.mesh.ibool, u_global)

    # ------------------------------------------------------------------ #
    # CPML building blocks: directional Galerkin operators without c².
    # ------------------------------------------------------------------ #
    def _cpml_grads(self, u_global, apsi_x, apsi_y, apsi_z):
        """Nine CPML directional quantities:
        ``(G_x u, G_y u, G_z u, K_x u, K_y u, K_z u,
           ∂(a·ψ_x)/∂x, ∂(a·ψ_y)/∂y, ∂(a·ψ_z)/∂z)``.

        Separate one-wide reference-derivative einsums (one field at a time) + a
        single batched scatter. Each ``a·ψ_d`` is only differentiated along its
        own direction, so only ``u`` needs all three. Splitting the derivatives
        into narrow einsums — rather than one 4-field stacked einsum per direction
        — is ~1.4× faster and ~15% less peak memory on the GPU: a wide multi-field
        einsum carries a component axis that hurts the contraction's memory access,
        so several narrow einsums beat one fat one here (measured, fp32, 40³/56³).
        """
        D = self._D
        ibool = self.mesh.ibool
        u_e = u_global[ibool]                                   # (E,kz,ky,kx)
        du_dxi   = torch.einsum("xX,ezyX->ezyx", D, u_e)
        du_deta  = torch.einsum("yY,ezYx->ezyx", D, u_e)
        du_dzeta = torch.einsum("zZ,eZyx->ezyx", D, u_e)
        gx_e = self._w_jac * (du_dxi   * self._inv_half_dx)
        gy_e = self._w_jac * (du_deta  * self._inv_half_dy)
        gz_e = self._w_jac * (du_dzeta * self._inv_half_dz)
        kx_e = torch.einsum("Xx,ezyX->ezyx", D, self._w_jx_noc2 * du_dxi)
        ky_e = torch.einsum("Yy,ezYx->ezyx", D, self._w_jy_noc2 * du_deta)
        kz_e = torch.einsum("Zz,eZyx->ezyx", D, self._w_jz_noc2 * du_dzeta)
        gax_e = self._w_jac * (torch.einsum("xX,ezyX->ezyx", D, apsi_x[ibool])
                               * self._inv_half_dx)
        gay_e = self._w_jac * (torch.einsum("yY,ezYx->ezyx", D, apsi_y[ibool])
                               * self._inv_half_dy)
        gaz_e = self._w_jac * (torch.einsum("zZ,eZyx->ezyx", D, apsi_z[ibool])
                               * self._inv_half_dz)
        return assemble_each(                           # nine (n_global,)
            [gx_e, gy_e, gz_e, kx_e, ky_e, kz_e, gax_e, gay_e, gaz_e],
            ibool, u_global)

    def step_cpml(self, u_now, psi_x, psi_y, psi_z, zeta_x, zeta_y, zeta_z):
        """Returns ``(w_sum, psi_x', psi_y', psi_z', zeta_x', zeta_y', zeta_z')``
        with ``w_sum`` the node-wise quantity multiplied by ``dt²·vp²``:

            u_next = 2 u_now − u_pre + dt² · vp²(x) · w_sum + source/M
        """
        cp = self.cpml
        if cp is None:
            raise RuntimeError("CPML not configured on this AcousticSEM3D")
        inv_M = self._inv_M
        gx, gy, gz, kx, ky, kz, gax, gay, gaz = self._cpml_grads(
            u_now, cp.a_x * psi_x, cp.a_y * psi_y, cp.a_z * psi_z)
        dudx = gx * inv_M; dudy = gy * inv_M; dudz = gz * inv_M
        lap_x = -kx * inv_M; lap_y = -ky * inv_M; lap_z = -kz * inv_M
        grad_apsi_x = gax * inv_M
        grad_apsi_y = gay * inv_M
        grad_apsi_z = gaz * inv_M

        one_pbx = 1.0 + cp.b_x
        one_pby = 1.0 + cp.b_y
        one_pbz = 1.0 + cp.b_z
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
    # SEMPropagator interface. State layout = field_specs order:
    #   (u, u_prev, psi_x, psi_y, psi_z, zeta_x, zeta_y, zeta_z)
    # ================================================================== #
    def step(self, fields: tuple, src, dt: float) -> tuple:
        u_now, u_prev, psi_x, psi_y, psi_z, zeta_x, zeta_y, zeta_z = fields
        dt2, inv_M = dt * dt, self._inv_M
        b = src[0]                                  # source -> field 0 (u)
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

    # ================================================================== #
    # Boundary-saving hooks (swap2nd reverse) — same shape as 2-D, with
    # six CPML memory fields zeroed in the reverse recompute.
    # ================================================================== #
    # Ring-save u AND the six CPML memories on the full band so the reverse
    # reconstruction restores the exact step input (memories are exactly zero
    # outside the band) and the BS gradient equals the full-tape one.
    bs_ring_save: ClassVar[list] = [0, 2, 3, 4, 5, 6, 7]
    bs_ring_restore: ClassVar[list] = [(0, 0, 0), (1, 0, -1), (2, 1, -1),
                                       (3, 2, -1), (4, 3, -1), (5, 4, -1),
                                       (6, 5, -1), (7, 6, -1)]
    # psi_d / zeta_d are exactly zero off b_d's support, so each rings on
    # band & (b_d != 0) -- the union of the three slabs is ~2.4x any one.
    bs_ring_support: ClassVar[dict] = {2: "b_x", 3: "b_y", 4: "b_z",
                                       5: "b_x", 6: "b_y", 7: "b_z"}

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
