"""1st-order velocity-stress Biot poroelastic SEM on a Hex27 deformed mesh.

The 3-D :class:`~sweep.sem.equations.biot_vs_deformed.BiotVSSEM2DDeformed`: the same
low-frequency Biot equations (solid velocity ``v``, Darcy flux ``q``, total stress ``sigma``,
pore pressure ``p``), the same placement -- ``v`` and ``p`` continuous nodal fields, ``q`` and
``sigma`` element-local, the drag integrated exactly over each step at every node -- and the
same sources and boundaries, on the geometry of
:class:`~sweep.sem.equations.elastic_vs3d_deformed.ElasticVSSEM3DDeformed` (the per-GLL
inverse Jacobian of a Hex27 mesh; an axis-aligned one is the flat box). See the 2-D class for
the equations and the discretisation. What differs:

* The flux copies at a node, one per element that shares it (1 inside an element, 2 on a
  face, 4 on an edge, 8 at a corner), are grouped by their number ``k``, and the drag
  operators ``E``, ``Psi`` are ``(n_k, k, k)`` per group: about ``2 x 343`` numbers per
  element at ``n_order = 4``, where padding every node to eight copies would take
  ``2 x 64 x 64``. The copies are a permutation of the element-local nodes, so moving a
  field between the two layouts is one gather.
* Porous and dry elastic elements only. A fluid element (``rho = 0``) is refused, because the
  fluid-solid interface is 2-D; a porous--dry interface needs nothing (sealed pores, as in
  2-D). The drag is isotropic (``b``).
* Sources as in 2-D, with ``y``: the moment tensor ``sxx`` ... ``syz`` on the total stress
  (``["sxx", "syy", "szz"]``, the explosion, is the default), ``vx`` / ``vy`` / ``vz`` and
  ``qx`` / ``qy`` / ``qz`` (``source_mode`` ``"velocity"``), ``fx`` / ``fy`` / ``fz`` (the
  force on both equations, ``"load"``) and the pore-pressure injection ``p``.
* The C-PML memories -- nine for the stress divergence, three for the flux divergence, three
  for the pressure gradient and nine for the velocity gradients -- live on every element.

``src`` is aligned with ``(v_x, v_y, v_z, p, q_x, q_y, q_z)``. Eager only; no boundary
saving (the drag is dissipative).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import ClassVar, Optional

import torch

from .._assembly import assemble, phys_grads3, stress_divergence3, vector_divergence3
from ..cpml.elastic_vs3d import SimpleElasticCPML3DProfile, build_simple_elastic_cpml_3d
from ..fields import FieldSpec, check_element_arrays, init_state_from_specs
from ..mesh.boundaries import free_surface_node_idx
from ..mesh.domains import masked_inv_mass
from ..mesh.hex27 import Hex27Mesh
from .biot_vs_deformed import (_SOURCE_MODES, BiotVSSEM2DDeformed, _media, _node_slots,
                               viscous_relaxation_times)

_COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")
_MEMORIES = ("m_sxx_x", "m_sxy_y", "m_sxz_z", "m_sxy_x", "m_syy_y", "m_syz_z",
             "m_sxz_x", "m_syz_y", "m_szz_z",                       # stress divergence
             "m_qx_x", "m_qy_y", "m_qz_z",                          # flux divergence
             "m_px", "m_py", "m_pz",                                # pressure gradient
             "m_vx_x", "m_vx_y", "m_vx_z", "m_vy_x", "m_vy_y", "m_vy_z",
             "m_vz_x", "m_vz_y", "m_vz_z")                          # velocity gradients


def _flux_blocks(ibool, n_global):
    """The element copies of the nodes, grouped by their number: ``(perm, blocks)``.

    ``perm`` orders the flattened element-local nodes by (copies at the node, node, rank of
    the copy) -- a permutation, every copy being in exactly one group. A group is ``(k,
    nodes)``: the nodes (ascending) that ``k`` elements share, whose ``len(nodes) * k``
    consecutive entries of ``perm`` are their copies, node by node."""
    node, slot, kmax = _node_slots(ibool)
    count = torch.bincount(node, minlength=n_global)
    perm = torch.argsort((count[node] * (n_global + 1) + node) * kmax + slot)
    blocks = [(int(k), torch.nonzero(count == k).squeeze(1))
              for k in torch.unique(count).tolist()]
    return perm, blocks


@dataclass
class BiotVSSEM3DDeformed:
    """1st-order velocity-stress Biot poroelastic SEM on a deformed Hex27 mesh."""

    field_specs: ClassVar[list] = [
        FieldSpec("v_x", supports_source=True, supports_receiver=True),
        FieldSpec("v_y", supports_source=True, supports_receiver=True),
        FieldSpec("v_z", supports_source=True, supports_receiver=True),
        FieldSpec("p", supports_source=True, supports_receiver=True),
        FieldSpec("q_x", supports_source=True, supports_receiver=True, layout="element"),
        FieldSpec("q_y", supports_source=True, supports_receiver=True, layout="element"),
        FieldSpec("q_z", supports_source=True, supports_receiver=True, layout="element"),
        *[FieldSpec(f"sigma_{c}", supports_receiver=True, layout="element")
          for c in _COMPONENTS],
        *[FieldSpec(f"r_{c}", internal=True, layout="element") for c in "xyz"],  # viscous_q
        *[FieldSpec(n, internal=True, boundary_related=True, layout="element")
          for n in _MEMORIES],
    ]

    # FD Biot's receiver names
    derived_receiver_types: ClassVar[dict] = {
        **{f"v{c}": (f"v_{c}", None) for c in "xyz"},
        **{f"q{c}": (f"q_{c}", None) for c in "xyz"},
        **{f"s{c}": (f"sigma_{c}", None) for c in _COMPONENTS}}

    # PropSEM's source names over (v_x, v_y, v_z, p, q_x, q_y, q_z), or the moment
    # (Mxx, Myy, Mzz, Mxy, Mxz, Myz): see the 2-D class
    propsem_source_names: ClassVar[dict] = {
        **{f"{name}{c}": ("velocity", tuple(float(j == i + off) for j in range(7)))
           for name, off in (("v", 0), ("v_", 0), ("q", 4), ("q_", 4))
           for i, c in enumerate("xyz")},
        **{f"f{c}": ("force", tuple(float(j in (i, i + 4)) for j in range(7)))
           for i, c in enumerate("xyz")},
        "p": ("injection", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
        **{f"s{c}": ("moment", tuple(float(j == i) for j in range(6)))
           for i, c in enumerate(_COMPONENTS)},
    }
    default_source_type: ClassVar[tuple] = ("sxx", "syy", "szz")
    default_receiver_type: ClassVar[tuple] = ("v_x", "v_y", "v_z")
    bs_refusal: ClassVar[str] = BiotVSSEM2DDeformed.bs_refusal

    mesh: Hex27Mesh
    H_e: torch.Tensor                  # (E, ngll, ngll, ngll) undrained P-wave modulus
    mu_e: torch.Tensor
    C_e: torch.Tensor
    M_e: torch.Tensor
    rho_e: torch.Tensor
    rho_f_e: torch.Tensor
    m_e: torch.Tensor
    b_e: torch.Tensor                  # the drag eta / kappa, relaxed
    dirichlet: tuple                   # the open-pore faces: p = 0 on their nodes
    dirichlet_idx: torch.Tensor
    viscous_q: Optional[tuple]

    mass_rho: torch.Tensor             # (n_global,) A(w rho)
    _inv_M: torch.Tensor               # 1 / mass_rho
    _inv_Mm: torch.Tensor              # 1 / A(w m) over the pore fluid: a flux injected at a point
    _inv_Mp: torch.Tensor              # 1 / A(w / M), the pressure's lumped compliance
    _inv_M1: torch.Tensor              # 1 / A(w) over the pore fluid: a pressure injected
    _lam_d: torch.Tensor               # drained lambda, H - 2 mu - C^2 / M
    _alpha: torch.Tensor               # C / M (0 in a dry solid)
    _solid_e: torch.Tensor             # (E,) bool, every element here
    _fluid_e: torch.Tensor             # (E,) bool, the porous ones
    _qmask: torch.Tensor               # (E, 1, 1, 1) the flux mask
    _tau: tuple                        # (tau_e, tau_s), equal without viscous_q
    # the flux copies: the permutation of the flattened element-local nodes, its inverse, the
    # inverse of the node order the groups list them in, and per group (k, nodes, w m,
    # w rho_f, w b tau_e / tau_s, a nodal flux load's share w / A(w))
    _perm: torch.Tensor
    _inv_perm: torch.Tensor
    _node_inv: torch.Tensor
    _blocks: tuple

    cpml: Optional[SimpleElasticCPML3DProfile]
    _D: torch.Tensor                   # (ngll, ngll)
    _w_J: torch.Tensor                 # (E, ngll, ngll, ngll) w |J|
    _inv_J: torch.Tensor               # (E, ngll, ngll, ngll, 3, 3)

    source_mode: str = "load"          # see the 2-D class
    _drag: tuple = None                # (dt, E per group, Psi per group), set by init_state

    @classmethod
    def build(
        cls,
        mesh: Hex27Mesh,
        H_e: torch.Tensor,
        mu_e: torch.Tensor,
        C_e: torch.Tensor,
        M_e: torch.Tensor,
        rho_e: torch.Tensor,
        rho_f_e: torch.Tensor,
        m_e: torch.Tensor,
        b_e: torch.Tensor,
        *,
        dirichlet: tuple = (),
        cpml: Optional[SimpleElasticCPML3DProfile] = None,
        viscous_q: Optional[tuple] = None,
    ) -> "BiotVSSEM3DDeformed":
        """Build the operators from the eight Biot coefficients at the GLL nodes;
        ``dirichlet`` names the open-pore faces, ``viscous_q = (Q0, f0)`` turns on the
        viscous attenuation."""
        check_element_arrays(mesh, H_e=H_e, mu_e=mu_e, C_e=C_e, M_e=M_e, rho_e=rho_e,
                             rho_f_e=rho_f_e, m_e=m_e, b_e=b_e)
        if bool((b_e < 0).any()):
            raise ValueError("the drag eta / kappa must be >= 0")
        solid, fluid = _media(H_e, mu_e, C_e, M_e, rho_e, rho_f_e, m_e,
                              (b_e, torch.zeros_like(b_e), b_e), False)
        if bool((fluid & ~solid).any()):
            raise NotImplementedError(
                f"{cls.__name__}: element {int((fluid & ~solid).nonzero()[0])} is a fluid "
                f"(rho = 0), and the fluid-solid interface is 2-D only "
                f"(BiotVSSEM2DDeformed); here every element has a solid frame")
        tau = (1.0, 1.0) if viscous_q is None else viscous_relaxation_times(*viscous_q)
        device = H_e.device
        w = mesh.w_jac
        ibool = mesh.ibool
        like = H_e.new_zeros(mesh.n_global)

        def lump(c):
            return assemble(w * c, ibool, like)

        p_alive = torch.zeros(mesh.n_global, dtype=torch.bool, device=device).index_fill(
            0, ibool[fluid].reshape(-1), True)
        fl = fluid.view(-1, 1, 1, 1)
        qmask = fl.to(H_e.dtype)
        # the coefficients a dry solid does not have are zero (checked above); through the
        # mask their gradient is zero too, so an update keeps them so
        C_e, rho_f_e, m_e, b_e = C_e * qmask, rho_f_e * qmask, m_e * qmask, b_e * qmask
        inv_M_e = torch.where(fl, 1.0 / torch.where(fl, M_e, torch.ones_like(M_e)), 0.0)
        mass_rho = lump(rho_e)
        inv_M1 = masked_inv_mass(lump(qmask.expand_as(M_e)), p_alive)
        alpha = C_e * inv_M_e
        unrelax = tau[0] / tau[1]

        perm, groups = _flux_blocks(ibool, mesh.n_global)
        sizes = [k * nodes.numel() for k, nodes in groups]

        def to_blocks(x):
            return [b.view(-1, k) for b, (k, _) in
                    zip(torch.split(x.reshape(-1)[perm], sizes), groups)]

        blocks = tuple(
            SimpleNamespace(k=k, nodes=nodes, wm=wm, wr=wr, wb=wb,
                            share=sh * inv_M1[nodes][:, None])
            for (k, nodes), wm, wr, wb, sh in zip(
                groups, to_blocks(torch.where(fl, w * m_e, torch.ones_like(w))),
                to_blocks(w * rho_f_e), to_blocks(w * b_e * unrelax),
                to_blocks(w * qmask)))
        node_order = torch.cat([nodes for _, nodes in groups])
        return cls(
            mesh=mesh, H_e=H_e, mu_e=mu_e, C_e=C_e, M_e=M_e, rho_e=rho_e,
            rho_f_e=rho_f_e, m_e=m_e, b_e=b_e,
            dirichlet=tuple(dirichlet),
            dirichlet_idx=free_surface_node_idx(mesh, dirichlet, device),
            viscous_q=None if viscous_q is None else tuple(float(v) for v in viscous_q),
            mass_rho=mass_rho, _inv_M=1.0 / mass_rho,
            _inv_Mm=masked_inv_mass(lump(m_e), p_alive),
            _inv_Mp=masked_inv_mass(lump(inv_M_e), p_alive), _inv_M1=inv_M1,
            _lam_d=H_e - 2.0 * mu_e - alpha * C_e, _alpha=alpha,
            _solid_e=solid, _fluid_e=fluid, _qmask=qmask, _tau=tau,
            _perm=perm, _inv_perm=torch.argsort(perm), _node_inv=torch.argsort(node_order),
            _blocks=blocks,
            cpml=cpml, _D=mesh.deriv, _w_J=w, _inv_J=mesh.inv_J,
        )

    # The C-PML builder ``cpml=`` expects, and the speed PropSEM designs it for.
    cpml_builder: ClassVar[object] = staticmethod(build_simple_elastic_cpml_3d)
    pml_speed: ClassVar[object] = staticmethod(BiotVSSEM2DDeformed.pml_speed)

    @property
    def source_field_elements(self):
        """The elements each source field ``(v_x, v_y, v_z, p, q_x, q_y, q_z)`` lives in."""
        s, f = self._solid_e, self._fluid_e
        return (s, s, s, f, f, f, f)

    # ------------------------------------------------------------------ #
    # The drag: one matrix exponential per node and group, for the run's dt
    # ------------------------------------------------------------------ #
    def drag_operators(self, dt: float, chunk: int = 1 << 15):
        """``(E, Psi)`` per group, each ``(n_k, k, k)``, advancing the flux copies at a node
        by ``q <- E q + Psi r`` over a step of frozen forces ``r`` (all three components
        alike): from ``A dq/dt + W_B q = r``, ``A = diag(w m) - w rho_f (w rho_f)^T /
        M_rho``, the exponential of the 2-D class, in float64. ``chunk`` nodes at a time."""
        inv_rho = 1.0 / self.mass_rho.double()
        dtype = self.mass_rho.dtype
        Es, Psis = [], []
        for blk in self._blocks:
            wm, wr, wb = blk.wm.double(), blk.wr.double(), blk.wb.double()
            ir = inv_rho[blk.nodes]
            e_parts, p_parts = [], []
            for i in range(0, wm.shape[0], chunk):
                n = slice(i, i + chunk)
                A = torch.diag_embed(wm[n]) - wr[n, :, None] * wr[n, None, :] * ir[n, None, None]
                A_inv = torch.linalg.inv(A)
                top = torch.cat([-A_inv @ torch.diag_embed(wb[n]), A_inv], 2)
                X = torch.linalg.matrix_exp(torch.cat([top, torch.zeros_like(top)], 1) * dt)
                e_parts.append(X[:, :blk.k, :blk.k].to(dtype))
                p_parts.append(X[:, :blk.k, blk.k:].to(dtype))
            Es.append(torch.cat(e_parts))
            Psis.append(torch.cat(p_parts))
        return tuple(Es), tuple(Psis)

    def init_state(self, dt: float, device, dtype) -> tuple:
        """Zero fields, and the drag operators for this ``dt`` -- computed here, once per
        run, outside any checkpointed chunk of the time loop."""
        if self._drag is None or self._drag[0] != dt:
            self._drag = (dt, *self.drag_operators(dt))
        return init_state_from_specs(self, device, dtype)

    def _to_blocks(self, x):
        """Element-local ``(E, ngll, ngll, ngll)`` -> the copies, ``(n_k, k)`` per group."""
        sizes = [blk.k * blk.nodes.numel() for blk in self._blocks]
        return [b.view(-1, blk.k) for b, blk in
                zip(torch.split(x.reshape(-1)[self._perm], sizes), self._blocks)]

    def _from_blocks(self, blocks, like):
        return torch.cat([b.reshape(-1) for b in blocks])[self._inv_perm].view_as(like)

    def _pin(self, p):
        """``p = 0`` on the open-pore surface."""
        idx = self.dirichlet_idx
        return p if idx.numel() == 0 else p.index_fill(0, idx, 0.0)

    # ================================================================== #
    # SEMPropagator interface: init_state + step (staggered leapfrog).
    # ================================================================== #
    def _frame_stress(self, s, grads, div_v, dt, extra):
        """The frame's share of the stress update, ``dt (lam_d div v I + mu (grad v +
        grad v^T))``, on ``s = (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz)`` from the nine
        ``grads`` in :func:`phys_grads3` order (``dvx/dx, dvx/dy, dvx/dz, dvy/dx, ...``).
        ``extra`` is the state after this class's fields, returned stepped (the 2-D seam)."""
        vx_x, vx_y, vx_z, vy_x, vy_y, vy_z, vz_x, vz_y, vz_z = grads
        lam_d, mu = self._lam_d, self.mu_e
        s_xx, s_yy, s_zz, s_xy, s_xz, s_yz = s
        s_xx = s_xx + dt * (lam_d * div_v + 2.0 * mu * vx_x)
        s_yy = s_yy + dt * (lam_d * div_v + 2.0 * mu * vy_y)
        s_zz = s_zz + dt * (lam_d * div_v + 2.0 * mu * vz_z)
        s_xy = s_xy + dt * mu * (vx_y + vy_x)
        s_xz = s_xz + dt * mu * (vx_z + vz_x)
        s_yz = s_yz + dt * mu * (vy_z + vz_y)
        return (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz), extra

    def step(self, fields: tuple, src, dt: float) -> tuple:
        """One staggered-leapfrog step (v, q at n+1/2; sigma, p, r at n+1)."""
        n = len(BiotVSSEM3DDeformed.field_specs)
        (v_x, v_y, v_z, p, q_x, q_y, q_z, s_xx, s_yy, s_zz, s_xy, s_xz, s_yz,
         r_x, r_y, r_z, *mem) = fields[:n]
        extra = fields[n:]
        mode = self.source_mode
        if mode not in _SOURCE_MODES:
            raise ValueError(f"source_mode must be one of {_SOURCE_MODES}, got {mode!r}")
        if self._drag is None or self._drag[0] != dt:
            raise RuntimeError(f"step(dt={dt}) needs init_state(dt) first: the drag "
                               f"operators are computed there, once per run")
        cp = self.cpml
        D, ibool, inv_J, w = self._D, self.mesh.ibool, self._inv_J, self._w_J
        like = torch.zeros_like(p)
        (m_sxx_x, m_sxy_y, m_sxz_z, m_sxy_x, m_syy_y, m_syz_z, m_sxz_x, m_syz_y, m_szz_z,
         m_qx_x, m_qy_y, m_qz_z, m_px, m_py, m_pz,
         m_vx_x, m_vx_y, m_vx_z, m_vy_x, m_vy_y, m_vy_z, m_vz_x, m_vz_y, m_vz_z) = mem

        # ----- pore pressure injected at t^n; the total stress does not move -----
        p = self._pin(p + dt * self._inv_M1 * src[3])

        # ----- forces at t^n: total-stress divergence and pressure gradient -----
        d_p_dx, d_p_dy, d_p_dz = phys_grads3(D, ibool, inv_J, p)
        if cp is not None:
            ax, bx, ay, by, az, bz = cp.a_x, cp.b_x, cp.a_y, cp.b_y, cp.a_z, cp.b_z
            m_sxx_x = ax * m_sxx_x + bx * s_xx
            m_sxy_y = ay * m_sxy_y + by * s_xy
            m_sxz_z = az * m_sxz_z + bz * s_xz
            m_sxy_x = ax * m_sxy_x + bx * s_xy
            m_syy_y = ay * m_syy_y + by * s_yy
            m_syz_z = az * m_syz_z + bz * s_yz
            m_sxz_x = ax * m_sxz_x + bx * s_xz
            m_syz_y = ay * m_syz_y + by * s_yz
            m_szz_z = az * m_szz_z + bz * s_zz
            m_px = ax * m_px + bx * d_p_dx
            m_py = ay * m_py + by * d_p_dy
            m_pz = az * m_pz + bz * d_p_dz
            d_p_dx, d_p_dy, d_p_dz = d_p_dx + m_px, d_p_dy + m_py, d_p_dz + m_pz
            sig = (s_xx + m_sxx_x, s_xy + m_sxy_y, s_xz + m_sxz_z,
                   s_xy + m_sxy_x, s_yy + m_syy_y, s_yz + m_syz_z,
                   s_xz + m_sxz_x, s_yz + m_syz_y, s_zz + m_szz_z)
        else:
            sig = (s_xx, s_xy, s_xz, s_xy, s_yy, s_yz, s_xz, s_yz, s_zz)
        f = [-r for r in stress_divergence3(D, ibool, inv_J, w, sig, like)]
        if mode == "load":
            f = [f[c] + src[c] for c in range(3)]

        # ----- v and the flux copies at every node, the drag integrated exactly -----
        inv_m, qmask = self._inv_M, self._qmask
        # w (-grad p + r) per copy; nothing moves the copies of a dry element
        g = [self._to_blocks(w * (r - d) * qmask)
             for r, d in ((r_x, d_p_dx), (r_y, d_p_dy), (r_z, d_p_dz))]
        q_old = [self._to_blocks(q) for q in (q_x, q_y, q_z)]
        fm = [fc * inv_m for fc in f]
        _, Es, Psis = self._drag
        news, sums = [], []
        for i, blk in enumerate(self._blocks):
            nodes = blk.nodes
            rb = [g[c][i] - blk.wr * fm[c][nodes][:, None] for c in range(3)]
            if mode == "load":
                rb = [rb[c] + blk.share * src[4 + c][nodes][:, None] for c in range(3)]
            qb = torch.stack([q_old[c][i] for c in range(3)], -1)          # (n_k, k, 3)
            nb = (torch.einsum("nij,njc->nic", Es[i], qb)
                  + torch.einsum("nij,njc->nic", Psis[i], torch.stack(rb, -1)))
            news.append(nb)
            sums.append((blk.wr[..., None] * (nb - qb)).sum(1))             # (n_k, 3)
        dq = torch.cat(sums)[self._node_inv]                                 # (n_global, 3)
        v_x = v_x + (dt * f[0] - dq[:, 0]) * inv_m
        v_y = v_y + (dt * f[1] - dq[:, 1]) * inv_m
        v_z = v_z + (dt * f[2] - dq[:, 2]) * inv_m
        q_x, q_y, q_z = (self._from_blocks([nb[..., c] for nb in news], q)
                         for c, q in enumerate((q_x, q_y, q_z)))
        if mode == "velocity":
            v_x = v_x + dt * inv_m * src[0]
            v_y = v_y + dt * inv_m * src[1]
            v_z = v_z + dt * inv_m * src[2]
            q_x = q_x + (dt * self._inv_Mm * src[4])[ibool] * qmask
            q_y = q_y + (dt * self._inv_Mm * src[5])[ibool] * qmask
            q_z = q_z + (dt * self._inv_Mm * src[6])[ibool] * qmask

        # ----- stress, pressure and the viscous memory from the new v, q -----
        (vx_x, vx_y, vx_z, vy_x, vy_y, vy_z,
         vz_x, vz_y, vz_z) = phys_grads3(D, ibool, inv_J, v_x, v_y, v_z)
        if cp is not None:
            m_vx_x = ax * m_vx_x + bx * vx_x
            m_vx_y = ay * m_vx_y + by * vx_y
            m_vx_z = az * m_vx_z + bz * vx_z
            m_vy_x = ax * m_vy_x + bx * vy_x
            m_vy_y = ay * m_vy_y + by * vy_y
            m_vy_z = az * m_vy_z + bz * vy_z
            m_vz_x = ax * m_vz_x + bx * vz_x
            m_vz_y = ay * m_vz_y + by * vz_y
            m_vz_z = az * m_vz_z + bz * vz_z
            vx_x, vx_y, vx_z = vx_x + m_vx_x, vx_y + m_vx_y, vx_z + m_vx_z
            vy_x, vy_y, vy_z = vy_x + m_vy_x, vy_y + m_vy_y, vy_z + m_vy_z
            vz_x, vz_y, vz_z = vz_x + m_vz_x, vz_y + m_vz_y, vz_z + m_vz_z
            m_qx_x = ax * m_qx_x + bx * q_x
            m_qy_y = ay * m_qy_y + by * q_y
            m_qz_z = az * m_qz_z + bz * q_z
            div_q = vector_divergence3(D, ibool, inv_J, w,
                                       (q_x + m_qx_x, q_y + m_qy_y, q_z + m_qz_z), like)
        else:
            div_q = vector_divergence3(D, ibool, inv_J, w, (q_x, q_y, q_z), like)
        div_v = vx_x + vy_y + vz_z
        alpha = self._alpha
        rhs = div_q - assemble(w * alpha * div_v, ibool, like)
        dp_free = self._pin(dt * self._inv_Mp * rhs)
        p = p + dp_free
        dsig_p = alpha * dp_free[ibool]                    # sigma = sigma' - alpha p
        (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz), extra = self._frame_stress(
            (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz),
            (vx_x, vx_y, vx_z, vy_x, vy_y, vy_z, vz_x, vz_y, vz_z), div_v, dt, extra)
        s_xx, s_yy, s_zz = s_xx - dsig_p, s_yy - dsig_p, s_zz - dsig_p
        if self.viscous_q is not None:
            # dr/dt = -(r - (tau_e / tau_s - 1) b q) / tau_s, exact over the step at q^{n+1/2}
            tau_e, tau_s = self._tau
            decay = math.exp(-dt / tau_s)
            gain = (1.0 - decay) * (tau_e / tau_s - 1.0)
            r_x = decay * r_x + gain * self.b_e * q_x
            r_y = decay * r_y + gain * self.b_e * q_y
            r_z = decay * r_z + gain * self.b_e * q_z

        return (v_x, v_y, v_z, p, q_x, q_y, q_z, s_xx, s_yy, s_zz, s_xy, s_xz, s_yz,
                r_x, r_y, r_z,
                m_sxx_x, m_sxy_y, m_sxz_z, m_sxy_x, m_syy_y, m_syz_z, m_sxz_x, m_syz_y,
                m_szz_z, m_qx_x, m_qy_y, m_qz_z, m_px, m_py, m_pz,
                m_vx_x, m_vx_y, m_vx_z, m_vy_x, m_vy_y, m_vy_z, m_vz_x, m_vz_y, m_vz_z,
                *extra)
