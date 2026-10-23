"""1st-order velocity-stress Biot poroelastic SEM on a Quad9 deformed mesh.

The low-frequency Biot equations in the form of the FD :class:`sweep.equations.Biot`
(solid velocity ``v``, Darcy flux ``q = phi (v_f - v)``, total stress ``sigma``, pore
pressure ``p``; Biot 1962, Carcione *Wave Fields in Real Media* ch. 7)::

    rho  dv/dt + rho_f dq/dt         = div sigma
    rho_f dv/dt + m dq/dt + b q      = -grad p
    d sigma/dt = (H - 2 mu) div v I + mu (grad v + grad v^T) + C div q I
    d p/dt     = -C div v - M div q

with the eight coefficients ``[H, mu, C, M, rho, rho_f, m, b]`` of
:func:`sweep.equations.biot.biot_from_rock`.

Discretisation
--------------
The solid is :class:`ElasticVSSEM2DDeformed`'s: ``v_x, v_z`` continuous (CG) nodal
fields, the total stress element-local, the momentum equation in weak form with a lumped
mass. The pore fluid is the pressure form of the acoustic SEM: ``p`` is a CG nodal field
whose equation is taken in weak form, and the Darcy flux ``q_x, q_z`` is element-local,
driven by the strong gradient of ``p``::

    M_p dp/dt = -A(w alpha div v) + D^T W q,      M_p = A(w / M),  alpha = C / M,
    sigma     = sigma' - alpha p I,               d sigma'/dt = lambda_d div v I + mu (grad v + grad v^T)

(``A`` assembles, ``D^T W`` is the weak divergence), so that a pore fluid on its own
(``rho_f = C = 0``) is exactly the acoustic pressure SEM. The other placement -- ``q``
continuous and ``p`` element-local, as in SPECFEM2D's poroelastic solver -- leaves the
relative flow, which has no shear stiffness, on a continuous field: a point source then
excites grid-scale modes that run along the grid lines through it, and the pore pressure
there grows like ``1 / dh`` under refinement. The two couplings (``alpha p`` in the
momentum, ``alpha div v`` in the pressure) are transposes of each other, as are ``D^T W q``
and the strong ``grad p``, so the semi-discrete system conserves the Biot energy up to the
viscous drag.

The two momentum equations meet at every node: the solid's (one, CG) and the element
copies of the flux there (one per element sharing the node). Eliminating ``dv/dt`` leaves,
per node, ``A dq/dt + W_b q = r`` over the copies, with
``A = diag(w m) - w rho_f (w rho_f)^T / M_rho``; with the forces frozen over the step this
is integrated exactly, ``q <- E q + Psi r`` (``E`` and ``Psi`` from one matrix exponential
per node, computed in :meth:`init_state` for the run's ``dt``), and then
``v <- v + (dt F - sum w rho_f (q_new - q)) / M_rho``. So any permeability is stable, the
time step is that of the inviscid fast P wave (see
:func:`sweep.equations.biot.biot_fast_p_velocity`), and ``b -> inf`` gives the undrained
elastic medium.

Boundaries: an open-pore free surface (traction-free, ``p = 0``) on the ``dirichlet``
sides -- ``p`` pinned to zero on their nodes, the traction natural; elsewhere the boundary
integrals are dropped (traction-free, no flow), behind the C-PML. C-PML: the profile of
:class:`ElasticVSSEM2DDeformed`, field memories for the weak forms (stress, flux) and
gradient memories for the strong ones (``grad v``, ``grad p``): 12 element fields.

Sources
-------
``src`` is aligned with ``(v_x, v_z, p)``. ``src[2]`` injects pore pressure: ``p`` jumps by
``dt src / A(w)`` at the start of the step and the total stress does not move (FD
``Biot``'s ``p``). ``src[0], src[1]`` follow ``source_mode``: ``"load"`` (default) is the
weak-form load on the total momentum -- a moment tensor ``-M : grad(phi)`` injects total
stress, FD's ``sxx`` / ``szz`` / ``sxz``; ``"velocity"`` is FD's ``vx`` / ``vz``: ``v`` jumps
by ``dt src / M_rho`` after the momentum update and the flux does not move.

Eager only, no boundary saving: the viscous drag is dissipative at every node, so the
reverse reconstruction does not exist (as on the FD ``Biot``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

import torch

from .._assembly import assemble, phys_grads
from ..cpml.elastic_vs import SimpleElasticCPMLProfile, build_simple_elastic_cpml
from ..fields import FieldSpec, check_element_arrays, init_state_from_specs
from ..mesh.boundaries import free_surface_node_idx
from ..mesh.quad9 import Quad9Mesh

_SOURCE_MODES = ("load", "velocity")


def _node_slots(ibool):
    """``(node, slot, kmax)`` of every element-local node in flattened ``ibool`` order: its
    global node and its rank among the element copies of that node."""
    g = ibool.reshape(-1)
    order = torch.argsort(g, stable=True)
    gs = g[order]
    pos = torch.arange(gs.numel(), device=g.device)
    start = torch.ones_like(gs, dtype=torch.bool)
    start[1:] = gs[1:] != gs[:-1]
    run = torch.cummax(torch.where(start, pos, torch.zeros_like(pos)), 0).values
    slot = torch.empty_like(pos)
    slot[order] = pos - run
    return g, slot, int(slot.max()) + 1


@dataclass
class BiotVSSEM2DDeformed:
    """1st-order velocity-stress Biot poroelastic SEM on a deformed Quad9 mesh."""

    # ---- Field specification: the positional state layout (see fields.py) ---- #
    field_specs: ClassVar[list] = [
        FieldSpec("v_x", supports_source=True, supports_receiver=True),
        FieldSpec("v_z", supports_source=True, supports_receiver=True),
        FieldSpec("p", supports_source=True, supports_receiver=True),
        FieldSpec("q_x", supports_receiver=True, layout="element"),
        FieldSpec("q_z", supports_receiver=True, layout="element"),
        FieldSpec("sigma_xx", supports_receiver=True, layout="element"),
        FieldSpec("sigma_zz", supports_receiver=True, layout="element"),
        FieldSpec("sigma_xz", supports_receiver=True, layout="element"),
        FieldSpec("m_sxx_x", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_sxz_z", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_sxz_x", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_szz_z", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_qx_x", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_qz_z", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_px", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_pz", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_vxx", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_vzz", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_vxz", internal=True, boundary_related=True, layout="element"),
        FieldSpec("m_vzx", internal=True, boundary_related=True, layout="element"),
    ]

    # FD Biot's receiver names
    derived_receiver_types: ClassVar[dict] = {
        "vx": ("v_x", None), "vz": ("v_z", None), "qx": ("q_x", None), "qz": ("q_z", None),
        "sxx": ("sigma_xx", None), "szz": ("sigma_zz", None), "sxz": ("sigma_xz", None)}

    # PropSEM's source names, FD Biot's meaning: a velocity or the pore pressure is
    # injected (source_mode "velocity", one payload entry per (v_x, v_z, p)); a stress
    # name is the total-stress moment tensor (Mxx, Mzz, Mxz) on the momentum.
    propsem_source_names: ClassVar[dict] = {
        "vx": ("velocity", 0), "vz": ("velocity", 1), "v_x": ("velocity", 0),
        "v_z": ("velocity", 1), "p": ("velocity", 2),
        "sxx": ("moment", (1.0, 0.0, 0.0)), "szz": ("moment", (0.0, 1.0, 0.0)),
        "sxz": ("moment", (0.0, 0.0, 1.0)),
    }
    propsem_source_refused: ClassVar[dict] = {
        n: "the Darcy flux is element-local on the SEM and takes no point source; "
           "inject the pore pressure ('p') or a velocity instead"
        for n in ("qx", "qz", "q_x", "q_z")}
    default_source_type: ClassVar[tuple] = ("sxx", "szz")
    default_receiver_type: ClassVar[tuple] = ("v_x", "v_z")

    # PropSEM refuses memory=BoundarySaving() with this reason
    bs_refusal: ClassVar[str] = (
        "BiotVSSEM2DDeformed has no boundary saving: the viscous drag is dissipative "
        "at every node, so stepping back multiplies round-off by exp(gamma dt) per "
        "step and the reverse reconstruction does not exist. Use memory=Full() or "
        "Ckpt(chunks=...).")

    mesh: Quad9Mesh
    H_e: torch.Tensor                  # (E, ngll, ngll) undrained P-wave modulus
    mu_e: torch.Tensor
    C_e: torch.Tensor
    M_e: torch.Tensor
    rho_e: torch.Tensor
    rho_f_e: torch.Tensor
    m_e: torch.Tensor
    b_e: torch.Tensor
    dirichlet: tuple                   # the open-pore sides: p = 0 on their nodes
    dirichlet_idx: torch.Tensor

    mass_rho: torch.Tensor             # (n_global,) A(w rho)
    _inv_M: torch.Tensor               # 1 / mass_rho
    _inv_Mp: torch.Tensor              # 1 / A(w / M), the pressure's lumped compliance
    _inv_M1: torch.Tensor              # 1 / A(w): a pressure injected at a point
    _lam_d: torch.Tensor               # drained lambda, H - 2 mu - C^2 / M
    _alpha: torch.Tensor               # C / M
    # the element copies of the flux at every node: flattened ibool -> node * kmax + slot
    _slot_idx: torch.Tensor            # (E ngll^2,)
    _kmax: int
    _wm: torch.Tensor                  # (n_global, kmax) w m per copy (1 on empty slots)
    _wb: torch.Tensor                  # w b (0 on empty slots)
    _wr: torch.Tensor                  # w rho_f (0 on empty slots)

    cpml: Optional[SimpleElasticCPMLProfile]

    _D: torch.Tensor                   # (ngll, ngll)
    _w_J: torch.Tensor
    _ixx: torch.Tensor
    _ixz: torch.Tensor
    _iex: torch.Tensor
    _iez: torch.Tensor

    source_mode: str = "load"          # see the module docstring
    _drag: tuple = None                # (dt, E, Psi), set by init_state

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
        *,
        dirichlet: tuple = (),
        cpml: Optional[SimpleElasticCPMLProfile] = None,
    ) -> "BiotVSSEM2DDeformed":
        """Build the operators from the eight Biot coefficients at the GLL nodes;
        ``dirichlet`` names the open-pore sides."""
        check_element_arrays(mesh, H_e=H_e, mu_e=mu_e, C_e=C_e, M_e=M_e, rho_e=rho_e,
                             rho_f_e=rho_f_e, m_e=m_e, b_e=b_e)
        device = H_e.device
        w = mesh.w_jac
        like = H_e.new_zeros(mesh.n_global)

        def lump(c):
            return assemble(w * c, mesh.ibool, like)

        node, slot, kmax = _node_slots(mesh.ibool)
        slot_idx = node * kmax + slot

        def slots(x, fill):
            out = torch.full((mesh.n_global * kmax,), fill, dtype=x.dtype, device=device)
            return out.index_copy(0, slot_idx, x.reshape(-1)).view(mesh.n_global, kmax)

        mass_rho = lump(rho_e)
        alpha = C_e / M_e
        return cls(
            mesh=mesh, H_e=H_e, mu_e=mu_e, C_e=C_e, M_e=M_e, rho_e=rho_e,
            rho_f_e=rho_f_e, m_e=m_e, b_e=b_e,
            dirichlet=tuple(dirichlet),
            dirichlet_idx=free_surface_node_idx(mesh, dirichlet, device),
            mass_rho=mass_rho, _inv_M=1.0 / mass_rho,
            _inv_Mp=1.0 / lump(1.0 / M_e), _inv_M1=1.0 / lump(torch.ones_like(M_e)),
            _lam_d=H_e - 2.0 * mu_e - alpha * C_e, _alpha=alpha,
            _slot_idx=slot_idx, _kmax=kmax,
            _wm=slots(w * m_e, 1.0), _wb=slots(w * b_e, 0.0), _wr=slots(w * rho_f_e, 0.0),
            cpml=cpml,
            _D=mesh.deriv, _w_J=w,
            _ixx=mesh.inv_J[..., 0, 0], _ixz=mesh.inv_J[..., 0, 1],
            _iex=mesh.inv_J[..., 1, 0], _iez=mesh.inv_J[..., 1, 1],
        )

    # The C-PML builder ``cpml=`` expects, and the speed PropSEM designs it for.
    cpml_builder: ClassVar[object] = staticmethod(build_simple_elastic_cpml)

    @staticmethod
    def pml_speed(models) -> float:
        """The largest fast-P velocity of ``[H, mu, C, M, rho, rho_f, m, b]``."""
        from ...equations.biot import biot_fast_p_velocity
        return biot_fast_p_velocity([torch.as_tensor(m).detach() for m in models])

    # ------------------------------------------------------------------ #
    # The drag: one matrix exponential per node, for the run's dt
    # ------------------------------------------------------------------ #
    def drag_operators(self, dt: float):
        """``(E, Psi)``, each ``(n_global, kmax, kmax)``: the flux copies at a node advance
        by ``q <- E q + Psi r`` over a step of frozen forces ``r``. From
        ``A dq/dt + W_b q = r`` with ``A = diag(w m) - w rho_f (w rho_f)^T / M_rho``:
        ``exp([[-A^-1 W_b, A^-1], [0, 0]] dt) = [[E, Psi], [0, I]]``, in float64."""
        k = self._kmax
        wm, wb, wr = (t.double() for t in (self._wm, self._wb, self._wr))
        mass = self.mass_rho.double()[:, None, None]
        A_inv = torch.linalg.inv(torch.diag_embed(wm) - wr[:, :, None] * wr[:, None, :] / mass)
        top = torch.cat([-A_inv * wb[:, None, :], A_inv], 2)
        X = torch.linalg.matrix_exp(torch.cat([top, torch.zeros_like(top)], 1) * dt)
        dtype = self.mass_rho.dtype
        return X[:, :k, :k].to(dtype), X[:, :k, k:].to(dtype)

    def init_state(self, dt: float, device, dtype) -> tuple:
        """Zero fields, and the drag operators for this ``dt`` -- computed here, once per
        run, outside any checkpointed chunk of the time loop."""
        if self._drag is None or self._drag[0] != dt:
            self._drag = (dt, *self.drag_operators(dt))
        return init_state_from_specs(self, device, dtype)

    def _to_slots(self, x):
        """Element-local ``(E, ngll, ngll)`` -> ``(n_global, kmax)``, zero on empty slots."""
        k = self._kmax
        out = x.new_zeros(self.mesh.n_global * k).index_add(0, self._slot_idx, x.reshape(-1))
        return out.view(self.mesh.n_global, k)

    def _from_slots(self, s, like):
        return s.reshape(-1)[self._slot_idx].view_as(like)

    def _weak_div(self, *pairs):
        """``sum_g w (d(phi)/dx X + d(phi)/dz Z)`` at the global nodes for every element-local
        pair ``(X, Z)`` -- :class:`ElasticVSSEM2DDeformed`'s stress divergence for the
        momentum, ``D^T W q`` for the pressure."""
        w_J, ixx, ixz, iex, iez = self._w_J, self._ixx, self._ixz, self._iex, self._iez
        F_beta = torch.stack([w_J * (X * ixx + Z * ixz) for X, Z in pairs], dim=-1)
        F_alpha = torch.stack([w_J * (X * iex + Z * iez) for X, Z in pairs], dim=-1)
        R_e = (torch.einsum("db,eadk->eabk", self._D, F_beta)
               + torch.einsum("ga,egbk->eabk", self._D, F_alpha))
        like = pairs[0][0].new_zeros(self.mesh.n_global)
        if len(pairs) == 1:
            return assemble(R_e[..., 0], self.mesh.ibool, like)
        return assemble(R_e, self.mesh.ibool, like)

    def _pin(self, p):
        """``p = 0`` on the open-pore surface."""
        idx = self.dirichlet_idx
        return p if idx.numel() == 0 else p.index_fill(0, idx, 0.0)

    # ================================================================== #
    # SEMPropagator interface: init_state + step (staggered leapfrog).
    # ================================================================== #
    def step(self, fields: tuple, src, dt: float) -> tuple:
        """One staggered-leapfrog step (v, q at n+1/2; sigma, p at n+1)."""
        (v_x, v_z, p, q_x, q_z, s_xx, s_zz, s_xz,
         m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z, m_qx_x, m_qz_z, m_px, m_pz,
         m_vxx, m_vzz, m_vxz, m_vzx) = fields
        mode = self.source_mode
        if mode not in _SOURCE_MODES:
            raise ValueError(f"source_mode must be one of {_SOURCE_MODES}, got {mode!r}")
        if self._drag is None or self._drag[0] != dt:
            raise RuntimeError(f"step(dt={dt}) needs init_state(dt) first: the drag "
                               f"operators are computed there, once per run")
        _, E, Psi = self._drag
        cp = self.cpml
        ibool = self.mesh.ibool
        w = self._w_J

        # ----- pore pressure injected at t^n; the total stress does not move -----
        p = self._pin(p + dt * self._inv_M1 * src[2])

        # ----- forces at t^n: total-stress divergence and pressure gradient -----
        d_p_dx, d_p_dz = phys_grads(self._D, ibool, self._ixx, self._ixz, self._iex,
                                    self._iez, p)
        if cp is not None:
            m_sxx_x = cp.a_x * m_sxx_x + cp.b_x * s_xx
            m_sxz_z = cp.a_z * m_sxz_z + cp.b_z * s_xz
            m_sxz_x = cp.a_x * m_sxz_x + cp.b_x * s_xz
            m_szz_z = cp.a_z * m_szz_z + cp.b_z * s_zz
            m_px = cp.a_x * m_px + cp.b_x * d_p_dx
            m_pz = cp.a_z * m_pz + cp.b_z * d_p_dz
            d_p_dx, d_p_dz = d_p_dx + m_px, d_p_dz + m_pz
            R_x, R_z = self._weak_div((s_xx + m_sxx_x, s_xz + m_sxz_z),
                                      (s_xz + m_sxz_x, s_zz + m_szz_z))
        else:
            R_x, R_z = self._weak_div((s_xx, s_xz), (s_xz, s_zz))
        f_x, f_z = -R_x, -R_z
        if mode == "load":
            f_x, f_z = f_x + src[0], f_z + src[1]

        # ----- v and the flux copies at every node, the drag integrated exactly -----
        wr, inv_m = self._wr, self._inv_M
        moved = []
        for q, f, dp in ((q_x, f_x, d_p_dx), (q_z, f_z, d_p_dz)):
            qs = self._to_slots(q)
            r = self._to_slots(-w * dp) - wr * (f * inv_m)[:, None]
            qs_new = torch.einsum("nij,nj->ni", E, qs) + torch.einsum("nij,nj->ni", Psi, r)
            moved.append((self._from_slots(qs_new, q), (wr * (qs_new - qs)).sum(1)))
        (q_x, dmom_x), (q_z, dmom_z) = moved
        v_x = v_x + (dt * f_x - dmom_x) * inv_m
        v_z = v_z + (dt * f_z - dmom_z) * inv_m
        if mode == "velocity":
            v_x = v_x + dt * inv_m * src[0]
            v_z = v_z + dt * inv_m * src[1]

        # ----- stress and pressure from the new v, q -----
        d_vx_dx, d_vx_dz, d_vz_dx, d_vz_dz = phys_grads(
            self._D, ibool, self._ixx, self._ixz, self._iex, self._iez, v_x, v_z)
        if cp is not None:
            m_vxx = cp.a_x * m_vxx + cp.b_x * d_vx_dx
            m_vzz = cp.a_z * m_vzz + cp.b_z * d_vz_dz
            m_vxz = cp.a_z * m_vxz + cp.b_z * d_vx_dz
            m_vzx = cp.a_x * m_vzx + cp.b_x * d_vz_dx
            d_vx_dx = d_vx_dx + m_vxx
            d_vz_dz = d_vz_dz + m_vzz
            d_vx_dz = d_vx_dz + m_vxz
            d_vz_dx = d_vz_dx + m_vzx
            m_qx_x = cp.a_x * m_qx_x + cp.b_x * q_x
            m_qz_z = cp.a_z * m_qz_z + cp.b_z * q_z
            div_q = self._weak_div((q_x + m_qx_x, q_z + m_qz_z))
        else:
            div_q = self._weak_div((q_x, q_z))
        div_v = d_vx_dx + d_vz_dz
        alpha = self._alpha
        dp_free = self._pin(dt * self._inv_Mp * (
            div_q - assemble(w * alpha * div_v, ibool, p.new_zeros(self.mesh.n_global))))
        p = p + dp_free
        lam_d, mu = self._lam_d, self.mu_e
        dsig_p = alpha * dp_free[ibool]                    # sigma = sigma' - alpha p
        s_xx = s_xx + dt * (lam_d * div_v + 2.0 * mu * d_vx_dx) - dsig_p
        s_zz = s_zz + dt * (lam_d * div_v + 2.0 * mu * d_vz_dz) - dsig_p
        s_xz = s_xz + dt * mu * (d_vx_dz + d_vz_dx)

        return (v_x, v_z, p, q_x, q_z, s_xx, s_zz, s_xz,
                m_sxx_x, m_sxz_z, m_sxz_x, m_szz_z, m_qx_x, m_qz_z, m_px, m_pz,
                m_vxx, m_vzz, m_vxz, m_vzx)
