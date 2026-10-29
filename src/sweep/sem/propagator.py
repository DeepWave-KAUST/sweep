"""Generic SEM propagator — one time loop for every SEM equation.

Mirrors the FieldSpec + positional-state pattern of ``sweep``'s FD propagator
(`sweep.propagator`): a single :class:`SEMPropagator` drives the time loop and
delegates the per-step physics to ``equation.step(...)``. Each equation declares
a ``field_specs`` list (see :mod:`sweep.sem.fields`) — the single source of
truth for which evolving fields it carries, their order, and each field's role
(``supports_source`` / ``supports_receiver`` / ``boundary_related`` / ``seis_key``).

    methods
    -------
    init_state(dt, device, dtype) -> tuple[Tensor, ...]
        Allocate the zero fields as a tuple in ``field_specs`` order (wavefields
        + history + CPML memory). Constants live on the equation instance.
    step(fields, src, dt) -> tuple[Tensor, ...]
        Advance one time step. ``fields`` is the positional tuple; ``src`` is a
        list aligned with the source fields (``src[k]`` = the k-th source
        field's body-force vector for this step). Returns the new field tuple.

The propagator derives the source/receiver field indices and output keys from
``field_specs``; it never inspects the field tensors' meaning.

Source / receiver point location is unified in :class:`PointStamp`,
which works for scalar (acoustic) and vector (elastic force / moment
tensor) sources on both :class:`Quad4Mesh` (closed-form (ξ, η)) and
:class:`Quad9Mesh` (Newton-Raphson on the Q2 map). A moment tensor on an
element edge is an :class:`EdgeStamp` over the elements that meet there.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch

from .mesh.quad4 import Quad4Mesh
from .mesh.quad9 import Quad9Mesh
from .mesh.hex8 import Hex8Mesh
from .mesh.hex27 import Hex27Mesh
from .basis.quad9 import q2_basis_at, q2_basis_deriv_at
from .._compile import compile_or_fallback
from ._assembly import element_mean, scatter_point_sources
from .fields import resolve_field_types


# --------------------------------------------------------------------------- #
# Lagrange basis helpers (scalar evaluation at a point in [-1, 1]).
# --------------------------------------------------------------------------- #
def receiver_gather(mesh, receivers, layout="global"):
    """``(rec_G, rec_W)`` of one shot: the global node ids behind every
    receiver's element and that receiver's Lagrange weights, both
    ``(nrec, ngll**dim)``. With ``layout="element"`` the ids index an
    element-local field flattened to ``(E * ngll**dim,)`` instead -- the
    receiver's own element's nodes (Biot's flux and stresses).

    Every receiver samples ``sum_ab w_eta[a] w_xi[b] . field[ibool[e, a, b]]``,
    so ONE gather plus a weighted sum answers all of them at once -- the
    per-receiver Python loop it replaced was ~5x slower on GPU for tens of
    receivers, its CUDA sync per receiver dominating.

    Left WHERE THEY ARE, uncast: the caller moves, so a shot batch stacks
    first and pays one host-to-device copy instead of one per shot, and the C
    path takes int64/fp32 where the eager one takes the mesh's dtype. Three
    callers wrote these two lines out (the eager forward, PropSEM's batch
    gather, SEMCudaPropagator's consts).
    """
    if layout == "element":
        m = mesh.ibool[0].numel()
        nodes = torch.arange(m, device=mesh.ibool.device)
        G = torch.stack([r.elem_id * m + nodes for r in receivers])
    elif layout == "global":
        G = torch.stack([mesh.ibool[r.elem_id].reshape(-1) for r in receivers])
    else:
        raise ValueError(f"receivers sample a 'global' or an 'element' field, "
                         f"not {layout!r}")
    return G, torch.stack([r.gather_weights() for r in receivers])


def field_layouts(eq, idxs):
    """The storage layout of each state field in ``idxs`` (``field_specs``)."""
    return [eq.field_specs[i].layout for i in idxs]


def lagrange_basis_at(nodes: torch.Tensor, x: float) -> torch.Tensor:
    """Lagrange basis ℓ_j(x), j = 0..len(nodes)-1, at scalar x ∈ [-1, 1]."""
    npx = nodes.detach().cpu().numpy().astype(np.float64)
    out = np.ones_like(npx)
    for j in range(npx.size):
        for k in range(npx.size):
            if k != j:
                out[j] *= (x - npx[k]) / (npx[j] - npx[k])
    return torch.as_tensor(out, dtype=nodes.dtype, device=nodes.device)


def lagrange_basis_deriv_at(nodes: torch.Tensor, x: float) -> torch.Tensor:
    """ℓ'_j(x), j = 0..len(nodes)-1, at scalar x ∈ [-1, 1]."""
    npx = nodes.detach().cpu().numpy().astype(np.float64)
    n = npx.size
    out = np.zeros_like(npx)
    for j in range(n):
        total = 0.0
        for m in range(n):
            if m == j:
                continue
            term = 1.0 / (npx[j] - npx[m])
            for k in range(n):
                if k == j or k == m:
                    continue
                term *= (x - npx[k]) / (npx[j] - npx[k])
            total += term
        out[j] = total
    return torch.as_tensor(out, dtype=nodes.dtype, device=nodes.device)


def _lagrange_np(nodes, t):
    """``ℓ_j(t)`` ``(P, n)`` at every entry of ``t`` ``(P,)`` -- the products
    of :func:`lagrange_basis_at` in the same order, so the same numbers."""
    out = np.ones((t.size, nodes.size))
    for j in range(nodes.size):
        for k in range(nodes.size):
            if k != j:
                out[:, j] *= (t - nodes[k]) / (nodes[j] - nodes[k])
    return out


def _lagrange_deriv_np(nodes, t):
    """``ℓ'_j(t)`` ``(P, n)`` -- :func:`lagrange_basis_deriv_at`'s sums in the
    same order, so the same numbers."""
    n = nodes.size
    out = np.zeros((t.size, n))
    for j in range(n):
        total = np.zeros(t.size)
        for m in range(n):
            if m == j:
                continue
            term = np.full(t.size, 1.0 / (nodes[j] - nodes[m]))
            for k in range(n):
                if k == j or k == m:
                    continue
                term *= (t - nodes[k]) / (nodes[j] - nodes[k])
            total += term
        out[:, j] = total
    return out


# --------------------------------------------------------------------------- #
# Q2 single-point evaluation (for the Quad9 Newton locator).
# --------------------------------------------------------------------------- #
def _q2_eval_one(control_xy: torch.Tensor, xi: float, eta: float) -> torch.Tensor:
    psi_xi  = q2_basis_at(torch.tensor([xi],  dtype=control_xy.dtype,
                                       device=control_xy.device))[0]
    psi_eta = q2_basis_at(torch.tensor([eta], dtype=control_xy.dtype,
                                       device=control_xy.device))[0]
    return torch.einsum("i,j,ijk->k", psi_eta, psi_xi, control_xy)


def _q2_jacobian_one(control_xy: torch.Tensor, xi: float, eta: float) -> torch.Tensor:
    psi_xi   = q2_basis_at(torch.tensor([xi],  dtype=control_xy.dtype,
                                         device=control_xy.device))[0]
    dpsi_xi  = q2_basis_deriv_at(torch.tensor([xi],  dtype=control_xy.dtype,
                                                device=control_xy.device))[0]
    psi_eta  = q2_basis_at(torch.tensor([eta], dtype=control_xy.dtype,
                                         device=control_xy.device))[0]
    dpsi_eta = q2_basis_deriv_at(torch.tensor([eta], dtype=control_xy.dtype,
                                                device=control_xy.device))[0]
    dxdxi  = torch.einsum("i,j,ijk->k", psi_eta, dpsi_xi, control_xy)
    dxdeta = torch.einsum("i,j,ijk->k", dpsi_eta, psi_xi, control_xy)
    return torch.stack([dxdxi, dxdeta], dim=-1)            # (2_xz, 2_ξη)


# --------------------------------------------------------------------------- #
# Newton on the Q2 map, for many points at once (Quad9 / Hex27 locators).
# --------------------------------------------------------------------------- #
def _q2_np(t):
    """``(ψ, ψ')`` of the three Q2 nodes at every entry of ``t``, each
    ``(n, 3)`` -- :func:`q2_basis_at` / :func:`q2_basis_deriv_at` in numpy."""
    return (np.stack([0.5 * t * (t - 1.0), 1.0 - t * t, 0.5 * t * (t + 1.0)], -1),
            np.stack([t - 0.5, -2.0 * t, t + 0.5], -1))


def _newton_q2(control, target, tol, start, *, max_iter=40):
    """Newton on the Q2 map ``x(ξ, η[, γ])`` of ``P`` points at once, in float64.

    ``control`` is ``(P, 3, 3, 2)`` (Quad9: ``[i(η), j(ξ), xz]``) or
    ``(P, 3, 3, 3, 3)`` (Hex27: ``[k(γ), i(η), j(ξ), xyz]``), ``target``
    ``(P, d)``, ``tol`` ``(P,)``, ``start`` ``(P, d)`` local coordinates in
    ``(ξ, η[, γ])`` order. Returns ``(local (P, d), converged (P,))``.

    Each point runs the one-point iteration and stops where it would: when the
    residual is below its ``tol`` (checked before the update). It does **not**
    clamp to [-1, 1] -- an overshoot past ±1 means the point lies in a
    neighbouring element, and the locators rely on seeing that -- only a ±3
    guard against runaway. (Clamping every step made Newton stall on the
    reference-square edge near a curved inter-element boundary and report
    non-convergence.) A singular Jacobian leaves that point unconverged.
    """
    d = target.shape[1]
    loc = np.array(start, dtype=np.float64)
    done = np.zeros(len(target), dtype=bool)
    ok = np.zeros(len(target), dtype=bool)
    for _ in range(max_iter):
        act = np.flatnonzero(~done)
        if act.size == 0:
            break
        (px, dpx), (pe, dpe) = _q2_np(loc[act, 0]), _q2_np(loc[act, 1])
        c = control[act]
        if d == 2:
            pos = np.einsum("ni,nj,nijk->nk", pe, px, c)
        else:
            pg, dpg = _q2_np(loc[act, 2])
            pos = np.einsum("nk,ni,nj,nkijd->nd", pg, pe, px, c)
        resid = target[act] - pos
        conv = np.sqrt((resid * resid).sum(-1)) < tol[act]
        ok[act[conv]] = done[act[conv]] = True
        go = ~conv
        if not go.any():
            break
        act, resid, c = act[go], resid[go], c[go]
        px, dpx, pe, dpe = px[go], dpx[go], pe[go], dpe[go]
        if d == 2:
            J = np.stack([np.einsum("ni,nj,nijk->nk", pe, dpx, c),
                          np.einsum("ni,nj,nijk->nk", dpe, px, c)], -1)        # (n, xz, ξη)
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            good = np.isfinite(det) & (det != 0.0)
            det = np.where(good, det, 1.0)
            delta = np.stack([(J[:, 1, 1] * resid[:, 0] - J[:, 0, 1] * resid[:, 1]) / det,
                              (J[:, 0, 0] * resid[:, 1] - J[:, 1, 0] * resid[:, 0]) / det], -1)
        else:
            pg, dpg = pg[go], dpg[go]
            J = np.stack([np.einsum("nk,ni,nj,nkijd->nd", pg, pe, dpx, c),
                          np.einsum("nk,ni,nj,nkijd->nd", pg, dpe, px, c),
                          np.einsum("nk,ni,nj,nkijd->nd", dpg, pe, px, c)], -1)  # (n, xyz, ξηγ)
            det = np.linalg.det(J)
            good = np.isfinite(det) & (det != 0.0)
            delta = np.zeros_like(resid)
            if good.any():
                delta[good] = np.linalg.solve(J[good], resid[good][..., None])[..., 0]
        done[act[~good]] = True
        loc[act] = np.clip(loc[act] + delta, -3.0, 3.0)
    return loc, ok


def _newton_xi_eta(control_xy: torch.Tensor, x_target: float, z_target: float,
                   *, max_iter: int = 40, tol: Optional[float] = None,
                   xi0: float = 0.0, eta0: float = 0.0,
                   ) -> tuple[float, float, bool]:
    """2D Newton on the Q2 map ``x(ξ, η)`` of one element. Returns
    ``(ξ, η, converged)`` -- :func:`_newton_q2` for one point."""
    if tol is None:
        tol = _newton_tol((x_target, z_target))
    loc, ok = _newton_q2(control_xy.detach().cpu().double().numpy()[None],
                         np.array([[x_target, z_target]], dtype=np.float64),
                         np.array([tol]), np.array([[xi0, eta0]], dtype=np.float64),
                         max_iter=max_iter)
    return float(loc[0, 0]), float(loc[0, 1]), bool(ok[0])


def _newton_tol(coords):
    """Newton's residual tolerance at a point: scaled to the coordinate
    magnitude and float64's round-off floor, where the iteration runs
    whatever the mesh dtype (a fixed absolute tol sits below the floor for
    large coordinates). ``coords`` may be one point or ``(P, d)``."""
    c = np.abs(np.asarray(coords, dtype=np.float64))
    return 32.0 * np.finfo(np.float64).eps * (c.sum(-1) + 1.0)


def _locate_geometry(mesh):
    """The locators' view of a deformed mesh, built once per mesh: float64
    control nodes, element-centre depths in the mesh dtype (the candidate
    order), the element column width."""
    geo = mesh.__dict__.get("_locate_geometry")
    if geo is None:
        control = mesh.control_xy if isinstance(mesh, Quad9Mesh) else mesh.control_xyz
        cz = mesh.z_e.mean(dim=tuple(range(1, mesh.z_e.dim())))
        geo = {
            "control": control.detach().cpu().double().numpy(),
            "cz": cz.detach().cpu().numpy(),
            "dx_e": (float(mesh.x_e[:, 0, :].max().item()) / mesh.nex
                     if isinstance(mesh, Quad9Mesh) else None),
        }
        mesh.__dict__["_locate_geometry"] = geo
    return geo


def _locate_many(mesh, coords, cols, n_rows, elem_of_row):
    """The shared search of :func:`_locate_in_quad9_many` /
    :func:`_locate_in_hex27_many`: every point tries the rows of its element
    column nearest in depth first (at most 8), takes the first element whose
    reference cube holds it (overshoot <= 1e-6), else the least overshoot
    below 0.05; local coordinates come back clamped to [-1, 1]."""
    geo = _locate_geometry(mesh)
    P, d = coords.shape
    cz = geo["cz"][elem_of_row(np.arange(n_rows)[None, :], cols[:, None])]       # (P, n_rows)
    z = coords[:, -1].astype(cz.dtype)
    order = np.argsort(np.abs(cz - z[:, None]), axis=1, kind="stable")[:, :min(8, n_rows)]
    tol = _newton_tol(coords)
    elem = np.full(P, -1, dtype=np.int64)
    loc = np.zeros((P, d))
    best = np.full(P, np.inf)
    best_elem = np.full(P, -1, dtype=np.int64)
    best_loc = np.zeros((P, d))
    pending = np.arange(P)
    for k in range(order.shape[1]):
        if pending.size == 0:
            break
        e = elem_of_row(order[pending, k], cols[pending])
        got, ok = _newton_q2(geo["control"][e], coords[pending], tol[pending], np.zeros((pending.size, d)))
        over = np.maximum((np.abs(got) - 1.0).max(-1), 0.0)
        inside = ok & (over <= 1e-6)
        elem[pending[inside]] = e[inside]
        loc[pending[inside]] = got[inside]
        closer = ok & ~inside & (over < best[pending])
        upd = pending[closer]
        best[upd], best_elem[upd], best_loc[upd] = over[closer], e[closer], got[closer]
        pending = pending[~inside]
    near = pending[best[pending] < 0.05]
    elem[near], loc[near] = best_elem[near], best_loc[near]
    return elem, np.clip(loc, -1.0, 1.0), pending[best[pending] >= 0.05], best


def _locate_in_quad9_many(mesh: Quad9Mesh, xz):
    """``(elem_id (P,), ξ (P,), η (P,))`` of ``P`` points ``(x, z)`` in a
    deformed Quad9 mesh, all located together (one batched Newton).

    The mesh has uniform x columns, so the column ``ex`` is exact; only the
    row ``ez`` is ambiguous near curved inter-element edges, which is why
    the rows are tried nearest-first until one reference square holds the
    point. A point exactly on a shared edge is accepted on whichever side
    converges first.
    """
    xz = np.asarray(xz, dtype=np.float64).reshape(-1, 2)
    nex, nez = mesh.nex, mesh.nez
    ex = np.clip((xz[:, 0] / _locate_geometry(mesh)["dx_e"]).astype(np.int64), 0, nex - 1)
    elem, loc, failed, best = _locate_many(mesh, xz, ex, nez, lambda ez, ex: ez * nex + ex)
    if failed.size:
        i = failed[0]
        raise RuntimeError(
            f"could not locate (x={xz[i, 0]}, z={xz[i, 1]}) in any Quad9 element of "
            f"column ex={ex[i]} (nearest overshoot {best[i]:.2e})")
    return elem, loc[:, 0], loc[:, 1]


def _locate_in_quad9(mesh: Quad9Mesh, x: float, z: float
                     ) -> tuple[int, float, float]:
    """``(elem_id, ξ, η)`` of one point -- :func:`_locate_in_quad9_many`."""
    e, xi, eta = _locate_in_quad9_many(mesh, [[x, z]])
    return int(e[0]), float(xi[0]), float(eta[0])


def _locate_in_hex27_many(mesh: Hex27Mesh, xyz):
    """``(elem_id, ξ, η, γ)`` (each ``(P,)``) of ``P`` points ``(x, y, z)`` in a
    deformed Hex27 mesh: uniform x/y columns, only the z-row ambiguous under
    topography -- :func:`_locate_in_quad9_many` in 3-D."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    nex, ney, nez = mesh.nex, mesh.ney, mesh.nez
    ex = np.clip((xyz[:, 0] / mesh.dx_e).astype(np.int64), 0, nex - 1)
    ey = np.clip((xyz[:, 1] / mesh.dy_e).astype(np.int64), 0, ney - 1)
    elem, loc, failed, _ = _locate_many(mesh, xyz, ey * nex + ex, nez,
                                        lambda ez, col: ez * (ney * nex) + col)
    if failed.size:
        i = failed[0]
        raise RuntimeError(
            f"could not locate (x={xyz[i, 0]}, y={xyz[i, 1]}, z={xyz[i, 2]}) in any "
            f"Hex27 element of column (ex={ex[i]}, ey={ey[i]})")
    return elem, loc[:, 0], loc[:, 1], loc[:, 2]


def _locate_in_hex27(mesh: Hex27Mesh, x, y, z):
    """``(elem_id, ξ, η, γ)`` of one point -- :func:`_locate_in_hex27_many`."""
    e, xi, eta, gam = _locate_in_hex27_many(mesh, [[x, y, z]])
    return int(e[0]), float(xi[0]), float(eta[0]), float(gam[0])


#: Points a mesh remembers -- located, and stamped -- least recently used out
#: first. A streamer brings new receiver positions at every shot and recorded
#: coordinates rarely repeat, so a record of every point ever asked for would
#: grow with the survey; a miss costs a batched Newton and a weight evaluation.
POINT_RECORD_SIZE = 1 << 16


class _PointRecord:
    """A least-recently-used ``{point: value}`` of at most ``size`` entries."""

    def __init__(self, size):
        self.size, self._d = size, OrderedDict()

    def get(self, key):
        value = self._d.get(key)
        if value is not None:
            self._d.move_to_end(key)
        return value

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.size:
            self._d.popitem(last=False)

    def __len__(self):
        return len(self._d)


def _record(mesh, name):
    """The mesh's :class:`_PointRecord` ``name`` (made on first use)."""
    rec = mesh.__dict__.get(name)
    if rec is None:
        rec = mesh.__dict__[name] = _PointRecord(POINT_RECORD_SIZE)
    return rec


def locate_points(mesh, points):
    """``[(elem_id, ξ, η[, γ]), ...]`` of ``points`` (``(x, z)`` / ``(x, y, z)``).
    On a deformed mesh the points it has not located yet are located together,
    one batched Newton for all of them, and every answer is kept on the mesh,
    so the stamps built next -- and the next shot's, at the same receivers --
    find it; callers that stamp many points call it first. The axis-aligned
    meshes locate in closed form."""
    if isinstance(mesh, Quad4Mesh):
        return [_locate_quad(mesh, *p) for p in points]
    if isinstance(mesh, Hex8Mesh):
        return [_hex8_local(mesh, *p) for p in points]
    rec = _record(mesh, "_located")
    keys = [tuple(map(float, p)) for p in points]
    found = {}
    for k in dict.fromkeys(keys):
        hit = rec.get(k)
        if hit is not None:
            found[k] = hit
    todo = [k for k in dict.fromkeys(keys) if k not in found]
    if todo:
        cols = (_locate_in_quad9_many if isinstance(mesh, Quad9Mesh) else _locate_in_hex27_many)(mesh, todo)
        for k, *row in zip(todo, *cols):
            found[k] = (int(row[0]),) + tuple(float(v) for v in row[1:])
            rec.put(k, found[k])
    return [found[k] for k in keys]


def _located(mesh, coords):
    """``(elem_id, ξ, η[, γ])`` of one point on a deformed mesh."""
    return locate_points(mesh, [coords])[0]


# --------------------------------------------------------------------------- #
# Unified point stamp (scalar + vector sources, Quad4 + Quad9 meshes).
# --------------------------------------------------------------------------- #
@dataclass
class PointStamp:
    """Lagrange weights + derivatives + local inv-Jacobian at one point.

    A single representation that covers every source/receiver need:
      * ``scalar_stamp`` — acoustic body source,
      * ``force_stamp``  — elastic body force vector,
      * ``moment_tensor_stamp`` — elastic moment-tensor source,
      * ``gather``       — interpolate one field component at the point.

    ``ixx, ixz, iex, iez`` are the local inverse-Jacobian components
    ``∂ξ/∂x, ∂ξ/∂z, ∂η/∂x, ∂η/∂z`` at the point. For an axis-aligned
    Quad4 element ``ixz = iex = 0`` so the moment-tensor formula reduces
    to the simple ``∂ξ/∂x, ∂η/∂z`` chain rule.
    """
    elem_id: int
    w_eta:  torch.Tensor      # (ngll,) ℓ_α(η_s)
    w_xi:   torch.Tensor      # (ngll,) ℓ_β(ξ_s)
    dw_eta: torch.Tensor      # (ngll,) ℓ'_α(η_s)
    dw_xi:  torch.Tensor      # (ngll,) ℓ'_β(ξ_s)
    ixx: float
    ixz: float
    iex: float
    iez: float

    def scalar_stamp(self, value):
        return self.w_eta[:, None] * self.w_xi[None, :] * value

    def force_stamp(self, F_x, F_z):
        outer = self.w_eta[:, None] * self.w_xi[None, :]
        return F_x * outer, F_z * outer

    def moment_tensor_stamp(self, Mxx, Mzz, Mxz):
        w_b_da = self.w_eta[:, None] * self.dw_xi[None, :]   # ∂/∂ξ pattern
        w_da_b = self.dw_eta[:, None] * self.w_xi[None, :]   # ∂/∂η pattern
        dw_dx = w_b_da * self.ixx + w_da_b * self.iex
        dw_dz = w_b_da * self.ixz + w_da_b * self.iez
        stamp_x = -(Mxx * dw_dx + Mxz * dw_dz)
        stamp_z = -(Mxz * dw_dx + Mzz * dw_dz)
        return stamp_x, stamp_z

    def gather(self, field_e: torch.Tensor) -> torch.Tensor:
        return torch.einsum("a,b,ab->", self.w_eta, self.w_xi, field_e)

    def gather_weights(self) -> torch.Tensor:
        """Flattened receiver weight stamp ``(ngll²,)`` aligned with
        ``ibool[elem].reshape(-1)`` for the vectorised receiver gather."""
        return (self.w_eta[:, None] * self.w_xi[None, :]).reshape(-1)


@dataclass
class PointStamp3D:
    """3-D Lagrange weights at one point for a :class:`Hex8Mesh` element.

    The axis order matches ``ibool[elem]`` = ``(kz, ky, kx)`` ↔ ``(z, y, x)``,
    so every stamp tensor is ``w_gamma ⊗ w_eta ⊗ w_xi`` (z ⊗ y ⊗ x). Covers the
    scalar (acoustic) source, the 3-component body force, and field gather.
    Moment-tensor sources are not yet surfaced in 3-D.
    """
    elem_id: int
    w_gamma: torch.Tensor     # (ngll,) ℓ_(z)(γ_s)
    w_eta:   torch.Tensor     # (ngll,) ℓ_(y)(η_s)
    w_xi:    torch.Tensor     # (ngll,) ℓ_(x)(ξ_s)

    def _outer(self) -> torch.Tensor:
        return (self.w_gamma[:, None, None]
                * self.w_eta[None, :, None]
                * self.w_xi[None, None, :])

    def scalar_stamp(self, value):
        return self._outer() * value

    def force_stamp(self, F_x, F_y, F_z):
        outer = self._outer()
        return F_x * outer, F_y * outer, F_z * outer

    def gather(self, field_e: torch.Tensor) -> torch.Tensor:
        return torch.einsum("z,y,x,zyx->", self.w_gamma, self.w_eta,
                            self.w_xi, field_e)

    def gather_weights(self) -> torch.Tensor:
        return self._outer().reshape(-1)


def _cached_stamp(mesh, kind, coords, build):
    """``build(mesh, *coords)``, kept on the mesh: a point stamps the same way
    every time, and a survey's receivers are stamped again at every shot."""
    rec = _record(mesh, "_stamps")
    key = (kind,) + tuple(map(float, coords))
    stamp = rec.get(key)
    if stamp is None:
        stamp = build(mesh, *coords)
        rec.put(key, stamp)
    return stamp


def build_point_stamps(mesh, points):
    """:func:`build_point_stamp` of every point in ``points``. The ones the mesh
    has not stamped yet are located together and stamped together -- their
    weights reach the mesh's device in one copy -- and every stamp is kept
    on the mesh for the next call."""
    rec = _record(mesh, "_stamps")
    keys = [("point",) + tuple(map(float, p)) for p in points]
    found = {}
    for k in dict.fromkeys(keys):
        hit = rec.get(k)
        if hit is not None:
            found[k] = hit
    todo = [k for k in dict.fromkeys(keys) if k not in found]
    if todo:
        for k, stamp in zip(todo, _build_point_stamps(mesh, [k[1:] for k in todo])):
            found[k] = stamp
            rec.put(k, stamp)
    return [found[k] for k in keys]


def _build_point_stamps(mesh, coords):
    """Uncached :func:`build_point_stamps`."""
    loc = np.array(locate_points(mesh, coords), dtype=np.float64)
    elem = loc[:, 0].astype(np.int64)
    if isinstance(mesh, (Hex8Mesh, Hex27Mesh)):
        return _hex_stamps(mesh, elem, loc[:, 1], loc[:, 2], loc[:, 3])
    return _quad_stamps(mesh, elem, loc[:, 1], loc[:, 2])


def build_point_stamp(mesh, *coords):
    """Locate a point and build the Lagrange/derivative weights.

    Coordinate convention follows the mesh: 2-D meshes take ``(x, z)``, the
    3-D :class:`Hex8Mesh` takes ``(x, y, z)``. Dispatches on mesh type:
    closed-form (ξ, η[, γ]) for axis-aligned :class:`Quad4Mesh` /
    :class:`Hex8Mesh`, Newton-Raphson on the Q2 map for a deformed
    :class:`Quad9Mesh` / :class:`Hex27Mesh` (call :func:`locate_points` with
    all the points first to locate them together). The stamp is kept on the
    mesh and handed back for the same point.
    """
    return _cached_stamp(mesh, "point", coords, _build_point_stamp)


def _build_point_stamp(mesh, *coords):
    return _build_point_stamps(mesh, [coords])[0]


def _hex8_local(mesh, x, y, z):
    """``(elem_id, ξ, η, γ)`` of a point in the axis-aligned :class:`Hex8Mesh`
    (closed form)."""
    if not (0.0 <= x <= mesh.x_max and 0.0 <= y <= mesh.y_max
            and 0.0 <= z <= mesh.z_max):
        raise ValueError(f"point ({x}, {y}, {z}) outside domain")
    ex = min(int(x / mesh.dx_e), mesh.nex - 1)
    ey = min(int(y / mesh.dy_e), mesh.ney - 1)
    ez = min(int(z / mesh.dz_e), mesh.nez - 1)
    return ((ez * mesh.ney + ey) * mesh.nex + ex,
            2.0 * (x - ex * mesh.dx_e) / mesh.dx_e - 1.0,
            2.0 * (y - ey * mesh.dy_e) / mesh.dy_e - 1.0,
            2.0 * (z - ez * mesh.dz_e) / mesh.dz_e - 1.0)


def _hex_stamps(mesh, elem, xi, eta, gam):
    """The :class:`PointStamp3D` of each ``(elem_id, ξ, η, γ)``, their weights
    moved to the mesh's device in one copy (the stamps hold views of it)."""
    nodes = mesh.xi.detach().cpu().numpy().astype(np.float64)
    w = np.stack([_lagrange_np(nodes, np.asarray(t, dtype=np.float64)) for t in (gam, eta, xi)], 1)
    W = torch.as_tensor(w, dtype=mesh.xi.dtype, device=mesh.xi.device)
    return [PointStamp3D(elem_id=int(e), w_gamma=W[p, 0], w_eta=W[p, 1], w_xi=W[p, 2])
            for p, e in enumerate(elem)]


def _locate_quad(mesh, x, z):
    """``(elem_id, ξ, η)`` of a point in a 2-D mesh: closed form on the
    axis-aligned :class:`Quad4Mesh`, Newton on the Q2 map of a
    :class:`Quad9Mesh`."""
    if isinstance(mesh, Quad4Mesh):
        if not (0.0 <= x <= mesh.x_max and 0.0 <= z <= mesh.z_max):
            raise ValueError(f"point ({x}, {z}) outside domain")
        ex = min(int(x / mesh.dx_e), mesh.nex - 1)
        ez = min(int(z / mesh.dz_e), mesh.nez - 1)
        xi_s  = 2.0 * (x - ex * mesh.dx_e) / mesh.dx_e - 1.0
        eta_s = 2.0 * (z - ez * mesh.dz_e) / mesh.dz_e - 1.0
        return ez * mesh.nex + ex, xi_s, eta_s
    if isinstance(mesh, Quad9Mesh):
        return _located(mesh, (x, z))
    raise TypeError(f"unsupported mesh type {type(mesh).__name__}")


def _quad_jacobian(mesh, elem_id, xi_s, eta_s):
    """``J[d, m] = ∂x_d/∂ξ_m`` of element ``elem_id`` at ``(ξ, η)``, as
    floats: ``((∂x/∂ξ, ∂x/∂η), (∂z/∂ξ, ∂z/∂η))``."""
    if isinstance(mesh, Quad4Mesh):
        return ((0.5 * mesh.dx_e, 0.0), (0.0, 0.5 * mesh.dz_e))
    J = _q2_jacobian_one(mesh.control_xy[elem_id], xi_s, eta_s)
    return ((J[0, 0].item(), J[0, 1].item()), (J[1, 0].item(), J[1, 1].item()))


def _quad_stamp(mesh, elem_id, xi_s, eta_s):
    """The :class:`PointStamp` of element ``elem_id`` at local ``(ξ, η)``."""
    return _quad_stamps(mesh, [elem_id], [xi_s], [eta_s])[0]


def _quad_stamps(mesh, elem, xi, eta):
    """The :class:`PointStamp` of each ``(elem_id, ξ, η)``: Lagrange weights
    and derivatives as :func:`lagrange_basis_at` gives them, the inverse
    Jacobian of the Q2 map in float64, and the weights moved to the mesh's
    device in one copy (the stamps hold views of it)."""
    nodes = mesh.xi.detach().cpu().numpy().astype(np.float64)
    xi, eta = np.asarray(xi, dtype=np.float64), np.asarray(eta, dtype=np.float64)
    w = np.stack([_lagrange_np(nodes, eta), _lagrange_np(nodes, xi),
                  _lagrange_deriv_np(nodes, eta), _lagrange_deriv_np(nodes, xi)], 1)
    W = torch.as_tensor(w, dtype=mesh.xi.dtype, device=mesh.xi.device)
    if isinstance(mesh, Quad4Mesh):
        inv = [(2.0 / mesh.dx_e, 0.0, 0.0, 2.0 / mesh.dz_e)] * len(xi)
    else:
        c = _locate_geometry(mesh)["control"][np.asarray(elem, dtype=np.int64)]
        (px, dpx), (pe, dpe) = _q2_np(xi), _q2_np(eta)
        dxi = np.einsum("ni,nj,nijk->nk", pe, dpx, c)              # (∂x/∂ξ, ∂z/∂ξ)
        deta = np.einsum("ni,nj,nijk->nk", dpe, px, c)             # (∂x/∂η, ∂z/∂η)
        det = dxi[:, 0] * deta[:, 1] - deta[:, 0] * dxi[:, 1]
        inv = np.stack([deta[:, 1] / det, -deta[:, 0] / det,
                        -dxi[:, 1] / det, dxi[:, 0] / det], -1).tolist()
    return [PointStamp(elem_id=int(e), w_eta=W[p, 0], w_xi=W[p, 1], dw_eta=W[p, 2], dw_xi=W[p, 3],
                       ixx=i[0], ixz=i[1], iex=i[2], iez=i[3])
            for p, (e, i) in enumerate(zip(elem, inv))]


# --------------------------------------------------------------------------- #
# A point on an element edge, for the sources that differentiate the basis.
# --------------------------------------------------------------------------- #
@dataclass
class EdgeStamp:
    """One source point stamped in every element that meets at it.

    A moment-tensor load is ``-M : ∇φ_a(x_s)``, and the gradient of a C0
    element's basis jumps across its edges. Stamped in the one element the
    locator returns, a source on an edge takes a one-sided derivative and
    radiates lopsided -- an explosion on a corner came out 7e-3 (v_z) and 2e-2
    (v_x) off mirror symmetry, and integer grid sources land on edges every
    ``n_order`` cells. The limit of a symmetric source shrunk onto the point
    instead gives each element the share of a small disk it holds: ``weights``
    are the sector angles over ``2π`` -- 1/2 a side on an edge, the corner
    angles at a corner, the half-disk inside the domain on its boundary.

    Force and scalar stamps need none of this: the Lagrange weights agree on
    the shared nodes and vanish off them, so one element carries them exactly.
    """
    parts: list          # one PointStamp per element whose closure holds the point
    weights: list        # each part's share, summing to 1


def _on_edge_tol(mesh):
    """How close to ±1 a local coordinate counts as on the edge. float32
    geometry rounds node positions by ~1e-5 of an element; float64 is exact
    for the grid points PropSEM hands down."""
    return 1e-8 if mesh.xi.dtype == torch.float64 else 1e-4


def _local_in(mesh, e, x, z, xi0, eta0):
    """``(ξ, η)`` of the point in element ``e`` (unclamped), or ``None`` if
    the locate fails. Quad9 is solved in float64 whatever the mesh dtype, so
    the edge test is not left to a float32 Newton stopping early."""
    if isinstance(mesh, Quad4Mesh):
        ex, ez = e % mesh.nex, e // mesh.nex
        return (2.0 * (x - ex * mesh.dx_e) / mesh.dx_e - 1.0,
                2.0 * (z - ez * mesh.dz_e) / mesh.dz_e - 1.0)
    xi, eta, ok = _newton_xi_eta(mesh.control_xy[e].double(), x, z,
                                 xi0=xi0, eta0=eta0)
    return (xi, eta) if ok else None


def _sector_angle(mesh, e, xi, eta):
    """The angle element ``e`` subtends at its local point ``(ξ, η)``: π on an
    edge (a straight cut to first order), the corner angle at a corner."""
    if abs(xi) != 1.0 or abs(eta) != 1.0:
        return float(np.pi)
    (a, b), (c, d) = _quad_jacobian(mesh, e, xi, eta)
    t1 = np.array([-xi * a, -xi * c])          # along +-ξ, into the element
    t2 = np.array([-eta * b, -eta * d])        # along +-η, into the element
    cos = float(t1 @ t2) / float(np.linalg.norm(t1) * np.linalg.norm(t2))
    return float(np.arccos(max(-1.0, min(1.0, cos))))


def build_moment_stamp(mesh, *coords):
    """:func:`build_point_stamp` for a source that differentiates the basis.

    Inside an element it IS that function's stamp. On an element edge or
    corner it is an :class:`EdgeStamp` over every element whose closure holds
    the point. 3-D meshes have no moment-tensor stamp; they get the plain one.
    Kept on the mesh, as :func:`build_point_stamp` keeps its stamps.
    """
    return _cached_stamp(mesh, "moment", coords, _build_moment_stamp)


def _build_moment_stamp(mesh, *coords):
    if not isinstance(mesh, (Quad4Mesh, Quad9Mesh)):
        return build_point_stamp(mesh, *coords)
    x, z = coords
    e0, xi0, eta0 = _locate_quad(mesh, x, z)
    stamp = _quad_stamp(mesh, e0, xi0, eta0)          # = build_point_stamp's
    tol = _on_edge_tol(mesh)
    loc = _local_in(mesh, e0, x, z, xi0, eta0) or (xi0, eta0)
    on_x, on_z = abs(loc[0]) >= 1 - tol, abs(loc[1]) >= 1 - tol
    if not (on_x or on_z):
        return stamp
    snap = lambda v: max(-1.0, min(1.0, v)) if abs(v) < 1 - tol else (1.0 if v > 0 else -1.0)
    ex0, ez0 = e0 % mesh.nex, e0 // mesh.nex
    found = []
    for dz in ((-1, 0, 1) if on_z else (0,)):
        for dx in ((-1, 0, 1) if on_x else (0,)):
            ex, ez = ex0 + dx, ez0 + dz
            if not (0 <= ex < mesh.nex and 0 <= ez < mesh.nez):
                continue
            e = ez * mesh.nex + ex
            got = _local_in(mesh, e, x, z, loc[0] - 2 * dx, loc[1] - 2 * dz)
            if got is None or max(abs(got[0]), abs(got[1])) > 1 + 10 * tol:
                continue
            found.append((e, snap(got[0]), snap(got[1])))
    if len(found) < 2:                     # on the outer boundary, one element
        return stamp
    angles = [_sector_angle(mesh, e, xi, eta) for e, xi, eta in found]
    total = sum(angles)
    return EdgeStamp(parts=[_quad_stamp(mesh, e, xi, eta) for e, xi, eta in found],
                     weights=[a / total for a in angles])


def _stamp_components(stamp, kind, payload):
    """Per-source-field stamp components for one point source (kind + payload).

    Returns a list aligned with the equation's source fields: ``[scalar]`` for an
    acoustic body source, ``[F_x, F_z(, F_y)]`` for an elastic body force, the
    two/three moment-tensor components for ``moment_tensor`` / ``explosion``.

    A ``force`` payload has one entry per source field (Biot's three: ``v_x,
    v_z, p``), a 2-D ``moment_tensor`` one ``(Mxx, Mzz, Mxz)`` block per pair
    of them; fields past the payload take no load.
    """
    if kind == "scalar":
        return [stamp.scalar_stamp(payload[0])]
    if kind == "force":
        # one Lagrange stamp per entry: force_stamp's ``F * outer`` product, for
        # any number of source fields
        return [stamp.scalar_stamp(f) for f in payload]
    if kind == "moment_tensor":
        if len(payload) % 3:
            raise ValueError(
                f"a moment_tensor payload is (Mxx, Mzz, Mxz) blocks; got "
                f"{len(payload)} entries")
        return [c for k in range(0, len(payload), 3)
                for c in stamp.moment_tensor_stamp(*payload[k:k + 3])]
    if kind == "explosion":                       # isotropic Mxx = Mzz = M0, Mxz = 0
        m0 = payload[0]
        return list(stamp.moment_tensor_stamp(m0, m0, 0.0))
    raise ValueError(f"unknown source kind {kind!r}")


# the source kinds whose stamp differentiates the basis (see EdgeStamp)
_DERIVATIVE_KINDS = ("moment_tensor", "explosion")


def _stamp_loads(stamp, kind, payload, alive=None, names=None):
    """``[(elem_id, components), ...]`` of one point source: its element for
    a :class:`PointStamp`, every element of an :class:`EdgeStamp`, each
    weighted by its share.

    ``alive`` (an equation's ``source_field_elements``: per source field, the
    ``(E,)`` bool of the elements it lives in) keeps each field's load to
    those elements, the shares renormalised over them -- a source on an
    interface between two media loads each field from its own side. A source
    left loading nothing is refused (``names``: the source fields)."""
    parts = getattr(stamp, "parts", None)
    weights = [1.0] if parts is None else list(stamp.weights)
    parts = [stamp] if parts is None else parts
    comps = [_stamp_components(p, kind, payload) for p in parts]
    if alive is None:
        if len(parts) == 1:
            return [(parts[0].elem_id, comps[0])]
        return [(p.elem_id, [c * w for c in cs])
                for p, w, cs in zip(parts, weights, comps)]
    out = [[] for _ in parts]
    loaded = False
    for f in range(len(comps[0])):
        keep = [bool(alive[f][p.elem_id]) if f < len(alive) else True for p in parts]
        total = sum(w for w, k in zip(weights, keep) if k)
        for i, k in enumerate(keep):
            c = comps[i][f] * (weights[i] / total) if k else comps[i][f] * 0.0
            loaded = loaded or bool(k and c.abs().max() > 0)
            out[i].append(c)
    if not loaded and any(bool(c.abs().max() > 0) for cs in comps for c in cs):
        fields = [n for f, n in enumerate(names or ())
                  if any(bool(cs[f].abs().max() > 0) for cs in comps if f < len(cs))]
        raise ValueError(
            f"a {kind!r} source in element(s) {[p.elem_id for p in parts]} loads "
            f"{fields or 'its fields'}, none of which lives there (a fluid element "
            f"carries no solid velocity or stress, a dry one no pore pressure or flux)")
    return [(p.elem_id, cs) for p, cs in zip(parts, out)]


# --------------------------------------------------------------------------- #
# PML placement guard: a source or receiver inside the absorbing band records /
# injects a *non-physical* (artificially damped) field — a silent seismogram
# corruption.  Detect it geometrically from the CPML band extent and warn/raise.
# --------------------------------------------------------------------------- #
def _cpml_geom(eq):
    """Return ``(sides, thickness)`` of the equation's CPML, or ``None``.

    Works for the standalone acoustic/elastic equations (``eq.cpml``) and the
    coupled fluid–solid equation (``eq.el.cpml`` / ``eq.ac.cpml``); both PML
    sub-profiles share the same bands, so the first one found suffices."""
    for obj in (eq, getattr(eq, "el", None), getattr(eq, "ac", None)):
        prof = getattr(obj, "cpml", None) if obj is not None else None
        sides = getattr(prof, "sides", None)
        thick = getattr(prof, "pml_thickness", None)
        if sides and thick:
            return tuple(sides), float(thick)
    return None


def _pml_offenders(mesh, sides, thickness, coords):
    """Indices/coords of points inside a PML band (``x``/``z`` bands; 2-D & 3-D).

    Uses ``thickness`` (= max axis band) for every damped side — conservative by
    at most one element on the finer axis, which correctly also flags the thin
    transition margin at the PML edge. An axis whose opposite bands OVERLAP
    (``2·thickness ≥ extent``) has no interior at all — a degenerate all-absorbing
    domain (e.g. a tiny unit-test box), where flagging every point is noise rather
    than a placement mistake — so that axis is skipped. Returns
    ``[(i, coord_tuple, reason), ...]``."""
    xmax, zmax = float(mesh.x_max), float(mesh.z_max)
    x_interior = 2.0 * thickness < xmax     # opposite bands leave an interior?
    z_interior = 2.0 * thickness < zmax
    out = []
    for i, c in enumerate(coords):
        x, z = float(c[0]), float(c[-1])          # (x, z) or (x, y, z)
        hits = []
        if x_interior and "left" in sides and x < thickness:
            hits.append(f"left band (x<{thickness:.1f})")
        if x_interior and "right" in sides and x > xmax - thickness:
            hits.append(f"right band (x>{xmax - thickness:.1f})")
        if z_interior and "top" in sides and z < thickness:
            hits.append(f"top band (z<{thickness:.1f})")
        if z_interior and "bottom" in sides and z > zmax - thickness:
            hits.append(f"bottom band (z>{zmax - thickness:.1f})")
        if hits:
            out.append((i, tuple(round(float(v), 2) for v in c), ", ".join(hits)))
    return out


def _check_pml_placement(eq, source_xz, receiver_xz, guard):
    """Warn/raise if any source or receiver falls inside the CPML absorbing band.

    ``guard`` ∈ {``"warn"`` (default), ``"error"``, ``"off"``}. Inside the PML the
    field is intentionally damped, so a receiver there records a non-physical
    trace and a source there drives the run from within the absorber — both are
    almost always a setup mistake (e.g. an edge shot/geophone, or a domain too
    small for the chosen ``pml_thickness_elements``)."""
    if guard == "off":
        return
    if guard not in ("warn", "error"):
        raise ValueError(f"pml_guard must be 'warn', 'error', or 'off'; got {guard!r}")
    geom = _cpml_geom(eq)
    if geom is None:
        return
    sides, thickness = geom
    for label, pts in (("source", source_xz), ("receiver", receiver_xz)):
        bad = _pml_offenders(eq.mesh, sides, thickness, pts)
        if not bad:
            continue
        detail = "; ".join(f"{label}[{i}] {c} in {why}" for i, c, why in bad)
        msg = (f"{len(bad)} {label}(s) fall inside the C-PML absorbing band, where "
               f"the field is artificially damped (non-physical): {detail}. "
               f"Move them into the interior, enlarge the domain / reduce "
               f"pml_thickness_elements, or pass pml_guard='off' to silence.")
        if guard == "error":
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=3)


# --------------------------------------------------------------------------- #
# Equation protocol (documentation / static check aid).
# --------------------------------------------------------------------------- #
@runtime_checkable
class SEMEquation(Protocol):
    field_specs: list                 # ordered FieldSpec list = positional layout

    def init_state(self, dt: float, device, dtype) -> tuple: ...
    def step(self, fields: tuple, src, dt: float) -> tuple: ...


# --------------------------------------------------------------------------- #
# Generic propagator.
# --------------------------------------------------------------------------- #
@dataclass
class SEMPropagator:
    eq: SEMEquation
    dt: float
    nt: int
    sources: list
    receivers: list
    source_kinds: list
    source_payloads: list
    receiver_type: list = None        # field names, in output-component order
    _compiled_step: object = None     # cached torch.compile(eq.step), lazy

    @classmethod
    def build(
        cls,
        eq: SEMEquation,
        *,
        dt: float,
        nt: int,
        source_xz: list,
        receiver_xz: list,
        source_kinds: list | None = None,
        source_payloads: list | None = None,
        receiver_type: list | None = None,
        pml_guard: str = "warn",
    ) -> "SEMPropagator":
        """``pml_guard`` ∈ {``"warn"``, ``"error"``, ``"off"``}: catch sources /
        receivers placed inside the C-PML absorbing band (where the field is
        artificially damped — a silent seismogram corruption). Default warns."""
        _check_pml_placement(eq, source_xz, receiver_xz, pml_guard)
        n_src_fields = sum(s.supports_source for s in eq.field_specs)
        if source_kinds is None:
            # Default kind: scalar for 1-field eqs (acoustic), force for vector.
            default = "scalar" if n_src_fields == 1 else "force"
            source_kinds = [default] * len(source_xz)
        if source_payloads is None:
            if n_src_fields == 1:
                source_payloads = [(1.0,)] * len(source_xz)
            else:
                # First component unit, rest zero (force along x). Generalises
                # to 2-D (1, 0) and 3-D (1, 0, 0).
                default = (1.0,) + (0.0,) * (n_src_fields - 1)
                source_payloads = [default] * len(source_xz)
        # a moment tensor differentiates the basis, so on an element edge it
        # is stamped in every element there (EdgeStamp); the rest locate once --
        # unless the fields live in different media (source_field_elements):
        # then every source meets all the elements at its point, to take each
        # field's load from the ones that carry it
        edge_all = getattr(eq, "source_field_elements", None) is not None
        locate_points(eq.mesh, list(source_xz) + list(receiver_xz))
        sources   = [(build_moment_stamp if edge_all or kind in _DERIVATIVE_KINDS
                      else build_point_stamp)(eq.mesh, *xz)
                     for xz, kind in zip(source_xz, source_kinds)]
        receivers = build_point_stamps(eq.mesh, receiver_xz)
        return cls(eq=eq, dt=dt, nt=nt, sources=sources, receivers=receivers,
                   source_kinds=source_kinds, source_payloads=source_payloads,
                   receiver_type=resolve_field_types(
                       type(eq), receiver_type, "receiver"))

    def _source_media(self):
        """``(alive, names)`` for :func:`_stamp_loads`: the equation's
        ``source_field_elements`` (``None`` when every field lives everywhere)
        and its source field names."""
        alive = getattr(self.eq, "source_field_elements", None)
        names = [s.name for s in self.eq.field_specs if s.supports_source]
        return alive, names

    # ---- source spatial pattern (assembled ONCE, scaled per step) ------- #
    def _build_source_patterns(self, n_global, device, dtype):
        """Assemble the constant per-field source spatial pattern.

        The body source is separable: ``b_fn(x, t) = pattern_fn(x) · s(t)``
        — the time function ``s(t) = wavelet[n]`` is a single scalar shared
        by every source at step ``n``, so the spatial scatter (zeros +
        index_add of the Lagrange/moment stamps) only needs doing once at
        unit amplitude. The hot loop then just multiplies by the scalar,
        which avoids re-allocating + re-scattering an ``n_global`` vector
        every time step.
        """
        eq = self.eq
        mesh = eq.mesh
        n_src = sum(s.supports_source for s in eq.field_specs)
        # A LIST in source-field order, not a name-keyed dict: every caller
        # immediately re-ordered such a dict back into this list anyway, and
        # the positional order is already the equation's own field order.
        patterns = [torch.zeros(n_global, dtype=dtype, device=device)
                    for _ in range(n_src)]
        alive, names = self._source_media()
        for stamp, kind, payload in zip(self.sources, self.source_kinds,
                                        self.source_payloads):
            for e, comps in _stamp_loads(stamp, kind, payload, alive, names):
                if len(comps) > n_src:
                    # zip would drop the extra components without a word
                    raise ValueError(
                        f"a {kind!r} source with payload {tuple(payload)} has "
                        f"{len(comps)} components; {type(eq).__name__} has "
                        f"{n_src} source fields")
                g = mesh.ibool[e].reshape(-1)
                for pat, comp in zip(patterns, comps):
                    pat.index_add_(0, g, comp.reshape(-1))
        return patterns

    # ---- per-point-source patterns (FD mode-B per-source wavelets) ------- #
    def _build_per_source_patterns(self, device, dtype):
        """Per-point-source stamp set, source axis kept (not summed).

        For the FD source-encoding mode with a per-source wavelet ``(P, nt)``:
        returns ``(src_G, src_W)`` where ``src_G`` is ``(P, m)`` global node
        indices (``m = ngll**dim``, times the widest :class:`EdgeStamp`'s
        element count, zero-padded) and ``src_W`` is a list (per source field,
        in ``field_specs`` order) of ``(P, m)`` stamp weights. The per-step source
        term is then ``Σ_k src_W[f][k] · wavelet[k, n]`` — see
        :func:`sweep.sem._assembly.scatter_point_sources`. Contrast
        :meth:`_build_source_patterns`, which sums the ``P`` sources into one
        ``(n_global,)`` pattern (valid only when they share one time function).
        """
        mesh = self.eq.mesh
        n_src = sum(s.supports_source for s in self.eq.field_specs)
        G_rows, W_rows = [], [[] for _ in range(n_src)]
        alive, names = self._source_media()
        for stamp, kind, payload in zip(self.sources, self.source_kinds,
                                        self.source_payloads):
            loads = _stamp_loads(stamp, kind, payload, alive, names)
            G_rows.append(torch.cat([mesh.ibool[e].reshape(-1) for e, _ in loads]))
            for f, rows in enumerate(W_rows):
                # a payload may cover the leading source fields only (a moment
                # tensor on Biot's v rows, none on p): the rest take no load
                rows.append(torch.cat([
                    (c[f] if f < len(c) else torch.zeros_like(c[0])).reshape(-1)
                    for _, c in loads]))
        width = max(g.numel() for g in G_rows)
        if any(g.numel() != width for g in G_rows):
            # an edge source spans several elements: pad the others with zero
            # weights on node 0, which the scatter adds as nothing
            pad = lambda t: torch.cat([t, t.new_zeros(width - t.numel())])
            G_rows = [pad(g) for g in G_rows]
            W_rows = [[pad(w) for w in rows] for rows in W_rows]
        src_G = torch.stack(G_rows).to(device)                    # (P, m)
        src_W = [torch.stack(rows).to(device=device, dtype=dtype)
                 for rows in W_rows]                              # per field (P, m)
        return src_G, src_W

    # ---- forward time loop ---------------------------------------------- #
    def forward(
        self,
        source_wavelet: torch.Tensor,
        *,
        record_field_every: int | None = None,
        snapshot_times=None,
        compile_step: bool = False,
        memory: str | None = None,
        bs_store_dtype=None,
        checkpoint_chunk: int | None = None,
    ):
        """Run the time loop.

        ``compile_step=True`` wraps ``eq.step`` in :func:`torch.compile`
        (cached on the propagator across calls) — the per-timestep tensor
        graph is fused, which on GPU is ~2× faster than eager for the
        elastic step. The first call pays a one-time compilation cost. With
        ``memory='boundary'`` it also fuses the reverse reconstruction and the
        per-step adjoint recompute (both call ``eq.step``), ~2× on the BS
        rollout with bit-identical gradients.

        ``memory='boundary'`` runs the backward pass with boundary-saving
        wavefield reconstruction (peak memory ~flat in ``nt``) for any equation
        exposing the BS hooks (acoustic swap2nd / velocity-stress substep).
        Snapshots are unavailable in this mode (the forward state is discarded).

        ``checkpoint_chunk=K`` (``PropSEM(memory=Ckpt(chunks=K))``) keeps one
        state per ``K`` steps and rebuilds the tape chunk by chunk in the
        backward pass, any equation, at one extra forward; on CUDA the chunks
        are replayed as CUDA graphs (:mod:`sweep.sem.ckpt_rollout`). No
        snapshots in this mode either.

        ``source_wavelet`` is ``(nt,)`` — one time function shared by every point
        source — or ``(P, nt)`` for the FD source-encoding mode: ``P = len(self.
        sources)`` superposed sources each with its own per-step amplitude
        ``s_k(t)`` (distinct polarity / amplitude / time-shift), injected as
        ``Σ_k stamp_k(x) · s_k(t)``.

        Returns the seismogram tensor ``(nt, nrec, ncomp)`` -- the FD
        ``PropTorch`` layout minus its leading shot axis, since this propagator
        is single-shot. ``ncomp`` and the component ORDER come from
        ``receiver_type``; it defaults to every receiver-capable field of the
        equation, so acoustic gives ``ncomp = 1`` and elastic gives 2 (2-D) or
        3 (3-D).

        With ``record_field_every=k`` it returns ``(seis, snaps)`` instead --
        the same shape as FD/JAX's ``(record, wavefield)`` pair -- where
        ``snaps`` is ``(nsnap, ncomp, n_global)`` sampled at time steps
        ``range(0, nt, k)``. ``snapshot_times`` names the steps instead (any
        order, duplicates collapse; ``snaps`` follows them ascending); the two
        are exclusive.
        """
        per_source = self._check_wavelet(source_wavelet)
        snap_at = self._snapshot_steps(record_field_every, snapshot_times)
        device, dtype = self._dev_dtype()
        recv = self._recv_components()
        rec_G, rec_W = self._receiver_gather(device, dtype)
        layouts = field_layouts(self.eq, recv)
        source_spec, like = self._source_term(per_source, device, dtype)
        step_fn = self._step_fn(compile_step)

        if checkpoint_chunk:
            return self._run_checkpointed(
                source_spec, like, source_wavelet, rec_G, rec_W, recv, layouts,
                snap_at is not None, memory, compile_step, checkpoint_chunk)
        if memory == "boundary":
            return self._run_boundary_saved(
                source_spec, source_wavelet, rec_G, rec_W, recv,
                snap_at is not None, compile_step, bs_store_dtype)

        n_rec = len(self.receivers)
        # an element-local field (Biot's flux, stresses) is gathered from its
        # own layout
        rec_GI = {i: rec_G if lay == "global" else
                  receiver_gather(self.eq.mesh, self.receivers, lay)[0].to(device)
                  for i, lay in zip(recv, layouts)} if n_rec else {}
        state = self.eq.init_state(self.dt, device, dtype)     # positional tuple
        seis = {i: torch.zeros(n_rec, self.nt, dtype=dtype, device=device)
                for i in recv}
        snaps = {i: [] for i in recv}
        for n in range(self.nt):
            if per_source:
                src = scatter_point_sources(*source_spec, source_wavelet[:, n], like)
            else:
                s_val = source_wavelet[n]
                src = [p * s_val for p in source_spec]
            state = step_fn(state, src, self.dt)

            if n_rec > 0:
                for i in recv:
                    seis[i][:, n] = (state[i].reshape(-1)[rec_GI[i]]
                                     * rec_W).sum(dim=1)

            if snap_at is not None and n in snap_at:
                for i, lay in zip(recv, layouts):
                    snaps[i].append(state[i].clone() if lay == "global" else
                                    element_mean(self.eq.mesh, state[i]))

        # (nrec, nt) per component -> (nt, nrec, ncomp), the FD layout.
        out = torch.stack([seis[i] for i in recv], -1).transpose(0, 1)
        if snap_at is None:
            return out
        if not snap_at:                       # snapshot_times=[]: none asked for
            return out, state[recv[0]].new_zeros((0, len(recv), self.eq.mesh.n_global))
        # (nsnap, ncomp, n_global), at the steps of snap_at in ascending order;
        # an element-local field as the |J|-weighted mean at each shared node
        return out, torch.stack([torch.stack(snaps[i], 0) for i in recv], 1)

    def _snapshot_steps(self, every, times):
        """The steps to snapshot at, as a set, or ``None`` for no snapshots."""
        if every is not None and times is not None:
            raise ValueError(
                "record_field_every and snapshot_times are exclusive; give one.")
        if every is not None:
            return set(range(0, self.nt, int(every)))
        if times is None:
            return None
        steps = {int(t) for t in times}
        bad = sorted(t for t in steps if not 0 <= t < self.nt)
        if bad:
            raise ValueError(
                f"snapshot_times {bad} outside the valid range [0, {self.nt - 1}].")
        return steps

    # ------------------------------------------------------------------ #
    # What forward() resolves before the time loop, one question each
    # ------------------------------------------------------------------ #
    def _check_wavelet(self, w):
        """Is the wavelet per-source? ``(nt,)`` is one time function shared by
        every point source; ``(P, nt)`` is the FD source-encoding mode, one
        per source. Anything else is refused here rather than misread as a
        batch by the loop below."""
        if w.dim() == 2:
            if w.shape[0] != len(self.sources) or w.shape[1] != self.nt:
                raise ValueError(
                    f"per-source wavelet must be (P={len(self.sources)}, "
                    f"{self.nt}); got {tuple(w.shape)}")
            return True
        if w.shape != (self.nt,):
            raise ValueError(
                f"source_wavelet shape {tuple(w.shape)} "
                f"!= ({self.nt},) or (P, {self.nt})"
            )
        return False

    def _dev_dtype(self):
        """Where the state lives, read off whichever mass array the equation
        has -- the propagator itself stores no device."""
        ref = getattr(self.eq, "mass", None)
        if ref is None:
            ref = getattr(self.eq, "mass_rho")
        return ref.device, ref.dtype

    def _recv_components(self):
        """Indices into the state tuple of the fields to record, IN THE ORDER
        the caller asked for. Falls back to every receiver-capable field for a
        propagator built before ``receiver_type`` existed (constructed
        directly rather than through :meth:`build`)."""
        specs = self.eq.field_specs
        rtype = self.receiver_type or resolve_field_types(
            type(self.eq), None, "receiver")
        by_name = {sp.name: i for i, sp in enumerate(specs)}
        return [by_name[nm] for nm in rtype]

    def _receiver_gather(self, device, dtype):
        """``(rec_G, rec_W)``: the global node ids behind every receiver's
        element and their Lagrange weights, so all receivers sample in ONE
        gather plus weighted sum per field --

            seis[r] = sum_ab w_eta[a] w_xi[b] . field[ibool[elem, a, b]]

        -- instead of a Python loop whose per-receiver CUDA sync dominated
        (~5x on GPU for tens of receivers). ``(None, None)`` with no
        receivers."""
        if not self.receivers:
            return None, None
        G, W = receiver_gather(self.eq.mesh, self.receivers)
        return G.to(device), W.to(device=device, dtype=dtype)

    def _source_term(self, per_source, device, dtype):
        """The source's spatial part, and the scratch the per-source form
        scatters into.

        A shared wavelet gives one ``(n_global,)`` pattern per field, scattered
        once here and merely scaled each step. A per-source wavelet keeps the
        point sources apart, because the step needs
        ``sum_k stamp_k(x) . s_k(t)`` rebuilt every time.
        """
        if not per_source:
            return self._build_source_patterns(
                self.eq.mesh.n_global, device, dtype), None
        return (self._build_per_source_patterns(device, dtype),
                torch.zeros(self.eq.mesh.n_global, dtype=dtype, device=device))

    def _step_fn(self, compile_step):
        """``eq.step``, or its ``torch.compile`` wrapper cached on the
        propagator so the compilation is paid once across calls. A compiler
        that cannot build the step (CPU with g++ < 10) warns once and leaves
        it uncompiled -- see :func:`sweep._compile.compile_or_fallback`."""
        if not compile_step:
            return self.eq.step
        if getattr(self, "_compiled_step", None) is None:
            self._compiled_step = compile_or_fallback(
                self.eq.step, self._dev_dtype()[0],
                f"the {type(self.eq).__name__} step")
        return self._compiled_step

    def _run_checkpointed(self, source_spec, like, wavelet, rec_G, rec_W, recv,
                          layouts, want_snapshots, memory, compile_step, chunk):
        """The ``checkpoint_chunk`` route: one state kept per ``chunk`` steps,
        the tape rebuilt chunk by chunk in backward -- see
        :mod:`sweep.sem.ckpt_rollout`. ``recv`` IS the output order."""
        from .ckpt_rollout import run_checkpointed
        if memory is not None:
            raise ValueError(f"checkpoint_chunk and memory={memory!r} are exclusive.")
        if rec_G is None:
            raise ValueError("checkpoint_chunk needs receivers.")
        if want_snapshots:
            raise ValueError(
                "record_field_every / snapshot_times are not available with "
                "checkpoint_chunk -- the rollout keeps one state per chunk.")
        if int(chunk) < 1:
            raise ValueError(f"checkpoint_chunk must be >= 1; got {chunk!r}.")
        rec_GI = {i: rec_G if lay == "global" else
                  receiver_gather(self.eq.mesh, self.receivers, lay)[0].to(rec_G.device)
                  for i, lay in zip(recv, layouts)}
        source = source_spec if like is None else (*source_spec, like)

        def step_of(eq):
            if not compile_step:
                return eq.step
            return compile_or_fallback(eq.step, rec_W.device,
                                       f"the {type(eq).__name__} step")

        return run_checkpointed(self.eq, step_of, source, wavelet, rec_GI, rec_W,
                                recv, self.dt, self.nt, int(chunk))

    def _run_boundary_saved(self, source_spec, wavelet, rec_G, rec_W, recv,
                            want_snapshots, compile_step, store_dtype):
        """The ``memory='boundary'`` route: peak memory flat in ``nt``, at one
        extra forward per step in the backward. ``recv`` IS the output order,
        so the gather already produces the components in it and nothing is
        re-ordered afterwards."""
        from .boundary_saving import run_boundary_saved
        if rec_G is None:
            raise ValueError("memory='boundary' needs receivers.")
        if want_snapshots:
            raise ValueError(
                "record_field_every / snapshot_times are not available with "
                "memory='boundary' -- the forward state is discarded and "
                "reconstructed.")
        seis = run_boundary_saved(
            self.eq, source_spec, wavelet, rec_G, rec_W, self.dt, self.nt,
            compile_step=compile_step, store_dtype=store_dtype, recv_idx=recv)
        return torch.stack(seis, -1).transpose(0, 1)
