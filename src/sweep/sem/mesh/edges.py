"""Per-node edge (face) metric for a deformed :class:`Quad9Mesh`.

Surface integrals — the free-surface traction, and the fluid–solid coupling —
need three things along one face of an element: the global node indices, the
1-D surface Jacobian, and the unit normal. Element *volume* integrals use
``w_jac = w_α w_β |J|``; a **surface** integral must instead use the 1-D GLL
weight times the *edge* Jacobian, never the 2-D weight product.

Geometry (cofactor identity). The mesh stores ``inv_J[e, jz, jx] = [[∂ξ/∂x,
∂ξ/∂z], [∂η/∂x, ∂η/∂z]]`` and ``det_J = |J|``. Inverting the 2×2 gives the
forward metric without ever forming ``J``::

    ∂x/∂ξ =  iez · det_J        ∂x/∂η = -ixz · det_J
    ∂z/∂ξ = -iex · det_J        ∂z/∂η =  ixx · det_J

so a ``η = const`` face (top/bottom) is parametrised by ξ and a ``ξ = const``
face (left/right) by η, and in both cases::

    jac1d = ‖∂x/∂s‖ ,   t = (∂x/∂s, ∂z/∂s)/jac1d ,   ∮ f dΓ ≈ Σ_q w_q jac1d_q f_q

**The returned normal is UNORIENTED.** It is the tangent rotated by −90°, which
picks one of the two face normals by an arbitrary convention. Our mesh is
z-**down** (``jz = 0`` is the top; see :mod:`sweep.sem.mesh.quad4`) while
SPECFEM2D is z-up, so transcribing an edge→sign table from SPECFEM silently
flips the normal. Callers **must** fix the sign with a geometric test against
their own reference direction (see :func:`orient_normal`); a wrong-sign coupling
produces a plausible-looking wavefield with an inverted reflection.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_SIDES = ("top", "bottom", "left", "right")


@dataclass(frozen=True)
class EdgeMetric:
    """Per-GLL-node quantities along one face of ``n_e`` elements.

    Every field is ``(n_e, ngll)``, indexed by the face's own reference
    coordinate ``s`` (ξ for top/bottom, η for left/right).
    """
    node_idx: torch.Tensor    # global node index (int64)
    jac1d: torch.Tensor       # 1-D surface Jacobian ‖∂x/∂s‖
    weight: torch.Tensor      # w_1D(s) · jac1d  — the surface quadrature weight
    tx: torch.Tensor          # unit tangent (direction of increasing s)
    tz: torch.Tensor
    nx: torch.Tensor          # unit normal — UNORIENTED (see module docstring)
    nz: torch.Tensor


def edge_metric(mesh, elem_ids, side: str) -> EdgeMetric:
    """Edge metric along ``side`` of the elements ``elem_ids`` of a Quad9 mesh.

    ``side`` ∈ {"top", "bottom", "left", "right"} in the mesh's z-down frame
    ("top" is ``jz = 0``). The normal is unoriented — see the module docstring.
    """
    if side not in _SIDES:
        raise ValueError(f"side must be one of {_SIDES}; got {side!r}")
    e = torch.as_tensor(elem_ids, dtype=torch.long,
                        device=mesh.ibool.device).reshape(-1)
    ng = mesh.ngll
    ixx = mesh.inv_J[..., 0, 0]      # ∂ξ/∂x
    ixz = mesh.inv_J[..., 0, 1]      # ∂ξ/∂z
    iex = mesh.inv_J[..., 1, 0]      # ∂η/∂x
    iez = mesh.inv_J[..., 1, 1]      # ∂η/∂z

    if side in ("top", "bottom"):                 # η = const face, varies in ξ
        jz = 0 if side == "top" else ng - 1
        node_idx = mesh.ibool[e, jz, :]
        dJ = mesh.det_J[e, jz, :]
        dx_ds = iez[e, jz, :] * dJ                # ∂x/∂ξ
        dz_ds = -iex[e, jz, :] * dJ               # ∂z/∂ξ
    else:                                         # ξ = const face, varies in η
        jx = 0 if side == "left" else ng - 1
        node_idx = mesh.ibool[e, :, jx]
        dJ = mesh.det_J[e, :, jx]
        dx_ds = -ixz[e, :, jx] * dJ               # ∂x/∂η
        dz_ds = ixx[e, :, jx] * dJ                # ∂z/∂η

    jac1d = torch.sqrt(dx_ds * dx_ds + dz_ds * dz_ds)
    tx, tz = dx_ds / jac1d, dz_ds / jac1d
    w1d = mesh.w_ref.reshape(1, ng).expand(e.numel(), ng)
    return EdgeMetric(node_idx=node_idx, jac1d=jac1d, weight=w1d * jac1d,
                      tx=tx, tz=tz, nx=tz, nz=-tx)   # rotate tangent by −90°


def orient_normal(em: EdgeMetric, mesh, elem_ids, toward_elem_ids) -> EdgeMetric:
    """Flip ``em``'s normal so it points **from** ``elem_ids`` **toward**
    ``toward_elem_ids`` (element-centroid test, per face).

    This is the geometric orientation the module docstring demands: it never
    consults an edge→sign table, so it is immune to the z-up/z-down frame trap.
    Raises if any face is degenerate (normal ⟂ the centroid offset), which would
    make the sign meaningless.
    """
    a = torch.as_tensor(elem_ids, dtype=torch.long,
                        device=mesh.ibool.device).reshape(-1)
    b = torch.as_tensor(toward_elem_ids, dtype=torch.long,
                        device=mesh.ibool.device).reshape(-1)
    cx = mesh.x_e.flatten(1).mean(1)              # (E,) element centroids
    cz = mesh.z_e.flatten(1).mean(1)
    dx = (cx[b] - cx[a]).reshape(-1, 1)
    dz = (cz[b] - cz[a]).reshape(-1, 1)
    proj = em.nx * dx + em.nz * dz                # (n_e, ngll)
    s = torch.sign(proj.mean(dim=1, keepdim=True))
    if bool((s == 0).any()) or bool((torch.sign(proj) != s).any()):
        raise ValueError(
            "degenerate/ambiguous face orientation: the normal is not "
            "consistently on one side of the element-centroid offset.")
    return EdgeMetric(node_idx=em.node_idx, jac1d=em.jac1d, weight=em.weight,
                      tx=em.tx, tz=em.tz, nx=s * em.nx, nz=s * em.nz)
