"""Deformed quadrilateral SEM mesh with Q2 (9-node) element geometry.

The element interior is mapped from the reference square [−1, 1]² to
physical space via biquadratic Lagrange shape functions on 9 control
nodes (4 corners, 4 edge midpoints, 1 centre). This is the SPECFEM2D
``NGNOD = 9`` convention.

The GLL solution nodes (used for quadrature *and* the basis on which u
is expanded) are still ``N + 1`` per direction (default ``N = 4``).
What changes from :class:`Quad4Mesh` is purely the geometry layer: the
Jacobian ``J(ξ, η)`` is now per-GLL, not constant per element.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .nodes import first_copy_slot, node_values
from .quad4 import BoundaryEdges

from ..basis.gll import gll_nodes_weights, lagrange_derivative_matrix
from ..basis.quad9 import (
    evaluate_q2_jacobian,
    evaluate_q2_map,
    q2_basis_at,
)


@dataclass
class Quad9Mesh:
    """Deformed-quad SEM mesh with Q2 geometry and per-GLL Jacobian."""

    n_order: int
    nex: int                          # column count of elements
    nez: int                          # row count
    x_max: float                      # bounding box (control-node extent)
    z_max: float

    # Reference-element data
    xi: torch.Tensor                  # (ngll,)
    w_ref: torch.Tensor               # (ngll,)
    deriv: torch.Tensor               # (ngll, ngll)

    # Element control geometry
    control_xy: torch.Tensor          # (E, 3, 3, 2)  — per-element Q2 control nodes

    # Per-element-per-GLL Jacobian
    inv_J: torch.Tensor               # (E, ngll, ngll, 2, 2)
    det_J: torch.Tensor               # (E, ngll, ngll)  positive
    w_jac: torch.Tensor               # (E, ngll, ngll)  = w_α w_β |J|

    # Connectivity (identical layout to :class:`Quad4Mesh`)
    ibool: torch.Tensor               # (E, ngll, ngll) int64
    x_global: torch.Tensor            # (n_global,)
    z_global: torch.Tensor            # (n_global,)
    n_global: int

    # Per-element GLL physical coords (handy for sampling models)
    x_e: torch.Tensor                 # (E, ngll, ngll)
    z_e: torch.Tensor                 # (E, ngll, ngll)

    @property
    def ngll(self) -> int:
        return self.n_order + 1

    @property
    def n_elements(self) -> int:
        return self.nex * self.nez

    # Nominal element size — well-defined for the uniform-x topography mesh
    # (and exact for the axis-aligned degenerate case). Used by CPML profile
    # builders that need a physical thickness ``L = N · dh_e``.
    @property
    def dx_e(self) -> float:
        return self.x_max / self.nex

    @property
    def dz_e(self) -> float:
        return self.z_max / self.nez


def _build_ibool_for_axis_grid(nex: int, nez: int, ngll: int) -> tuple[torch.Tensor, int]:
    """Standard Quad4-style ibool laid out in row-major (z, x).

    Works for any conforming mesh whose element-grid topology is a
    regular ``nez × nex`` Cartesian array — the *physical* coords of
    nodes may be arbitrary, only the connectivity matters here.
    """
    n_order = ngll - 1
    nx_g = nex * n_order + 1
    nz_g = nez * n_order + 1
    n_global = nx_g * nz_g

    ez = torch.arange(nez).view(nez, 1, 1, 1)
    ex = torch.arange(nex).view(1, nex, 1, 1)
    jz = torch.arange(ngll).view(1, 1, ngll, 1)
    jx = torch.arange(ngll).view(1, 1, 1, ngll)
    gz = ez * n_order + jz
    gx = ex * n_order + jx
    ibool = (gz * nx_g + gx).reshape(nex * nez, ngll, ngll).to(torch.long)
    return ibool, n_global


def quad9_from_control_nodes(
    control_xy: torch.Tensor,         # (nez, nex, 3, 3, 2)
    *,
    n_order: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Quad9Mesh:
    """Build a deformed mesh from a per-element control-node tensor.

    ``control_xy[ez, ex, i, j, :]`` is the (x, z) coordinate of the
    ``(i, j)``-th Q2 control node of element ``(ez, ex)``. Shared GLL nodes on
    inter-element edges are de-duplicated by the analytic ``ibool`` indexing and
    take their coordinate from their lowest-numbered element copy (the mesh is
    assumed conforming, so the copies agree up to rounding); the build checks
    only ``det_J > 0`` (no inversion), then assembles Jacobian + ibool + global
    node coords.
    """
    if control_xy.ndim != 5 or control_xy.shape[2:] != (3, 3, 2):
        raise ValueError(
            f"control_xy must have shape (nez, nex, 3, 3, 2), got {tuple(control_xy.shape)}"
        )
    nez, nex = control_xy.shape[:2]
    control_xy = control_xy.to(dtype=dtype, device=device)
    control_flat = control_xy.reshape(nez * nex, 3, 3, 2)

    # Reference-element bits
    xi, w_ref = gll_nodes_weights(n_order, dtype=dtype, device=device)
    deriv = lagrange_derivative_matrix(xi)
    ngll = n_order + 1

    # Sample physical (x, z) and inv_J / det_J at every GLL node
    phys_xz = evaluate_q2_map(control_flat, xi, xi)            # (E, ngll, ngll, 2)
    _, inv_J, det_J = evaluate_q2_jacobian(control_flat, xi, xi)
    if (det_J <= 0).any():
        bad = (det_J <= 0).nonzero(as_tuple=False)[0].tolist()
        raise ValueError(
            f"Negative or zero Jacobian determinant at element/GLL {bad} — "
            "element is inverted or degenerate"
        )
    w_outer = w_ref[:, None] * w_ref[None, :]                  # (ngll, ngll)
    w_jac = w_outer * det_J                                    # (E, ngll, ngll)

    x_e = phys_xz[..., 0].contiguous()
    z_e = phys_xz[..., 1].contiguous()

    # Connectivity (axis-aligned element topology). We still rely on the
    # element grid being rectangular in (ez, ex) — only the coordinates
    # are deformed.
    ibool, n_global = _build_ibool_for_axis_grid(nex, nez, ngll)
    ibool = ibool.to(device=device)

    # Global node coordinates from the per-element GLL coords: a shared
    # node takes its lowest-numbered element copy, which is deterministic
    # where the old average of the copies was not (see nodes.py)
    x_global, z_global = node_values(first_copy_slot(ibool, n_global), x_e, z_e)

    return Quad9Mesh(
        n_order=n_order, nex=nex, nez=nez,
        x_max=float(x_global.max().item()),
        z_max=float(z_global.max().item()),
        xi=xi, w_ref=w_ref, deriv=deriv,
        control_xy=control_flat,
        inv_J=inv_J, det_J=det_J, w_jac=w_jac,
        ibool=ibool, x_global=x_global, z_global=z_global,
        n_global=n_global,
        x_e=x_e, z_e=z_e,
    )


def build_axis_aligned_quad9(
    nex: int, nez: int, x_max: float, z_max: float,
    *,
    n_order: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Quad9Mesh:
    """Affine Quad9 mesh = Quad4 with Q2 control nodes; used for degeneracy tests."""
    dx_e = x_max / nex
    dz_e = z_max / nez
    control = torch.zeros(nez, nex, 3, 3, 2, dtype=dtype, device=device)
    for ez in range(nez):
        for ex in range(nex):
            x0 = ex * dx_e
            z0 = ez * dz_e
            for i in range(3):     # η direction
                for j in range(3): # ξ direction
                    control[ez, ex, i, j, 0] = x0 + 0.5 * j * dx_e
                    control[ez, ex, i, j, 1] = z0 + 0.5 * i * dz_e
    return quad9_from_control_nodes(control, n_order=n_order,
                                    dtype=dtype, device=device)


def build_topography_quad9(
    nex: int, nez: int, x_max: float, z_bottom: float,
    z_top_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    n_order: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Quad9Mesh:
    """Build a conforming Quad9 mesh with the top edge following ``z_top_fn(x)``.

    The bottom edge is flat at ``z = z_bottom``. Each column of elements
    is graded linearly in z between the (curved) top and the (flat)
    bottom — the simplest conforming mesh that exactly carries the
    topography along the free surface.

    ``z_top_fn`` must accept a 1-D torch tensor of x coordinates and
    return the surface z (in the same SEM down-positive convention,
    so ``z_top_fn(x) ≥ 0`` everywhere and ``z_top_fn(x) < z_bottom``).

    The element-corner / edge-midpoint x-positions are uniformly spaced
    in x; z-positions inside each column are linearly graded.
    """
    # 1-D control-node grid: 2 · nex + 1 columns, 2 · nez + 1 rows
    nc_x = 2 * nex + 1
    nc_z = 2 * nez + 1
    x_axis = torch.linspace(0.0, x_max, nc_x, dtype=dtype, device=device)
    z_top  = z_top_fn(x_axis)                # (nc_x,)
    # Linear grading in z, per x column (uniform fractions between top
    # and bottom independent of x → top is curved, all other edges follow).
    frac_z = torch.linspace(0.0, 1.0, nc_z, dtype=dtype, device=device)
    z_grid = z_top[None, :] + frac_z[:, None] * (z_bottom - z_top[None, :])
    # z_grid[i_z, i_x] is the z at row i_z, column i_x of the global control
    # node grid. x is uniform so x_grid[i_z, i_x] = x_axis[i_x].
    x_grid = x_axis[None, :].expand_as(z_grid)

    # Re-pack into per-element (ez, ex, i, j, 2)
    control = torch.zeros(nez, nex, 3, 3, 2, dtype=dtype, device=device)
    for ez in range(nez):
        for ex in range(nex):
            for i in range(3):     # η: 0 = top of element (i_z = 2*ez)
                for j in range(3): # ξ: 0 = left   (i_x = 2*ex)
                    i_z = 2 * ez + i
                    i_x = 2 * ex + j
                    control[ez, ex, i, j, 0] = x_grid[i_z, i_x]
                    control[ez, ex, i, j, 1] = z_grid[i_z, i_x]
    return quad9_from_control_nodes(control, n_order=n_order,
                                    dtype=dtype, device=device)


def boundary_element_edges(mesh: Quad9Mesh) -> BoundaryEdges:
    """Same convention as for Quad4: returns flat element indices on each side."""
    nex, nez = mesh.nex, mesh.nez
    ez = torch.arange(nez, device=mesh.ibool.device)
    ex = torch.arange(nex, device=mesh.ibool.device)
    flat = lambda ez_, ex_: ez_ * nex + ex_
    return BoundaryEdges(
        left=flat(ez, torch.zeros_like(ez)),
        right=flat(ez, torch.full_like(ez, nex - 1)),
        top=flat(torch.zeros_like(ex), ex),
        bottom=flat(torch.full_like(ex, nez - 1), ex))
