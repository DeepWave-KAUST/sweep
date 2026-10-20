"""Deformed hexahedral SEM mesh with Q2 (27-node) element geometry — the 3-D
analogue of :mod:`sweep.sem.mesh.quad9` (SPECFEM3D ``NGNOD = 27``).

The element interior is mapped from the reference cube [−1, 1]³ to physical space
via triquadratic Lagrange shape functions on 27 control nodes. The GLL solution
nodes stay ``N + 1`` per direction; what changes from :class:`Hex8Mesh` is purely
the geometry layer — the Jacobian ``J(ξ, η, γ)`` is per-GLL (3×3), not constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ..basis.gll import gll_nodes_weights, lagrange_derivative_matrix
from ..basis.hex27 import evaluate_hex27_map, evaluate_hex27_jacobian
from .nodes import first_copy_slot, node_values


@dataclass
class Hex27Mesh:
    """Deformed-hex SEM mesh with Q2 geometry and per-GLL 3×3 Jacobian."""

    n_order: int
    nex: int
    ney: int
    nez: int
    x_max: float
    y_max: float
    z_max: float

    xi: torch.Tensor                  # (ngll,)
    w_ref: torch.Tensor               # (ngll,)
    deriv: torch.Tensor               # (ngll, ngll)

    control_xyz: torch.Tensor         # (E, 3, 3, 3, 3)
    inv_J: torch.Tensor               # (E, ngll, ngll, ngll, 3, 3)
    det_J: torch.Tensor               # (E, ngll, ngll, ngll)  positive
    w_jac: torch.Tensor               # (E, ngll, ngll, ngll)  = w₃ |J|

    ibool: torch.Tensor               # (E, ngll, ngll, ngll) int64
    x_global: torch.Tensor
    y_global: torch.Tensor
    z_global: torch.Tensor
    n_global: int

    x_e: torch.Tensor                 # (E, ngll, ngll, ngll)
    y_e: torch.Tensor
    z_e: torch.Tensor

    @property
    def ngll(self) -> int:
        return self.n_order + 1

    @property
    def n_elements(self) -> int:
        return self.nex * self.ney * self.nez

    # Nominal element sizes (exact for the axis-aligned case; used by CPML
    # profile builders that need a physical thickness L = N · dh_e).
    @property
    def dx_e(self) -> float:
        return self.x_max / self.nex

    @property
    def dy_e(self) -> float:
        return self.y_max / self.ney

    @property
    def dz_e(self) -> float:
        return self.z_max / self.nez


def _build_ibool_hex(nex, ney, nez, ngll):
    """Hex8/Hex27 ibool, row-major (z, y, x) — element topology only."""
    n_order = ngll - 1
    nx_g, ny_g, nz_g = nex * n_order + 1, ney * n_order + 1, nez * n_order + 1
    n_global = nx_g * ny_g * nz_g
    ez = torch.arange(nez).view(nez, 1, 1, 1, 1, 1)
    ey = torch.arange(ney).view(1, ney, 1, 1, 1, 1)
    ex = torch.arange(nex).view(1, 1, nex, 1, 1, 1)
    kz = torch.arange(ngll).view(1, 1, 1, ngll, 1, 1)
    ky = torch.arange(ngll).view(1, 1, 1, 1, ngll, 1)
    kx = torch.arange(ngll).view(1, 1, 1, 1, 1, ngll)
    gz = ez * n_order + kz
    gy = ey * n_order + ky
    gx = ex * n_order + kx
    ibool = ((gz * ny_g + gy) * nx_g + gx).reshape(
        nex * ney * nez, ngll, ngll, ngll).to(torch.long)
    return ibool, n_global


def _pack_control(X, Y, Z, nez, ney, nex):
    """Per-element Q2 control nodes from global control grids ``(nc_z,nc_y,nc_x)``.

    Returns ``(E, 3, 3, 3, 3)`` with axis order ``(k=γ/z, i=η/y, j=ξ/x, xyz)``.
    """
    ez = torch.arange(nez).view(nez, 1, 1, 1, 1, 1)
    ey = torch.arange(ney).view(1, ney, 1, 1, 1, 1)
    ex = torch.arange(nex).view(1, 1, nex, 1, 1, 1)
    k = torch.arange(3).view(1, 1, 1, 3, 1, 1)
    i = torch.arange(3).view(1, 1, 1, 1, 3, 1)
    j = torch.arange(3).view(1, 1, 1, 1, 1, 3)
    gz = (2 * ez + k).expand(nez, ney, nex, 3, 3, 3)
    gy = (2 * ey + i).expand(nez, ney, nex, 3, 3, 3)
    gx = (2 * ex + j).expand(nez, ney, nex, 3, 3, 3)
    ctrl = torch.stack([X[gz, gy, gx], Y[gz, gy, gx], Z[gz, gy, gx]], dim=-1)
    return ctrl.reshape(nez * ney * nex, 3, 3, 3, 3)


def hex27_from_control_nodes(
    control_xyz: torch.Tensor,        # (E, 3, 3, 3, 3)
    nex: int, ney: int, nez: int,
    *,
    n_order: int = 4,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Hex27Mesh:
    """Assemble Jacobian + ibool + global node coords from element control nodes."""
    control_xyz = control_xyz.to(dtype=dtype, device=device)
    xi, w_ref = gll_nodes_weights(n_order, dtype=dtype, device=device)
    deriv = lagrange_derivative_matrix(xi)
    ngll = n_order + 1

    phys = evaluate_hex27_map(control_xyz, xi, xi, xi)         # (E,gz,gy,gx,3)
    _, inv_J, det_J = evaluate_hex27_jacobian(control_xyz, xi, xi, xi)
    if (det_J <= 0).any():
        bad = (det_J <= 0).nonzero(as_tuple=False)[0].tolist()
        raise ValueError(
            f"Negative or zero Jacobian determinant at element/GLL {bad} — "
            "element is inverted or degenerate")
    w = w_ref
    w3 = w[:, None, None] * w[None, :, None] * w[None, None, :]
    w_jac = w3 * det_J

    x_e = phys[..., 0].contiguous()
    y_e = phys[..., 1].contiguous()
    z_e = phys[..., 2].contiguous()

    ibool, n_global = _build_ibool_hex(nex, ney, nez, ngll)
    ibool = ibool.to(device=device)
    # a shared node's coordinate from its lowest-numbered element copy:
    # deterministic where the old average of the copies was not (nodes.py)
    xg, yg, zg = node_values(first_copy_slot(ibool, n_global), x_e, y_e, z_e)

    return Hex27Mesh(
        n_order=n_order, nex=nex, ney=ney, nez=nez,
        x_max=float(xg.max()), y_max=float(yg.max()), z_max=float(zg.max()),
        xi=xi, w_ref=w_ref, deriv=deriv,
        control_xyz=control_xyz, inv_J=inv_J, det_J=det_J, w_jac=w_jac,
        ibool=ibool, x_global=xg, y_global=yg, z_global=zg, n_global=n_global,
        x_e=x_e, y_e=y_e, z_e=z_e)


def build_axis_aligned_hex27(
    nex, ney, nez, x_max, y_max, z_max,
    *, n_order=4, dtype=torch.float32, device="cpu",
) -> Hex27Mesh:
    """Affine Hex27 mesh = Hex8 with Q2 control nodes (degeneracy tests)."""
    nc = lambda ne: 2 * ne + 1
    x_axis = torch.linspace(0.0, x_max, nc(nex), dtype=dtype, device=device)
    y_axis = torch.linspace(0.0, y_max, nc(ney), dtype=dtype, device=device)
    z_axis = torch.linspace(0.0, z_max, nc(nez), dtype=dtype, device=device)
    Z, Y, X = torch.meshgrid(z_axis, y_axis, x_axis, indexing="ij")
    control = _pack_control(X, Y, Z, nez, ney, nex)
    return hex27_from_control_nodes(control, nex, ney, nez, n_order=n_order,
                                    dtype=dtype, device=device)


def build_topography_hex27(
    nex, ney, nez, x_max, y_max, z_bottom,
    z_top_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *, n_order=4, dtype=torch.float32, device="cpu",
) -> Hex27Mesh:
    """Conforming Hex27 mesh whose top face follows ``z_top_fn(x, y)``.

    The bottom face is flat at ``z = z_bottom``; each vertical column is graded
    linearly in z between the curved top surface and the flat bottom. ``z_top_fn``
    takes two equal-shaped tensors of (x, y) and returns the surface z (SEM
    down-positive: ``0 ≤ z_top < z_bottom``). x and y control positions are
    uniform.
    """
    nc = lambda ne: 2 * ne + 1
    x_axis = torch.linspace(0.0, x_max, nc(nex), dtype=dtype, device=device)
    y_axis = torch.linspace(0.0, y_max, nc(ney), dtype=dtype, device=device)
    YY, XX = torch.meshgrid(y_axis, x_axis, indexing="ij")     # (nc_y, nc_x)
    z_top = z_top_fn(XX, YY)                                   # (nc_y, nc_x)
    frac_z = torch.linspace(0.0, 1.0, nc(nez), dtype=dtype, device=device)
    # Z[iz, iy, ix] = z_top[iy,ix] + frac[iz] * (z_bottom - z_top[iy,ix])
    Z = z_top[None] + frac_z[:, None, None] * (z_bottom - z_top[None])
    X = x_axis.view(1, 1, -1).expand_as(Z)
    Y = y_axis.view(1, -1, 1).expand_as(Z)
    control = _pack_control(X, Y, Z, nez, ney, nex)
    return hex27_from_control_nodes(control, nex, ney, nez, n_order=n_order,
                                    dtype=dtype, device=device)
