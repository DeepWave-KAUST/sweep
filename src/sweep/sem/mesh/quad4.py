"""Axis-aligned rectangular quad4 mesh with GLL node connectivity.

Phase-1 simplification: all elements have identical size and are axis-
aligned. Jacobian is constant per element (= diag(dx/2, dz/2)). For a
deformed mesh we'd need per-(element, GLL) Jacobian tensors and metric
derivatives — out of scope for the PoC.

Convention
----------
- Domain is [0, x_max] × [0, z_max], with z increasing **downward**
  (matching sweep's FD propagator and SPECFEM-style geophysics).
- Element index ``e`` is row-major over (ez, ex), 0-based:
  ``e = ez * nex + ex``, with ez = 0 at the top (z = 0).
- Within each element, local indices are ``(jz, jx)`` over GLL nodes;
  jz = 0 is the top edge.
- ``ibool[e, jz, jx]`` returns the global node index for that GLL point.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import NamedTuple

import torch

from ..basis.gll import gll_nodes_weights, lagrange_derivative_matrix


@dataclass
class Quad4Mesh:
    """Uniform axis-aligned rectangular mesh with GLL node connectivity."""

    n_order: int
    nex: int
    nez: int
    x_max: float
    z_max: float

    # Reference-element data
    xi: torch.Tensor       # (n_order+1,)  GLL nodes on [-1, 1]
    w_ref: torch.Tensor    # (n_order+1,)  GLL weights on [-1, 1]
    deriv: torch.Tensor    # (n_order+1, n_order+1)  D[i,j] = ℓ'_j(xi_i)

    # Physical element size and constant Jacobian factor per element
    dx_e: float
    dz_e: float
    jacobian_det: float    # |J| = (dx_e/2) * (dz_e/2)

    # Connectivity
    ibool: torch.Tensor    # (E, ngll, ngll)  int64
    x_global: torch.Tensor # (n_global,)
    z_global: torch.Tensor # (n_global,)
    n_global: int

    # Per-element GLL coords (handy for sampling model fields)
    x_e: torch.Tensor      # (E, ngll, ngll)
    z_e: torch.Tensor      # (E, ngll, ngll)

    @property
    def ngll(self) -> int:
        return self.n_order + 1

    @property
    def n_elements(self) -> int:
        return self.nex * self.nez


def build_quad4_mesh(
    nex: int,
    nez: int,
    x_max: float,
    z_max: float,
    n_order: int = 4,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Quad4Mesh:
    """Build a uniform Cartesian quad mesh with GLL connectivity.

    Parameters
    ----------
    nex, nez : number of elements in x, z.
    x_max, z_max : physical domain extents (z increases downward, top is z=0).
    n_order : polynomial order N. Default 4 → 5×5 GLL nodes per element.
    """
    ngll = n_order + 1
    xi, w_ref = gll_nodes_weights(n_order, dtype=dtype, device=device)
    deriv = lagrange_derivative_matrix(xi)

    dx_e = x_max / nex
    dz_e = z_max / nez

    # Global node grid: shared GLL points on element interfaces.
    nx_global = nex * n_order + 1
    nz_global = nez * n_order + 1
    n_global = nx_global * nz_global

    # Global node coords (we compute them analytically; alternative would be
    # to flatten per-element coords then unique() — slower and floating-point fragile)
    # In x: positions are union over elements of (x_e_left + (xi + 1) * dx_e / 2);
    # because element interfaces share an endpoint, the union de-duplicates.
    ix = torch.arange(nx_global, dtype=dtype, device=device)
    iz = torch.arange(nz_global, dtype=dtype, device=device)
    # Map global integer index to (element_ix, local_jx)
    e_ix = (ix.long() // n_order).clamp_max(nex - 1)
    j_ix = ix.long() - e_ix * n_order
    e_iz = (iz.long() // n_order).clamp_max(nez - 1)
    j_iz = iz.long() - e_iz * n_order
    x_axis = e_ix.to(dtype) * dx_e + (xi[j_ix] + 1.0) * (dx_e * 0.5)
    z_axis = e_iz.to(dtype) * dz_e + (xi[j_iz] + 1.0) * (dz_e * 0.5)
    # Outer product gives (nz_global, nx_global) coord grids
    z_grid, x_grid = torch.meshgrid(z_axis, x_axis, indexing="ij")
    x_global = x_grid.reshape(-1).contiguous()
    z_global = z_grid.reshape(-1).contiguous()

    # ibool[e, jz, jx] = global index in row-major (z, x) ordering
    # Global integer index of node (jz, jx) of element (ez, ex):
    #   gz = ez * n_order + jz
    #   gx = ex * n_order + jx
    #   g  = gz * nx_global + gx
    ez = torch.arange(nez, device=device).view(nez, 1, 1, 1)
    ex = torch.arange(nex, device=device).view(1, nex, 1, 1)
    jz = torch.arange(ngll, device=device).view(1, 1, ngll, 1)
    jx = torch.arange(ngll, device=device).view(1, 1, 1, ngll)
    gz = ez * n_order + jz
    gx = ex * n_order + jx
    ibool = (gz * nx_global + gx).reshape(nex * nez, ngll, ngll).to(torch.long)

    # Per-element GLL coords (E, ngll, ngll)
    x_e = x_global[ibool]
    z_e = z_global[ibool]

    return Quad4Mesh(
        n_order=n_order,
        nex=nex,
        nez=nez,
        x_max=float(x_max),
        z_max=float(z_max),
        xi=xi,
        w_ref=w_ref,
        deriv=deriv,
        dx_e=dx_e,
        dz_e=dz_e,
        jacobian_det=(dx_e * 0.5) * (dz_e * 0.5),
        ibool=ibool,
        x_global=x_global,
        z_global=z_global,
        n_global=n_global,
        x_e=x_e,
        z_e=z_e,
    )


class BoundaryEdges(NamedTuple):
    """Element indices whose edges touch each physical boundary.

    A fixed four-field record, so it is a NamedTuple rather than a dict: the
    side names are part of the type, ``edges.top`` autocompletes, and a
    misspelling is an AttributeError at the call site instead of a KeyError
    deep inside an equation's constructor. Look-up by a runtime side name
    still works via ``getattr(edges, side)``.
    """

    left: torch.Tensor      # x = 0
    right: torch.Tensor     # x = x_max
    top: torch.Tensor       # z = 0
    bottom: torch.Tensor    # z = z_max


def boundary_element_edges(mesh: Quad4Mesh) -> BoundaryEdges:
    """Return element indices whose edges touch each physical boundary.

    Edges:  ``'left'`` (x=0), ``'right'`` (x=x_max),
            ``'top'`` (z=0), ``'bottom'`` (z=z_max).
    Values are 1-D int64 tensors of element-flat indices ``e = ez*nex + ex``.
    """
    nex, nez = mesh.nex, mesh.nez
    ez = torch.arange(nez, device=mesh.ibool.device)
    ex = torch.arange(nex, device=mesh.ibool.device)
    flat = lambda ez_, ex_: ez_ * nex + ex_
    return BoundaryEdges(
        left=flat(ez, torch.zeros_like(ez)),
        right=flat(ez, torch.full_like(ez, nex - 1)),
        top=flat(torch.zeros_like(ex), ex),
        bottom=flat(torch.full_like(ex, nez - 1), ex))


def assemble_mass_vector(
    mesh: Quad4Mesh,
    rho_inv_c2: torch.Tensor,  # (E, ngll, ngll) — 1/(rho c^2) at GLL nodes
) -> torch.Tensor:
    """Assemble the global diagonal mass vector  M[g] = Σ_e w_i w_j |J_e| · ρc²-inv.

    Phase 1: constant ρ acoustic, so user passes 1/(ρ c²) sampled at GLL points.
    Returns a (n_global,) tensor.
    """
    w_ij = mesh.w_ref[:, None] * mesh.w_ref[None, :]                # (ngll, ngll)
    m_e = rho_inv_c2 * w_ij * mesh.jacobian_det                     # (E, ngll, ngll)
    out = torch.zeros(mesh.n_global, dtype=m_e.dtype, device=m_e.device)
    out.index_add_(0, mesh.ibool.reshape(-1), m_e.reshape(-1))
    return out
