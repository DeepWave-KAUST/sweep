"""Axis-aligned rectangular hex8 mesh with GLL node connectivity (3-D).

The 3-D analogue of :mod:`sweep.sem.mesh.quad4`. All elements have identical
size and are axis-aligned, so the Jacobian is constant per element
(= diag(dx/2, dy/2, dz/2)) and |J| = dx·dy·dz/8. A deformed (Hex27) mesh would
carry a per-(element, GLL) Jacobian — out of scope here, mirroring the
Quad4/Quad9 split in 2-D.

Convention
----------
- Domain is [0, x_max] × [0, y_max] × [0, z_max], with z increasing
  **downward** (matching sweep's FD propagator and the 2-D SEM path).
- Element index ``e`` is row-major over (ez, ey, ex), 0-based:
  ``e = (ez * ney + ey) * nex + ex``, with ez = 0 at the top (z = 0).
- Within each element, local GLL indices are ``(kz, ky, kx)`` over the
  reference axes (ζ ↔ z, η ↔ y, ξ ↔ x); kz = 0 is the top face.
- ``ibool[e, kz, ky, kx]`` returns the global node index for that GLL point,
  in row-major (z, y, x) ordering of the global node grid.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import NamedTuple

import torch

from ..basis.gll import gll_nodes_weights, lagrange_derivative_matrix


@dataclass
class Hex8Mesh:
    """Uniform axis-aligned rectangular hex mesh with GLL node connectivity."""

    n_order: int
    nex: int
    ney: int
    nez: int
    x_max: float
    y_max: float
    z_max: float

    # Reference-element data
    xi: torch.Tensor       # (n_order+1,)  GLL nodes on [-1, 1]
    w_ref: torch.Tensor    # (n_order+1,)  GLL weights on [-1, 1]
    deriv: torch.Tensor    # (n_order+1, n_order+1)  D[i,j] = ℓ'_j(xi_i)

    # Physical element size and constant Jacobian factor per element
    dx_e: float
    dy_e: float
    dz_e: float
    jacobian_det: float    # |J| = (dx_e/2)·(dy_e/2)·(dz_e/2)

    # Connectivity
    ibool: torch.Tensor    # (E, ngll, ngll, ngll)  int64
    x_global: torch.Tensor # (n_global,)
    y_global: torch.Tensor # (n_global,)
    z_global: torch.Tensor # (n_global,)
    n_global: int

    # Per-element GLL coords (handy for sampling model fields)
    x_e: torch.Tensor      # (E, ngll, ngll, ngll)
    y_e: torch.Tensor
    z_e: torch.Tensor

    # The O(nx+ny+nz) description this mesh was tiled from (None on a mesh
    # assembled by hand). Lets a consumer that must not touch the arrays --
    # the DD strips -- describe THIS mesh rather than re-derive one.
    layout: "Hex8Layout | None" = None

    @property
    def ngll(self) -> int:
        return self.n_order + 1

    @property
    def n_elements(self) -> int:
        return self.nex * self.ney * self.nez


@dataclass
class Hex8Layout:
    """The O(nx + ny + nz) description of a uniform axis-aligned hex8 mesh.

    Everything :func:`build_hex8_mesh` materialises per element-node -- ``ibool``,
    ``x_e/y_e/z_e``, ``x/y/z_global``, measured 50.7 B per global node -- is a
    function of what is stored here: the reference element, the element size
    and the three 1-D node-coordinate axes. The coordinate arrays are a
    meshgrid of the axes, so ``x_global[g] == x_axis[g % nx_global]`` exactly,
    and ``ibool`` is the row-major formula in :func:`hex8_ibool`.

    Why it exists: the domain-decomposition path used to hold the global mesh
    on every rank while computing on one strip of it, which made the mesh the
    largest term that did not shrink with the number of GPUs. A rank now reads
    THIS and builds only its strip's arrays from it; the fused path still gets
    the full :class:`Hex8Mesh`, built lazily from the same object.
    """

    n_order: int
    nex: int
    ney: int
    nez: int
    x_max: float
    y_max: float
    z_max: float

    xi: torch.Tensor       # (n_order+1,)
    w_ref: torch.Tensor    # (n_order+1,)
    deriv: torch.Tensor    # (n_order+1, n_order+1)

    dx_e: float
    dy_e: float
    dz_e: float
    jacobian_det: float

    x_axis: torch.Tensor   # (nx_global,) node x-coordinates, one per column
    y_axis: torch.Tensor   # (ny_global,)
    z_axis: torch.Tensor   # (nz_global,)

    @property
    def ngll(self) -> int:
        return self.n_order + 1

    @property
    def n_elements(self) -> int:
        return self.nex * self.ney * self.nez

    @property
    def nx_global(self) -> int:
        return self.nex * self.n_order + 1

    @property
    def ny_global(self) -> int:
        return self.ney * self.n_order + 1

    @property
    def nz_global(self) -> int:
        return self.nez * self.n_order + 1

    @property
    def n_global(self) -> int:
        return self.nx_global * self.ny_global * self.nz_global

    @property
    def device(self) -> torch.device:
        return self.x_axis.device

    @property
    def dtype(self) -> torch.dtype:
        return self.x_axis.dtype


def hex8_layout(
    nex: int,
    ney: int,
    nez: int,
    x_max: float,
    y_max: float,
    z_max: float,
    n_order: int = 4,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> Hex8Layout:
    """The axes and scalars of :func:`build_hex8_mesh` without its arrays.

    Same arguments, same arithmetic: ``build_hex8_mesh`` calls this and then
    tiles the result, so a layout and the mesh built from it never disagree.
    """
    xi, w_ref = gll_nodes_weights(n_order, dtype=dtype, device=device)
    deriv = lagrange_derivative_matrix(xi)

    dx_e = x_max / nex
    dy_e = y_max / ney
    dz_e = z_max / nez

    # Global node grid: shared GLL points on element interfaces.
    nx_global = nex * n_order + 1
    ny_global = ney * n_order + 1
    nz_global = nez * n_order + 1

    # 1-D global node coordinates per axis (interfaces de-duplicate because
    # adjacent elements share an endpoint GLL node), computed analytically.
    def axis_coords(n_axis_global, n_elem, d_e):
        ig = torch.arange(n_axis_global, dtype=dtype, device=device)
        e_i = (ig.long() // n_order).clamp_max(n_elem - 1)
        j_i = ig.long() - e_i * n_order
        return e_i.to(dtype) * d_e + (xi[j_i] + 1.0) * (d_e * 0.5)

    return Hex8Layout(
        n_order=n_order,
        nex=nex, ney=ney, nez=nez,
        x_max=float(x_max), y_max=float(y_max), z_max=float(z_max),
        xi=xi, w_ref=w_ref, deriv=deriv,
        dx_e=dx_e, dy_e=dy_e, dz_e=dz_e,
        jacobian_det=(dx_e * 0.5) * (dy_e * 0.5) * (dz_e * 0.5),
        x_axis=axis_coords(nx_global, nex, dx_e),
        y_axis=axis_coords(ny_global, ney, dy_e),
        z_axis=axis_coords(nz_global, nez, dz_e),
    )


def hex8_ibool(nex: int, ney: int, nez: int, n_order: int,
               device: torch.device | str = "cpu",
               *, int32: bool = False) -> torch.Tensor:
    """``ibool[e, kz, ky, kx]`` for the row-major (z, y, x) global node grid.

    Global integer index of node ``(kz, ky, kx)`` of element ``(ez, ey, ex)``::

        gz = ez*n_order + kz ; gy = ey*n_order + ky ; gx = ex*n_order + kx
        g  = (gz*ny_global + gy)*nx_global + gx

    with ``e = (ez*ney + ey)*nex + ex``. This formula IS the mesh's numbering;
    the domain-decomposition strips rely on it being x-fastest (a cut is a
    whole YZ plane) and check their rebuilt strip against it.

    ``int32=True`` returns it narrowed, for a consumer that hands it straight
    to the kernels (which take int32) and never wants the int64 copy as well
    -- a DD strip held both, measured 15.5 B per global node per rank. The
    range is checked here, not assumed: the ceiling is on ``E*ngll^3``, which
    exceeds ``n_global`` and so overflows first.
    """
    ngll = n_order + 1
    nx_global = nex * n_order + 1
    ny_global = ney * n_order + 1
    ez = torch.arange(nez, device=device).view(nez, 1, 1, 1, 1, 1)
    ey = torch.arange(ney, device=device).view(1, ney, 1, 1, 1, 1)
    ex = torch.arange(nex, device=device).view(1, 1, nex, 1, 1, 1)
    kz = torch.arange(ngll, device=device).view(1, 1, 1, ngll, 1, 1)
    ky = torch.arange(ngll, device=device).view(1, 1, 1, 1, ngll, 1)
    kx = torch.arange(ngll, device=device).view(1, 1, 1, 1, 1, ngll)
    gz = ez * n_order + kz
    gy = ey * n_order + ky
    gx = ex * n_order + kx
    out = ((gz * ny_global + gy) * nx_global + gx).reshape(
        nex * ney * nez, ngll, ngll, ngll)
    if not int32:
        return out.to(torch.long)
    n_el_nodes = nex * ney * nez * ngll ** 3
    if n_el_nodes > 2 ** 31 - 1:
        raise ValueError(
            f"int32 ibool needs E*ngll^3 = {n_el_nodes} <= 2^31-1; this mesh "
            "is past what the int32 kernel offsets can address")
    return out.to(torch.int32)


def build_hex8_mesh(
    nex: int,
    ney: int,
    nez: int,
    x_max: float,
    y_max: float,
    z_max: float,
    n_order: int = 4,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    geometry: bool = True,
    ibool32: bool = False,
) -> Hex8Mesh:
    """Build a uniform Cartesian hex mesh with GLL connectivity.

    Parameters
    ----------
    nex, ney, nez : number of elements in x, y, z.
    x_max, y_max, z_max : physical domain extents (z increases downward,
        top is z=0).
    n_order : polynomial order N. Default 4 → 5×5×5 GLL nodes per element.
    geometry : with ``False`` the six coordinate arrays (``x_e/y_e/z_e``,
        ``x/y/z_global`` -- 35 of the mesh's 50.7 B per node) are left
        ``None`` and only the connectivity is built. For a consumer that
        reads ``ibool`` and the element counts alone, such as a
        domain-decomposition strip whose coordinates come from the global
        :class:`Hex8Layout`.
    ibool32 : store the connectivity as int32 (the width the kernels take).
        Opt-in, and only for such a consumer: a DD strip otherwise holds an
        int64 mesh copy AND the int32 kernel copy, measured 15.5 B per global
        node per rank. The single-domain path and the boundary helpers read
        ``Hex8Mesh.ibool`` expecting the default int64.
    """
    lay = hex8_layout(nex, ney, nez, x_max, y_max, z_max, n_order,
                      dtype=dtype, device=device)
    ibool = hex8_ibool(nex, ney, nez, n_order, device, int32=ibool32)

    if geometry:
        # (nz_global, ny_global, nx_global) coord grids, row-major (z, y, x).
        z_grid, y_grid, x_grid = torch.meshgrid(
            lay.z_axis, lay.y_axis, lay.x_axis, indexing="ij")
        x_global = x_grid.reshape(-1).contiguous()
        y_global = y_grid.reshape(-1).contiguous()
        z_global = z_grid.reshape(-1).contiguous()
        # Per-element GLL coords (E, ngll, ngll, ngll)
        x_e = x_global[ibool]
        y_e = y_global[ibool]
        z_e = z_global[ibool]
    else:
        x_global = y_global = z_global = None
        x_e = y_e = z_e = None

    return Hex8Mesh(
        n_order=n_order,
        nex=nex, ney=ney, nez=nez,
        x_max=lay.x_max, y_max=lay.y_max, z_max=lay.z_max,
        xi=lay.xi, w_ref=lay.w_ref, deriv=lay.deriv,
        dx_e=lay.dx_e, dy_e=lay.dy_e, dz_e=lay.dz_e,
        jacobian_det=lay.jacobian_det,
        ibool=ibool,
        x_global=x_global, y_global=y_global, z_global=z_global,
        n_global=lay.n_global,
        x_e=x_e, y_e=y_e, z_e=z_e,
        layout=lay,
    )


class BoundaryFaces(NamedTuple):
    """Element indices whose faces touch each physical boundary.

    The 3-D twin of :class:`~sweep.sem.mesh.quad4.BoundaryEdges`; a NamedTuple
    for the same reason -- the six side names are part of the type, so a typo
    is an AttributeError at the call site rather than a KeyError inside an
    equation's constructor. ``getattr(faces, side)`` still serves a runtime
    side name.
    """

    left: torch.Tensor      # x = 0
    right: torch.Tensor     # x = x_max
    front: torch.Tensor     # y = 0
    back: torch.Tensor      # y = y_max
    top: torch.Tensor       # z = 0
    bottom: torch.Tensor    # z = z_max


def boundary_element_faces(mesh: Hex8Mesh) -> BoundaryFaces:
    """Element indices whose faces touch each physical boundary.

    Faces:  ``'left'`` (x=0), ``'right'`` (x=x_max),
            ``'front'`` (y=0), ``'back'`` (y=y_max),
            ``'top'`` (z=0), ``'bottom'`` (z=z_max).
    Values are 1-D int64 tensors of element-flat indices
    ``e = (ez*ney + ey)*nex + ex``.
    """
    nex, ney, nez = mesh.nex, mesh.ney, mesh.nez
    dev = mesh.ibool.device
    ez = torch.arange(nez, device=dev)
    ey = torch.arange(ney, device=dev)
    ex = torch.arange(nex, device=dev)
    flat = lambda ez_, ey_, ex_: (ez_ * ney + ey_) * nex + ex_

    # left/right: vary (ez, ey) at fixed ex
    EZ, EY = torch.meshgrid(ez, ey, indexing="ij")
    EZ, EY = EZ.reshape(-1), EY.reshape(-1)
    # front/back: vary (ez, ex) at fixed ey
    EZ2, EX2 = torch.meshgrid(ez, ex, indexing="ij")
    EZ2, EX2 = EZ2.reshape(-1), EX2.reshape(-1)
    # top/bottom: vary (ey, ex) at fixed ez
    EY3, EX3 = torch.meshgrid(ey, ex, indexing="ij")
    EY3, EX3 = EY3.reshape(-1), EX3.reshape(-1)

    return BoundaryFaces(
        left=flat(EZ, EY, torch.zeros_like(EZ)),
        right=flat(EZ, EY, torch.full_like(EZ, nex - 1)),
        front=flat(EZ2, torch.zeros_like(EZ2), EX2),
        back=flat(EZ2, torch.full_like(EZ2, ney - 1), EX2),
        top=flat(torch.zeros_like(EY3), EY3, EX3),
        bottom=flat(torch.full_like(EY3, nez - 1), EY3, EX3))
