"""Per-axis SLAB storage of the 3-D acoustic C-PML memory fields.

A C-PML memory field is updated as ``psi_d' = b_d * q + a_d * psi_d`` from a
zero start, and ``b_d`` is exactly ``+0.0`` outside the absorbing layer of
axis d (``cpml/acoustic3d.py::_ab_coeffs``: ``torch.where(sigma > 0, .., 0)``)
while ``a_d = exp(-dt*(sigma+alpha))`` is ``1.0`` there. So ``psi_d``,
``zeta_d`` and ``a_d * psi_d`` are exactly ``+0.0`` at every node off the
layer, for all time -- checked on the stored bytes of the full-length tapes
by ``test/sem/test_sem_psi_slab.py`` -- and
the full-length arrays the drivers used to allocate (fifteen of them in the
forward, six restores plus three products in the backward) held ~93% zeros
at production size.

The compact layout keeps each field on the index SLAB of its axis: the set
of planes ``i_d`` on which some node has ``b_d != 0`` (on the flat mesh
``b_d`` is a function of ``i_d`` alone, so the slab IS the support; on a
deformed mesh the support sits inside a slab one plane wider, still exact).
The node numbering ``g = (iz*ny + iy)*nx + ix`` -- shared by the flat Hex8
and the deformed Hex27 meshes and by every DD strip -- turns a node into its
slot with integer arithmetic plus one 1-D table per axis, so there is no
per-node map to store:

    x:  slot = (iz*ny + iy) * nsx + tab_x[ix]        length nz*ny*nsx
    y:  slot = (iz*nsy + tab_y[iy]) * nx + ix        length nz*nsy*nx
    z:  slot = (tab_z[iz]*ny + iy) * nx + ix         length nsz*ny*nx

with ``tab_d[i] = -1`` off the slab (the kernels read the literal 0.0f there
and drop the +0.0 they would have written). ``planes_d`` is the inverse
table, ``ring_slot_d`` re-expresses the boundary-saving ring index
``band_idx_d`` in slot coordinates, ``dir_slot_d`` the Dirichlet nodes that
lie on the slab. The CUDA side mirrors these formulas in
``sem3d/kernels.cuh`` (``SemSlabs``); ``slab_slot`` here is the reference
the tests hold it to.
"""
from __future__ import annotations

import torch

AXES = ("x", "y", "z")
_DIM = {"x": 2, "y": 1, "z": 0}


def node_grid_dims(ibool, ngll: int, n_global: int) -> tuple[int, int, int]:
    """``(nx, ny, nz)`` of the structured node grid behind ``ibool``.

    Read off the element-local strides of the first element (x fastest, then
    y, then z) and then CHECKED on every element: the node numbering must be
    ``base + kx + ky*nx + kz*nx*ny`` everywhere, or the slot arithmetic would
    silently address the wrong node. A mesh that is not numbered this way
    raises here, and the caller keeps the full-length layout."""
    E = ibool.shape[0]
    ib = ibool.reshape(E, ngll, ngll, ngll).to(torch.int64)   # [e, kz, ky, kx]
    sx = int(ib[0, 0, 0, 1] - ib[0, 0, 0, 0])                  # along kx: 1
    nx = int(ib[0, 0, 1, 0] - ib[0, 0, 0, 0])                  # along ky: nx
    nxy = int(ib[0, 1, 0, 0] - ib[0, 0, 0, 0])                 # along kz: nx*ny
    if sx != 1 or nx <= 0 or nxy <= 0 or nxy % nx:
        raise ValueError(f"ibool is not a structured x-fastest grid: strides {sx}, {nx}, {nxy}")
    ny = nxy // nx
    if n_global % nxy:
        raise ValueError(f"n_global {n_global} is not a multiple of nx*ny {nxy}")
    nz = n_global // nxy
    kx = torch.arange(ngll, device=ib.device)
    want = (ib[:, :1, :1, :1] + kx.reshape(1, 1, 1, -1)
            + (kx * nx).reshape(1, 1, -1, 1) + (kx * nxy).reshape(1, -1, 1, 1))
    if not torch.equal(want, ib):
        raise ValueError("ibool is not the structured numbering g = (iz*ny + iy)*nx + ix")
    return nx, ny, nz


def slab_slot(axis: str, g, dims, tab):
    """Reference of the device formula: slots of global nodes ``g`` on the
    slab of ``axis`` (``-1`` off it). ``tab`` is that axis's table."""
    nx, ny, _nz = dims
    g = g.to(torch.int64)
    ix = g % nx
    r = torch.div(g, nx, rounding_mode="floor")
    iy = r % ny
    iz = torch.div(r, ny, rounding_mode="floor")
    t = tab.to(g.device)
    ns = int((t >= 0).sum())
    if axis == "x":
        tt = t[ix]
        s = r * ns + tt
    elif axis == "y":
        tt = t[iy]
        s = (iz * ns + tt) * nx + ix
    else:
        tt = t[iz]
        s = (tt * ny + iy) * nx + ix
    return torch.where(tt >= 0, s, torch.full_like(s, -1))


def slab_length(axis: str, dims, ns: int) -> int:
    nx, ny, nz = dims
    return {"x": nz * ny * ns, "y": nz * ns * nx, "z": ns * ny * nx}[axis]


def plane_mask(b, axis: str, dims, db=None):
    """Which planes of ``axis`` the coefficients damp: ``b != 0`` (union
    ``db != 0`` when ``db`` is given) reduced over the other two axes of the
    ``(nx, ny, nz)`` node grid.

    One place, because getting the reduced axes right is easy to get wrong by
    hand -- the array is stored ``(nz, ny, nx)`` while ``dims`` reads
    ``(nx, ny, nz)``, so z reduces over ``(1, 2)`` and x over ``(0, 1)``.

    ``db``'s support is NOT inside ``b``'s: it reaches exactly one plane
    further at each end of the band (measured, 2 x 65^2 nodes per horizontal
    axis at nex=16). The STORAGE slab is their union, because ``g_d`` is read
    wherever either is non-zero; the a*psi element flags stay on ``b`` alone,
    which is what feeds ``psi_d``.
    """
    nx, ny, nz = dims
    other = tuple(d for d in (0, 1, 2) if d != _DIM[axis])
    on = (b.reshape(nz, ny, nx) != 0).any(dim=other)
    if db is not None:
        on = on | (db.reshape(nz, ny, nx) != 0).any(dim=other)
    return on


def band_node_mask(c: dict):
    """``(n_global,) uint8``: 1 where ANY C-PML coefficient is non-zero.

    The node kernels take an interior fast path wherever all six are zero --
    62% of nodes at nex=48, 70% at nex=64 -- and, because a warp is 32
    consecutive nodes of the x-fastest numbering and takes the branch only if
    ALL of them do, 50% of WARPS at nex=48 and none at all at nex=16.

    (The fraction is the PRODUCT of the three axes' complements, not one minus
    their mean: the mask is a union and the three bands overlap. Measured
    interior node fraction by nex: 16 -> 0.168, 32 -> 0.471, 48 -> 0.619,
    64 -> 0.703; interior warps: 0.000, 0.317, 0.498, 0.605. An earlier
    version of this docstring said 86%/90%, which is one minus the mean.)

    Deciding it by loading the six
    fp32 arrays costs 24 B at EVERY node, against the 40 B the interior branch
    then actually uses; one byte answers the same question and the branch
    taken is identical, so nothing about the arithmetic changes.

    The band path still reads the coefficients it needs, but only where it is
    taken.
    """
    m = torch.zeros_like(c["b_x"], dtype=torch.bool)
    for ax in AXES:
        m |= c[f"b_{ax}"] != 0
        m |= c[f"db{ax}_d{ax}"] != 0
    return m.to(torch.uint8).contiguous()


def cpml_slab_tables(c: dict, planes: dict | None = None) -> dict:
    """The slab bookkeeping for one propagator or strip, from its consts:
    ``ibool`` (int32 (E, m)), ``n_global``, ``ngll``, the coefficient arrays
    ``b_x/b_y/b_z`` (n_global), the ring indices ``band_idx_{x,y,z}`` and
    ``dirichlet_idx``. Returns the keys the C drivers read.

    ``planes`` overrides which planes of an axis count as in-band, per axis.
    A strip derives its own from ITS slice of the profile, and on a deformed
    mesh two strips can then disagree about a shared y or z plane -- the band
    curves, so a plane can carry a damped node in one strip's x range and not
    in another's. Nothing noticed while the slab fields were strip-local; a
    slab field that is EXCHANGED across a cut needs the two sides to agree
    entry for entry, so the DD builder passes the union over strips here. A
    wider slab is still exact (the extra planes hold exact zeros), which is
    why the override is safe; the strip's own set must be a SUBSET of it.
    """
    dev = c["b_x"].device
    nx, ny, nz = node_grid_dims(c["ibool"], int(c["ngll"]), int(c["n_global"]))
    out, cols = {"nx": nx, "ny": ny, "nz": nz}, {}
    for ax in AXES:
        # the STORAGE slab is b's planes union db's (see plane_mask); the
        # a*psi element flags below stay on b's alone
        on_b = plane_mask(c[f"b_{ax}"], ax, (nx, ny, nz))
        on = plane_mask(c[f"b_{ax}"], ax, (nx, ny, nz),
                        c[f"db{ax}_d{ax}"])              # (n_ax,) planes
        if planes is not None and ax in planes:
            wide = planes[ax].to(on.device)
            if bool((on & ~wide).any()):
                raise AssertionError(
                    f"the {ax} plane override drops planes this strip damps")
            on = wide
        ns = int(on.sum())
        tab = torch.full((on.numel(),), -1, dtype=torch.int32, device=dev)
        tab[on] = torch.arange(ns, dtype=torch.int32, device=dev)
        out[f"tab_{ax}"] = tab
        out[f"planes_{ax}"] = torch.nonzero(on).reshape(-1).to(torch.int32)
        out[f"ns{ax}"] = ns
        out[f"slab_n_{ax}"] = slab_length(ax, (nx, ny, nz), ns)
        ring = slab_slot(ax, c[f"band_idx_{ax}"], (nx, ny, nz), tab)
        if ring.numel() and bool((ring < 0).any()):
            raise AssertionError(f"band_idx_{ax} has nodes off the {ax} slab")
        out[f"ring_slot_{ax}"] = ring.contiguous()
        d = c["dirichlet_idx"]
        ds = slab_slot(ax, d, (nx, ny, nz), tab) if d.numel() else d.to(torch.int64)
        out[f"dir_slot_{ax}"] = ds[ds >= 0].contiguous()
        cols[ax] = _elem_columns(on_b, int(c["ngll"]) - 1)
    out["elem_slab"] = _elem_slab(cols, dev)
    return out


def _elem_columns(on, no):
    """Which ELEMENT columns along one axis touch the slab: element ``i``
    spans the node planes ``[i*no, i*no + no]``, so it touches the slab as
    soon as one of those planes carries a non-zero coefficient."""
    n_el = (on.numel() - 1) // no
    assert n_el * no + 1 == on.numel(), (
        f"{on.numel()} node planes is not n_elements * {no} + 1: the element "
        "grid is not the structured one the flags are derived from")
    return on.unfold(0, no + 1, no).any(dim=1)          # (n_el,)


def _elem_slab(cols, dev):
    """``(E,)`` int32, bit d set where element e may carry a non-zero
    ``psi_d`` -- the element grid is ``e = (ez*ney + ey)*nex + ex``.

    ``psi_d`` starts at zero and grows only through ``b_d``, so an element
    whose every node has ``b_d == 0`` carries ``a_d*psi_d == +0.0`` on all
    of them: the assembly's whole ``a*psi`` half contributes exactly +0.0
    there and the kernels skip it. At production sizes that is most of the
    mesh -- 79% of the (element, axis) pairs at nex=32, 93% at nex=96.
    """
    x, y, z = (cols[ax].to(torch.int32) for ax in AXES)
    f = (x[None, None, :] | (y[None, :, None] << 1) | (z[:, None, None] << 2))
    return f.reshape(-1).contiguous().to(dev)


SLAB_KEYS = tuple(["nx", "ny", "nz", "elem_slab"]
                  + [f"{k}_{ax}" for ax in AXES for k in
                     ("tab", "planes", "slab_n", "ring_slot", "dir_slot")]
                  + [f"ns{ax}" for ax in AXES])

FIELD_AXIS = {"psi_x": "x", "psi_y": "y", "psi_z": "z",
              "zeta_x": "x", "zeta_y": "y", "zeta_z": "z",
              "psi_x_n": "x", "psi_y_n": "y", "psi_z_n": "z",
              "zeta_x_n": "x", "zeta_y_n": "y", "zeta_z_n": "z",
              "apsi_x": "x", "apsi_y": "y", "apsi_z": "z",
              "apx": "x", "apy": "y", "apz": "z",
              "r_psi_x": "x", "r_psi_y": "y", "r_psi_z": "z",
              "r_zeta_x": "x", "r_zeta_y": "y", "r_zeta_z": "z",
              # the adjoint carries: NOT zero off the layer (combine writes
              # a_d*(l + lap) there with a_d = 1) but dead -- step_T consumes
              # them only as b_d*l, and b_d is +0.0 off the slab. Dropping
              # them changes at most the sign of an exact-zero term.
              # lap_d: written by grads_T on slab d alone and read by combine
              # only where the slot exists (NaN-poison test).
              # g_d: the assembly's own gradient, read only where b_d or db_d
              # is non-zero -- which is exactly what the storage slab is.
              # k_d: read alone only as b_d*k_d, and b_d is +0.0 off the slab
              # (the interior reads the fused m_d = ga_d - k_d instead).
              "lapx": "x", "lapy": "y", "lapz": "z",
              "gx": "x", "gy": "y", "gz": "z",
              "kx": "x", "ky": "y", "kz": "z",
              "lpsi_x": "x", "lpsi_y": "y", "lpsi_z": "z",
              "lzeta_x": "x", "lzeta_y": "y", "lzeta_z": "z",
              "lpsi_x_n": "x", "lpsi_y_n": "y", "lpsi_z_n": "z",
              "lzeta_x_n": "x", "lzeta_y_n": "y", "lzeta_z_n": "z"}


def set_slab_fields(inp, c: dict, on: bool) -> None:
    """Copy the slab bookkeeping onto a pybind input struct (either layout;
    the tables are cheap and the kernels only read them when psi_slab=1)."""
    inp.psi_slab = int(on)
    for k in ("nx", "ny", "nz", "nsx", "nsy", "nsz"):
        setattr(inp, k, int(c[k]))
    for ax in AXES:
        for k in ("tab", "planes", "ring_slot", "dir_slot"):
            setattr(inp, f"{k}_{ax}", c[f"{k}_{ax}"])


def field_len(name: str, c: dict, on: bool) -> int:
    """Allocation length of a memory-field array under either layout."""
    return int(c[f"slab_n_{FIELD_AXIS[name]}"]) if on else int(c["n_global"])
