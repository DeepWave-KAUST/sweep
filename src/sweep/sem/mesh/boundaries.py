"""Global-node index sets on the mesh boundary — one implementation for 2-D and 3-D.

Four equation modules carried a copy of this (``acoustic2d``,
``acoustic2d_deformed``, ``acoustic3d``, ``acoustic3d_deformed``), and a fifth
reached across module boundaries to import one of the private copies. The 2-D
pair was byte-identical; so was the 3-D pair once a trailing comment is ignored.
Only two things actually varied: whether the boundary elements come from
*edges* (2-D) or *faces* (3-D), and which ``ibool`` axis the side slices.

Both are data, so they live in :data:`_SIDE_AXIS` rather than in a chain of
``elif``\\ s repeated four times. A new side, or a new mesh family, is a table
entry.
"""

from __future__ import annotations

import torch

from .quad4 import boundary_element_edges as _edges_2d
from .hex8 import boundary_element_faces as _faces_3d

# side -> (ibool axis to slice, take the FIRST or the LAST GLL index on it).
# ``ibool`` is (element, z, [y,] x): axis 0 is z, and in 3-D axis 1 is y.
_SIDE_AXIS = {
    2: {"top": (0, 0), "bottom": (0, -1),
        "left": (1, 0), "right": (1, -1)},
    3: {"top": (0, 0), "bottom": (0, -1),
        "front": (1, 0), "back": (1, -1),
        "left": (2, 0), "right": (2, -1)},
}


def mesh_ndim(mesh) -> int:
    """2 or 3, from the mesh's own structure rather than from its class name."""
    return mesh.ibool.dim() - 1


def boundary_elements(mesh):
    """Element indices touching each boundary — edges in 2-D, faces in 3-D."""
    return _edges_2d(mesh) if mesh_ndim(mesh) == 2 else _faces_3d(mesh)


def free_surface_node_idx(mesh, sides, device) -> torch.Tensor:
    """Sorted, de-duplicated global-node indices on ``sides``.

    ``sides=()`` is a live configuration -- ``PropSEM(free_surface=False)``
    resolves to an empty tuple and passes it straight through -- so the empty
    case returns an empty index tensor rather than tripping over an empty
    ``torch.cat``.

    Stays on the device throughout. The four copies this replaces went through
    ``set(...tolist())``, forcing a GPU->host sync inside every ``build()``;
    that mattered because ``build()`` is called once per FWI iteration.
    """
    empty = torch.zeros(0, dtype=torch.long, device=device)
    if not sides:
        return empty
    ndim = mesh_ndim(mesh)
    table = _SIDE_AXIS[ndim]
    bnd = boundary_elements(mesh)
    ng = mesh.ngll
    picked = []
    for side in sides:
        if side not in table:
            raise ValueError(
                f"unknown boundary side {side!r} in {ndim}-D; "
                f"valid: {sorted(table)}")
        axis, end = table[side]
        elems = getattr(bnd, side).to(device=device, dtype=torch.long)
        # index_select on the ibool axis (offset by 1: axis 0 is the element)
        sel = mesh.ibool.to(device).index_select(0, elems).index_select(
            axis + 1, torch.tensor([ng - 1 if end else 0], device=device))
        picked.append(sel.reshape(-1))
    return torch.unique(torch.cat(picked)) if picked else empty
