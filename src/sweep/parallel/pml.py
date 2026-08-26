"""Rank-local PML widths for a model-parallel mesh.

PML is a physical absorbing boundary — it must be applied **only on faces
that are on the global grid's physical boundary**. For a rank whose tile
sits in the middle of the mesh, the faces adjacent to a neighbour tile are
NOT physical boundaries; they're computational split lines that
:class:`HaloExchange` fills with neighbour interior data. Adding PML on
such a face would incorrectly attenuate waves that are physically
travelling through.

This module gives :func:`build_rank_pml_widths` — a helper that returns
the per-side PML widths for this rank's tile, with zeros on the sides that
face a neighbour and ``abcn`` on the sides that face the physical edge.

The z axis is never split (v1 keeps depth whole), so every rank gets the
same z widths and the existing image-method / free-surface treatment is
untouched.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING


if TYPE_CHECKING:
    from sweep.parallel.mesh import ModelParallelMesh


def build_rank_pml_widths(
    mesh: "ModelParallelMesh",
    abcn: int,
    ndim: int,
    *,
    image_method_active: bool = False,
) -> List[int]:
    """Per-side PML widths for THIS rank, in SWEEP's
    ``[z_low, z_high, (y_low, y_high,) x_low, x_high]`` layout.

    Parameters
    ----------
    mesh : ModelParallelMesh
        Mesh topology (we only consult ``is_edge``).
    abcn : int
        Global PML width (the value PML would have on every side if the
        rank were running un-split).
    ndim : int
        2 or 3.
    image_method_active : bool
        If True, the z-low (top) PML is suppressed (the image method
        handles the free surface). Mirrors the existing single-rank
        behaviour in ``PropBase.init_abc``.

    Returns
    -------
    list of int
        Length ``2 * ndim``. Pass directly to
        ``set_cpml_profiles_{s,r}(pml_width=...)``. Sides that don't have
        a neighbour-facing constraint get ``abcn``; sides that face a
        neighbour get 0 (no PML).

    Notes
    -----
    A 1-tile axis (``px==1`` or ``py==1``) is treated as having both
    edges, so PML is applied on both ends — matches the single-rank
    behaviour for that axis. :meth:`MeshTopology.is_edge` already returns
    True for both low and high in that case.
    """
    if ndim not in (2, 3):
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")

    z_low = 0 if image_method_active else abcn
    z_high = abcn

    x_low = abcn if mesh.is_edge("x", "low") else 0
    x_high = abcn if mesh.is_edge("x", "high") else 0

    if ndim == 2:
        return [z_low, z_high, x_low, x_high]

    y_low = abcn if mesh.is_edge("y", "low") else 0
    y_high = abcn if mesh.is_edge("y", "high") else 0
    return [z_low, z_high, y_low, y_high, x_low, x_high]


# ---------------------------------------------------------------------------
# Cut-face bitmask
# ---------------------------------------------------------------------------
# ``SolverContext::cut_mask`` (csrc/cuda/common/context.h) numbers its faces
# x, z, y -- NOT the axis order used by the free-surface bitmask in
# ``equations/_edges.py::fs_faces_to_c_bitmask``, which is z, y, x. The two
# masks live in the same struct and are literally reversed with respect to each
# other, so they must never be "harmonised" without changing the CUDA side to
# match. ``test_cut_face_mask.py`` parses the header and asserts these agree.
CUT_X_LO = 1
CUT_X_HI = 2
CUT_Z_LO = 4
CUT_Z_HI = 8
CUT_Y_LO = 16
CUT_Y_HI = 32


def dd_cut_face_mask(mesh, ndim: int) -> int:
    """Which of this rank's faces are cut lines rather than physical boundaries.

    A cut face cannot be inferred from ``pad == 0``: a free-surface face also has
    zero pad, and the two need opposite treatment -- a free surface reflects, a
    cut face is filled from the neighbour. Hence an explicit mask.

    Only x (and y in 3-D) can be cut; the z axis is never split, so no z bit is
    ever set here even though the C side defines them.
    """
    if mesh is None:
        return 0
    mask = 0
    if not mesh.is_edge("x", "low"):
        mask |= CUT_X_LO
    if not mesh.is_edge("x", "high"):
        mask |= CUT_X_HI
    if ndim == 3:
        if not mesh.is_edge("y", "low"):
            mask |= CUT_Y_LO
        if not mesh.is_edge("y", "high"):
            mask |= CUT_Y_HI
    return mask


def dd_cut_pad(mesh, pad, abcn: int, ndim: int):
    """Shrink `pad` to zero on faces that are cut lines.

    ``image_method_active=False`` is passed on purpose. That flag makes
    :func:`build_rank_pml_widths` zero the ``z_lo`` entry, and its contract
    ("the top face is a free surface") only held on the old DD branch, where
    ``_image_method_active`` meant exactly that. Under the per-edge free-surface
    feature, ``resolve_topo_method`` returns ``image=True`` for ANY free face, so
    passing it here would delete the top PML whenever e.g. only the LEFT face is
    free -- a face that is neither a free surface nor ever cut (the cut mask only
    sets x/y bits).

    ``normalize_pad`` has already zeroed every genuine free-surface face, so this
    call must contribute cut faces and nothing else. The ``min`` is what keeps it
    additive-only: a face already zeroed stays zero.
    """
    if mesh is None:
        return tuple(pad)
    cut_pad = build_rank_pml_widths(mesh, abcn=abcn, ndim=ndim,
                                    image_method_active=False)
    return tuple(min(p, c) for p, c in zip(pad, cut_pad))
