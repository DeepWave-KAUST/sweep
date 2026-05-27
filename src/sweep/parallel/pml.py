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
