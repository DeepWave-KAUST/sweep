"""Source / receiver coordinate partitioning for a model-parallel mesh.

The SWEEP source / receiver coordinate layout is::

    coords[..., 0]  = x       (lateral / split)
    coords[..., 1]  = y       (crossline / split, 3-D only)
    coords[..., -1] = z       (depth, never split)

Inferred from ``src/sweep/sources/torch.py``: a ``torch.flip(coords, [-1])``
followed by ``index_put_`` into a wavefield shaped ``(B, 1, Nz, [Ny,] Nx)``
only matches if ``coords[..., 0]`` is x and ``coords[..., -1]`` is z.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch

from sweep.parallel._topology import MeshTopology


def partition_global_coords(
    coords_global: torch.Tensor,
    topology: MeshTopology,
    global_shape: Tuple[int, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select and shift the coords that fall inside THIS rank's tile.

    Parameters
    ----------
    coords_global : torch.Tensor
        Shape ``(nshots, npts, ndim)`` integer tensor of GLOBAL indices.
        ``ndim`` must equal ``len(global_shape)`` (2 or 3).
    topology : MeshTopology
        Mesh layout; only the local ``(yi, xi)`` offset is consulted.
    global_shape : tuple of int
        ``(Nz, [Ny,] Nx)`` — used to compute the tile extent.

    Returns
    -------
    local_coords : torch.Tensor
        Same shape as ``coords_global``. For on-tile entries the split axes
        are shifted to local indexing (``x - ox``, ``y - oy``); off-tile
        entries are zeroed so they cannot be silently used as valid indices.
    mask : torch.Tensor
        Shape ``(nshots, npts)`` boolean tensor, ``True`` where the coord is
        inside this rank's tile.
    """
    if coords_global.ndim != 3:
        raise ValueError(
            "coords_global must be (nshots, npts, ndim); "
            f"got shape {tuple(coords_global.shape)}"
        )
    ndim = coords_global.shape[-1]
    if ndim != len(global_shape):
        raise ValueError(
            f"coords ndim={ndim} does not match global_shape ndim={len(global_shape)}"
        )

    local_shape, offsets = topology.local_extent(global_shape)
    # ``local_shape``/``offsets`` use SWEEP wavefield order (z, [y,] x).
    # ``coords`` use (x, [y,] z). Map between them explicitly.

    mask = torch.ones(
        coords_global.shape[0], coords_global.shape[1],
        dtype=torch.bool, device=coords_global.device,
    )
    local = coords_global.clone()

    if ndim == 2:
        nx_loc = local_shape[1]
        ox = offsets[1]
        x = coords_global[..., 0]
        in_x = (x >= ox) & (x < ox + nx_loc)
        mask = mask & in_x
        local[..., 0] = x - ox
    else:  # ndim == 3
        ny_loc = local_shape[1]
        nx_loc = local_shape[2]
        oy = offsets[1]
        ox = offsets[2]
        x = coords_global[..., 0]
        y = coords_global[..., 1]
        in_x = (x >= ox) & (x < ox + nx_loc)
        in_y = (y >= oy) & (y < oy + ny_loc)
        mask = mask & in_x & in_y
        local[..., 0] = x - ox
        local[..., 1] = y - oy

    local = torch.where(
        mask.unsqueeze(-1), local, torch.zeros_like(local)
    )
    return local, mask


def gather_tile_records(
    tile_record: torch.Tensor,
    own_rec_idx: Sequence[int],
    mesh: Any,
) -> Optional[torch.Tensor]:
    """Reassemble one shot group's record from its tiles, on the group root.

    The inverse of :func:`partition_global_coords`: that call handed each tile
    the receiver columns whose global coordinates fall inside it, this one puts
    those columns back at their global index.

    The gather runs over ``mesh.model_pg`` -- the ``py * px`` ranks that
    decompose ONE shot -- and NOT over the world. With ``shot_groups > 1``
    every group propagates a DIFFERENT shot through the SAME tile grid, so two
    ranks sharing a tile coordinate carry the same global receiver indices
    holding different shots' traces. A world-wide gather writes both into one
    array and whichever tile is assembled later silently wins, which is a
    corrupt record rather than an error.

    Parameters
    ----------
    tile_record
        This rank's record, ``(..., nrec_tile, nt)``.
    own_rec_idx
        Global receiver indices of this tile's columns, in tile order.
    mesh
        A :class:`sweep.parallel.ModelParallelMesh`.

    Returns
    -------
    torch.Tensor or None
        The assembled record on the shot group's root rank
        (``topology.tile_rank == 0``, i.e. global rank
        ``shot_group * py * px``), ``None`` on every other rank.
    """
    import torch.distributed as dist

    topo = mesh.topology
    is_root = topo.tile_rank == 0
    payload = (list(own_rec_idx), tile_record.detach().cpu())
    gathered: List[Any] = [None] * topo.tile_world_size
    # ``dst`` is a GLOBAL rank even when ``group`` restricts the collective
    # (torch keeps ``group_dst`` for the group-relative spelling, and torch is
    # unpinned here), so name the group root the long way round.
    dist.gather_object(
        payload,
        gathered if is_root else None,
        dst=topo.rank_at(topo.shot_group, 0, 0),
        group=mesh.model_pg,
    )
    if not is_root:
        return None
    return assemble_tile_records(gathered)


def assemble_tile_records(
    gathered: Sequence[Tuple[Sequence[int], torch.Tensor]],
) -> Optional[torch.Tensor]:
    """Place each tile's receiver columns at their global index.

    Pure and collective-free so it can be tested without ``torch.distributed``.
    Tiles that own no receiver contribute nothing; a global index no tile owns
    stays zero.
    """
    full = None
    owned = [(idx, rc) for idx, rc in gathered if len(idx)]
    if not owned:
        return None
    nrec_global = max(max(idx) for idx, _ in owned) + 1
    for idx, rc in owned:
        if full is None:
            shape = list(rc.shape)
            shape[-2] = nrec_global
            full = torch.zeros(shape, dtype=rc.dtype)
        for j, gi in enumerate(idx):
            full[..., gi, :] = rc[..., j, :]
    return full
