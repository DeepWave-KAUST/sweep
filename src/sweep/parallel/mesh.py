"""Process-group-aware mesh wrapping :class:`MeshTopology`.

The arithmetic part of the rank grid (coords, neighbours, tile extent) lives
in :class:`sweep.parallel.MeshTopology` and is reachable via ``self.topology``.
This wrapper adds the two ``torch.distributed`` sub-groups used by halo
exchange and the gradient reductions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch.distributed as dist

from sweep.parallel._topology import MeshTopology


class ModelParallelMesh:
    """Topology + ``ProcessGroups`` for ``(P_shot, Py, Px)`` decomposition.

    Two sub-groups are created at construction time (one ``new_group`` call
    per ranks-list — every rank participates in every ``new_group`` call,
    even ones it isn't a member of, as required by PyTorch's collective
    contract):

    ``model_pg``
        The ``py * px`` tile ranks of THIS rank's shot group. Used by the
        halo exchange and the model-parallel gradient gather.

    ``shot_pg``
        The ``shot_groups`` ranks holding the same ``(yi, xi)`` tile across
        different shot groups. Used by the existing shot-parallel
        ``all_reduce`` on ``vp.grad``.

    Parameters
    ----------
    grid : (Py, Px)
        Tile-grid extents. ``Py == 1`` for 2-D.
    shot_groups : int, optional
        Number of shot-parallel groups. Defaults to ``world_size // (Py*Px)``.
    world_size, rank : int, optional
        Override the values pulled from ``torch.distributed``. Useful only
        for testing or non-default process-group setups.
    """

    def __init__(
        self,
        grid: Tuple[int, int],
        shot_groups: Optional[int] = None,
        *,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        py, px = grid
        if world_size is None or rank is None:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "ModelParallelMesh requires torch.distributed to be "
                    "initialised (or pass world_size & rank explicitly)."
                )
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        if shot_groups is None:
            tile = py * px
            if world_size % tile != 0:
                raise ValueError(
                    f"world_size={world_size} not divisible by py*px={tile}; "
                    f"specify shot_groups explicitly."
                )
            shot_groups = world_size // tile

        self.topology = MeshTopology(
            py=py, px=px, shot_groups=shot_groups,
            world_size=world_size, rank=rank,
        )
        self._model_pg: Optional[dist.ProcessGroup] = None
        self._shot_pg: Optional[dist.ProcessGroup] = None

        if dist.is_available() and dist.is_initialized():
            self._build_subgroups()

    # ------------------------------------------------------------------
    # Sub-group construction
    # ------------------------------------------------------------------
    def _build_subgroups(self) -> None:
        topo = self.topology
        tile = topo.tile_world_size

        for sg in range(topo.shot_groups):
            ranks = list(range(sg * tile, (sg + 1) * tile))
            pg = dist.new_group(ranks=ranks)
            if sg == topo.shot_group:
                self._model_pg = pg

        for yi in range(topo.py):
            for xi in range(topo.px):
                ranks = [
                    sg * tile + yi * topo.px + xi
                    for sg in range(topo.shot_groups)
                ]
                pg = dist.new_group(ranks=ranks)
                if yi == topo.yi and xi == topo.xi:
                    self._shot_pg = pg

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    @property
    def model_pg(self) -> Optional[dist.ProcessGroup]:
        return self._model_pg

    @property
    def shot_pg(self) -> Optional[dist.ProcessGroup]:
        return self._shot_pg

    # Convenience pass-throughs (so callers don't need ``mesh.topology.xxx``).
    @property
    def rank(self) -> int:
        return self.topology.rank

    @property
    def world_size(self) -> int:
        return self.topology.world_size

    @property
    def coord(self) -> Tuple[int, int, int]:
        return self.topology.coord

    def neighbour_rank(self, axis: str, direction: int) -> Optional[int]:
        return self.topology.neighbour_rank(axis, direction)

    def is_edge(self, axis: str, side: str) -> bool:
        return self.topology.is_edge(axis, side)

    def local_extent(self, global_shape):
        return self.topology.local_extent(global_shape)
