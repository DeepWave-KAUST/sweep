"""Model-parallel helpers for SWEEP.

Splits the wavefield in space (``x``, optionally ``y``) across ranks so each
rank holds only a tile. Halo exchange between time steps is implemented in
:mod:`sweep.parallel.halo` (added in PR-2).

The two top-level objects are:

:class:`MeshTopology`
    Pure rank-grid arithmetic, no ``torch.distributed`` dependency. Safe to
    construct in single-process unit tests.

:class:`ModelParallelMesh`
    Topology + ``ProcessGroups`` (``model_pg``, ``shot_pg``). Construction
    requires ``torch.distributed`` to be initialised.

See :doc:`docs/dev-plans/model_parallel_README.md` for the design overview.
"""

from sweep.parallel._topology import MeshTopology
from sweep.parallel.mesh import ModelParallelMesh
from sweep.parallel.routing import partition_global_coords

__all__ = [
    "MeshTopology",
    "ModelParallelMesh",
    "partition_global_coords",
]
