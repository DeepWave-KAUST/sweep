"""Model-parallel helpers for SWEEP.

Splits the wavefield in space (``x``, optionally ``y``) across ranks so each
rank holds only a tile. :class:`HaloExchange` ships the per-step halo
strips between neighbour ranks.

Top-level objects:

:class:`MeshTopology`
    Pure rank-grid arithmetic, no ``torch.distributed`` dependency. Safe to
    construct in single-process unit tests.

:class:`ModelParallelMesh`
    Topology + ``ProcessGroups`` (``model_pg``, ``shot_pg``). Construction
    requires ``torch.distributed`` to be initialised.

:class:`HaloExchange`
    ``autograd.Function`` that fills halo strips from neighbour interiors
    in forward and accumulates the adjoint in backward.

:func:`partition_global_coords`
    Filter and shift global source / receiver coords into a rank's tile.

:func:`exchange_halos`
    Convenience: call :class:`HaloExchange` for each field in a list.
"""

from sweep.parallel._topology import MeshTopology, balanced_grid
from sweep.parallel.halo import HaloExchange, exchange_halos
from sweep.parallel.mesh import ModelParallelMesh
from sweep.parallel.pml import build_rank_pml_widths
from sweep.parallel.routing import partition_global_coords

__all__ = [
    "ModelParallel",
    "HaloExchange",
    "MeshTopology",
    "ModelParallelMesh",
    "balanced_grid",
    "build_rank_pml_widths",
    "exchange_halos",
    "partition_global_coords",
]


def __getattr__(name):  # lazy: ModelParallel pulls in PropTorch (heavy import)
    if name == "ModelParallel":
        from sweep.parallel.dd_propagator import ModelParallel
        return ModelParallel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
