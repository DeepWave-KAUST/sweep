"""``gather_record`` must gather per shot group, not over the world.

With ``shot_groups > 1`` every group propagates a DIFFERENT shot through the
SAME tile grid, so two ranks that share a tile coordinate carry the same global
receiver indices holding different shots' traces.  A world-wide gather wrote
both into one array and whichever tile landed later silently won, so rank 0
returned a record spliced from several shots and nothing raised.

Runs on gloo/CPU: the defect is in which process group the collective uses, so
no GPU and no propagator are needed to expose it.
"""
import os
import tempfile

import pytest
import torch

dist = pytest.importorskip("torch.distributed")
if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("gloo is required", allow_module_level=True)

import torch.multiprocessing as mp

from sweep.parallel import assemble_tile_records

PY, PX, SHOT_GROUPS = 1, 2, 2
WORLD = PY * PX * SHOT_GROUPS
NT, NFIELD = 3, 1
# tile xi owns these global receiver columns
OWN = {0: [0, 1], 1: [2, 3]}


def _trace_value(shot_group, global_rec):
    """Distinct per (shot, receiver) so a mixed record cannot pass by luck."""
    return 100.0 * shot_group + global_rec


def _worker(rank, init_file):
    from sweep.parallel import ModelParallelMesh, gather_tile_records

    dist.init_process_group("gloo", init_method=f"file://{init_file}",
                            world_size=WORLD, rank=rank)
    try:
        mesh = ModelParallelMesh(grid=(PY, PX))
        topo = mesh.topology
        assert topo.shot_groups == SHOT_GROUPS, topo.shot_groups

        own = OWN[topo.xi]
        tile = torch.zeros(1, NT, len(own), NFIELD)
        for j, gi in enumerate(own):
            tile[:, :, j, :] = _trace_value(topo.shot_group, gi)

        full = gather_tile_records(tile, own, mesh)

        if topo.tile_rank != 0:
            assert full is None, f"rank {rank} is not a group root but got a record"
            return
        assert full is not None, f"rank {rank} is a group root and got None"
        assert full.shape == (1, NT, 4, NFIELD), full.shape
        for gi in range(4):
            want = _trace_value(topo.shot_group, gi)
            got = full[0, 0, gi, 0].item()
            assert got == want, (
                f"rank {rank} (shot group {topo.shot_group}) receiver {gi}: "
                f"got {got} (shot group {int(got // 100)}), want {want}. "
                "The gather crossed shot groups."
            )
    finally:
        dist.destroy_process_group()


def test_gather_record_does_not_cross_shot_groups():
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "store")
        mp.spawn(_worker, args=(init_file,), nprocs=WORLD, join=True)


def test_assemble_tile_records_places_columns_by_global_index():
    """The pure half: tiles land at their global index, gaps stay zero."""
    a = torch.full((1, NT, 2, NFIELD), 7.0)
    b = torch.full((1, NT, 1, NFIELD), 9.0)
    full = assemble_tile_records([([0, 3], a), ([], torch.empty(0)), ([1], b)])
    assert full.shape == (1, NT, 4, NFIELD)
    assert full[0, 0, 0, 0] == 7.0 and full[0, 0, 3, 0] == 7.0
    assert full[0, 0, 1, 0] == 9.0
    assert full[0, 0, 2, 0] == 0.0      # no tile owns column 2
