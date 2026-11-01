"""sweep.autodiff under ModelParallel: every rank takes the same difference step.

Each rank may pass its own tile, so the step's peaks max|m| and max|h| are
all-reduced over the model's process group.  Before, every rank took the step
of its own tile, so the collective forward ran m + diag(e_k) h: on 2 GPUs, with
the direction ten times weaker on one tile, H h came back with an error eight
times its norm (test/dd_second_order_check.py), and a rank whose tile
direction was zero raised while the others entered the collective.  Runs on
gloo/CPU: only the step and its collective are exercised, with no propagator.
"""
import os
import tempfile

import pytest
import torch

dist = pytest.importorskip("torch.distributed")
if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("gloo is required", allow_module_level=True)

import torch.multiprocessing as mp

WORLD = 2
REL_STEP = 1e-2


def _tiles(mesh):
    """A ModelParallel as _step sees it: its mesh and its device."""
    from sweep.parallel.dd_propagator import ModelParallel

    prop = ModelParallel.__new__(ModelParallel)
    prop.mesh, prop.dev = mesh, torch.device("cpu")
    return prop


def _worker(rank, init_file):
    from sweep.autodiff import _step
    from sweep.parallel import ModelParallelMesh

    dist.init_process_group("gloo", init_method=f"file://{init_file}", world_size=WORLD, rank=rank)
    try:
        prop = _tiles(ModelParallelMesh(grid=(1, WORLD)))
        tile = torch.full((4, 3), 2000.0 + 500.0 * rank)            # peaks 2000 and 2500
        h = torch.ones(4, 3) * (1.0 if rank == 0 else 0.1)           # peaks 1 and 0.1
        expected = REL_STEP * 2500.0 / 1.0

        def all_steps(e):
            out = [None] * WORLD
            dist.all_gather_object(out, e)
            return out

        e = _step(prop, [tile], [h], REL_STEP)
        assert all_steps(e) == [expected] * WORLD, (rank, e)

        h_zero_here = h * (rank == 0)                                # zero on rank 1's tile only
        assert all_steps(_step(prop, [tile], [h_zero_here], REL_STEP)) == [expected] * WORLD

        for bad, message in ((torch.zeros(4, 3), "nothing to differentiate"),
                             (h * (float("nan") if rank == 1 else 1.0), "not finite")):
            with pytest.raises(ValueError, match=message):           # on every rank, not one
                _step(prop, [tile], [bad], REL_STEP)
    finally:
        dist.destroy_process_group()


def test_every_rank_takes_the_step_of_the_whole_model():
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(_worker, args=(os.path.join(d, "store"),), nprocs=WORLD, join=True)
