"""ModelParallelMesh's sub-groups run on the caller's process-group timeout.

``new_group`` does not inherit the timeout given to ``init_process_group``, so
the mesh reads it back off the default group (``_default_pg_timeout``) and
passes it on.  That lookup asked the group's CUDA backend whenever torch saw a
GPU: on a GPU host every rank of a gloo (CPU) run initialised CUDA there, and a
``cpu:gloo`` group, which has no CUDA backend, lost the timeout outright -- its
sub-groups were back on torch's 30-minute default.

Two gloo ranks pretend to see a GPU (whatever this host has), build a mesh
under each spelling of the gloo backend, and check that both sub-groups carry
the timeout the ranks were initialised with and that CUDA was left alone.
"""
from __future__ import annotations

import datetime
import json
import os
import traceback

import pytest
import torch

dist = pytest.importorskip("torch.distributed")
if not dist.is_available() or not dist.is_gloo_available():
    pytest.skip("gloo is required", allow_module_level=True)

import torch.multiprocessing as mp

WORLD = 2
#: No backend's default (gloo 30 min, NCCL 10 min).
TIMEOUT = datetime.timedelta(seconds=123)
BACKENDS = ("gloo", "cpu:gloo")


def _worker(rank, out_dir):
    from sweep.parallel import ModelParallelMesh

    torch.cuda.is_available = lambda: True    # a GPU host, as far as the mesh can tell
    for i, backend in enumerate(BACKENDS):
        dist.init_process_group(backend, init_method=f"file://{out_dir}/store{i}",
                                world_size=WORLD, rank=rank, timeout=TIMEOUT)
        try:
            mesh = ModelParallelMesh(grid=(1, WORLD))
            res = {"timeouts": [g._get_backend(torch.device("cpu")).options._timeout.total_seconds()
                                for g in (mesh.model_pg, mesh.shot_pg)],
                   "cuda_initialized": torch.cuda.is_initialized()}
        except Exception:
            res = {"error": traceback.format_exc()}
        with open(os.path.join(out_dir, f"{backend}.rank{rank}.json"), "w") as f:
            json.dump(res, f)
        dist.barrier()
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def mesh_runs(tmp_path_factory):
    """Both backends through one spawn of the two ranks: the output dir."""
    out = tmp_path_factory.mktemp("mesh_pg_timeout")
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(r, str(out))) for r in range(WORLD)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    hung = [p for p in procs if p.is_alive()]
    for p in hung:
        p.terminate()
    if hung:
        pytest.fail(f"{len(hung)} rank(s) did not finish in 120 s")
    codes = [p.exitcode for p in procs]
    if any(codes):
        pytest.fail(f"rank exit codes {codes}")
    return out


@pytest.mark.parametrize("backend", BACKENDS)
def test_subgroups_inherit_the_timeout(mesh_runs, backend):
    for rank in range(WORLD):
        res = json.loads((mesh_runs / f"{backend}.rank{rank}.json").read_text())
        if "error" in res:
            pytest.fail(f"rank {rank}:\n{res['error']}", pytrace=False)
        assert res["timeouts"] == [TIMEOUT.total_seconds()] * 2, (
            f"rank {rank}: model_pg / shot_pg timeouts {res['timeouts']} s, "
            f"the ranks were initialised with {TIMEOUT.total_seconds()} s")
        assert not res["cuda_initialized"], f"rank {rank}: building the mesh initialised CUDA"
