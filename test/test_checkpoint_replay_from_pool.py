"""The checkpoint modes' replay buffers are allocated by the propagator.

Same sentinel instrument as the scratch pools: the buffers Python allocated
next to the checkpoint snapshots are filled with a sentinel between forward
and backward, and the backward must overwrite it -- a driver that kept
allocating its own replay buffer would leave every sentinel in place. The
run's gradient is meaningless under the sentinel (the replay overwrites the
rows it reads, so it is in fact unaffected, but that is not asserted).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from conftest import requires_binding

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver_gradient_mode_suite as suite  # noqa: E402

SENTINEL = 1e-3

# suite key -> (compiled binding prefix, declared buffer count, modes)
CASES = {
    "acoustic_vti_1st_2d": ("acoustic_vti_1st_2d", 1, ("ckpt_chunk",)),
    "acoustic_vti_1st_3d": ("acoustic_vti_1st_3d", 1, ("ckpt_chunk",)),
    "vrz3d": ("acoustic_vrz3d", 1, ("ckpt_chunk", "ckpt_recursive")),
    "lsrtm2d": ("acoustic_lsrtm2d", 1, ("ckpt_chunk",)),
    "lsrtm3d": ("acoustic_lsrtm3d", 1, ("ckpt_chunk",)),
    "visco2d": ("visco_acoustic2d", 1, ("ckpt_chunk",)),
    "elastic_tti_2nd2d": ("elastic_tti_2nd2d", 2, ("ckpt_chunk",)),
    "das2d": ("das2d", 1, ("ckpt_chunk", "ckpt_recursive")),
    "das3d": ("das3d", 1, ("ckpt_chunk",)),
}


@pytest.mark.parametrize("key, mode", [
    pytest.param(k, m, marks=requires_binding(f"{v[0]}_forward"))
    for k, v in CASES.items() for m in v[2]
])
def test_the_backward_writes_the_handed_over_replay_buffers(key, mode):
    _prefix, n_buffers, _modes = CASES[key]
    dev = torch.device("cuda:0")
    ns = suite.build_parser().parse_args([])
    ns.nt = 60
    spec = suite.SOLVERS[key]
    shape = suite.shape_for(spec, ns)
    scenario = suite.SCENARIOS["interior"]
    _true, models_init, grad_flags = suite.make_models(spec, shape)
    sources, receivers = suite.make_geometry(spec, shape, scenario, ns)
    wavelet = torch.tensor(suite.ricker(ns.nt, ns.dt, ns.freq, ns.delay), device=dev)
    solver = suite.build_solver(spec, "c", mode, scenario, shape, dev, ns,
                                Path("/tmp"), "replaypool")

    models = suite.tensors_from_models(models_init, grad_flags, dev)
    syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    torch.cuda.synchronize()
    replay = solver.checkpoint_replay
    assert len(replay) == n_buffers, f"{key}/{mode}: expected {n_buffers} replay buffers, got {len(replay)}"
    for t in replay:
        t.fill_(SENTINEL)
    (syn.double() ** 2).sum().backward()
    torch.cuda.synchronize()

    touched = [bool((t != SENTINEL).any()) for t in replay]
    assert any(touched), (
        f"{key}/{mode}: every replay buffer still holds the sentinel after a backward: "
        "the driver allocated its own, and this binding is inert")
    assert all(m.grad is None or torch.isfinite(m.grad).all() for m in models)
