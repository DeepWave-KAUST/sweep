"""ElasticTTISG's backward scratch comes from the Python pool, and provably so.

A bit-exact gate cannot see this change: when the pool is NOT bound the C++
side allocates a fresh zero tensor of the same geometry, and the numbers are
identical either way. Green therefore proves nothing about whether the move
took effect (lesson_optimisation_flag_silently_inactive). The instrument here
is direct: the backward writes its scratch into the workspace, so after a
backward the Python-side pool must hold non-zero data. If the C++ side had
fallen back to its own zeros_like, the pool would still be all zeros.

One class serves both dimensions with different scratch (2-D: six plain
tensors, 3-D: the shared 18-tensor elastic workspace), so both are checked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from conftest import requires_binding

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver_gradient_mode_suite as suite  # noqa: E402


def _backward_once(key, nt):
    dev = torch.device("cuda:0")
    ns = suite.build_parser().parse_args([])
    ns.nt = nt
    spec = suite.SOLVERS[key]
    shape = suite.shape_for(spec, ns)
    scenario = suite.SCENARIOS["interior"]

    _true, models_init, grad_flags = suite.make_models(spec, shape)
    sources, receivers = suite.make_geometry(spec, shape, scenario, ns)
    wavelet = torch.tensor(suite.ricker(ns.nt, ns.dt, ns.freq, ns.delay), device=dev)
    solver = suite.build_solver(spec, "c", "full", scenario, shape, dev, ns,
                                Path("/tmp"), "wspool")

    models = suite.tensors_from_models(models_init, grad_flags, dev)
    syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    (syn.double() ** 2).sum().backward()
    torch.cuda.synchronize()
    return solver, models


@pytest.mark.parametrize("key, expected_n", [
    pytest.param("elastic_tti_sg2d", 6, marks=requires_binding("elastic_tti_sg2d_forward")),
    pytest.param("elastic_tti_sg3d", 18, marks=requires_binding("elastic_tti_sg3d_forward")),
])
def test_backward_writes_into_the_python_pool(key, expected_n):
    solver, models = _backward_once(key, nt=60)

    pool = solver.adjoint_workspace
    assert len(pool) == expected_n, (
        f"{key}: expected the {expected_n} declared workspace tensors, got {len(pool)}")
    touched = [bool((t != 0).any()) for t in pool]
    assert any(touched), (
        f"{key}: every pool tensor is still all-zero after a backward: the C++ side "
        "did not take the pool -- it fell back to its own zeros_like, and this "
        "change is inert")
    # And the gradient is a real one, so the backward that wrote the pool did work.
    g = [m.grad for m in models if m.grad is not None]
    assert g and all(torch.isfinite(x).all() for x in g)
    assert max(float(x.abs().max()) for x in g) > 0
