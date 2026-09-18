"""The compiled backward's scratch comes from the Python pool, and provably so.

A bit-exact gate cannot see this change: when the pool is NOT bound the C++
side allocates a fresh zero tensor of the same geometry, and the numbers are
identical either way. Green therefore proves nothing about whether the move
took effect (lesson_optimisation_flag_silently_inactive). The instrument here
is direct: the backward writes its scratch into the workspace, so after a
backward the Python-side pool must hold non-zero data. If the C++ side had
fallen back to its own zeros_like, the pool would still be all zeros.

One further invariant rides along for the equations whose pool carries a
read-only zero buffer (the "previous stress" at the first reverse step):
that slot must still be zero afterwards, because nothing else guarantees it
is zero the next time -- the pool is zeroed per forward, not per backward.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from conftest import requires_binding

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver_gradient_mode_suite as suite  # noqa: E402


def _backward_once(key, mode, nt):
    dev = torch.device("cuda:0")
    ns = suite.build_parser().parse_args([])
    ns.nt = nt
    spec = suite.SOLVERS[key]
    shape = suite.shape_for(spec, ns)
    scenario = suite.SCENARIOS["interior"]

    _true, models_init, grad_flags = suite.make_models(spec, shape)
    sources, receivers = suite.make_geometry(spec, shape, scenario, ns)
    wavelet = torch.tensor(suite.ricker(ns.nt, ns.dt, ns.freq, ns.delay), device=dev)
    solver = suite.build_solver(spec, "c", mode, scenario, shape, dev, ns,
                                Path("/tmp"), "wspool")

    models = suite.tensors_from_models(models_init, grad_flags, dev)
    syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    (syn.double() ** 2).sum().backward()
    torch.cuda.synchronize()
    return solver, models


# suite key -> (declared pool size per suite mode, index of the read-only zero
# slot or None, modes to run). The modes differ in which slots they touch (the
# checkpoint seeds only exist in ckpt_chunk; DAS 2-D's boundary-saving mode is
# the only one that needs its six extra grids).
ALL = ("full", "bs_gpu", "ckpt_chunk")
CASES = {
    "elastic_tti_sg2d": (lambda mode: 6, None, ALL),
    "elastic_tti_sg3d": (lambda mode: 18, None, ALL),
    "acoustic_vti_1st_2d": (lambda mode: 5, 2, ALL),
    "acoustic_vti_1st_3d": (lambda mode: 6, 3, ALL),
    "das2d": (lambda mode: 15 if mode == "bs_gpu" else 9, 0, ALL),
    "das3d": (lambda mode: 19, 0, ("full", "ckpt_chunk")),
}


@pytest.mark.parametrize("key, mode", [
    pytest.param(k, m, marks=requires_binding(f"{k}_forward"))
    for k, (_n, _z, modes) in CASES.items() for m in modes
])
def test_backward_writes_into_the_python_pool(key, mode):
    n_of, zero_slot, _modes = CASES[key]
    expected_n = n_of(mode)
    solver, models = _backward_once(key, mode, nt=60)

    pool = solver.adjoint_workspace
    assert len(pool) == expected_n, (
        f"{key}/{mode}: expected the {expected_n} declared workspace tensors, got {len(pool)}")
    touched = [bool((t != 0).any()) for t in pool]
    assert any(touched), (
        f"{key}/{mode}: every pool tensor is still all-zero after a backward: the "
        "C++ side did not take the pool -- it fell back to its own zeros_like, and "
        "this change is inert")
    if zero_slot is not None:
        assert not touched[zero_slot], (
            f"{key}/{mode}: slot {zero_slot} is the read-only zero buffer and was written")
    # And the gradient is a real one, so the backward that wrote the pool did work.
    g = [m.grad for m in models if m.grad is not None]
    assert g and all(torch.isfinite(x).all() for x in g)
    assert max(float(x.abs().max()) for x in g) > 0
