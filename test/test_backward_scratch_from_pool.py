"""The compiled backward's scratch comes from the Python pool, and provably so.

A bit-exact gate cannot see this change: when the pool is NOT bound the C++
side allocates a fresh zero tensor of the same geometry, and the numbers are
identical either way. Green therefore proves nothing about whether the move
took effect (lesson_optimisation_flag_silently_inactive). The instrument here
is direct: after the forward (which zeroes the pool) every slot is filled
with a sentinel, and after the backward at least one slot must have lost it
-- the driver wrote there. A driver that had fallen back to its own
zeros_like would leave every sentinel in place. The final CONTENTS are not
the criterion: a per-step scratch holds whatever the last reverse step left
(visco's carrier is vp^2*Lap(u) of the initial, all-zero state, i.e. exactly
zero), so "non-zero at the end" would be a false alarm there.

The read-only zero slots (the "previous stress" at the first reverse step)
must keep their sentinel in every cell: nothing may ever write them, because
nothing re-zeroes the pool between forward and backward. The run's gradient
is meaningless under the sentinel and is only required to finish.
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
    torch.cuda.synchronize()
    for t in solver.adjoint_workspace:          # after the forward's zeroing, before the backward
        t.fill_(SENTINEL)
    (syn.double() ** 2).sum().backward()
    torch.cuda.synchronize()
    return solver, models


SENTINEL = 1e-3

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
    "vrz3d": (lambda mode: 10, None, ALL),
    # slot 8 is the read-only zero field in full mode and a stress workspace
    # in the others -- the modes never share a pool
    "elastic_tti_2nd2d": (lambda mode: 9 if mode == "full" else 11,
                          lambda mode: 8 if mode == "full" else None, ALL),
    "lsrtm2d": (lambda mode: 1, None, ALL),
    "lsrtm3d": (lambda mode: 2 if mode == "bs_gpu" else 1, None, ALL),
    "visco2d": (lambda mode: 1, None, ("full", "ckpt_chunk")),
}


# suite keys whose compiled binding carries a family prefix
BINDING = {"vrz3d": "acoustic_vrz3d", "lsrtm2d": "acoustic_lsrtm2d", "lsrtm3d": "acoustic_lsrtm3d",
           "visco2d": "visco_acoustic2d"}


@pytest.mark.parametrize("key, mode", [
    pytest.param(k, m, marks=requires_binding(f"{BINDING.get(k, k)}_forward"))
    for k, (_n, _z, modes) in CASES.items() for m in modes
])
def test_backward_writes_into_the_python_pool(key, mode):
    n_of, zero_slot, _modes = CASES[key]
    expected_n = n_of(mode)
    if callable(zero_slot):
        zero_slot = zero_slot(mode)
    solver, models = _backward_once(key, mode, nt=60)

    pool = solver.adjoint_workspace
    assert len(pool) == expected_n, (
        f"{key}/{mode}: expected the {expected_n} declared workspace tensors, got {len(pool)}")
    touched = [bool((t != SENTINEL).any()) for t in pool]
    assert any(touched), (
        f"{key}/{mode}: every pool tensor still holds the sentinel after a backward: the "
        "C++ side did not take the pool -- it fell back to its own zeros_like, and "
        "this change is inert")
    if zero_slot is not None:
        assert not touched[zero_slot], (
            f"{key}/{mode}: slot {zero_slot} is the read-only zero buffer and was written")
    assert all(m.grad is None or torch.isfinite(m.grad).all() for m in models)
