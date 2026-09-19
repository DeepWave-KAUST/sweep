"""A compiled call refuses to run when the propagator did not bind a buffer.

The campaign's contract is that the Python propagator owns every buffer the
compiled forward and backward use. That used to be observed (a census showed
zero allocations inside C++) rather than enforced: each binding site read "use
the bound list, else allocate my own", so a layout that silently stopped
declaring a pool would keep working -- slower and quietly self-allocating --
and nothing would say so.

This test proves the enforcement. It captures the params object the propagator
built for a real call, clears ONE bound list, and requires the compiled entry
point to refuse with a message that names the buffer. Clearing the list is
exactly what "the propagator did not bind it" looks like from C++.

The equations here cover the three shapes of the contract: the acoustic
template skeleton, the staggered template skeleton and a hand-written driver.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from conftest import capture_both, requires_binding

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver_gradient_mode_suite as suite  # noqa: E402

# key -> (binding prefix, the BackwardInput list the driver must be handed)
CASES = [
    ("acoustic2d", "acoustic2d", "adjoint_wavefields"),
    ("acoustic2d", "acoustic2d", "grads_out"),
    ("elastic2d", "elastic2d", "adjoint_wavefields"),
    ("elastic2d", "elastic2d", "adjoint_workspace"),
    ("das2d", "das2d", "grads_out"),
]


def _run_once(key, mode="bs_gpu", nt=40):
    """One real call, returning the captured backward params and the closure
    that replays the compiled backward with them."""
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
                                Path("/tmp"), "mandatory")
    models = suite.tensors_from_models(models_init, grad_flags, dev)
    cap = capture_both(solver)
    syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    (syn.double() ** 2).sum().backward()
    return cap


@pytest.mark.parametrize("key, binding, field", [
    pytest.param(k, b, f, marks=requires_binding(f"{b}_forward"), id=f"{k}-{f}")
    for k, b, f in CASES
])
def test_the_backward_refuses_an_unbound_buffer(key, binding, field):
    cap = _run_once(key)
    params, run = cap["bp"], cap["bwd_func"]
    bound = list(getattr(params, field))
    assert bound, f"{key}: the propagator bound nothing as {field}; the case is vacuous"
    setattr(params, field, [])
    with pytest.raises(RuntimeError) as excinfo:
        run(params)
    message = str(excinfo.value)
    assert field in message or field.replace("_", " ") in message, (
        f"{key}: the backward refused an unbound {field}, but its message does not name it: "
        f"{message.splitlines()[0][:200]}")
    # and it still runs once the binding is restored: the refusal is about the
    # missing list, not about the state the failed call left behind
    setattr(params, field, bound)
    run(params)
