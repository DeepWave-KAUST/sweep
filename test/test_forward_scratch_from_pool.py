"""The compiled forward's scratch comes from the propagator, and provably so.

Same instrument as test_backward_scratch_from_pool.py, on the forward side:
the propagator hands the driver ``forward_workspace_nvar`` transient grids per
call, and the driver writes its per-step scratch into them. If the driver had
kept allocating its own zeros_like, the tensors Python handed over would still
be all zero afterwards. They are transient (one-call lifetime, dropped after
the forward), so the test keeps its own reference by wrapping the allocator.

A forward WITHOUT gradients is used on purpose: this scratch is paid by pure
forward users, and the lazy adjoint pool must stay empty on that path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from conftest import requires_binding

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver_gradient_mode_suite as suite  # noqa: E402

from sweep.propagator._c import _CompiledPropagator  # noqa: E402

CASES = {
    "das2d": 4,
    "das3d": 9,
    "elastic_tti_2nd2d": 3,
}


@pytest.mark.parametrize("key", [
    pytest.param(k, marks=requires_binding(f"{k}_forward")) for k in CASES
])
def test_forward_writes_into_the_handed_over_scratch(key, monkeypatch):
    dev = torch.device("cuda:0")
    ns = suite.build_parser().parse_args([])
    ns.nt = 40
    spec = suite.SOLVERS[key]
    shape = suite.shape_for(spec, ns)
    scenario = suite.SCENARIOS["interior"]
    _true, models_init, grad_flags = suite.make_models(spec, shape)
    sources, receivers = suite.make_geometry(spec, shape, scenario, ns)
    wavelet = torch.tensor(suite.ricker(ns.nt, ns.dt, ns.freq, ns.delay), device=dev)
    solver = suite.build_solver(spec, "c", "full", scenario, shape, dev, ns,
                                Path("/tmp"), "fwdpool")

    handed = []
    orig = _CompiledPropagator._transient_forward_workspace

    def keep(self, batch_size):
        ws = orig(self, batch_size)
        handed.append(ws)
        return ws
    monkeypatch.setattr(_CompiledPropagator, "_transient_forward_workspace", keep)

    models = suite.tensors_from_models(models_init, [False] * len(grad_flags), dev)
    with torch.no_grad():
        syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    torch.cuda.synchronize()

    assert len(handed) == 1 and len(handed[0]) == CASES[key], (
        f"{key}: expected {CASES[key]} scratch grids handed over, got "
        f"{[len(w) for w in handed]}")
    touched = [bool((t != 0).any()) for t in handed[0]]
    assert any(touched), (
        f"{key}: every handed-over scratch grid is still all-zero after a forward: the "
        "driver did not take the pool -- it allocated its own, and this change is inert")
    assert solver.adjoint_workspace == (), "a forward-only call must not allocate the adjoint pool"
    assert torch.isfinite(syn).all() and float(syn.abs().max()) > 0


@pytest.mark.parametrize("key", [
    pytest.param(k, marks=requires_binding(f"{k}_forward")) for k in CASES
])
def test_the_scratch_does_not_outlive_the_forward(key, monkeypatch):
    """With gradients on, ``Wrapper.forward`` saves its params object on ``ctx``
    for the backward. The forward scratch must not ride along: it is dead once
    the forward returns, and keeping it would sit idle at the backward's peak.
    Only weak references are kept here, so the tensors are collectable as soon
    as the propagator lets go of them."""
    import gc
    import weakref

    dev = torch.device("cuda:0")
    ns = suite.build_parser().parse_args([])
    ns.nt = 40
    spec = suite.SOLVERS[key]
    shape = suite.shape_for(spec, ns)
    scenario = suite.SCENARIOS["interior"]
    _true, models_init, grad_flags = suite.make_models(spec, shape)
    sources, receivers = suite.make_geometry(spec, shape, scenario, ns)
    wavelet = torch.tensor(suite.ricker(ns.nt, ns.dt, ns.freq, ns.delay), device=dev)
    solver = suite.build_solver(spec, "c", "full", scenario, shape, dev, ns,
                                Path("/tmp"), "fwdpool")

    weak = []
    orig = _CompiledPropagator._transient_forward_workspace

    def watch(self, batch_size):
        ws = orig(self, batch_size)
        weak.extend(weakref.ref(t) for t in ws)
        return ws
    monkeypatch.setattr(_CompiledPropagator, "_transient_forward_workspace", watch)

    models = suite.tensors_from_models(models_init, grad_flags, dev)
    syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    torch.cuda.synchronize()
    assert len(weak) == CASES[key]
    gc.collect()
    alive = [w() is not None for w in weak]
    assert not any(alive), (
        f"{key}: {sum(alive)} of {len(weak)} forward scratch grids are still referenced "
        "after the forward returned -- the autograd ctx is keeping them alive into the backward")
    # the graph itself is intact: the backward still runs and gives a gradient
    (syn.double() ** 2).sum().backward()
    assert any(m.grad is not None and float(m.grad.abs().max()) > 0 for m in models)
