"""The derived model coefficients come from the propagator, provably and bit-exactly.

Every elastic-family driver derives per-call coefficients from the bound
models (Lame parameters from vp/vs/rho, VTI stiffness from vp/eps/delta/rho,
1/z for VRZ). They used to be torch expressions inside the compiled call -- two
kept tensors plus their temporaries allocated by the backend every forward and
every backward. Now the propagator hands the driver ``derived_model_nvar``
model-shaped ``torch.empty`` slots (``derived_models``) and one fused kernel
per family fills them.

Three things are checked per equation, in one instrument around the compiled
entry points (``_backend_impl.forward_func`` / ``backward_func``):

* the slots are handed over and WRITTEN: they are filled with NaN right before
  the call and must come back finite -- a driver that still computed its own
  tensors would leave the sentinel in place;
* the kernel is bit-identical to the torch expression it replaced
  (``torch.equal`` against the same formula evaluated with torch on the models
  the call received);
* the call allocates nothing through the CUDA caching allocator any more --
  the campaign's actual goal, measured, not inferred from the bit-exact gate.

The slots must also not outlive the call (weak references after the forward).
"""
from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path

import pytest
import torch

from conftest import requires_binding

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver_gradient_mode_suite as suite  # noqa: E402


def _lame(models):
    vp, vs, rho = models[0], models[1], models[2]
    return [rho * vs * vs, rho * (vp * vp - 2 * vs * vs)]


def _vti(models):
    vp, eps, delta, rho = models[0], models[1], models[2], models[3]
    vp_sq = vp * vp
    rho_vp2 = rho * vp_sq
    return [rho_vp2 * (1.0 + 2.0 * eps), rho_vp2 * torch.sqrt(1.0 + 2.0 * delta),
            rho_vp2, 1.0 / rho]


def _vrz(models):
    return [torch.reciprocal(models[1])]


# key -> (slots the forward gets, slots the full-mode backward gets, reference
# formula in slot order, binding prefix). The DAS full backward reads the stored
# strain history and derives nothing; its forward, bs and checkpoint paths do.
CASES = {
    "elastic2d": (2, 2, _lame, "elastic2d"),
    "elastic3d": (2, 2, _lame, "elastic3d"),
    "das2d": (2, 0, _lame, "das2d"),
    "das3d": (2, 0, _lame, "das3d"),
    "das_mu2d": (2, 2, _lame, "das_mu2d"),
    "das_mu3d": (2, 2, _lame, "das_mu3d"),
    "acoustic_vti_1st_2d": (4, 4, _vti, "acoustic_vti_1st_2d"),
    "acoustic_vti_1st_3d": (4, 4, _vti, "acoustic_vti_1st_3d"),
    "vrz2d": (1, 1, _vrz, "acoustic_vrz2d"),
    "vrz3d": (1, 1, _vrz, "acoustic_vrz3d"),
}
PARAMS = [pytest.param(k, marks=requires_binding(f"{CASES[k][3]}_forward")) for k in CASES]


def _build(key, nt=40):
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
                                Path("/tmp"), "derivedpool")
    models = suite.tensors_from_models(models_init, grad_flags, dev)
    return solver, wavelet, sources, receivers, models


class _Probe:
    """Wraps one compiled entry point: NaN-fills the handed-over slots before the
    call, records the allocator's call count across it, and keeps the models
    and (weakly) the slots for the checks afterwards."""

    def __init__(self, impl, name):
        self.impl, self.name, self.fn = impl, name, getattr(impl, name)
        self.calls = []
        setattr(impl, name, self)

    def __call__(self, params):
        slots = list(params.derived_models)
        for t in slots:
            t.fill_(float("nan"))
        torch.cuda.synchronize()
        allocs = torch.cuda.memory_stats()["allocation.all.allocated"]
        out = self.fn(params)
        torch.cuda.synchronize()
        allocs = torch.cuda.memory_stats()["allocation.all.allocated"] - allocs
        self.calls.append(dict(models=[m.detach().clone() for m in params.models],
                               values=[t.detach().clone() for t in slots],
                               weak=[weakref.ref(t) for t in slots], allocs=allocs))
        return out

    def restore(self):
        setattr(self.impl, self.name, self.fn)


@pytest.mark.parametrize("key", PARAMS)
def test_forward_and_backward_derive_into_the_handed_over_slots(key):
    n_forward, n_backward, reference, _ = CASES[key]
    solver, wavelet, sources, receivers, models = _build(key)
    impl = getattr(solver, "_backend_impl", solver)
    probes = [_Probe(impl, "forward_func"), _Probe(impl, "backward_func")]
    try:
        syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
        (syn.double() ** 2).sum().backward()
    finally:
        for p in probes:
            p.restore()
    for probe, n in zip(probes, (n_forward, n_backward)):
        assert len(probe.calls) == 1, f"{key}: {probe.name} ran {len(probe.calls)} times"
        call = probe.calls[0]
        assert len(call["values"]) == n, (
            f"{key}: {probe.name} was handed {len(call['values'])} derived slots, expected {n}")
        for i, (got, ref) in enumerate(zip(call["values"], reference(call["models"]))):
            assert torch.isfinite(got).all(), (
                f"{key}: {probe.name} left the NaN sentinel in derived slot {i}: the driver "
                "did not fill the handed-over slot")
            assert torch.equal(got, ref), (
                f"{key}: {probe.name} derived slot {i} differs from the torch formula "
                f"(max |diff| {float((got - ref).abs().max()):.3e}): the fused kernel is not bit-exact")
    assert any(m.grad is not None and float(m.grad.abs().max()) > 0 for m in models)


@pytest.mark.parametrize("key", PARAMS)
def test_the_slots_do_not_outlive_the_call(key):
    solver, wavelet, sources, receivers, models = _build(key)
    impl = getattr(solver, "_backend_impl", solver)
    probe = _Probe(impl, "forward_func")
    try:
        syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    finally:
        probe.restore()
    torch.cuda.synchronize()
    weak = probe.calls[0]["weak"]
    probe.calls.clear()
    gc.collect()
    alive = [w() is not None for w in weak]
    assert not any(alive), (
        f"{key}: {sum(alive)} of {len(weak)} derived slots are still referenced after the "
        "forward returned -- the autograd ctx is keeping them alive into the backward")
    (syn.double() ** 2).sum().backward()
    assert any(m.grad is not None and float(m.grad.abs().max()) > 0 for m in models)


@pytest.mark.parametrize("key", PARAMS)
def test_the_full_mode_forward_allocates_nothing(key):
    """With the coefficients bound, nothing inside the compiled full-mode
    forward touches the caching allocator: every buffer it needs -- wavefields,
    scratch, history, record, derived coefficients -- comes from the propagator."""
    solver, wavelet, sources, receivers, models = _build(key)
    impl = getattr(solver, "_backend_impl", solver)
    # one warm-up call: lazily created CUDA state (handles, the first CPU
    # transfer of the field-index tables) is not what is being measured
    syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    (syn.double() ** 2).sum().backward()
    for m in models:
        m.grad = None
    probe = _Probe(impl, "forward_func")
    try:
        syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    finally:
        probe.restore()
    assert probe.calls[0]["allocs"] == 0, (
        f"{key}: the compiled forward made {probe.calls[0]['allocs']} allocations of its own")
    (syn.double() ** 2).sum().backward()
