"""The boundary-saving backward's reconstruction state comes from the propagator.

In boundary-saving mode the backward re-runs the forward backwards from
``u_last_two`` with the saved boundaries injected. The wavefield struct it steps
(the physical fields, plus the carriers the imaging reads) used to be allocated
inside the compiled call. Now ``cuda_layout.reconstruction_nvar`` (an explicit
``bs_reconstruction_nvar``, or the slot table's ``recon`` list -- the DD path's
source of the same count) says how many zeroed grids the propagator hands the
backward as ``forward_wavefields``, and every driver binds them.

Instrument (the same as the other pool tests): wrap ``backward_bs_func``, fill
each handed-over grid with a sentinel right before the call, and require that
every grid was written afterwards -- a driver that still allocated its own
struct would leave every sentinel in place. The absorbing rim is never written
by the reverse stencil, so the criterion is "some cell changed", never "all".
The same wrapper counts the caching allocator's calls across the backward: the
remaining per-call allocations are pinned per equation (RESIDUAL) so a driver
that quietly starts allocating again fails here, not in a memory profile weeks
later.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from conftest import requires_binding

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver_gradient_mode_suite as suite  # noqa: E402

SENTINEL = 12345.0

# key -> (reconstruction grids handed over, allocations the bs backward still
# makes inside C++ -- none since slice 16 (the negated source became a signed
# injection kernel), binding prefix)
# The DAS-mu drivers hand over the elastic fields only: their strain grids are
# dead in the bs reverse loop, and a dead grid is exactly what this test's
# sentinel would expose.
CASES = {
    "acoustic2d": (3, 0, "acoustic2d"),
    "acoustic3d": (3, 0, "acoustic3d"),
    "lsrtm2d": (3, 0, "acoustic_lsrtm2d"),
    "lsrtm3d": (3, 0, "acoustic_lsrtm3d"),
    "vrz2d": (3, 0, "acoustic_vrz2d"),
    "vrz3d": (3, 0, "acoustic_vrz3d"),
    "elastic2d": (7, 0, "elastic2d"),
    "elastic3d": (12, 0, "elastic3d"),
    "das_mu2d": (7, 0, "das_mu2d"),
    "das_mu3d": (12, 0, "das_mu3d"),
    "elastic_tti_sg2d": (11, 0, "elastic_tti_sg2d"),
    "elastic_tti_sg3d": (12, 0, "elastic_tti_sg3d"),
    "das2d": (9, 0, "das2d"),
    "acoustic_vti_1st_2d": (4, 0, "acoustic_vti_1st_2d"),
    "acoustic_vti_1st_3d": (5, 0, "acoustic_vti_1st_3d"),
    "elastic_tti_2nd2d": (6, 0, "elastic_tti_2nd2d"),
}
PARAMS = [pytest.param(k, marks=requires_binding(f"{CASES[k][2]}_forward")) for k in CASES]
# cpu-staged boundaries keep u_last_two on the host; the reconstruction grids
# must still be allocated on the GPU (found the hard way: an illegal address)
STAGED = [pytest.param(k, marks=requires_binding(f"{CASES[k][2]}_forward"))
          for k in ("acoustic2d", "elastic2d", "das2d")]


def _build(key, nt=40, mode="bs_gpu"):
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
                                Path("/tmp"), "reconpool")
    models = suite.tensors_from_models(models_init, grad_flags, dev)
    return solver, wavelet, sources, receivers, models


class _Probe:
    def __init__(self, impl, name):
        self.impl, self.name, self.fn = impl, name, getattr(impl, name)
        self.calls = []
        setattr(impl, name, self)

    def __call__(self, params):
        grids = list(params.forward_wavefields)
        for t in grids:
            t.fill_(SENTINEL)
        torch.cuda.synchronize()
        allocs = torch.cuda.memory_stats()["allocation.all.allocated"]
        out = self.fn(params)
        torch.cuda.synchronize()
        allocs = torch.cuda.memory_stats()["allocation.all.allocated"] - allocs
        self.calls.append(dict(written=[bool((t != SENTINEL).any()) for t in grids],
                               shapes=[tuple(t.shape) for t in grids], allocs=allocs))
        return out

    def restore(self):
        setattr(self.impl, self.name, self.fn)


@pytest.mark.parametrize("key", PARAMS)
def test_bs_backward_reconstructs_into_the_handed_over_grids(key):
    n, residual, _ = CASES[key]
    solver, wavelet, sources, receivers, models = _build(key)
    impl = getattr(solver, "_backend_impl", solver)
    # warm-up: lazily created CUDA state is not what the allocation count measures
    syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    (syn.double() ** 2).sum().backward()
    for m in models:
        m.grad = None
    probe = _Probe(impl, "backward_bs_func")
    try:
        syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
        (syn.double() ** 2).sum().backward()
    finally:
        probe.restore()
    assert len(probe.calls) == 1, f"{key}: backward_bs ran {len(probe.calls)} times"
    call = probe.calls[0]
    assert len(call["written"]) == n, (
        f"{key}: {len(call['written'])} reconstruction grids handed over, expected {n}")
    assert all(call["written"]), (
        f"{key}: reconstruction grids {[i for i, w in enumerate(call['written']) if not w]} "
        "still hold the sentinel after the backward: the driver did not bind them")
    assert call["allocs"] <= residual, (
        f"{key}: the bs backward made {call['allocs']} allocations of its own, "
        f"expected at most {residual}")
    assert any(m.grad is not None and float(m.grad.abs().max()) > 0 for m in models)


@pytest.mark.parametrize("key", STAGED)
def test_bs_backward_with_cpu_staged_boundaries_reconstructs_on_the_gpu(key):
    n = CASES[key][0]
    solver, wavelet, sources, receivers, models = _build(key, mode="bs_cpu")
    impl = getattr(solver, "_backend_impl", solver)
    probe = _Probe(impl, "backward_bs_func")
    try:
        syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
        (syn.double() ** 2).sum().backward()
    finally:
        probe.restore()
    call = probe.calls[0]
    assert len(call["written"]) == n and all(call["written"]), (
        f"{key}: cpu-staged bs backward did not reconstruct into the handed-over grids")
    assert any(m.grad is not None and torch.isfinite(m.grad).all() and float(m.grad.abs().max()) > 0
               for m in models)


def test_reconstruction_count_comes_from_the_slot_table_where_one_exists():
    """One authority per count: a class with a slot table derives the
    reconstruction count from ``slots.recon`` (what the DD driver binds), a
    class without one declares ``bs_reconstruction_nvar`` explicitly."""
    from sweep.equations.cuda_layout import CUDALayoutSpec
    from sweep.equations import slot_table

    spec = CUDALayoutSpec(base_nvar=3, pml_nvar=6, last_two_nvar=2, slots=slot_table.ACOUSTIC2D)
    assert spec.reconstruction_nvar == slot_table.ACOUSTIC2D.nrecon == 3
    assert CUDALayoutSpec(base_nvar=3, pml_nvar=6, last_two_nvar=2).reconstruction_nvar == 0
    assert CUDALayoutSpec(base_nvar=3, pml_nvar=6, last_two_nvar=2,
                          bs_reconstruction_nvar=5).reconstruction_nvar == 5
