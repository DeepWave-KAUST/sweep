"""The full-mode forward history (``u_allt``) is allocated by the propagator.

Same instrument as the scratch and gradient pools: capture the tensor Python
hands the compiled forward as ``u_allt_out`` and check the forward wrote into
it. A driver that kept allocating its own history would leave it all zero.
The declared shape must also be the driver's shape -- the C++ side checks it
loudly, so a wrong declaration fails here rather than overruns.

The no-gradient forward must not allocate a history at all: the buffer only
exists so that the backward can replay from it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from conftest import requires_binding

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver_gradient_mode_suite as suite  # noqa: E402

import sweep.propagator._c as c_prop  # noqa: E402

# suite key -> compiled binding prefix
CASES = {
    "acoustic2d": "acoustic2d",
    "vrz2d": "acoustic_vrz2d",
    "lsrtm2d": "acoustic_lsrtm2d",
    "elastic2d": "elastic2d",
    "das2d": "das2d",
    "das_mu2d": "das_mu2d",
    "elastic_tti_sg2d": "elastic_tti_sg2d",
    "elastic_tti_2nd2d": "elastic_tti_2nd2d",
    "acoustic_vti_1st_2d": "acoustic_vti_1st_2d",
    "visco2d": "visco_acoustic2d",
    "vrz3d": "acoustic_vrz3d",
}


def _run(key, monkeypatch, *, with_grad):
    dev = torch.device("cuda:0")
    ns = suite.build_parser().parse_args([])
    ns.nt = 40
    spec = suite.SOLVERS[key]
    shape = suite.shape_for(spec, ns)
    scenario = suite.SCENARIOS["interior"]
    _true, models_init, grad_flags = suite.make_models(spec, shape)
    if key == "lsrtm2d":
        models_init = list(models_init)   # a non-zero reflectivity, so the Born record is non-zero
        mp = models_init[1].copy(); mp[mp.shape[0] // 2:mp.shape[0] // 2 + 2, :] = 0.05
        models_init[1] = mp
    sources, receivers = suite.make_geometry(spec, shape, scenario, ns)
    wavelet = torch.tensor(suite.ricker(ns.nt, ns.dt, ns.freq, ns.delay), device=dev)
    solver = suite.build_solver(spec, "c", "full", scenario, shape, dev, ns,
                                Path("/tmp"), "histpool")

    handed = []
    orig = c_prop._history_buffer

    def keep(shape_, device):
        t = orig(shape_, device)
        handed.append(t)
        return t
    monkeypatch.setattr(c_prop, "_history_buffer", keep)

    flags = grad_flags if with_grad else [False] * len(grad_flags)
    models = suite.tensors_from_models(models_init, flags, dev)
    syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    torch.cuda.synchronize()
    return solver, handed, syn


@pytest.mark.parametrize("key", [
    pytest.param(k, marks=requires_binding(f"{v}_forward")) for k, v in CASES.items()
])
def test_the_forward_writes_into_the_handed_over_history(key, monkeypatch):
    solver, handed, syn = _run(key, monkeypatch, with_grad=True)
    assert len(handed) == 1, f"{key}: expected one history buffer for one full-mode forward, got {len(handed)}"
    u_allt = handed[0]
    expected = solver._history_shape(int(syn.shape[0]))
    assert tuple(u_allt.shape) == tuple(expected), (key, tuple(u_allt.shape), expected)
    assert bool((u_allt != 0).any()), (
        f"{key}: the handed-over history is still all-zero after a forward -- the "
        "driver allocated its own, and this binding is inert")
    assert torch.isfinite(syn).all() and float(syn.abs().max()) > 0


@pytest.mark.parametrize("key", [
    pytest.param(k, marks=requires_binding(f"{v}_forward")) for k, v in list(CASES.items())[:3]
])
def test_a_forward_without_gradients_allocates_no_history(key, monkeypatch):
    _solver, handed, _syn = _run(key, monkeypatch, with_grad=False)
    assert handed == [], f"{key}: {len(handed)} history buffers allocated for a forward nobody will replay"
