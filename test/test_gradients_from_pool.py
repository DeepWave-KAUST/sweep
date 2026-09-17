"""The compiled backward accumulates into gradient buffers the propagator
allocated, and provably so.

``BackwardInput.grads_out`` has always been bound on the stepped/DD path; the
monolithic path now binds it too, so no driver allocates its own gradient
tensors. The instrument is the same as for the scratch pools: capture the
buffers Python hands over and check the driver wrote into them -- a driver
that silently kept allocating its own would leave them all zero.

``grads_out`` is ``[grad_wavelet?] + one per (prepared) model``; the wavelet
slot exists only where the equation declares ``grads_out_has_wavelet``. The
VRZ3D case pins a subtlety of that layout: the acoustic family reserves slot
0 for the wavelet gradient, which VRZ3D never produces, so its slot 0 must
stay zero.
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

# suite key -> (compiled binding prefix, has wavelet slot, slots that must stay zero)
CASES = {
    "acoustic2d": ("acoustic2d", True, ()),
    "elastic2d": ("elastic2d", False, ()),
    "elastic_tti_sg2d": ("elastic_tti_sg2d", False, ()),
    "acoustic_vti_1st_2d": ("acoustic_vti_1st_2d", False, ()),
    "lsrtm2d": ("acoustic_lsrtm2d", True, ()),
    "das2d": ("das2d", False, ()),
    "elastic_tti_2nd2d": ("elastic_tti_2nd2d", False, ()),
    "vrz3d": ("acoustic_vrz3d", True, (0,)),
}
MODES = ("full", "bs_gpu")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("key", [
    pytest.param(k, marks=requires_binding(f"{v[0]}_forward")) for k, v in CASES.items()
])
def test_backward_accumulates_into_the_handed_over_gradients(key, mode, monkeypatch):
    _prefix, has_wavelet, zero_slots = CASES[key]
    dev = torch.device("cuda:0")
    ns = suite.build_parser().parse_args([])
    ns.nt = 200
    spec = suite.SOLVERS[key]
    shape = suite.shape_for(spec, ns)
    scenario = suite.SCENARIOS["interior"]
    _true, models_init, grad_flags = suite.make_models(spec, shape)
    if key == "lsrtm2d":
        # The suite's Born case has a zero reflectivity, so its record -- and
        # every gradient -- is identically zero and a "was it written" check
        # could not tell binding from allocation. A reflector band fixes that.
        models_init = list(models_init)
        mp = models_init[1].copy()
        mp[mp.shape[0] // 2:mp.shape[0] // 2 + 2, :] = 0.05
        models_init[1] = mp
    sources, receivers = suite.make_geometry(spec, shape, scenario, ns)
    wavelet = torch.tensor(suite.ricker(ns.nt, ns.dt, ns.freq, ns.delay), device=dev)
    solver = suite.build_solver(spec, "c", mode, scenario, shape, dev, ns,
                                Path("/tmp"), "gradpool")

    handed = []
    orig = c_prop._gradient_buffers

    def keep(has_wavelet_, forward_source, models):
        grads = orig(has_wavelet_, forward_source, models)
        handed.append((has_wavelet_, len(models), grads))
        return grads
    monkeypatch.setattr(c_prop, "_gradient_buffers", keep)

    models = suite.tensors_from_models(models_init, grad_flags, dev)
    syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    (syn.double() ** 2).sum().backward()
    torch.cuda.synchronize()

    assert len(handed) == 1, f"{key}/{mode}: expected one backward, saw {len(handed)}"
    flag, n_models, grads = handed[0]
    assert flag == has_wavelet, f"{key}: layout declares grads_out_has_wavelet={flag}"
    assert len(grads) == n_models + int(has_wavelet)
    touched = [bool((g != 0).any()) for g in grads]
    model_slots = touched[int(has_wavelet):]
    assert any(model_slots), (
        f"{key}/{mode}: no handed-over gradient buffer was written -- the driver "
        "allocated its own, and the binding is inert")
    for i in zero_slots:
        assert not touched[i], f"{key}/{mode}: slot {i} is never produced by this equation and was written"
    g = [m.grad for m in models if m.grad is not None]
    assert g and all(torch.isfinite(x).all() for x in g)
    assert max(float(x.abs().max()) for x in g) > 0
