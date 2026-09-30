"""The compiled forward's record is allocated by the propagator.

The stepped/DD path has always bound ``record_out``; the monolithic path now
does too, in the driver's own layout (``cuda_layout.record_shape``). Same
sentinel instrument as the other pools: the buffer Python hands over is
filled with a sentinel, and the forward must overwrite it. Both record
layouts are covered -- the acoustic family's ``(B, nrec, nt)`` and the
staggered family's ``(nfield, B, nrec, nt)`` -- so a wrong declaration fails
loudly in the driver's shape check rather than here.
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

SENTINEL = 1e-3

# suite key -> (compiled binding prefix, record ndim)
CASES = {
    "acoustic2d": ("acoustic2d", 3),
    "acoustic3d": ("acoustic3d", 3),
    "vrz2d": ("acoustic_vrz2d", 3),
    "vrz3d": ("acoustic_vrz3d", 3),
    "lsrtm2d": ("acoustic_lsrtm2d", 3),
    "visco2d": ("visco_acoustic2d", 3),
    "elastic2d": ("elastic2d", 4),
    "das2d": ("das2d", 4),
    "das_mu2d": ("das_mu2d", 4),
    "acoustic_vti_1st_2d": ("acoustic_vti_1st_2d", 4),
    "elastic_tti_sg2d": ("elastic_tti_sg2d", 4),
    "elastic_tti_2nd2d": ("elastic_tti_2nd2d", 4),
}


@pytest.mark.parametrize("key", [
    pytest.param(k, marks=requires_binding(f"{v[0]}_forward")) for k, v in CASES.items()
])
def test_the_forward_writes_the_handed_over_record(key, monkeypatch):
    _prefix, ndim = CASES[key]
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
                                Path("/tmp"), "recpool")

    handed = []
    orig = c_prop._record_buffer

    def keep(shape_, device):
        t = orig(shape_, device)
        t.fill_(SENTINEL)
        handed.append(t)
        return t
    monkeypatch.setattr(c_prop, "_record_buffer", keep)

    models = suite.tensors_from_models(models_init, [False] * len(grad_flags), dev)
    with torch.no_grad():
        syn = suite.guarded_solver_call(solver, wavelet, sources, receivers, models=models)
    torch.cuda.synchronize()

    assert len(handed) == 1, f"{key}: expected one record buffer for one forward, got {len(handed)}"
    rec = handed[0]
    assert rec.ndim == ndim and rec.shape[-1] == ns.nt and rec.shape[-2] == len(receivers[0]), \
        (key, tuple(rec.shape))
    assert not bool((rec == SENTINEL).all()), (
        f"{key}: the handed-over record still holds the sentinel everywhere after a forward -- "
        "the driver allocated its own, and this binding is inert")
    assert torch.isfinite(syn).all()
