"""impl='c' keeps no backward state on a call whose backward never runs.

Each C driver Function decided inside its forward whether a backward would
run, from its inputs' ``requires_grad``. ``Function.forward`` runs with grad
mode off whatever the caller's mode, and ``requires_grad`` does not see an
outer ``torch.no_grad()``: a wavelet that requires grad (a source-inversion
leaf) passed in under ``no_grad`` made the forward keep its save-all tape, or
its C-PML band ring under ``BoundarySaving()``, for a backward that never came.
The call site decides now, so such a call peaks where the same call with a
plain wavelet does.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, Full
from sweep.sem import (AcousticPotentialSEM2D, AcousticSEM3D, ElasticSEM2DDeformed,
                       ElasticVSSEM2DDeformed, PropSEM)
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NT, DT, F0, DH = 300, 1e-3, 10.0, 10.0
VALUES = {"vp": 2400.0, "vs": 1300.0, "rho": 2000.0}

# id: (equation, shape, abcn, memory, nshots) -- the driver each one reaches
CASES = {
    "acoustic2d_cpml_tape": (AcousticPotentialSEM2D, (41, 61), 12, Full(), 1),
    "acoustic2d_cpml_ring": (AcousticPotentialSEM2D, (41, 61), 12, BoundarySaving(), 1),
    "acoustic2d_cpml_batch": (AcousticPotentialSEM2D, (41, 61), 12, BoundarySaving(), 3),
    "acoustic2d_bare": (AcousticPotentialSEM2D, (41, 61), 0, Full(), 1),
    "elastic2d_cpml": (ElasticSEM2DDeformed, (41, 61), 12, Full(), 1),
    "vs2d_cpml": (ElasticVSSEM2DDeformed, (41, 61), 12, Full(), 1),
    "acoustic3d_strip": (AcousticSEM3D, (13, 17, 21), 8, Full(), 1),
    "acoustic3d_strip_ring": (AcousticSEM3D, (13, 17, 21), 8, BoundarySaving(), 1),
}


def _ricker():
    t = np.arange(NT) * DT - 1.2 / F0
    a = (np.pi * F0 * t) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device="cuda")


def _peak(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base, out


@pytest.mark.parametrize("case", list(CASES))
def test_a_grad_leaf_under_no_grad_keeps_no_tape(case):
    equation, shape, abcn, memory, nshots = CASES[case]
    p = PropSEM(equation, shape=shape, dh=DH, dt=DT, nt=NT, n_order=4,
                free_surface=False, abcn=abcn, f_dominant=F0, dev="cuda",
                dtype=torch.float32, impl="c", memory=memory)
    models = [torch.full(shape, VALUES[n], device="cuda") for n in p.model_names]
    nx = shape[-1]
    if len(shape) == 2:
        src = np.array([[nx // 2 + 6 * (k - nshots // 2), 4] for k in range(nshots)])
        rec = np.array([[[x, 3] for x in range(4, nx - 4, 4)]] * nshots)
    else:
        src = np.array([[nx // 2, shape[1] // 2, 4]])
        rec = np.array([[[x, shape[1] // 2, 3] for x in range(4, nx - 4, 4)]])
    wav = _ricker()
    leaf = wav.clone().requires_grad_(True)
    with torch.no_grad():
        p(wav, src, rec, models=models)                          # warm the consts
        plain_peak, plain = _peak(lambda: p(wav, src, rec, models=models))
        leaf_peak, out = _peak(lambda: p(leaf, src, rec, models=models))
    assert not out.requires_grad
    # the same run (the kernels' atomicAdd order aside)
    assert float((out - plain).norm() / plain.norm()) < 1e-5
    # Two runs of the same call can peak a fraction of a MiB apart (the caching
    # allocator's rounding; once 0.26 MiB in a full run); the tapes this guards
    # against kept 2.9-160 MiB more.
    assert leaf_peak <= plain_peak + 2**20, (
        f"{case}: {(leaf_peak - plain_peak) / 2**20:.2f} MiB more with a grad leaf "
        "under no_grad")
