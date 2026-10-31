"""The compiled step (``EagerOptions(use_compile=True)``) against the uncompiled one,
equation by equation (CUDA).

Compiling for a GPU changes the arithmetic: the reference-derivative contractions
are unrolled into multiply-adds (``sweep.sem._assembly.contract``) and Inductor
fuses and reorders the rest. The records and every model gradient must still agree
with the uncompiled step to fp32 round-off.
"""
import math

import numpy as np
import pytest
import torch

from sweep.propagator import EagerOptions
from sweep.sem import (PropSEM, AcousticSEM2DDeformed, ElasticSEM2DDeformed, ElasticVSSEM2DDeformed,
                       BiotVSSEM2DDeformed, AcousticSEM3D, ElasticVSSEM3D)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled GPU step")

DEV, F32 = torch.device("cuda"), torch.float32
EL = (2500.0, 1400.0, 2000.0)


def _biot():
    from sweep.equations.biot import biot_from_biot_constants
    from test_biot import DAI1
    return tuple(float(c) for c in biot_from_biot_constants(**DAI1, b=7e4, phi=0.3))


# name: (equation, shape, dt, model values, source_type, receiver_type, topography, abcn)
CASES = {
    "acoustic2d_deformed": (AcousticSEM2DDeformed, (41, 81), 1e-3, (2500.0,), None, None, True, None),
    "elastic2d_deformed": (ElasticSEM2DDeformed, (41, 81), 1e-3, EL, ["sxx", "szz"], ["vx", "vz"], True, None),
    "elastic_vs2d_deformed": (ElasticVSSEM2DDeformed, (41, 81), 1e-3, EL, ["sxx", "szz"], ["vx", "vz"], True,
                              None),
    "biot2d": (BiotVSSEM2DDeformed, (41, 81), 5e-4, None, ["sxx", "szz"], ["vx", "vz", "p"], False, None),
    "acoustic3d": (AcousticSEM3D, (17, 17, 17), 8e-4, (2500.0,), None, None, False, 8),
    "elastic_vs3d": (ElasticVSSEM3D, (17, 17, 17), 8e-4, EL, None, None, False, 8),
}


def _run(name, compile, nt=80):
    eq, shape, dt, vals, st, rt, topo, abcn = CASES[name]
    vals = _biot() if vals is None else vals
    kw = dict(shape=shape, dh=10.0, dt=dt, n_order=4, free_surface=True, f_dominant=12.0, dev=DEV,
              dtype=F32, impl="eager", eager_options=EagerOptions(use_compile=compile))
    if st:
        kw.update(source_type=st, receiver_type=rt)
    if abcn:
        kw["abcn"] = abcn
    if topo:
        x = np.arange(shape[-1])
        kw["topography"] = 4.0 - 4.0 * np.exp(-((x - shape[-1] / 2) / (shape[-1] / 8)) ** 2)
    prop = PropSEM(eq, **kw)
    if len(shape) == 2:
        src = np.array([[shape[1] // 2, 8.0]])
        rx = np.linspace(shape[1] * 0.1, shape[1] * 0.9, 12)
        rec = np.stack([rx, np.full_like(rx, 10.0)], 1)[None]
    else:
        src = np.array([[shape[2] // 2, shape[1] // 2, 6.0]])
        rx = np.linspace(shape[2] * 0.2, shape[2] * 0.8, 6)
        rec = np.stack([rx, np.full_like(rx, shape[1] / 2), np.full_like(rx, 8.0)], 1)[None]
    t = torch.arange(nt, device=DEV, dtype=F32) * dt - 1.2 / 12.0
    a = (math.pi * 12.0 * t) ** 2
    wav = (1 - 2 * a) * torch.exp(-a)
    ms = [torch.full(shape, v, device=DEV, dtype=F32, requires_grad=True) for v in vals]
    out = prop(wav, src, rec, models=ms)
    scale = 1.0 / float(out.detach().abs().max())          # keep the fp32 gradients away from underflow
    (out * scale).square().sum().backward()
    return out.detach(), [m.grad for m in ms]


@pytest.mark.parametrize("name", list(CASES))
def test_compiled_step_matches_uncompiled(name):
    torch._dynamo.reset()
    out_e, g_e = _run(name, False)
    out_c, g_c = _run(name, True)
    assert torch.isfinite(out_c).all()
    rel = lambda a, b: float((a.double() - b.double()).norm() / b.double().norm())
    assert rel(out_c, out_e) < 1e-5, rel(out_c, out_e)
    for k, (a, b) in enumerate(zip(g_c, g_e)):
        assert torch.isfinite(a).all()
        assert rel(a, b) < 1e-4, (k, rel(a, b))
