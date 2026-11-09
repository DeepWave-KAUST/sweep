"""Recursive checkpointing of the staggered family bisects each segment.

``sg_driver.cuh``'s recursive backward used to replay from the nearest
checkpoint at every reverse step, about nt^2 / (2 count) forward steps in all;
it now bisects each checkpoint segment like the acoustic skeleton.  Both replay
the same states with the same kernels, so the gradient has to stay
bit-identical to full storage for every checkpoint count: one checkpoint (a
single deep bisection), counts that split nt into uneven segments, and host
checkpoint storage -- with each equation's default source and with a force on
a velocity field (the leaf reads v(it) before the step's injection) under a
free surface.
``test_checkpoint_state_from_pool`` pins that every bisection scratch set the
propagator hands over gets written.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import requires_binding
from sweep.equations._registry import get_equation
from sweep.propagator.options import Ckpt, Full
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

DEV = "cuda"
# registry name -> (compiled name, ndim, velocity field for a force source)
EQS = {
    "Elastic": ("elastic2d", 2, "vz"),
    "ElasticVRR": ("elastic_vr2d", 2, "pz"),
    "ViscoElastic": ("visco_elastic2d", 2, "vz"),
    "DASMu": ("das_mu2d", 2, "vz"),
    "Elastic3D": ("elastic3d", 3, "vz"),
    "DASMu3D": ("das_mu3d", 3, "vz"),
}
RECIPE = {"vp": (2000.0, 500.0), "vs": (1100.0, 300.0), "rho": (1900.0, 300.0),
          "Qp": (60.0, 30.0), "Qs": (40.0, 25.0),
          "Rp_x": (0.0, 0.01), "Rp_z": (0.0, 0.01), "Rs_x": (0.0, 0.01), "Rs_z": (0.0, 0.01)}


def _models(names, shape):
    z = torch.linspace(0, 1, shape[0], device=DEV).reshape([-1] + [1] * (len(shape) - 1))
    box = torch.zeros(shape, device=DEV)
    box[tuple(slice(n // 3, n // 3 + max(2, n // 4)) for n in shape)] = 1.0
    return [(RECIPE[n][0] + RECIPE[n][1] * (z * torch.ones(shape, device=DEV) - 0.1 * box)).contiguous()
            for n in names]


def _geometry(shape):
    if len(shape) == 2:
        nx = shape[1]
        rx = np.arange(4, nx - 4, 3)
        return np.array([[nx // 2, 6]], np.int64), np.stack([rx, np.full_like(rx, 3)], -1)[None]
    ny, nx = shape[1:]
    gx, gy = np.meshgrid(np.arange(3, nx - 3, 4), np.arange(3, ny - 3, 4), indexing="ij")
    rec = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, 3)], -1)[None]
    return np.array([[nx // 2, ny // 2, 5]], np.int64), rec


def _run(name, memory, free_surface, source_type):
    ndim = EQS[name][1]
    shape, nt = ((44, 52), 157) if ndim == 2 else ((22, 18, 20), 97)   # nt: uneven segments
    eq = get_equation(name)(4, DEV)
    ms = [m.requires_grad_(True) for m in _models(list(eq.models), shape)]
    wav = torch.tensor((1e3 * ricker(np.arange(nt) * 1e-3 - 0.06, f=15.0)).astype(np.float32),
                       device=DEV)
    src, rec = _geometry(shape)
    kw = dict(free_surface=True) if free_surface else {}
    if source_type:
        kw["source_type"] = [source_type]
    prop = PropTorch(eq, shape=shape, abcn=12, dh=10.0, dt=1e-3, impl="c", memory=memory, **kw)
    d = prop(wav, src.copy(), rec.copy(), models=ms)
    w = torch.randn(d.shape, device=DEV, generator=torch.Generator(device=DEV).manual_seed(7))
    (d * w).sum().backward()
    return [d.detach()] + [m.grad.detach() for m in ms]


@pytest.mark.parametrize("name", [pytest.param(n, marks=requires_binding(f"{c}_backward_recursive_ckpt"))
                                  for n, (c, _, _) in EQS.items()])
@pytest.mark.parametrize("scenario", ["default_source", "velocity_source_free_surface"])
def test_recursive_bisection_is_bitwise_full(name, scenario):
    ndim, vel = EQS[name][1], EQS[name][2]
    fs, src = (False, None) if scenario == "default_source" else (True, vel)
    full = _run(name, Full(), fs, src)
    counts = (1, 3, 7) if ndim == 2 else (1, 4)
    runs = [Ckpt(mode="recursive", count=c) for c in counts]
    runs.append(Ckpt(mode="recursive", count=counts[1], storage="cpu"))
    for ckpt in runs:
        out = _run(name, ckpt, fs, src)
        assert all(torch.equal(a, b) for a, b in zip(full, out)), (name, scenario, ckpt)
