"""Boundary saving of the velocity-stress SEM classes against the full tape.

Their ring holds the velocities only: the band's stresses and C-PML memories
are element-local, and the reverse march rebuilds them wrong (the memories
zeroed). The per-step VJP therefore pins the velocity update's output to the
ring (``boundary_saving.pin_to_ring``). Before that, the model gradient next to
the band was off -- 4.1e-4 of the 3-D vp gradient with no free surface, 9e-5
with one, 1e-3 of the 2-D rho gradient -- while the records were right; every
case below was red then. Checked here to fp64 round-off: both band splits, both
ring modes, topography, and the vmapped multi-shot rollout.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, EagerOptions, Full
from sweep.sem import (PropSEM, ElasticVSSEM2DDeformed, ElasticVSSEM3D,
                       ElasticVSSEM3DDeformed)

NT, DT, F0 = 120, 5e-4, 20.0


def _run(eq, memory, fs, split=None, topo=False, mode="full", nshots=1, compile=False):
    three_d = eq is not ElasticVSSEM2DDeformed
    if three_d:
        shape, src, rec = (9, 13, 17), [8, 6, 4.0], [[x, 6, 2] for x in (2, 6, 10, 14)]
        rt = ["vx", "vy", "vz"]
    else:
        shape, src, rec = (9, 17), [8, 4.0], [[x, 2] for x in (2, 6, 10, 14)]
        rt = ["vx", "vz"]
    kw = dict(shape=shape, dh=10.0, dt=DT, n_order=4, abcn=8, f_dominant=F0, dev="cpu",
              dtype=torch.float64, impl="eager", source_type=["vz"], receiver_type=rt,
              free_surface=fs, memory=memory,
              eager_options=EagerOptions(use_compile=compile))
    if isinstance(memory, BoundarySaving):
        kw["bs_band_mode"] = mode
    if split is not None:
        kw["equation_options"] = {"band_split": split}
    if topo:
        x = np.arange(shape[-1])
        if three_d:
            yy, xx = np.meshgrid(np.arange(shape[1]), x, indexing="ij")
            kw["topography"] = 2.0 - 2.0 * np.exp(-(((xx - 8) / 4.0) ** 2 + ((yy - 6) / 4.0) ** 2))
        else:
            kw["topography"] = 2.0 - 2.0 * np.exp(-((x - 8) / 4.0) ** 2)
    p = PropSEM(eq, **kw)
    z = torch.linspace(0, 1, shape[0], dtype=torch.float64).reshape(-1, *([1] * (len(shape) - 1)))
    ms = [(a + b * z).expand(shape).clone().requires_grad_(True)
          for a, b in ((2400.0, 200.0), (1300.0, 100.0), (2000.0, 0.0))]
    t = torch.arange(NT, dtype=torch.float64) * DT - 0.05
    w = (1 - 2 * (math.pi * F0 * t) ** 2) * torch.exp(-(math.pi * F0 * t) ** 2)
    srcs = np.array([[src[0] + k - nshots // 2] + src[1:] for k in range(nshots)])
    out = p(w, srcs, np.array([rec] * nshots), models=ms)
    out.square().sum().backward()
    return out.detach(), [m.grad for m in ms]


_FULL = {}


def _check(eq, fs, split=None, topo=False, **bs):
    """BoundarySaving(**bs) against the full tape of the same configuration."""
    key = (eq.__name__, fs, split, topo, bs.get("nshots", 1))
    if key not in _FULL:
        _FULL[key] = _run(eq, Full(), fs, split, topo, nshots=bs.get("nshots", 1))
    r0, g0 = _FULL[key]
    r1, g1 = _run(eq, BoundarySaving(), fs, split, topo, **bs)
    assert float((r1 - r0).norm() / r0.norm()) < 1e-12
    for name, a, b in zip(("vp", "vs", "rho"), g1, g0):
        rel = float((a - b).norm() / b.norm())
        assert rel < 1e-12, f"{name}: boundary saving vs full tape {rel:.2e}"


@pytest.mark.parametrize("split", [True, False], ids=["split", "unsplit"])
@pytest.mark.parametrize("fs", [False, True], ids=["no_free_surface", "free_top"])
def test_vs3d(fs, split):
    _check(ElasticVSSEM3D, fs, split)


@pytest.mark.parametrize("split", [True, False], ids=["split", "unsplit"])
def test_vs3d_deformed_topography(split):
    _check(ElasticVSSEM3DDeformed, True, split, topo=True)


@pytest.mark.parametrize("split", [True, False], ids=["split", "unsplit"])
@pytest.mark.parametrize("fs", [False, True], ids=["no_free_surface", "free_top"])
def test_vs2d(fs, split):
    _check(ElasticVSSEM2DDeformed, fs, split)


def test_vs2d_topography():
    _check(ElasticVSSEM2DDeformed, True, topo=True)


@pytest.mark.parametrize("eq", [ElasticVSSEM2DDeformed, ElasticVSSEM3D], ids=["2d", "3d"])
def test_rim_ring(eq):
    """bs_band_mode='rim': the ring is the band's first element layer, the
    deeper band nodes are zeroed and never restored."""
    _check(eq, False, mode="rim")


@pytest.mark.parametrize("eq", [ElasticVSSEM2DDeformed, ElasticVSSEM3D], ids=["2d", "3d"])
def test_multishot_vmapped_rollout(eq):
    """Three shots on one model take the vmapped rollout
    (run_boundary_saved_batched), whose ring masks are shared by the shots."""
    _check(eq, True, nshots=3)


def test_compiled_step():
    """The pinned recompute through torch.compile (the default eager path)."""
    _check(ElasticVSSEM2DDeformed, False, compile=True)
