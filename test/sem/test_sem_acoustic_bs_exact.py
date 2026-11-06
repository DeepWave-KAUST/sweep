"""Boundary saving of the scalar SEM classes against the full tape.

The per-step VJP re-runs the step from the reconstructed frame, and the model's
edge nodes read the C-PML memories of the band element next to them. Those
came back wrong, so the model gradient on the edge line was off while the
records were right: ``AcousticSEM2D``, ``AcousticSEM2DDeformed`` and
``AcousticSEM3DDeformed`` ring ``u`` only and zeroed them (4e-5 of the 2-D vp
gradient); ``AcousticSEM3D`` and ``AcousticPotentialSEM2D`` -- and
``ElasticSHSEM2DDeformed``, which shares its hooks -- ring them but restored
them one step stale (3e-6, 7e-5; 1e-6 of the SH rho gradient); 5e-5 to 6e-4
with the source next to the band. Every case below was red then (the smallest
2.6e-9). Checked here to fp64 round-off: both ring modes, a free
top, topography, the source next to the band on a strong contrast over a long
record, the vmapped multi-shot rollout, the compiled step and source encoding.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import sweep.sem.boundary_saving as BS
from sweep.propagator import BoundarySaving, EagerOptions, Full
from sweep.sem import (PropSEM, AcousticPotentialSEM2D, AcousticSEM2D,
                       AcousticSEM2DDeformed, AcousticSEM3D, AcousticSEM3DDeformed,
                       ElasticSHSEM2DDeformed)

EQS = [AcousticSEM2D, AcousticSEM2DDeformed, AcousticPotentialSEM2D,
       ElasticSHSEM2DDeformed, AcousticSEM3D, AcousticSEM3DDeformed]
IDS = [e.__name__ for e in EQS]
THREE_D = (AcousticSEM3D, AcousticSEM3DDeformed)


def _case(eq, stress):
    """Geometry, time axis and models. ``stress``: the source one cell off
    the band on a random strong contrast, over a long record -- what puts
    energy into the band."""
    td = eq in THREE_D
    if not stress:
        shape = (9, 13, 17) if td else (9, 17)
        nt, dt, f0, t0 = 120, 5e-4, 20.0, 0.05
        src = [8, 6, 4.0] if td else [8, 4.0]
        rec = [[x, 6, 2] for x in (2, 6, 10, 14)] if td else [[x, 2] for x in (2, 6, 10, 14)]
        z = torch.linspace(0, 1, shape[0], dtype=torch.float64).reshape(-1, *([1] * (len(shape) - 1)))
        vals = {"vp": 2400.0 + 200.0 * z, "vs": 1300.0 + 100.0 * z, "rho": 2000.0 + 0.0 * z}
        vals = {k: v.expand(shape) for k, v in vals.items()}
    else:
        shape, nt = ((9, 9, 13), 240) if td else ((25, 41), 600)
        dt, f0, t0 = 4e-4, 15.0, 0.08
        src = [1, 1, 2.0] if td else [1, 3.0]
        rec = ([[x, 4, 2] for x in range(1, 13, 3)] if td
               else [[x, 2] for x in range(1, 41, 4)])
        g = torch.Generator().manual_seed(7)
        vp = 1500.0 + 3000.0 * torch.rand(shape, generator=g, dtype=torch.float64)
        rho = 1800.0 + 600.0 * torch.rand(shape, generator=g, dtype=torch.float64)
        vals = {"vp": vp, "vs": vp / 1.8, "rho": rho}
    return shape, nt, dt, f0, t0, src, rec, vals


def _run(eq, memory, fs, *, topo=False, mode="full", nshots=1, compile=False,
         stress=False):
    shape, nt, dt, f0, t0, src, rec, vals = _case(eq, stress)
    kw = dict(shape=shape, dh=10.0, dt=dt, n_order=4, abcn=8, f_dominant=f0, dev="cpu",
              dtype=torch.float64, impl="eager", free_surface=fs, memory=memory,
              eager_options=EagerOptions(use_compile=compile))
    if isinstance(memory, BoundarySaving):
        kw["bs_band_mode"] = mode
    if topo:
        x = np.arange(shape[-1])
        if eq in THREE_D:
            yy, xx = np.meshgrid(np.arange(shape[1]), x, indexing="ij")
            kw["topography"] = 2.0 - 2.0 * np.exp(-(((xx - 8) / 4.0) ** 2 + ((yy - 6) / 4.0) ** 2))
        else:
            kw["topography"] = 2.0 - 2.0 * np.exp(-((x - 8) / 4.0) ** 2)
    p = PropSEM(eq, **kw)
    ms = [vals[n].clone().contiguous().requires_grad_(True) for n in p.model_names]
    t = torch.arange(nt, dtype=torch.float64) * dt - t0
    w = (1 - 2 * (math.pi * f0 * t) ** 2) * torch.exp(-(math.pi * f0 * t) ** 2)
    srcs = np.array([[src[0] + k - nshots // 2] + src[1:] for k in range(nshots)])
    out = p(w, srcs, np.array([rec] * nshots), models=ms)
    out.square().sum().backward()
    return out.detach(), [m.grad for m in ms], p.model_names


_FULL = {}


def _check(eq, fs, *, topo=False, stress=False, **bs):
    """BoundarySaving(**bs) against the full tape of the same configuration."""
    nshots = bs.get("nshots", 1)
    key = (eq.__name__, fs, topo, stress, nshots)
    if key not in _FULL:
        _FULL[key] = _run(eq, Full(), fs, topo=topo, stress=stress, nshots=nshots)
    r0, g0, names = _FULL[key]
    r1, g1, _ = _run(eq, BoundarySaving(), fs, topo=topo, stress=stress, **bs)
    assert float((r1 - r0).norm() / r0.norm()) < 1e-12
    for name, a, b in zip(names, g1, g0):
        rel = float((a - b).norm() / b.norm())
        assert rel < 1e-12, f"{name}: boundary saving vs full tape {rel:.2e}"


# The free top on the two classes without a topography case. Without a free top every class
# runs the same check in test_source_next_to_the_band, with the band loaded; the other four
# run the free top in test_topography.
@pytest.mark.parametrize("fs", [True], ids=["free_top"])
@pytest.mark.parametrize("eq", [AcousticSEM2D, AcousticSEM3D], ids=["AcousticSEM2D", "AcousticSEM3D"])
def test_matches_full_tape(eq, fs):
    _check(eq, fs)


@pytest.mark.parametrize("eq", EQS, ids=IDS)
def test_rim_ring(eq):
    """bs_band_mode='rim': the ring is the band's first element layer, the
    deeper band nodes are zeroed and never restored."""
    _check(eq, False, mode="rim")


@pytest.mark.parametrize("eq", EQS, ids=IDS)
def test_source_next_to_the_band(eq):
    """Energy in the band: 5e-5 (AcousticSEM3D) to 6e-4 (AcousticSEM3DDeformed)
    before."""
    _check(eq, False, stress=True)


@pytest.mark.parametrize("eq", [AcousticSEM2DDeformed, AcousticPotentialSEM2D,
                                ElasticSHSEM2DDeformed, AcousticSEM3DDeformed],
                         ids=["acoustic2d", "potential2d", "sh2d", "acoustic3d"])
def test_topography(eq):
    _check(eq, True, topo=True)


@pytest.mark.parametrize("eq", [AcousticSEM2D, AcousticPotentialSEM2D, AcousticSEM3D],
                         ids=["pinned_2d", "ringed_2d", "ringed_3d"])
def test_multishot_vmapped_rollout(eq, monkeypatch):
    """Three shots on one model take the vmapped rollout, whose ring mask is
    shared by the shots."""
    calls = []
    real = BS.run_boundary_saved_batched
    monkeypatch.setattr(BS, "run_boundary_saved_batched",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    _check(eq, True, nshots=3)
    assert calls, "the three shots did not take the vmapped rollout"


def test_compiled_step():
    """The pinned recompute through torch.compile (the default eager path)."""
    _check(AcousticSEM2D, False, compile=True)


def test_source_encoding_and_wavelet_gradient():
    """Two sources in one shot with their own wavelets (the per-source
    rollout), the wavelets a leaf too."""
    shape, nt, dt, f0, _, _, rec, vals = _case(AcousticSEM2D, False)
    t = torch.arange(nt, dtype=torch.float64) * dt

    def run(memory):
        p = PropSEM(AcousticSEM2D, shape=shape, dh=10.0, dt=dt, n_order=4, abcn=8,
                    f_dominant=f0, dev="cpu", dtype=torch.float64, impl="eager",
                    memory=memory, eager_options=EagerOptions(use_compile=False))
        vp = vals["vp"].clone().contiguous().requires_grad_(True)
        a = torch.stack([(math.pi * f0 * (t - 0.05)) ** 2, (math.pi * 1.3 * f0 * (t - 0.04)) ** 2])
        w = ((1 - 2 * a) * torch.exp(-a)).requires_grad_(True)
        out = p(w, np.array([[[4, 4.0], [12, 3.0]]]), np.array([rec]), models=[vp])
        out.square().sum().backward()
        return out.detach(), vp.grad, w.grad

    full, bs = run(Full()), run(BoundarySaving())
    for name, b, f in zip(("record", "vp", "wavelet"), bs, full):
        rel = float((b - f).norm() / f.norm())
        assert rel < 1e-12, f"{name}: boundary saving vs full tape {rel:.2e}"
