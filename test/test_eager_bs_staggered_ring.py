"""Boundary saving of the staggered ('substep') equations is exact in the
interior once the wavefield has crossed into the PML.

A staggered field at index ``i`` lives at ``i + 1/2``, so on the high side of
every axis the last physical cell holds half-grid values that sit in the first
half cell of the PML (``azh``/``axh`` nonzero there).  The reconstruction zeroes
CPML, so that cell has to come back from the saved ring, and each sub-step's
fields have to be restored as soon as it is inverted (the inverted velocity
half reads the ring's stresses).  Neither was done: from the steps in which
the wavefield reached the PML the error crept inward, and the interior
gradients drifted off the full tape (fp64, NT=300: 1.6e-5 Acoustic1st,
3.7e-4 Elastic).  The interior now matches to round-off; the outermost one or
two physical cells keep the near-PML error of the design (the CPML memories
are not stored).

CPU, fp32 (the eager propagator steps in fp32).  Max error beyond 2 cells of
the physical boundary, relative to the gradient's max, measured before / after
the fix: Acoustic1st 1.3e-5 / <=3.8e-7, Elastic 5.1e-5 / <=3.6e-6, Elastic with
a free surface 4.4e-5 / <=1.3e-6, JAX Acoustic1st 1.6e-5 / 5.1e-8, JAX Elastic
3.6e-4 / 1.7e-5 (fp32 rounding: in x64 the JAX interior matches to 1e-15).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import ricker
from sweep.equations import Acoustic1st, Elastic
from sweep.propagator.options import BoundaryOptions, EagerOptions, MemoryOptions
from sweep.propagator.torch import PropTorch

DT, DH, NT, SO, ABCN = 0.0015, 10.0, 300, 4, 20
SHAPE = (32, 40)
INSET = 2            # cells of the physical boundary left out of the comparison
BOUNDARY = MemoryOptions(strategy="boundary", boundary=BoundaryOptions(storage="gpu"))

# tol: eager; jax_tol: JAX, whose scan autodiff and per-step VJP round apart
CASES = {
    "acoustic1st": dict(cls=Acoustic1st, st=["p"], rt=["p"], kind="acoustic1st", fs=False,
                        tol=4e-6, jax_tol=4e-6),
    "elastic2d": dict(cls=Elastic, st=["sxx", "szz"], rt=["vx", "vz"], kind="elastic", fs=False,
                      tol=1.5e-5, jax_tol=5e-5),
    "elastic2d_fs": dict(cls=Elastic, st=["sxx", "szz"], rt=["vx", "vz"], kind="elastic", fs=True,
                         tol=1.5e-5),
}


def _models(kind):
    nz, nx = SHAPE
    ramp = lambda top, bot: np.broadcast_to(
        np.linspace(top, bot, nz, dtype=np.float32)[:, None], SHAPE).copy()
    vp = ramp(1800.0, 2400.0)
    true_vp = vp.copy()
    true_vp[nz // 3:2 * nz // 3, nx // 4:3 * nx // 4] += 180.0
    rho = ramp(1000.0, 1200.0)
    if kind == "acoustic1st":
        return [true_vp, rho], [vp, rho]
    return [true_vp, true_vp / 1.73, rho], [vp, vp / 1.73, rho]


def _geometry():
    nz, nx = SHAPE
    src = np.array([[nx // 2, nz // 4]], dtype=np.int32)
    rx = np.arange(2, nx - 2, 6, dtype=np.int32)
    rec = np.stack([rx, np.full(rx.size, SO // 2, np.int32)], -1)[None]
    return src, rec


def _interior_error(g_ref, g):
    g_ref, g = np.asarray(g_ref, np.float64), np.asarray(g, np.float64)
    inner = (slice(INSET, -INSET),) * g.ndim
    return float(np.abs(g - g_ref)[inner].max() / np.abs(g_ref).max())


def _eager_grads(cfg, memory, init_m, obs, w, src, rec):
    eq = cfg["cls"](spatial_order=SO, device="cpu", backend="torch")
    prop = PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False),
                     memory=memory, shape=SHAPE, dev="cpu", dh=DH, dt=DT,
                     source_type=cfg["st"], receiver_type=cfg["rt"], abcn=ABCN,
                     pml_type=eq.default_pml_type, free_surface=cfg["fs"], nt=NT, B=1)
    ms = [torch.tensor(a, requires_grad=True) for a in init_m]
    out = prop(w, src.copy(), rec.copy(), models=ms)
    if obs is None:
        return out.detach()
    (out - obs).pow(2).sum().backward()
    return [m.grad for m in ms]


@pytest.mark.parametrize("name", list(CASES))
def test_eager_boundary_saving_interior_matches_full_tape(name):
    cfg = CASES[name]
    true_m, init_m = _models(cfg["kind"])
    w = torch.tensor(ricker(NT, DT))
    src, rec = _geometry()
    with torch.no_grad():
        obs = _eager_grads(cfg, None, true_m, None, w, src, rec)
    full = _eager_grads(cfg, None, init_m, obs, w, src, rec)
    bs = _eager_grads(cfg, BOUNDARY, init_m, obs, w, src, rec)
    for k, (g_full, g_bs) in enumerate(zip(full, bs)):
        err = _interior_error(g_full, g_bs)
        assert err < cfg["tol"], f"{name} model {k}: interior error {err:.2e}"


@pytest.mark.parametrize("name", ["acoustic1st", "elastic2d"])
def test_jax_boundary_saving_interior_matches_autodiff(name):
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from sweep.propagator.jax import PropJax

    cfg = CASES[name]
    true_m, init_m = _models(cfg["kind"])
    # a unit stress source's velocities sit near the fp32 floor (see
    # test_jax_boundary_saving)
    w = ricker(NT, DT, scale=1e6 if cfg["kind"] == "elastic" else 1.0)[None, :]
    src, rec = _geometry()

    def build(memory=None):
        eq = cfg["cls"](spatial_order=SO, device="cpu", backend="jax")
        return PropJax(eq, SHAPE, memory=memory, dh=DH, dt=DT, source_type=cfg["st"],
                       receiver_type=cfg["rt"], abcn=ABCN, pml_type=eq.default_pml_type,
                       nt=NT, B=1)

    obs = build()(w, src[None], rec, models=[jnp.asarray(a) for a in true_m])

    def grad(prop):
        rest = [jnp.asarray(a) for a in init_m[1:]]

        def loss(m0):
            return jnp.sum((prop(w, src[None], rec, models=[m0] + rest) - obs) ** 2)

        return jax.grad(loss)(jnp.asarray(init_m[0]))

    err = _interior_error(grad(build()), grad(build(BOUNDARY)))
    assert err < cfg["jax_tol"], f"{name}: interior error {err:.2e}"
