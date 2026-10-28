"""ElasticBorn (eager): the scattered record is the directional derivative of
the Elastic record along the model perturbation.

1. background record == Elastic record, bit-exact;
2. scattered record == the Elastic Jacobian along (dvp, dvs, drho), as the
   adjoint identity against Elastic's autograd gradient, for no free surface /
   flat top / per-edge / topography;
3. boundary-saving gradients == full-storage gradients (exact reverse substeps).
"""

import numpy as np
import pytest
import torch

from sweep.equations import Elastic, ElasticBorn
from sweep.propagator.options import BoundarySaving, EagerOptions, Full
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

NZ, NX = 40, 48
DH, DT, NT = 10.0, 1.0e-3, 220
ABCN, SO = 12, 4
DTYPE = torch.float64   # inputs only: the eager propagator steps in fp32
UNCOMPILED = EagerOptions(use_compile=False)

FS_CASES = {
    "none": dict(free_surface=False),
    "top": dict(free_surface=True),
    "per_edge": dict(free_surface=(True, False, True, False)),
    "topography": dict(topography=(2 + 2 * np.sin(np.linspace(0, np.pi, NX))).astype(np.int64),
                       topo_method="image", free_surface=False),
}


def _wavelet():
    t = np.arange(NT) * DT - 0.05
    return torch.tensor(1.0e3 * ricker(t, f=15.0), dtype=DTYPE)


def _geometry():
    sources = torch.tensor([[NX // 2, 8]])
    rec_x = torch.arange(3, NX - 3, 4)
    receivers = torch.stack([rec_x, torch.full_like(rec_x, 6)], -1)[None]
    return sources, receivers


def _models():
    z = torch.linspace(0, 1, NZ, dtype=DTYPE)[:, None].expand(NZ, NX)
    x = torch.linspace(0, 1, NX, dtype=DTYPE)[None, :].expand(NZ, NX)
    vp = (1800 + 600 * z).contiguous()
    vs = vp / 1.8
    rho = 1800 + 300 * z
    g = torch.exp(-((z - 0.6) ** 2 + (x - 0.45) ** 2) / 0.01)
    lens = torch.exp(-((z - 0.35) ** 2 + (x - 0.7) ** 2) / 0.004)
    return (vp, vs, rho), (80 * g, -50 * lens + 30 * g, 60 * g - 40 * lens)


def _prop(eq_cls, fs, memory=None, **kw):
    eq = eq_cls(spatial_order=SO, device="cpu", backend="torch")
    return PropTorch(eq, shape=(NZ, NX), abcn=ABCN, dh=DH, dt=DT, impl="eager",
                     eager_options=UNCOMPILED, memory=memory or Full(), **FS_CASES[fs], **kw)


@pytest.mark.parametrize("fs", list(FS_CASES))
def test_scattered_record_is_elastic_jacobian(fs):
    """<J dm, r> = <dm, J^T r>: the scattered record against the Elastic
    autograd gradient of <d(m), r>.  No finite-difference cancellation, so it
    holds to the fp32 round-off of the eager propagator."""
    sources, receivers = _geometry()
    w = _wavelet()
    bg, pert = _models()

    born = _prop(ElasticBorn, fs, receiver_type=["vx", "vz", "svx", "svz"])
    with torch.no_grad():
        rec = born(w, sources, receivers, models=[*bg, *pert])
    rec_bg, rec_sc = rec[..., :2], rec[..., 2:]

    el = _prop(Elastic, fs, receiver_type=["vx", "vz"])
    models = [m.clone().requires_grad_(True) for m in bg]
    d = el(w, sources, receivers, models=models)
    assert torch.equal(rec_bg, d.detach()), "background record differs from Elastic"

    r = torch.randn(d.shape, dtype=d.dtype, generator=torch.Generator().manual_seed(1))
    (d * r).sum().backward()
    lhs = float((rec_sc.to(r.dtype) * r).sum())
    rhs = float(sum((dm * m.grad).sum() for dm, m in zip(pert, models)))
    assert abs(lhs - rhs) < 1e-4 * abs(rhs), (lhs, rhs)


@pytest.mark.parametrize("fs", ["none", "top"])
def test_boundary_saving_matches_full_storage(fs):
    """The reverse substeps invert both fields exactly: every gradient, the
    background (RWI) ones included, matches full storage.  Short record, central
    source: the wavefield has not reached the absorbing layer, past which the
    eager boundary saving of Elastic itself is no longer exact (Elastic and
    ElasticBorn alike drift to ~1e-3 by the end of the 220-step record above,
    fp64 included)."""
    nt = 40
    t = np.arange(nt) * DT - 0.03
    w = torch.tensor(1.0e3 * ricker(t, f=25.0), dtype=DTYPE)
    sources = torch.tensor([[NX // 2, NZ // 2]])
    rec_x = torch.arange(14, NX - 14, 3)
    receivers = torch.stack([rec_x, torch.full_like(rec_x, NZ // 2 - 6)], -1)[None]
    bg, pert = _models()
    grads = {}
    for name, mem in (("full", Full()), ("bs", BoundarySaving())):
        models = [m.clone().requires_grad_(True) for m in (*bg, *pert)]
        rec = _prop(ElasticBorn, fs, memory=mem)(w, sources, receivers, models=models)
        obs = torch.randn(rec.shape, dtype=rec.dtype, generator=torch.Generator().manual_seed(0))
        ((rec - obs) ** 2).sum().backward()
        grads[name] = [m.grad for m in models]
    for spec, gf, gb in zip(ElasticBorn.MODEL_SPECS, grads["full"], grads["bs"]):
        assert gf.abs().max() > 0, spec.name
        rel = float((gf - gb).abs().max() / gf.abs().max())
        assert rel < 1e-4, (spec.name, rel)
