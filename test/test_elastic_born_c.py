"""ElasticBorn impl='c' (CUDA).

1. background record == Elastic impl='c' record;
2. scattered record == the derivative of the Elastic impl='c' record;
3. the (dvp, dvs, drho) gradients == the Elastic impl='c' (vp, vs, rho)
   gradients for the same adjoint source, in every memory mode -- the Born
   backward IS elastic2d's adjoint imaged against the background;
4. the adjoint identity <J dm, r> = <dm, J^T r> with the Elastic c gradient;
5. the background and wavelet gradients are refused.
"""

import numpy as np
import pytest
import torch

from sweep.equations import Elastic, ElasticBorn
from sweep.propagator.options import BoundarySaving, Ckpt, Full
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="impl='c' needs CUDA")

NZ, NX = 48, 56
DH, DT, NT = 10.0, 1.0e-3, 300
ABCN, SO = 16, 4
DEV = "cuda"

FS_CASES = {
    "none": dict(free_surface=False),
    "top": dict(free_surface=True),
    "per_edge": dict(free_surface=(True, False, True, False)),
    "topography": dict(topography=(2 + 2 * np.sin(np.linspace(0, np.pi, NX))).astype(np.int64),
                       topo_method="image", free_surface=False),
}
MEMORY = {
    "full": Full(),
    "bs": BoundarySaving(),
    "ckpt": Ckpt(mode="chunk", chunks=4),
    "recursive": Ckpt(mode="recursive", count=6),
}


def _wavelet():
    t = np.arange(NT) * DT - 0.05
    return torch.tensor(1.0e3 * ricker(t, f=15.0), dtype=torch.float32, device=DEV)


def _geometry():
    # integer numpy coordinates: what impl='c' takes
    sources = np.array([[NX // 2, 8], [NX // 3, 10]])
    rec_x = np.arange(3, NX - 3, 4)
    receivers = np.repeat(np.stack([rec_x, np.full_like(rec_x, 6)], -1)[None], 2, axis=0)
    return sources, receivers


def _models():
    z = torch.linspace(0, 1, NZ, device=DEV)[:, None].expand(NZ, NX)
    x = torch.linspace(0, 1, NX, device=DEV)[None, :].expand(NZ, NX)
    vp = (1800 + 600 * z).contiguous()
    vs = vp / 1.8
    rho = 1800 + 300 * z
    g = torch.exp(-((z - 0.6) ** 2 + (x - 0.45) ** 2) / 0.01)
    lens = torch.exp(-((z - 0.35) ** 2 + (x - 0.7) ** 2) / 0.004)
    return (vp, vs, rho), (80 * g, -50 * lens + 30 * g, 60 * g - 40 * lens)


def _prop(eq_cls, fs, memory=None, **kw):
    eq = eq_cls(spatial_order=SO, device=DEV, backend="torch")
    return PropTorch(eq, shape=(NZ, NX), abcn=ABCN, dh=DH, dt=DT, impl="c",
                     memory=memory or Full(), **FS_CASES[fs], **kw)


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


@pytest.mark.parametrize("fs", list(FS_CASES))
def test_c_forward(fs):
    """The background record is Elastic's; the scattered one is the derivative
    of the Elastic impl='c' record along the perturbation (Richardson-extrapolated
    central differences, fp32: ~2e-4..4e-4 measured).  Not against eager: Elastic
    itself differs between the backends (1.4e-4 flat, ~0.5 with topography)."""
    sources, receivers = _geometry()
    w = _wavelet()
    bg, pert = _models()
    with torch.no_grad():
        born = _prop(ElasticBorn, fs, receiver_type=["vx", "vz", "svx", "svz"])(
            w, sources, receivers, models=[*bg, *pert])
        el = _prop(Elastic, fs, receiver_type=["vx", "vz"])

        def central(eps):
            plus = el(w, sources, receivers, models=[m + eps * d for m, d in zip(bg, pert)])
            minus = el(w, sources, receivers, models=[m - eps * d for m, d in zip(bg, pert)])
            return (plus - minus) / (2 * eps)

        fd = (4 * central(0.25) - central(0.5)) / 3
        assert _rel(born[..., :2], el(w, sources, receivers, models=list(bg))) < 1e-6
    assert _rel(born[..., 2:], fd) < 1e-3


def _c_grads(cls, fs, mode, models, r):
    ms = [m.clone().requires_grad_(g) for m, g in models]
    d = _prop(cls, fs, memory=MEMORY[mode])(_wavelet(), *_geometry(), models=ms)
    (d * r).sum().backward()
    return d.detach(), [m.grad for m, (_, g) in zip(ms, models) if g]


# boundary saving + topography is refused on impl='c' (Elastic alike)
@pytest.mark.parametrize("fs,mode", [(fs, mode) for fs in ("none", "top", "topography")
                                     for mode in MEMORY if (fs, mode) != ("topography", "bs")])
def test_c_gradients_are_elastic_imaging(fs, mode):
    bg, pert = _models()
    shape = (2, NT, _geometry()[1].shape[1], 2)
    r = torch.randn(shape, generator=torch.Generator().manual_seed(2)).to(DEV)
    sc, g_born = _c_grads(ElasticBorn, fs, mode, [(m, False) for m in bg] + [(m, True) for m in pert], r)
    _, g_el = _c_grads(Elastic, fs, mode, [(m, True) for m in bg], r)
    # same adjoint and imaging; the background comes from the fused Born kernel,
    # whose FMA contraction differs from elastic2d's at the fp32 round-off
    for name, a, b in zip(("dvp", "dvs", "drho"), g_born, g_el):
        assert b.abs().max() > 0, name
        assert _rel(a, b) < 1e-5, (name, _rel(a, b))
    # adjoint identity: <J dm, r> = <dm, J^T r>
    lhs = float((sc * r).sum())
    rhs = float(sum((dm * g).sum() for dm, g in zip(pert, g_el)))
    assert abs(lhs - rhs) < 1e-3 * abs(rhs), (lhs, rhs)


@pytest.mark.parametrize("which", ["vp", "wavelet"])
def test_c_refuses_background_and_wavelet_gradients(which):
    bg, pert = _models()
    models = [m.clone() for m in (*bg, *pert)]
    w = _wavelet()
    if which == "vp":
        models[0].requires_grad_(True)
    else:
        w.requires_grad_(True)
    with pytest.raises(NotImplementedError, match="impl='eager'"):
        _prop(ElasticBorn, "none")(w, *_geometry(), models=models)
