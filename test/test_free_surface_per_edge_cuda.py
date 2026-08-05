"""Per-edge free surface on ``impl='c'`` (CUDA): forward record + adjoint gradient
consistency vs the eager reference, for every face and z∩x corner, across ALL
backward memory modes — full, boundary-saving, and checkpointing.

CRITICAL — the geometry must DISCRIMINATE.  A central source with near-surface
receivers and short ``NT`` makes the free-surface reflection never reach the
receivers, so every ``free_surface`` config yields ~the same record/gradient and
a backend that silently ignores a face (e.g. runs top-only) passes trivially.
This test uses a central source, a receiver GRID over the whole physical
interior, and a long ``NT`` (many boundary round-trips), and asserts each
non-top config's record genuinely DIFFERS from the top-only reference before
trusting its c-vs-eager match.  The gradient tolerance (rel_l2 < 3e-3) is well
below the error a top-only-collapsed backend produces (>=2.8e-2 for a single x
face, 0.06+ for a z face), so this test fails loudly on that regression.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic, Elastic
from sweep.propagator.torch import PropTorch
from sweep.propagator.options import (
    CUDAOptions, MemoryOptions, BoundaryOptions, CkptOptions,
)
from sweep.signal import ricker

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="per-edge free surface on impl='c' requires CUDA",
)

DEV = "cuda"
NZ, NX, DH, DT, NT, SO, ABCN = 64, 64, 8.0, 1.0e-3, 450, 4, 16
PHYS = NZ - 2 * (ABCN + SO)      # physical interior of the no-FS layout (coords fit all configs)
DISC = 0.10                      # a non-top config must move the record at least this much

CONFIGS = [
    (True, "top"),
    (["bottom"], "bottom"),
    (["left"], "left"),
    (["right"], "right"),
    (["top", "left"], "corner"),
    (["top", "bottom", "left", "right"], "all4"),
]
MODES = ["full", "bs", "ckpt"]


def _wav():
    t = np.arange(NT, dtype=np.float32) * DT - 0.05
    return torch.tensor((1e3 * ricker(t, f=10.0)).astype(np.float32), device=DEV)


def _src_rec():
    src = np.array([[PHYS // 2, PHYS // 2]], dtype=np.int64)
    g = np.arange(2, PHYS - 1, 4, dtype=np.int64)
    rx, rz = np.meshgrid(g, g, indexing="xy")
    rec = np.stack([rx.ravel(), rz.ravel()], -1)[None]
    return src, rec


def _models(eqkind, req):
    d = np.linspace(0, 1, NZ, dtype=np.float32)
    vp = np.broadcast_to((2200 + 700 * d)[:, None], (NZ, NX)).astype(np.float32).copy()
    vp_t = torch.tensor(vp, device=DEV, requires_grad=req)
    if eqkind == "acoustic":
        return [vp_t]
    return [vp_t,
            torch.tensor((vp / 1.7).astype(np.float32).copy(), device=DEV),
            torch.full((NZ, NX), 1000.0, device=DEV)]


def _prop(eqkind, fs, mode):
    if eqkind == "acoustic":
        eq = Acoustic(spatial_order=SO, device=DEV, backend="torch")
        stype, rtype = ["p"], ["p"]
    else:
        eq = Elastic(spatial_order=SO, device=DEV, backend="torch")
        stype, rtype = ["sxx", "szz"], ["vx", "vz"]
    common = dict(shape=(NZ, NX), free_surface=fs, abcn=ABCN, dh=DH, dt=DT,
                  source_type=stype, receiver_type=rtype)
    if mode == "eager":
        return PropTorch(eq, backend="torch", impl="eager", use_ckpt=False, **common)
    if mode == "full":
        return PropTorch(eq, backend="torch", impl="c", use_ckpt=False,
                         boundary_saving_config={"enabled": False}, **common)
    if mode == "bs":
        return PropTorch(eq, backend="torch", impl="c",
                         cuda_options=CUDAOptions(memory=MemoryOptions(
                             strategy="boundary", boundary=BoundaryOptions(storage="gpu"))),
                         **common)
    if mode == "ckpt":
        return PropTorch(eq, backend="torch", impl="c",
                         cuda_options=CUDAOptions(memory=MemoryOptions(
                             strategy="ckpt", ckpt=CkptOptions(mode="chunk", chunks=10))),
                         **common)
    raise ValueError(mode)


# --- cache the (expensive, pure-python) eager reference per (eqkind, tag) ------
_CACHE = {}


def _record(eqkind, fs, mode, tag):
    key = ("rec", eqkind, tag, mode)
    if mode == "eager" and key in _CACHE:
        return _CACHE[key]
    prop = _prop(eqkind, fs, mode)
    src, rec = _src_rec()
    out = prop(_wav(), src, rec, models=_models(eqkind, False)).detach()
    if mode == "eager":
        _CACHE[key] = out
    return out


def _grad(eqkind, fs, mode, tag):
    key = ("grad", eqkind, tag, mode)
    if mode == "eager" and key in _CACHE:
        return _CACHE[key]
    prop = _prop(eqkind, fs, mode)
    src, rec = _src_rec()
    models = _models(eqkind, True)
    syn = prop(_wav(), src, rec, models=models)
    (0.5 * (syn ** 2).sum()).backward()
    g = models[0].grad
    if mode == "eager":
        _CACHE[key] = g
    return g


def _cos(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def _rel(a, b):
    return (a - b).norm().item() / max(b.norm().item(), 1e-30)


@pytest.mark.parametrize("eqkind", ["acoustic", "elastic"])
@pytest.mark.parametrize("fs,tag", CONFIGS)
def test_per_edge_cuda_forward_matches_eager(eqkind, fs, tag):
    """c forward record ~ eager, in a geometry where the free surface matters."""
    re = _record(eqkind, fs, "eager", tag)
    rc = _record(eqkind, fs, "full", tag)
    if tag != "top":
        rtop = _record(eqkind, True, "eager", "top")
        assert _rel(re, rtop) > DISC, (
            f"{eqkind}/{tag}: non-discriminating geometry "
            f"(eager record == top-only, rel_l2={_rel(re, rtop):.3f})")
    r = _rel(rc, re)
    assert r < 5e-3, f"{eqkind}/{tag}: c forward vs eager rel_l2={r:.2e}"


@pytest.mark.parametrize("eqkind", ["acoustic", "elastic"])
@pytest.mark.parametrize("fs,tag", CONFIGS)
@pytest.mark.parametrize("mode", MODES)
def test_per_edge_cuda_gradient_matches_eager(eqkind, fs, tag, mode):
    """c adjoint gradient ~ eager across full / bs / ckpt for every face + corner."""
    ge = _grad(eqkind, fs, "eager", tag)
    gc = _grad(eqkind, fs, mode, tag)
    c = _cos(gc, ge)
    r = _rel(gc, ge)
    assert c > 0.999 and r < 3e-3, f"{eqkind}/{tag}/{mode}: cos={c:.5f} rel_l2={r:.2e}"
