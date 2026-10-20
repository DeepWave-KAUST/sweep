"""Per-shot batched models (B, nz, nx) for ViscoAcoustic: no cross-shot coupling.

The NCQ step freezes the fractional exponent at the model average ``gbar``.
With per-shot models that average must be taken per batch element: a
whole-batch mean made every shot's |k| filter depend on the other shots' Q
(issue #123: ~1.7e-3 data / ~8e-4 gradient error vs running each model alone
when Q differs across the batch).  Covered here:

* the batched forward and the vp / Q gradients equal looping the shots one at
  a time, on impl='eager' and impl='c', with Q DIFFERENT per shot;
* ``prepare_models`` keeps one scalar exponent for a shared model (and for a
  shared model the C path repeated across the batch), so the shared path is
  bit-for-bit the historical one.
"""
import numpy as np
import pytest
import torch

from sweep.equations import ViscoAcoustic
from sweep.propagator.torch import PropTorch
from sweep.propagator.options import CUDAOptions, MemoryOptions, CkptOptions
from sweep.signal import ricker

HAS_CUDA = torch.cuda.is_available()
DEV = "cuda" if HAS_CUDA else "cpu"
NZ, NX, DH, DT, NT, SO, ABCN = 48, 64, 8.0, 1.0e-3, 400, 4, 16
F0 = 12.0
B = 3

needs_cuda = pytest.mark.skipif(not HAS_CUDA, reason="impl='c' requires CUDA")
# impl='c' in each gradient memory mode (every backward reads the filter grids).
IMPLS = [pytest.param(m, marks=needs_cuda) for m in ("c", "c_ckpt", "c_rec")] + ["eager"]
CKPT = {"c_ckpt": CkptOptions(mode="chunk", chunks=10),
        "c_rec": CkptOptions(mode="recursive", count=8)}


def _wavelet():
    t = np.arange(NT, dtype=np.float32) * DT - 0.06
    return torch.tensor((1e3 * ricker(t, f=F0)).astype(np.float32), device=DEV)


def _geometry():
    sx = np.array([16, 32, 48], dtype=np.int64)
    sources = np.stack([sx, np.full(B, 4, np.int64)], -1)            # (B, [x, z])
    rx = np.arange(4, NX - 4, 2, dtype=np.int64)
    rec = np.stack([rx, np.full_like(rx, 2)], -1)
    return sources, np.broadcast_to(rec, (B, *rec.shape)).copy()     # (B, nrec, 2)


def _models():
    """B models, heterogeneous in vp and Q, with a different Q level per shot
    (the case the whole-batch exponent coupled)."""
    rng = np.random.default_rng(0)
    d = np.linspace(0, 1, NZ, dtype=np.float32)[:, None]
    vp = np.stack([2100 + 150 * b + 600 * d + rng.uniform(-40, 40, (NZ, NX))
                   for b in range(B)]).astype(np.float32)
    Q = np.stack([(15 + 25 * b) * (1 + 0.5 * d) + rng.uniform(-3, 3, (NZ, NX))
                  for b in range(B)]).astype(np.float32)
    om = np.full_like(vp, 2 * np.pi * F0)
    return [torch.tensor(m, device=DEV) for m in (vp, Q, om)]


def _prop(impl):
    eq = ViscoAcoustic(spatial_order=SO, device=DEV, backend="torch")
    if impl == "eager":
        kw = dict(impl="eager", use_compile=False, use_ckpt=False)
    elif impl in CKPT:
        kw = dict(impl="c", cuda_options=CUDAOptions(memory=MemoryOptions(
            strategy="ckpt", ckpt=CKPT[impl])))
    else:
        kw = dict(impl="c", use_ckpt=False)
    return PropTorch(eq, shape=(NZ, NX), abcn=ABCN, dh=DH, dt=DT, nt=NT, **kw)


def _run(impl, models, sources, receivers):
    ms = [m.clone().requires_grad_(i < 2) for i, m in enumerate(models)]
    d = _prop(impl)(_wavelet(), sources, receivers, models=ms)
    (d ** 2).sum().backward()
    return d.detach().cpu(), [m.grad.detach().cpu() for m in ms[:2]]


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("impl", IMPLS)
def test_visco_batched_matches_loop(impl):
    """Batched forward and vp / Q gradients == one shot at a time (Q differs)."""
    sources, receivers = _geometry()
    models = _models()
    d_b, g_b = _run(impl, models, sources, receivers)
    for b in range(B):
        d_1, g_1 = _run(impl, [m[b] for m in models], sources[b:b + 1], receivers[b:b + 1])
        ed = _rel(d_b[b:b + 1], d_1)
        eg = [_rel(gb[b], g1) for gb, g1 in zip(g_b, g_1)]
        print(f"[{impl}] shot {b}: rel(data)={ed:.2e} rel(grad vp, Q)={eg[0]:.2e}, {eg[1]:.2e}")
        # The coupling bug sat at ~1e-3; what is left is float noise (cuFFT
        # batch plans, atomicAdd order in the compiled gradient).
        assert ed < 1e-5, f"[{impl}] shot {b} data rel={ed:.2e}"
        assert max(eg) < 1e-4, f"[{impl}] shot {b} grad rel={eg}"


def test_visco_per_shot_exponent_is_per_element():
    """prepare_models: one exponent per batch element for per-shot models,
    each the same reduction a single-shot run takes; one scalar otherwise."""
    eq = ViscoAcoustic(spatial_order=SO, device="cpu", backend="torch")
    vp, Q, om = [m.cpu() for m in _models()]
    gamma = torch.arctan(1.0 / Q) / np.pi

    eq.prepare_models([vp[:, None], Q[:, None], om[:, None]])          # eager per-shot layout
    assert eq._gbar.shape == (B, 1, 1, 1)
    for b in range(B):
        eq_1 = ViscoAcoustic(spatial_order=SO, device="cpu", backend="torch")
        eq_1.prepare_models([vp[b], Q[b], om[b]])
        assert torch.equal(eq._gbar[b].reshape(()), eq_1._gbar)

    eq.prepare_models([vp[0], Q[0], om[0]])                              # shared (nz, nx)
    assert eq._gbar.ndim == 0 and torch.equal(eq._gbar, torch.mean(gamma[0]))

    rep = lambda m: m[0][None, None].repeat(B, 1, 1, 1)                  # shared, as impl='c' lays it out
    eq.prepare_models([rep(vp), rep(Q), rep(om)])
    assert eq._gbar.ndim == 0
    assert torch.equal(eq._gbar, torch.mean(rep(gamma)))


@needs_cuda
def test_visco_shared_model_c_unchanged():
    """impl='c': a shared model keeps (nz, nx) filter grids and equals an
    identical per-shot batch bit-for-bit; per-shot Q gives (B, nz, nx) grids."""
    sources, receivers = _geometry()
    vp, Q, om = _models()
    prop = _prop("c")
    d_shared = prop(_wavelet(), sources, receivers, models=[vp[0], Q[0], om[0]]).cpu()
    assert all(g.ndim == 2 for g in prop.equation.c_eq_aux(prop))
    rep = lambda m: m[0][None].repeat(B, 1, 1)
    d_rep = _prop("c")(_wavelet(), sources, receivers, models=[rep(vp), rep(Q), rep(om)]).cpu()
    assert torch.equal(d_shared, d_rep)

    prop = _prop("c")
    prop(_wavelet(), sources, receivers, models=[vp, Q, om])
    shapes = [tuple(g.shape) for g in prop.equation.c_eq_aux(prop)]
    assert shapes == [(B, *shapes[1]), shapes[1], (B, *shapes[1])], shapes
