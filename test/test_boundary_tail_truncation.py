"""Boundary tail truncation (BoundaryOptions.tail_steps) for steady-state /
frequency-selection objectives: only the last K steps' boundary strips are
saved and back-propagated.

Covers: bit-exact degeneration when the tail covers the whole record,
steady-state margin convergence of the truncated gradient, storage parity,
the 3-D backend, and the unsupported-combination guards.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic, Acoustic3D, Elastic
from sweep.propagator.torch import PropTorch

if not torch.cuda.is_available():
    pytest.skip("boundary tail truncation is impl='c' CUDA only", allow_module_level=True)

DEV = torch.device("cuda:0")


def _steady_wavelet(nt, dt, f=10.0, rise=0.3):
    t = np.arange(nt) * dt
    ramp = np.clip(t / rise, 0.0, 1.0) ** 2
    return torch.as_tensor((np.sin(2 * np.pi * f * t) * ramp).astype(np.float32),
                           device=DEV)[None]


def _grad_2d(tail, nt=1200, n_probe=300, storage="gpu"):
    torch.manual_seed(0)
    cfg = {"enabled": True, "storage": storage}
    if tail:
        cfg["tail_steps"] = tail
    shape, abcn, dt = (140, 160), 20, 1e-3
    eq = Acoustic(spatial_order=4, device=DEV, backend="torch")
    p = PropTorch(eq, backend="torch", impl="c", shape=shape, dev=DEV, dh=10.0,
                  dt=dt, nt=nt, abcn=abcn, boundary_saving_config=cfg,
                  use_ckpt=False)
    z = torch.linspace(0, 1, shape[0], device=DEV)
    vp = (1800 + 600 * z)[:, None].expand(shape).contiguous()
    vp[40:70, 50:90] += 180.0
    vp = vp.clone().requires_grad_(True)
    src = np.array([[[45, 80]]], dtype=np.int64)
    rec = np.array([[[22, x] for x in range(4, 156, 6)]], dtype=np.int64)
    syn = p(_steady_wavelet(nt, dt), src, rec, models=[vp])
    g = torch.Generator(device="cpu").manual_seed(7)
    w = torch.rand(syn.shape[-2], generator=g).to(DEV)[None, None, :, None]
    (syn[:, -n_probe:] * syn[:, -n_probe:] * w).sum().backward()
    return vp.grad.detach().clone()


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.double().flatten(), b.double().flatten(), dim=0).item()


def test_tail_covering_full_record_is_bitwise():
    g_full = _grad_2d(None)
    g_ge = _grad_2d(1200 + 7)     # tail >= nt must degenerate exactly
    assert torch.equal(g_full, g_ge)


def test_steady_state_margin_convergence():
    g_full = _grad_2d(None)
    cs = [_cos(_grad_2d(300 + m), g_full) for m in (0, 200, 600)]
    # truncation discards only the adjoint x ring-up-transient correlation;
    # with a growing drain margin the gradient converges to the full one
    assert cs[0] > 0.98
    assert cs[-1] > 0.9995
    assert cs == sorted(cs)


def test_storage_parity_gpu_cpu():
    g_gpu = _grad_2d(500, storage="gpu")
    g_cpu = _grad_2d(500, storage="cpu")
    assert torch.equal(g_gpu, g_cpu)


def test_3d_tail_and_margin():
    def g3(tail):
        torch.manual_seed(0)
        cfg = {"enabled": True, "storage": "gpu"}
        if tail:
            cfg["tail_steps"] = tail
        nt, dt, shape, abcn = 400, 1e-3, (56, 48, 56), 12
        eq = Acoustic3D(spatial_order=4, device=DEV, backend="torch")
        p = PropTorch(eq, backend="torch", impl="c", shape=shape, dev=DEV,
                      dh=10.0, dt=dt, nt=nt, abcn=abcn,
                      boundary_saving_config=cfg, use_ckpt=False)
        z = torch.linspace(0, 1, shape[0], device=DEV)
        vp = (1800 + 600 * z)[:, None, None].expand(shape).contiguous()
        vp = vp.clone().requires_grad_(True)
        src = np.array([[[24, 24, 28]]], dtype=np.int64)
        rec = np.array([[[14, 24, x] for x in range(6, 50, 4)]], dtype=np.int64)
        syn = p(_steady_wavelet(nt, dt, rise=0.15), src, rec, models=[vp])
        (syn[:, -120:] ** 2).sum().backward()
        return vp.grad.detach().clone()

    g_full = g3(None)
    assert torch.equal(g_full, g3(400 + 9))
    assert _cos(g3(120 + 180), g_full) > 0.999


def test_guards():
    def mk(eq_cls, **kw):
        eq = eq_cls(spatial_order=4, device=DEV, backend="torch")
        return PropTorch(eq, backend="torch", impl="c", shape=(140, 160),
                         dev=DEV, dh=10.0, dt=1e-3, nt=200, abcn=20, **kw)

    def run(p, elastic=False):
        vp = torch.full((140, 160), 2000.0, device=DEV).requires_grad_(True)
        models = [vp]
        if elastic:
            models = [vp, (vp.detach() / 1.73).requires_grad_(True),
                      torch.full((140, 160), 1500.0, device=DEV).requires_grad_(True)]
        src = np.array([[[70, 80]]], dtype=np.int64)
        rec = np.array([[[22, 80]]], dtype=np.int64)
        syn = p(_steady_wavelet(200, 1e-3), src, rec, models=models)
        (syn * syn).sum().backward()

    cfg = {"enabled": True, "storage": "gpu", "tail_steps": 100}
    with pytest.raises(NotImplementedError):
        run(mk(Elastic, boundary_saving_config=cfg, use_ckpt=False,
               source_type=["vz"], receiver_type=["vz"]), elastic=True)
    with pytest.raises(ValueError):
        run(mk(Acoustic, boundary_saving_config={"enabled": False, "tail_steps": 100},
               use_ckpt=False))
    with pytest.raises(NotImplementedError):
        run(mk(Acoustic, boundary_saving_config=cfg, use_ckpt=True))
