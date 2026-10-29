"""``memory=Ckpt(chunks=K)`` on single shots: the checkpointed rollout of
:mod:`sweep.sem.ckpt_rollout`.

The forward keeps one state per chunk and the backward re-runs each chunk under
autograd. It is a memory strategy, not a different computation: the records equal
the full tape's and the model and wavelet gradients agree to round-off -- for any
equation (no boundary-saving hooks needed; Biot keeps model-derived operators
built by ``init_state``), any ``K`` (one that does not divide ``nt``, 1, more
than ``nt``), per-source wavelets and per-shot models. On CUDA the chunks are
CUDA graphs: the same checks run there, plus the fallback for a step that
cannot be captured.
"""
import gc
import math
import warnings

import numpy as np
import pytest
import torch

import sweep.sem.ckpt_rollout as CR
from sweep.propagator import Ckpt, Full
from sweep.sem import (PropSEM, AcousticSEM2D, ElasticSEM2DDeformed, BiotVSSEM2DDeformed,
                       AcousticSEM3D)

NT, DT, F0 = 60, 6e-4, 15.0


def _biot():
    from sweep.equations.biot import biot_from_biot_constants
    from test_biot import DAI1
    return tuple(float(c) for c in biot_from_biot_constants(**DAI1, b=7e4, phi=0.3))


# name: (equation, shape, model values, source_type, receiver_type, topography, dt)
CASES = {
    "acoustic2d": (AcousticSEM2D, (21, 33), (2000.0,), None, None, False, DT),
    "elastic2d_deformed": (ElasticSEM2DDeformed, (21, 33), (2400.0, 1300.0, 2000.0),
                           ["sxx", "szz"], ["vx", "vz"], True, DT),
    "biot2d": (BiotVSSEM2DDeformed, (21, 33), None, ["sxx", "szz"], ["vx", "vz", "p"], False, 2.5e-4),
    "acoustic3d": (AcousticSEM3D, (13, 13, 13), (2000.0,), None, None, False, DT),
}


def _wavelet(nt, dt, dtype, device, n=None):
    t = torch.arange(nt, dtype=dtype, device=device) * dt - 1.2 / F0
    w = (1 - 2 * (math.pi * F0 * t) ** 2) * torch.exp(-(math.pi * F0 * t) ** 2)
    if n is None:
        return w
    # n distinct time functions (sign and delay): the source-encoding mode
    return torch.stack([(-1) ** k * torch.roll(w, 3 * k) for k in range(n)])


def _setup(name, memory, dtype=torch.float64, device="cpu", nt=NT):
    eq, shape, vals, st, rt, topo, dt = CASES[name]
    vals = _biot() if vals is None else vals
    kw = dict(shape=shape, dh=10.0, dt=dt, n_order=4, abcn=8, free_surface=True, f_dominant=F0,
              dev=device, dtype=dtype, impl="eager", memory=memory)
    if st:
        kw.update(source_type=st, receiver_type=rt)
    if topo:
        x = np.arange(shape[-1])
        kw["topography"] = 3.0 - 3.0 * np.exp(-((x - shape[-1] / 2) / (shape[-1] / 8)) ** 2)
    prop = PropSEM(eq, **kw)
    if len(shape) == 2:
        src = np.array([[shape[1] // 2, 6.0]])
        rx = np.arange(3, shape[1] - 3, 3)
        rec = np.stack([rx, np.full_like(rx, 4)], 1)[None]
    else:
        src = np.array([[shape[2] // 2, shape[1] // 2, 5.0]])
        rx = np.arange(3, shape[2] - 3, 2)
        rec = np.stack([rx, np.full_like(rx, shape[1] // 2), np.full_like(rx, 3)], 1)[None]
    g = torch.Generator().manual_seed(0)
    models = [(v * (1 + 0.05 * torch.rand(shape, generator=g, dtype=dtype))).to(device) for v in vals]
    return prop, src, rec, models, dt


def _run(name, memory, *, grad=True, wav_grad=False, compile_step=False, **kw):
    prop, src, rec, models, dt = _setup(name, memory, **kw)
    ms = [m.clone().requires_grad_(grad) for m in models]
    w = _wavelet(NT, dt, models[0].dtype, models[0].device).requires_grad_(wav_grad)
    with torch.set_grad_enabled(grad):
        out = prop(w, src, rec, models=ms, compile_step=compile_step)
    if not grad:
        return out, None, None
    scale = 1.0 / float(out.detach().abs().max())
    (out * scale).square().sum().backward()
    return out.detach(), [m.grad for m in ms], w.grad


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


@pytest.fixture(autouse=True)
def one_thread():
    """Records compared bitwise on CPU: SEM scatters sum in a varying order on several threads."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("K", [7, 1, 100], ids=["K7", "K1", "K100"])
def test_matches_the_full_tape(name, K):
    if name == "acoustic3d" and K != 7:
        pytest.skip("3-D once is enough")
    r0, g0, w0 = _run(name, Full(), wav_grad=True)
    r1, g1, w1 = _run(name, Ckpt(chunks=K), wav_grad=True)
    assert torch.equal(r1, r0), "the forward is the same computation"
    for a, b in zip(g1, g0):
        assert _rel(a, b) < 1e-10
    assert _rel(w1, w0) < 1e-10


def test_without_grad_is_the_forward():
    r0, _, _ = _run("elastic2d_deformed", Full(), grad=False)
    r1, _, _ = _run("elastic2d_deformed", Ckpt(chunks=8), grad=False)
    assert torch.equal(r1, r0)


def test_per_source_wavelets():
    """The source-encoding mode: (1, nsrc, 2) sources, one time function each."""
    out = {}
    for mem in (Full(), Ckpt(chunks=9)):
        prop, _, rec, models, dt = _setup("acoustic2d", mem)
        src = np.array([[[8, 6.0], [16, 6.0], [24, 6.0]]])
        vp = models[0].clone().requires_grad_(True)
        w = _wavelet(NT, dt, vp.dtype, vp.device, n=3).requires_grad_(True)
        r = prop(w, src, rec, models=[vp])
        (r / r.detach().abs().max()).square().sum().backward()
        out[type(mem).__name__] = (r.detach(), vp.grad, w.grad)
    (r0, g0, w0), (r1, g1, w1) = out["Full"], out["Ckpt"]
    assert torch.equal(r1, r0)
    assert _rel(g1, g0) < 1e-10 and _rel(w1, w0) < 1e-10


def test_per_shot_models():
    out = {}
    for mem in (Full(), Ckpt(chunks=8)):
        prop, _, rec, models, dt = _setup("acoustic2d", mem)
        src = np.array([[10, 6.0], [22, 6.0]])
        vp = torch.stack([models[0], 1.03 * models[0]]).requires_grad_(True)     # (2, nz, nx)
        r = prop(_wavelet(NT, dt, vp.dtype, vp.device), src, np.repeat(rec, 2, 0), models=[vp])
        (r / r.detach().abs().max()).square().sum().backward()
        out[type(mem).__name__] = (r.detach(), vp.grad)
    (r0, g0), (r1, g1) = out["Full"], out["Ckpt"]
    assert torch.equal(r1, r0)
    assert _rel(g1, g0) < 1e-10


def test_second_derivative_is_refused():
    prop, src, rec, models, dt = _setup("acoustic2d", Ckpt(chunks=8))
    vp = models[0].clone().requires_grad_(True)
    loss = prop(_wavelet(NT, dt, vp.dtype, vp.device), src, rec, models=[vp]).square().sum()
    with pytest.raises(RuntimeError, match="first derivatives only"):
        torch.autograd.grad(loss, vp, create_graph=True)


# ------------------------------- CUDA graphs ------------------------------- #
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
# fp32 with atomic scatters: two runs of one configuration differ by up to ~1e-6
# (records) and ~2e-6 (gradients) on these meshes, so the bounds sit above that.
TOL = 1e-5


@cuda
@pytest.mark.parametrize("name", ["elastic2d_deformed", "biot2d", "acoustic3d"])
def test_graphs_match_the_full_tape(name, monkeypatch):
    built = []
    real = CR._Rollout._build_graphs

    def spy(self):
        real(self)
        built.append(self.g_f is not None and (self.g_b is not None or not self.need_grad))

    monkeypatch.setattr(CR._Rollout, "_build_graphs", spy)
    kw = dict(dtype=torch.float32, device="cuda")
    r0, g0, w0 = _run(name, Full(), wav_grad=True, **kw)
    r1, g1, w1 = _run(name, Ckpt(chunks=8), wav_grad=True, **kw)
    assert built == [True], "the chunks ran as CUDA graphs"
    assert _rel(r1, r0) < TOL
    for a, b in zip(g1, g0):
        assert _rel(a, b) < TOL
    assert _rel(w1, w0) < TOL


@cuda
def test_graphs_with_the_compiled_step():
    kw = dict(dtype=torch.float32, device="cuda")
    torch._dynamo.reset()
    r0, g0, _ = _run("elastic2d_deformed", Full(), **kw)
    r1, g1, _ = _run("elastic2d_deformed", Ckpt(chunks=8), compile_step=True, **kw)
    assert _rel(r1, r0) < 1e-5
    for a, b in zip(g1, g0):
        assert _rel(a, b) < 1e-4


@cuda
def test_graphs_equal_the_plain_chunks(monkeypatch):
    kw = dict(dtype=torch.float32, device="cuda")
    r1, g1, _ = _run("elastic2d_deformed", Ckpt(chunks=8), **kw)
    monkeypatch.setattr(CR, "USE_CUDA_GRAPHS", False)
    r0, g0, _ = _run("elastic2d_deformed", Ckpt(chunks=8), **kw)
    assert _rel(r1, r0) < TOL
    for a, b in zip(g1, g0):
        assert _rel(a, b) < TOL


class _Syncing(AcousticSEM2D):
    """A step with a host synchronisation: it cannot be captured."""

    def step(self, fields, src, dt):
        if float(fields[0].abs().sum()) < 0:      # never true; the .item() is the point
            raise AssertionError
        return super().step(fields, src, dt)


@cuda
def test_a_step_that_cannot_be_captured_runs_without_graphs(monkeypatch):
    monkeypatch.setattr(CR, "_CAPTURE_WARNED", set())
    monkeypatch.setitem(CASES, "syncing", (_Syncing,) + CASES["acoustic2d"][1:])
    kw = dict(dtype=torch.float32, device="cuda")
    r0, g0, _ = _run("acoustic2d", Full(), **kw)
    stream = torch.cuda.current_stream()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r1, g1, _ = _run("syncing", Ckpt(chunks=8), **kw)
    assert any("could not be captured as a CUDA graph" in str(w.message) for w in caught)
    assert _rel(r1, r0) < TOL and _rel(g1[0], g0[0]) < TOL
    # the failed capture must not leave the CUDA generator in capture mode (a random
    # draw would raise in whatever the caller runs next), nor the caller on the
    # capture stream (its later work would run unordered with the default stream's)
    torch.rand(4, device="cuda")
    assert torch.cuda.current_stream() == stream


@cuda
def test_repeated_calls_hold_no_more_memory():
    """The graphs are warmed up on one side stream per device. A new stream per
    call left a cuBLAS workspace behind every time (PyTorch keeps one per handle
    and stream for the process), and an FWI loop grew by 8-17 MB an iteration."""
    kw = dict(dtype=torch.float32, device="cuda")
    _run("elastic2d_deformed", Ckpt(chunks=8), **kw)      # the side stream's own first workspaces
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(3):
        _run("elastic2d_deformed", Ckpt(chunks=8), **kw)
    gc.collect()
    torch.cuda.synchronize()
    grown = torch.cuda.memory_allocated() - before
    assert grown < 2 ** 20, f"{grown / 2 ** 20:.1f} MiB more after three calls"
