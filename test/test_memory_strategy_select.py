"""Unified three-way gradient-memory mode (full / boundary / ckpt).

One resolver (options.resolve_memory_strategy) decides the mode identically
for the eager and CUDA backends; conflicting knob combinations raise instead
of checkpointing silently winning (the historical use_ckpt=True default beat
an explicitly enabled boundary_saving_config, so "boundary" scripts silently
ran the checkpoint backward).
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic
from sweep.propagator.options import MemoryOptions, BoundaryOptions, CkptOptions, resolve_memory_strategy
from sweep.propagator.torch import PropTorch

R = resolve_memory_strategy


def test_resolver_matrix():
    assert R("c") == "boundary"
    assert R("eager") == "ckpt"
    assert R("c", use_ckpt=False) == "boundary"
    assert R("eager", use_ckpt=False) == "full"
    assert R("c", boundary_saving_config={"enabled": True}) == "boundary"
    assert R("c", boundary_saving_config={"enabled": False}) == "full"
    assert R("c", use_ckpt=True) == "ckpt"
    assert R("eager", memory=MemoryOptions(strategy="full")) == "full"
    assert R("c", boundary_saving_config={"enabled": False}, use_ckpt=True) == "ckpt"
    assert R("c", boundary_saving_config={"enabled": True}, use_ckpt=False) == "boundary"
    with pytest.raises(ValueError, match="Conflicting"):
        R("c", boundary_saving_config={"enabled": True}, use_ckpt=True)
    with pytest.raises(ValueError):
        R("c", memory=MemoryOptions(strategy="ckpt"), boundary_saving_config={"enabled": True})
    with pytest.raises(ValueError):
        R("c", memory=MemoryOptions(strategy="ckpt"), use_ckpt=False)


def test_memory_options_full_takes_no_blocks():
    MemoryOptions(strategy="full")
    with pytest.raises(ValueError):
        MemoryOptions(strategy="full", boundary=BoundaryOptions())
    with pytest.raises(ValueError):
        MemoryOptions(strategy="full", ckpt=CkptOptions())


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="impl='c' needs CUDA")
DEV = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _mk(impl, **kw):
    eq = Acoustic(spatial_order=4, device=DEV, backend="torch")
    return PropTorch(eq, backend="torch", impl=impl, shape=(100, 120), dev=DEV,
                     dh=10.0, dt=1e-3, nt=160, abcn=20, **kw)


def _grad(p):
    torch.manual_seed(0)
    vp = torch.full((100, 120), 2000.0, device=DEV)
    vp[40:60, 40:80] += 150.0
    vp = vp.clone().requires_grad_(True)
    src = np.array([[[50, 60]]], dtype=np.int64)
    rec = np.array([[[22, x] for x in range(4, 116, 6)]], dtype=np.int64)
    from sweep.signal import ricker
    wav = torch.as_tensor(ricker(np.arange(160) * 1e-3, f=10.0),
                          dtype=torch.float32, device=DEV)[None]
    syn = p(wav, src, rec, models=[vp])
    (syn * syn).sum().backward()
    return vp.grad.detach().clone()


@needs_cuda
def test_c_boundary_dict_alone_runs_boundary():
    # THE trap regression: dict-style boundary config alone must select the
    # boundary backward (deterministic), never checkpointing.
    p = _mk("c", boundary_saving_config={"enabled": True, "storage": "gpu"})
    assert p.memory_strategy == "boundary"
    assert p.use_ckpt is False
    g1, g2 = _grad(p), _grad(p)
    assert torch.equal(g1, g2)


@needs_cuda
def test_c_defaults_and_full_and_ckpt():
    assert _mk("c").memory_strategy == "boundary"
    p_full = _mk("c", memory=MemoryOptions(strategy="full"))
    assert p_full.memory_strategy == "full" and p_full.use_ckpt is False
    _grad(p_full)
    p_ck = _mk("c", memory=MemoryOptions(strategy="ckpt", ckpt=CkptOptions()))
    assert p_ck.memory_strategy == "ckpt" and p_ck.use_ckpt is True
    _grad(p_ck)


@needs_cuda
def test_c_conflicts_raise():
    with pytest.raises(ValueError, match="Conflicting"):
        _mk("c", boundary_saving_config={"enabled": True}, use_ckpt=True)
    with pytest.raises(ValueError, match="not both"):
        _mk("c", memory=MemoryOptions(strategy="full"), use_ckpt=False)


@needs_cuda
def test_c_per_call_conflict_raises():
    p = _mk("c", memory=MemoryOptions(strategy="ckpt", ckpt=CkptOptions()))
    vp = torch.full((100, 120), 2000.0, device=DEV).requires_grad_(True)
    src = np.array([[[50, 60]]], dtype=np.int64)
    rec = np.array([[[22, 60]]], dtype=np.int64)
    from sweep.signal import ricker
    wav = torch.as_tensor(ricker(np.arange(160) * 1e-3, f=10.0),
                          dtype=torch.float32, device=DEV)[None]
    with pytest.raises(ValueError, match="three-way"):
        syn = p(wav, src, rec, models=[vp],
                boundary_saving_config={"enabled": True})
        (syn * syn).sum().backward()


def test_eager_strategies():
    assert _mk("eager").memory_strategy == "ckpt"
    p_full = _mk("eager", memory=MemoryOptions(strategy="full"))
    assert p_full.memory_strategy == "full"
    assert p_full.use_ckpt is False and not getattr(p_full._backend_impl, "_eager_bs", False)
    p_bs = _mk("eager", memory=MemoryOptions(strategy="boundary",
                                             boundary=BoundaryOptions(storage="gpu")))
    assert p_bs.memory_strategy == "boundary"
    assert p_bs.use_ckpt is False and p_bs._backend_impl._eager_bs
    # dict style now reaches the eager backend too
    p_bs2 = _mk("eager", boundary_saving_config={"enabled": True, "storage": "gpu"})
    assert p_bs2.memory_strategy == "boundary" and p_bs2._backend_impl._eager_bs
    p_off = _mk("eager", use_ckpt=False)
    assert p_off.memory_strategy == "full"
