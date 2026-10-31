"""``PropSEM(memory=...)`` as on PropTorch: the default strategy, ``memory_strategy``,
and what PropSEM refuses.

Without ``memory=`` PropTorch saves the boundary on ``impl='c'`` (the full tape for
equations whose kernels have no boundary saving) and checkpoints on eager; PropSEM
does the same, in chunks of ``_DEFAULT_CKPT_CHUNK`` steps on eager. The test suite
runs ``memory=None`` as the full tape (test/conftest.py); these tests undo that.
"""
import numpy as np
import pytest
import torch

import sweep.sem.prop_base as PB
from sweep.propagator import BoundarySaving, Ckpt, EagerOptions, Full
from sweep.sem import PropSEM, AcousticSEM2D, AcousticPotentialSEM2D, ElasticVSSEM3D

SHAPE, NT = (21, 33), 40
SRC = np.array([[16, 2.0]])
REC = np.stack([np.arange(2, 31, 4), np.full(8, 2.0)], 1)[None]


@pytest.fixture
def real_default(monkeypatch):
    monkeypatch.setattr(PB, "_DEFAULT_MEMORY", None)       # undo the suite's full-tape default


def _prop(**kw):
    kw.setdefault("dev", "cpu")
    return PropSEM(AcousticSEM2D, shape=SHAPE, dh=10.0, dt=1e-3, n_order=4, abcn=8,
                   free_surface=True, f_dominant=15.0, impl="eager", **kw)


def _wavelet(dev="cpu"):
    w = torch.zeros(NT, device=dev)
    w[3] = 1.0
    return w


def test_eager_checkpoints_by_default(real_default):
    p = _prop()
    assert p.memory == Ckpt(chunks=PB._DEFAULT_CKPT_CHUNK)
    assert p.memory_strategy == "ckpt"


@pytest.mark.parametrize("memory, strategy", [(Full(), "full"), (BoundarySaving(), "boundary"),
                                              (Ckpt(chunks=4), "ckpt")])
def test_memory_strategy_names_the_one_in_force(real_default, memory, strategy):
    p = _prop(memory=memory)
    assert p.memory == memory and p.memory_strategy == strategy


def test_snapshots_run_under_the_default(real_default):
    """The default strategy gives way to the full tape for return_wavefield=True;
    an explicit Ckpt() still refuses it."""
    vp = torch.full(SHAPE, 2000.0)
    seis, snaps = _prop()(_wavelet(), SRC, REC, models=[vp], return_wavefield=True,
                          snapshot_times=[10, 20])
    assert snaps.shape[0] == 2 and torch.isfinite(seis).all()
    with pytest.raises(ValueError, match="Ckpt"):
        _prop(memory=Ckpt(chunks=8))(_wavelet(), SRC, REC, models=[vp],
                                     return_wavefield=True, snapshot_times=[10, 20])


def test_fp16_rings_are_refused():
    with pytest.raises(NotImplementedError, match="fp16"):
        _prop(memory=BoundarySaving(storage_dtype="fp16"))


def test_a_second_backward_through_the_rollout_says_so():
    vp = torch.full(SHAPE, 2000.0, requires_grad=True)
    loss = _prop(memory=Ckpt(chunks=8))(_wavelet(), SRC, REC, models=[vp]).square().sum()
    loss.backward(retain_graph=True)
    with pytest.raises(RuntimeError, match="retain_graph"):
        loss.backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-graph compile modes")
def test_a_cuda_graph_compile_mode_takes_the_full_tape_by_default(real_default):
    p = _prop(dev="cuda", eager_options=EagerOptions(compile_mode="reduce-overhead"))
    assert p.memory == Full() and p.memory_strategy == "full"


def _sem_c():
    try:
        from sweep.sem.prop_c import sem_c_available
        return torch.cuda.is_available() and sem_c_available()
    except Exception:
        return False


@pytest.mark.skipif(not _sem_c(), reason="the SEM CUDA backend")
def test_impl_c_saves_the_boundary_where_its_kernels_can(real_default):
    p = PropSEM(AcousticPotentialSEM2D, shape=(25, 25), dh=25.0, dt=2e-4, nt=20, n_order=4,
                free_surface=("top",), abcn=8, f_dominant=8.0, dev="cuda", impl="c")
    assert p.memory == BoundarySaving() and p.memory_strategy == "boundary"
    # kernels that keep the whole history: the full tape
    q = PropSEM(ElasticVSSEM3D, shape=(17, 17, 17), dh=10.0, dt=8e-4, n_order=4, abcn=8,
                free_surface=True, f_dominant=12.0, dev="cuda", impl="c")
    assert q.memory == Full() and q.memory_strategy == "full"
