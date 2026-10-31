"""A compiler that cannot build the SEM step costs speed, not the run.

The compiled step on CPU goes through Inductor, which calls g++ with
``-std=c++20``; g++ 9 (Ubuntu 20.04) rejects it and every compiled step raised
``InductorError: CppCompileError``. As on the FD eager path, the step now runs
uncompiled, warns once, and returns exactly what ``use_compile=False`` does --
for the single-shot loop, boundary saving and the vmapped multi-shot path.
"""
import warnings

import pytest
import torch

CppCompileError = pytest.importorskip("torch._inductor.exc").CppCompileError

import sweep._compile as C
import sweep.propagator._torch_eager as TE
from sweep.propagator import BoundarySaving, EagerOptions
from sweep.sem import PropSEM, AcousticSEM2D

SHAPE, NT = (21, 33), 60
SRC1 = torch.tensor([[16, 2]])
SRC2 = torch.tensor([[10, 2], [22, 2]])
REC = torch.stack([torch.arange(2, 31, 4), torch.full((8,), 2)], 1)[None]


def _wavelet():
    t = torch.arange(NT, dtype=torch.float64) * 1e-3 - 0.04
    a = (torch.pi * 15.0 * t) ** 2
    return (1 - 2 * a) * torch.exp(-a)


def _run(compile, memory=None, sources=SRC1):
    kw = {} if memory is None else {"memory": memory}
    prop = PropSEM(AcousticSEM2D, shape=SHAPE, dh=10.0, dt=1e-3, n_order=4, abcn=8,
                   free_surface=True, f_dominant=15.0, dev="cpu", dtype=torch.float64,
                   impl="eager", eager_options=EagerOptions(use_compile=compile), **kw)
    vp = torch.full(SHAPE, 2000.0, dtype=torch.float64, requires_grad=True)
    rec = REC.expand(sources.shape[0], -1, -1)
    out = prop(_wavelet(), sources, rec, models=[vp])
    out.square().sum().backward()
    return out.detach(), vp.grad


@pytest.fixture(autouse=True)
def one_thread():
    """Bitwise comparisons on CPU: SEM scatters sum in a varying order on several threads."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.fixture
def compiler_cannot_build(monkeypatch):
    """torch.compile whose compiled function fails the way g++ 9 does; counts the tries."""
    calls = []

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            calls.append(1)
            raise CppCompileError(["g++"], "g++: error: unrecognized command line option '-std=c++20'")
        return compiled

    monkeypatch.setattr(C, "_FAILED", {})
    monkeypatch.setattr(TE, "INDUCTOR_23", False)   # torch 2.3 would not try on CPU
    monkeypatch.setattr(torch, "compile", fake_compile)
    return calls


@pytest.mark.parametrize("memory, sources", [(None, SRC1), (BoundarySaving(), SRC1), (None, SRC2)],
                         ids=["loop", "boundary_saving", "vmapped"])
def test_falls_back_to_the_uncompiled_step(compiler_cannot_build, memory, sources):
    ref_out, ref_grad = _run(False, memory, sources)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out, grad = _run(True, memory, sources)
        again, grad2 = _run(True, memory, sources)       # a second build skips the attempt
    assert compiler_cannot_build == [1], "the compiled step is tried once per process, then never again"
    msgs = [str(w.message) for w in caught if "torch.compile could not build" in str(w.message)]
    assert len(msgs) == 1 and "torch._inductor.config.cpp.cxx" in msgs[0]
    assert all(issubclass(w.category, RuntimeWarning) for w in caught
               if "torch.compile could not build" in str(w.message))
    for o, g in ((out, grad), (again, grad2)):
        assert torch.equal(o, ref_out) and torch.equal(g, ref_grad)


def test_step_errors_still_propagate(monkeypatch):
    """Only compiler failures fall back: an error of the step itself is the caller's."""
    monkeypatch.setattr(C, "_FAILED", {})
    monkeypatch.setattr(TE, "INDUCTOR_23", False)

    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            raise ValueError("a bug in the step")
        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    with pytest.raises(ValueError, match="a bug in the step"):
        _run(True)
