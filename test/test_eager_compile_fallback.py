"""A compiler that cannot build the eager step costs speed, not the run.

``EagerOptions.use_compile`` defaults to True, so a plain CPU forward goes
through torch.compile, and Inductor's CPU backend shells out to g++. With g++ 9
(Ubuntu 20.04) it fails on ``-std=c++20`` and the very first forward raised
``InductorError: CppCompileError``. The step now falls back to running
uncompiled, warns once, and returns exactly what ``use_compile=False`` does.
"""
import warnings

import numpy as np
import pytest
import torch
CppCompileError = pytest.importorskip("torch._inductor.exc").CppCompileError   # torch < 2.0: nothing to fall back from

from sweep.equations import Acoustic
from sweep.propagator import BoundarySaving
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch

DEV = torch.device("cpu")
SHAPE = (40, 48)
WAVELET = np.zeros(120, dtype=np.float32)
WAVELET[5] = 1.0
SRC = np.array([[20, 4]])
REC = np.array([[[ix, 4] for ix in range(0, 48, 4)]])


def _compile_that_cannot_build(calls):
    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            calls.append(1)
            raise CppCompileError(["g++"], "g++: error: unrecognized command line option '-std=c++20'")
        return compiled
    return fake_compile


def _run(use_compile, memory=None):
    kw = {} if memory is None else {"memory": memory}
    solver = PropTorch(Acoustic(device=DEV), shape=SHAPE, dh=10.0, dt=1e-3, dev=DEV, impl="eager",
                       eager_options=EagerOptions(use_compile=use_compile), **kw)
    vp = torch.full(SHAPE, 2000.0, requires_grad=True)
    out = solver(WAVELET, SRC, REC, models=[vp])
    out.square().sum().backward()
    return solver, out.detach(), vp.grad


@pytest.fixture
def compiles_on_cpu(monkeypatch):
    """These tests stand in their own torch.compile; on torch 2.3 the CPU step
    would skip compiling before reaching it (INDUCTOR_23)."""
    import sweep.propagator._torch_eager as TE
    monkeypatch.setattr(TE, "INDUCTOR_23", False)


@pytest.mark.usefixtures("compiles_on_cpu")
@pytest.mark.parametrize("memory", [None, BoundarySaving()], ids=["full", "boundary_saving"])
def test_falls_back_to_the_uncompiled_step(monkeypatch, memory):
    _, ref_out, ref_grad = _run(False, memory)

    calls = []
    monkeypatch.setattr(torch, "compile", _compile_that_cannot_build(calls))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver, out, grad = _run(True, memory)

    assert calls == [1], "the compiled step is tried once, then never again"
    assert solver.use_compile is False
    fallback = [w for w in caught if "torch.compile could not build" in str(w.message)]
    assert len(fallback) == 1 and issubclass(fallback[0].category, RuntimeWarning)
    assert "CppCompileError" in str(fallback[0].message)
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)


@pytest.mark.usefixtures("compiles_on_cpu")
def test_an_error_in_the_step_itself_is_not_swallowed(monkeypatch):
    def fake_compile(fn, **kwargs):
        def compiled(*args, **kw):
            raise ValueError("bad model")
        return compiled
    monkeypatch.setattr(torch, "compile", fake_compile)
    with pytest.raises(ValueError, match="bad model"):
        _run(True)


@pytest.mark.parametrize("memory", [None, BoundarySaving()], ids=["full", "boundary_saving"])
def test_a_torch_without_compile_runs_uncompiled_and_says_so_once(monkeypatch, memory):
    """torch < 2.0 has no torch.compile at all.  The propagator used to skip
    compiling there in silence, leaving users on the slow path unawares."""
    import sweep.propagator._torch_eager as TE
    _, ref_out, ref_grad = _run(False, memory)

    monkeypatch.delattr(torch, "compile")
    monkeypatch.setattr(TE, "_NO_COMPILE_WARNED", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver, out, grad = _run(True, memory)
        _run(True, memory)
    notices = [w for w in caught if "has no torch.compile" in str(w.message)]
    assert len(notices) == 1, "once per process, not per propagator or rollout"
    assert issubclass(notices[0].category, RuntimeWarning)
    assert solver.use_compile is False
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)


def _compile_never_called(calls):
    def fake_compile(fn, **kwargs):
        calls.append(kwargs)
        return fn
    return fake_compile


@pytest.mark.parametrize("memory", [None, BoundarySaving()], ids=["full", "boundary_saving"])
def test_torch_23_runs_uncompiled_on_cpu_and_says_so_once(monkeypatch, memory):
    """torch 2.3's Inductor miscompiles the step on CPU (an assertion, mostly
    while compiling the backward, where no fallback catches it), and nothing
    turns the faulty fusion off."""
    import sweep.propagator._torch_eager as TE
    _, ref_out, ref_grad = _run(False, memory)

    calls = []
    monkeypatch.setattr(TE, "INDUCTOR_23", True)
    monkeypatch.setattr(TE, "_NO_COMPILE_WARNED", False)
    monkeypatch.setattr(torch, "compile", _compile_never_called(calls))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver, out, grad = _run(True, memory)
        _run(True, memory)
    assert calls == []
    notices = [w for w in caught if "fixed in torch 2.4" in str(w.message)]
    assert len(notices) == 1 and issubclass(notices[0].category, RuntimeWarning)
    assert solver.use_compile is False
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(grad, ref_grad, rtol=0, atol=0)


@pytest.mark.parametrize("memory", [None, BoundarySaving()], ids=["full", "boundary_saving"])
def test_torch_23_on_cpu_still_compiles_for_other_backends(monkeypatch, memory):
    import sweep.propagator._torch_eager as TE
    calls = []
    monkeypatch.setattr(TE, "INDUCTOR_23", True)
    monkeypatch.setattr(torch, "compile", _compile_never_called(calls))
    solver = PropTorch(Acoustic(device=DEV), shape=SHAPE, dh=10.0, dt=1e-3, dev=DEV, impl="eager",
                       eager_options=EagerOptions(compile_backend="aot_eager"),
                       **({} if memory is None else {"memory": memory}))
    solver(WAVELET, SRC, REC, models=[torch.full(SHAPE, 2000.0, requires_grad=True)]).sum().backward()
    assert calls and all(c["backend"] == "aot_eager" and "options" not in c for c in calls)


@pytest.mark.parametrize("mode", ["default", None, "max-autotune"])
def test_torch_23_options_keep_the_mode_and_turn_reuse_off(mode):
    from sweep.propagator._torch_eager import _no_buffer_reuse
    options = _no_buffer_reuse(mode, None)
    assert options.pop("allow_buffer_reuse") is False
    if mode == "max-autotune":
        assert options.get("max_autotune") is True
    else:
        assert options == {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_torch_23_compiles_on_gpu_with_buffer_reuse_off(monkeypatch):
    """Buffer reuse hands a complex result still read through ``.real`` to the
    next FFT (ViscoAcoustic's forward turned NaN); with reuse off the GPU step
    still compiles."""
    import sweep.propagator._torch_eager as TE
    calls = []
    monkeypatch.setattr(TE, "INDUCTOR_23", True)
    monkeypatch.setattr(torch, "compile", _compile_never_called(calls))
    dev = torch.device("cuda")
    solver = PropTorch(Acoustic(device=dev), shape=SHAPE, dh=10.0, dt=1e-3, dev=dev, impl="eager")
    solver(WAVELET, SRC, REC, models=[torch.full(SHAPE, 2000.0, device=dev)])
    assert calls and all(c["options"] == {"allow_buffer_reuse": False} and "mode" not in c for c in calls)
