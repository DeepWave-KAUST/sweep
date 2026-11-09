"""Inductor options of the compiled eager step: ``EagerOptions(compile_cpp_wrapper=...,
compile_combo_kernels=...)``.

For a 2-D grid on a CUDA device from torch 2.9 the step compiles with Inductor's
``cpp_wrapper`` (the host code of each compiled call generated as C++) and
``combo_kernels`` (independent small kernels launched as one); elsewhere, 3-D
included, it compiles as before.
``cpp_wrapper`` keeps the kernels, so the records and gradients stay bit for bit
those of a compile without it; ``combo_kernels`` keeps the records and moves the
gradients by float32 round-off.  The comparisons compile both sides in one
process and so one Inductor cache: Inductor times the candidate configs of a
reduction kernel on its first compile, and two independent compiles of the same
graph can pick different ones, which alone moves a gradient by ~1e-7.

A step whose C++ wrapper does not build (no CUDA toolkit to build it against, a
C++ compiler that rejects Inductor's flags) compiles again without it -- not
uncompiled, ten times slower -- and later compiles of the process leave it out
at once.  Without a CUDA toolkit the default leaves it out from the start.
"""
import stat
import warnings
from functools import lru_cache

import numpy as np
import pytest
import torch

import sweep.propagator._torch_eager as TE
from sweep.equations import Acoustic, Acoustic3D
from sweep.equations.elastic import Elastic
from sweep.propagator import BoundarySaving, Full
from sweep.propagator.options import EagerOptions
from sweep.propagator.torch import PropTorch

from conftest import requires_compile, ricker

CppCompileError = pytest.importorskip("torch._inductor.exc").CppCompileError

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
pytestmark = requires_compile

SHAPE, NT, ABCN = (48, 56), 48, 10
OFF = dict(compile_cpp_wrapper=False, compile_combo_kernels=False)


def _solver(dev, eq="acoustic", memory=None, shape=SHAPE, **eager):
    kw = dict(shape=shape, dh=10.0, dt=1e-3, nt=NT, dev=dev, impl="eager", abcn=ABCN,
              eager_options=EagerOptions(**eager))
    if memory == "bs":
        kw["memory"] = BoundarySaving()
    elif memory == "full":
        kw["memory"] = Full()
    if eq == "elastic":
        kw.update(source_type=["sxx", "szz"], receiver_type=["vz"])
        return PropTorch(Elastic(4, device=dev), **kw)
    return PropTorch(Acoustic(4, device=dev), **kw)


def _models(dev, eq, shape=SHAPE):
    g = torch.Generator().manual_seed(0)
    vp = 2000 + 500 * torch.rand(shape, generator=g)
    models = [vp] + ([vp / 1.9, 1800 + 400 * torch.rand(shape, generator=g)] if eq == "elastic" else [])
    return [m.to(dev) for m in models]


def _gradient(solver, dev, eq="acoustic", shape=SHAPE):
    """Record and gradients (every model, then the wavelet) of sum(record**2), one shot."""
    models = [m.clone().requires_grad_(True) for m in _models(dev, eq, shape)]
    wavelet = torch.tensor(ricker(NT, 1e-3, fm=15.0), device=dev, requires_grad=True)
    src = np.array([[shape[1] // 2, 2]])
    rec = np.array([[[ix, 2] for ix in range(0, shape[1], 4)]])
    out = solver(wavelet, src, rec, models=models)
    out.square().sum().backward()
    return out.detach(), [m.grad for m in models] + [wavelet.grad]


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


def _extras(kwargs):
    """The two options among torch.compile's ``kwargs`` (a C++ wrapper brings
    the settings torch puts next to it: see _cpp_wrapper_companions)."""
    return {k: v for k, v in kwargs.get("options", {}).items() if k in TE.INDUCTOR_EXTRAS}


def _spy_compile(monkeypatch, fail=lambda options: False):
    """torch.compile replaced by one that records its keywords and returns the step
    uncompiled, raising Inductor's C++ compile error from the call when
    ``fail(options)`` -- as a compiler that cannot build does on the first call."""
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append(kwargs)
        options = kwargs.get("options", {})

        def compiled(*args, **kw):
            if fail(options):
                raise CppCompileError(["g++"], "g++: error: unrecognized command-line option '-std=c++20'")
            return fn(*args, **kw)
        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    return calls


@pytest.fixture
def fresh_failures(monkeypatch):
    """No C++ wrapper failure remembered from earlier tests, no warning given yet."""
    monkeypatch.setattr(TE, "_CPP_WRAPPER_FAILED", {})
    monkeypatch.setattr(TE, "_COMBO_FAILED_WARNED", False)
    monkeypatch.setattr(TE, "INDUCTOR_23", False)      # torch 2.3 would not compile on CPU at all


# --------------------------------------------------------------------------- #
# Defaults and refusals
# --------------------------------------------------------------------------- #

def test_on_cpu_the_default_compiles_as_before(monkeypatch):
    monkeypatch.setattr(TE, "INDUCTOR_23", False)
    calls = _spy_compile(monkeypatch)
    solver = _solver("cpu")
    assert solver._backend_impl._inductor_extras == {}
    _gradient(solver, "cpu")
    assert calls and all(c == {"mode": "default", "dynamic": False, "fullgraph": False} for c in calls)


@pytest.fixture
def toolkit(monkeypatch):
    """torch finds a CUDA toolkit to build a C++ wrapper against, whatever this machine has."""
    monkeypatch.setattr(TE, "_cuda_toolkit", lambda: True)


needs_toolkit = pytest.mark.skipif(not torch.cuda.is_available() or not TE._cuda_toolkit(),
                                   reason="no CUDA toolkit (CUDA_HOME) to build a C++ wrapper against")


@gpu
@pytest.mark.usefixtures("toolkit")
@pytest.mark.parametrize("mode, extras", [
    ("default", {"cpp_wrapper": True, "combo_kernels": True}),
    ("max-autotune-no-cudagraphs", {"cpp_wrapper": True, "combo_kernels": True}),
])      # the modes that replay CUDA graphs are refused on a GPU (test_eager_cuda_graph_modes)
def test_on_a_gpu_both_are_on_by_default(mode, extras):
    assert _solver("cuda", compile_mode=mode)._backend_impl._inductor_extras == extras


@gpu
@pytest.mark.usefixtures("toolkit")
def test_in_3d_the_default_compiles_as_before():
    """A 3-D step keeps the GPU busy: the two gained nothing there (combo_kernels
    made a 3-D elastic gradient 1.5 % slower), so the default leaves them off;
    asked for, they are on."""
    def solver(**eager):
        return PropTorch(Acoustic3D(4, device="cuda"), shape=(16, 14, 12), dh=10.0, dt=1e-3, nt=NT,
                         dev="cuda", impl="eager", abcn=4, eager_options=EagerOptions(**eager))
    assert solver()._backend_impl._inductor_extras == {}
    assert solver(compile_cpp_wrapper=True, compile_combo_kernels=True)._backend_impl._inductor_extras == \
        {"cpp_wrapper": True, "combo_kernels": True}


@gpu
def test_without_a_cuda_toolkit_the_default_leaves_the_cpp_wrapper_out(monkeypatch):
    """Its build would fail in every process; asked for, it is still tried."""
    monkeypatch.setattr(TE, "_cuda_toolkit", lambda: False)
    assert _solver("cuda")._backend_impl._inductor_extras == {"combo_kernels": True}
    assert _solver("cuda", compile_cpp_wrapper=True)._backend_impl._inductor_extras == \
        {"cpp_wrapper": True, "combo_kernels": True}


def test_the_toolkit_is_one_with_the_headers(monkeypatch, tmp_path):
    cpp_extension = pytest.importorskip("torch.utils.cpp_extension")     # imports setuptools
    for home, found in ((None, False), (str(tmp_path), False)):
        monkeypatch.setattr(cpp_extension, "CUDA_HOME", home)
        TE._cuda_toolkit.cache_clear()
        assert TE._cuda_toolkit() is found
    (tmp_path / "include").mkdir()
    (tmp_path / "include" / "cuda.h").write_text("")
    TE._cuda_toolkit.cache_clear()
    assert TE._cuda_toolkit() is True
    monkeypatch.undo()
    TE._cuda_toolkit.cache_clear()


@gpu
def test_before_torch_29_the_gpu_default_compiles_as_before(monkeypatch):
    monkeypatch.setattr(TE, "INDUCTOR_EXTRAS_DEFAULT", False)
    assert _solver("cuda")._backend_impl._inductor_extras == {}
    assert _solver("cuda", compile_combo_kernels=True)._backend_impl._inductor_extras == {"combo_kernels": True}


@gpu
@pytest.mark.usefixtures("fresh_failures", "toolkit")
def test_the_options_reach_torch_compile(monkeypatch):
    calls = _spy_compile(monkeypatch)
    _gradient(_solver("cuda", compile_mode="max-autotune-no-cudagraphs", compile_combo_kernels=False), "cuda")
    assert calls
    for c in calls:
        assert "mode" not in c and (c["dynamic"], c["fullgraph"]) == (False, False)
        assert _extras(c) == {"cpp_wrapper": True, "combo_kernels": False}
        assert c["options"]["max_autotune"] is True and c["options"]["coordinate_descent_tuning"] is True
        assert c["options"] == {**c["options"], **TE._cpp_wrapper_companions()}


def test_explicit_values_reach_torch_compile_on_cpu(monkeypatch):
    monkeypatch.setattr(TE, "INDUCTOR_23", False)
    calls = _spy_compile(monkeypatch)
    _gradient(_solver("cpu", compile_cpp_wrapper=True, compile_combo_kernels=False), "cpu")
    assert calls and all("mode" not in c and _extras(c) == {"cpp_wrapper": True, "combo_kernels": False}
                         for c in calls)
    _gradient(_solver("cpu", compile_combo_kernels=False), "cpu")
    assert calls[-1]["options"] == {"combo_kernels": False}, "no C++ wrapper: nothing next to it"


@pytest.mark.parametrize("eager, match", [
    (dict(use_compile=False, compile_cpp_wrapper=False), "use_compile=False compiles nothing"),
    (dict(use_compile=False, compile_combo_kernels=True), "use_compile=False compiles nothing"),
    (dict(compile_backend="aot_eager", compile_cpp_wrapper=True), "compile_backend='aot_eager' does not read"),
    (dict(compile_backend="aot_eager", compile_combo_kernels=False), "compile_backend='aot_eager' does not read"),
    (dict(compile_mode="reduce-overhead", compile_cpp_wrapper=True), "CUDA graphs"),
    (dict(compile_mode="max-autotune", compile_cpp_wrapper=True), "CUDA graphs"),
    (dict(compile_combo_kernels=True), "needs a CUDA device"),
], ids=["no_compile_cpp", "no_compile_combo", "aot_eager_cpp", "aot_eager_combo", "reduce_overhead",
        "max_autotune", "combo_on_cpu"])
def test_an_explicit_value_nothing_reads_raises(eager, match):
    with pytest.raises(ValueError, match=match):
        _solver("cpu", **eager)


def test_the_defaults_of_other_paths_raise_nothing():
    for eager in (dict(use_compile=False), dict(compile_backend="aot_eager"),
                  dict(compile_mode="reduce-overhead"), dict(compile_cpp_wrapper=False, compile_mode="max-autotune")):
        _solver("cpu", **eager)


# --------------------------------------------------------------------------- #
# A step that does not compile with them (stand-in compiler, CPU)
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=None)
def _uncompiled(memory):
    """The uncompiled run a compile that falls back must give bit for bit: with
    the step's time-invariant coefficients computed in place, as a rollout that
    started compiled computes them (``_hoists_invariants``)."""
    before, TE.HOIST_INVARIANTS = TE.HOIST_INVARIANTS, False
    try:
        return _gradient(_solver("cpu", memory=memory, use_compile=False), "cpu")
    finally:
        TE.HOIST_INVARIANTS = before


def _both_on(monkeypatch):
    """Both options asked for on CPU, where combo kernels cannot be (the order of
    the fallback is what is tested here, not Inductor)."""
    monkeypatch.setattr(TE._PropTorchEager, "_resolve_inductor_extras",
                        lambda self: {"cpp_wrapper": True, "combo_kernels": True})


def _same(a, b):
    (out_a, grads_a), (out_b, grads_b) = a, b
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
    for ga, gb in zip(grads_a, grads_b):
        torch.testing.assert_close(ga, gb, rtol=0, atol=0)


@pytest.mark.usefixtures("fresh_failures")
@pytest.mark.parametrize("memory", [None, "bs"], ids=["ckpt", "boundary_saving"])
def test_a_cpp_wrapper_that_does_not_build_is_dropped_for_the_process(monkeypatch, memory):
    _both_on(monkeypatch)
    reference = _uncompiled(memory)
    calls = _spy_compile(monkeypatch, fail=lambda o: o.get("cpp_wrapper", False))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = _solver("cpu", memory=memory)
        result = _gradient(solver, "cpu")
        later = len(calls)
        _gradient(_solver("cpu", memory=memory), "cpu")
    assert solver._backend_impl.use_compile, "compiled without it, not uncompiled"
    assert _extras(calls[0]) == {"cpp_wrapper": True, "combo_kernels": True}
    assert {"combo_kernels": True} in [c["options"] for c in calls[1:later]]
    assert all(c["options"] == {"combo_kernels": True} for c in calls[later:]), \
        "a later propagator leaves the C++ wrapper out at once"
    notes = [w for w in caught if "C++ wrapper" in str(w.message)]
    assert len(notes) == 1 and issubclass(notes[0].category, RuntimeWarning), [str(w.message) for w in caught]
    assert "CppCompileError" in str(notes[0].message) and "-std=c++20" in str(notes[0].message)
    assert "compile_cpp_wrapper=False" in str(notes[0].message)
    assert isinstance(TE._CPP_WRAPPER_FAILED.get("cpu"), CppCompileError)
    _same(result, reference)


@pytest.mark.usefixtures("fresh_failures")
def test_then_combo_kernels_and_only_then_the_compile(monkeypatch):
    _both_on(monkeypatch)
    reference = _uncompiled(None)
    calls = _spy_compile(monkeypatch, fail=lambda o: o.get("cpp_wrapper") or o.get("combo_kernels"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = _solver("cpu")
        result = _gradient(solver, "cpu")
    assert [_extras(c) for c in calls[:3]] == [
        {"cpp_wrapper": True, "combo_kernels": True}, {"combo_kernels": True}, {}]
    assert solver._backend_impl.use_compile
    assert TE._CPP_WRAPPER_FAILED == {}, "not the C++ wrapper's doing: later compiles try it again"
    notes = [str(w.message) for w in caught if "combo_kernels" in str(w.message)]
    assert len(notes) == 1 and "without cpp_wrapper and combo_kernels" in notes[0]
    _same(result, reference)

    calls = _spy_compile(monkeypatch, fail=lambda o: True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = _solver("cpu")
        result = _gradient(solver, "cpu")
    assert [_extras(c) for c in calls] == [
        {"cpp_wrapper": True, "combo_kernels": True}, {"combo_kernels": True}, {}]
    assert solver._backend_impl.use_compile is False, "nothing left to drop: uncompiled, as before"
    assert any("running it uncompiled" in str(w.message) for w in caught)
    _same(result, reference)


# --------------------------------------------------------------------------- #
# Numerics against a compile without them (Inductor, GPU)
# --------------------------------------------------------------------------- #

CASES = [("acoustic", None), ("acoustic", "full"), ("acoustic", "bs"), ("elastic", None), ("elastic", "bs")]
CASE_IDS = ["acoustic_ckpt", "acoustic_full", "acoustic_bs", "elastic_ckpt", "elastic_bs"]


@pytest.fixture
def deterministic(monkeypatch):
    """The first steps of a rollout run uncompiled, on cuDNN convolutions."""
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)


@lru_cache(maxsize=None)
def _compiled(eq, memory, cpp, combo):
    """Record and gradients of a compile with the options asked for: not one
    that dropped them (a C++ wrapper that failed before goes without a word)."""
    torch._dynamo.reset()            # each variant its own compile, under the recompile cap
    assert "cuda" not in TE._CPP_WRAPPER_FAILED
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = _solver("cuda", eq, memory, compile_cpp_wrapper=cpp, compile_combo_kernels=combo)
        out, grads = _gradient(solver, "cuda", eq)
    assert solver._backend_impl.use_compile, "it compiled"
    dropped = [str(w.message) for w in caught if "could not build" in str(w.message)]
    assert dropped == [], dropped
    return out.cpu(), [g.cpu() for g in grads]


@gpu
@needs_toolkit
@pytest.mark.usefixtures("deterministic")
@pytest.mark.parametrize("eq, memory", CASES, ids=CASE_IDS)
def test_cpp_wrapper_keeps_records_and_gradients_bit_for_bit(eq, memory):
    _same(_compiled(eq, memory, True, False), _compiled(eq, memory, False, False))


@gpu
@pytest.mark.usefixtures("deterministic")
@pytest.mark.parametrize("eq, memory", CASES, ids=CASE_IDS)
def test_combo_kernels_keep_records_and_move_gradients_by_round_off(eq, memory):
    (out, grads), (out0, grads0) = _compiled(eq, memory, False, True), _compiled(eq, memory, False, False)
    torch.testing.assert_close(out, out0, rtol=0, atol=0)
    for g, g0 in zip(grads, grads0):
        # float32 round-off accumulated over the steps (measured <= 2e-7 on 1000^2, nt=1000)
        assert _rel(g, g0) <= 1e-5


# --------------------------------------------------------------------------- #
# A C++ compiler that cannot build (GPU, Inductor)
# --------------------------------------------------------------------------- #

FAKE_GXX = """#!/bin/sh
# Answers --version like a g++ and rejects what Inductor compiles, as g++ 9
# rejects the -std=c++20 of Inductor's CPU kernels.
if [ "$1" = "--version" ]; then echo "g++ (stand-in) 9.4.0"; exit 0; fi
echo "g++: error: unrecognized command-line option '-std=c++20'" >&2
exit 1
"""


@gpu
@needs_toolkit
@pytest.mark.usefixtures("deterministic", "fresh_failures")
def test_a_cpp_wrapper_that_does_not_build_compiles_without_it(monkeypatch, tmp_path):
    """A grid of its own (no graph of it compiled before in this process) and a
    fresh Inductor cache, so that the C++ wrapper is really built."""
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    shape = (44, 52)
    torch._dynamo.reset()
    reference = _gradient(_solver("cuda", shape=shape, compile_cpp_wrapper=False), "cuda", shape=shape)

    gxx = tmp_path / "g++"
    gxx.write_text(FAKE_GXX)
    gxx.chmod(gxx.stat().st_mode | stat.S_IXUSR)
    import torch._inductor.config as inductor_config
    real_compile, calls = torch.compile, []
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: calls.append(kw) or real_compile(fn, **kw))
    torch._dynamo.reset()
    # Inductor's own patch, not monkeypatch: pytest >= 9.2 restores an attribute missing
    # from the target's __dict__ by deleting it, which deletes the config entry.
    with inductor_config.patch({"cpp.cxx": (str(gxx),)}), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = _solver("cuda", shape=shape)
        result = _gradient(solver, "cuda", shape=shape)
        later = len(calls)
        _gradient(_solver("cuda", shape=shape, memory="bs"), "cuda", shape=shape)

    assert solver._backend_impl.use_compile, "compiled without it, not uncompiled"
    notes = [str(w.message) for w in caught if "C++ wrapper" in str(w.message)]
    assert len(notes) == 1, [str(w.message) for w in caught]
    assert "-std=c++20" in notes[0] or "CppCompileError" in notes[0], notes[0]
    assert "cuda" in TE._CPP_WRAPPER_FAILED
    assert calls[0]["options"].get("cpp_wrapper") is True
    assert all(not c["options"].get("cpp_wrapper") for c in calls[later:]), \
        "a later propagator leaves the C++ wrapper out at once"
    _same(result, reference)


# --------------------------------------------------------------------------- #
# The default compile on each equation family (GPU, Inductor)
# --------------------------------------------------------------------------- #
# A graph Inductor compiles while autograd runs the backward pass fails out of
# the fallback's reach: the default is run once with gradients on every family
# test_compiled_stencil_slices compiles (there through aot_eager).
from test_compiled_stencil_slices import (EQUATIONS, _assert_matches,  # noqa: E402
                                          _release_compiled_state, _run as _run_family)  # noqa: F401


@gpu
@pytest.mark.usefixtures("deterministic", "fresh_failures", "_release_compiled_state")
@pytest.mark.parametrize("cls,shape,surface", EQUATIONS)
def test_the_default_compile_of_each_family_matches_uncompiled(cls, shape, surface):
    torch._dynamo.reset()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run = _run_family(cls, shape, surface, True)
    dropped = [str(w.message) for w in caught if "could not build" in str(w.message)]
    assert dropped == [], dropped
    _assert_matches(run, _run_family(cls, shape, surface, False))
