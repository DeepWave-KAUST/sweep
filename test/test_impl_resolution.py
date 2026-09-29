"""Which impl ``PropTorch`` resolves to, per device and per build.

The compiled backend runs on CUDA everywhere.  It runs on CPU only when
``sweep._C`` carries the compiled CPU engine (``csrc/cpu/**``): under
``SWEEP_JIT_FULL=1`` (the pybind developer shim) or with an AOT-built extension
(``SWEEP_BUILD_CUDA=1`` install).  The default path -- the ctypes layer over
the prebuilt CUDA core -- has no CPU engine, so there a CPU propagator's
``impl='auto'`` is eager and an explicit ``impl='c'`` is demoted with a warning.
Even where the CPU engine exists, ``'auto'`` on a CPU stays eager: the engine is
a developer path with known accuracy gaps, reached only by an explicit
``impl='c'``.

The regression this guards: the device gate once treated every non-CUDA
device alike, so on the JIT_FULL path an explicit ``impl='c'`` on CPU was
silently swapped for eager and the c-cpu column of
``backend_gradient_matrix.py`` compared eager against itself.

No kernel runs here: the build flags and the binding probe are monkeypatched,
and only the resolver is called.
"""
import re
import warnings
from pathlib import Path

import pytest

import sweep
from sweep.backend.c import jit
from sweep.propagator import torch as ptorch


class _Eq:
    """Minimal equation: a ``_C`` hook and a ``C_NAME``."""

    def __init__(self, c_name):
        self.C_NAME = c_name

    def _C(self):  # pragma: no cover - never called by the resolver
        raise AssertionError("the resolver must not load kernels")


CPU_EQ = _Eq("acoustic2d")            # has csrc/cpu/equations/acoustic2d/
CUDA_ONLY_EQ = _Eq("elastic_tti_2nd2d")  # bound straight to the CUDA core


@pytest.fixture
def build(monkeypatch):
    """Pin the build: binding available, then pick default / jit_full / aot."""
    monkeypatch.delenv("SWEEP_JIT_FULL", raising=False)
    monkeypatch.setattr(ptorch, "_compiled_binding_available", lambda: True)

    def _set(kind):
        assert kind in ("default", "jit_full", "aot")
        monkeypatch.setattr(jit, "jit_full", lambda: kind == "jit_full")
        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: kind == "aot")
    return _set


def _resolve(impl, device, equation=CPU_EQ):
    """Resolve like PropTorch does; return (impl, [warning messages])."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _, got = ptorch._normalize_backend_impl("torch", impl, equation=equation, device=device)
    return got, [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


# (build, impl, device) -> (resolved impl, warned?)
MATRIX = {
    ("default", None, "cpu"): ("eager", False),
    ("default", "c", "cpu"): ("eager", True),
    ("default", None, "cuda"): ("c", False),
    ("default", "c", "cuda"): ("c", False),
    ("jit_full", None, "cpu"): ("eager", False),   # auto never picks the CPU engine
    ("jit_full", "c", "cpu"): ("c", False),
    ("jit_full", None, "cuda"): ("c", False),
    ("jit_full", "c", "cuda"): ("c", False),
    ("aot", None, "cpu"): ("eager", False),        # auto never picks the CPU engine
    ("aot", "c", "cpu"): ("c", False),
    ("aot", None, "cuda"): ("c", False),
    ("aot", "c", "cuda"): ("c", False),
}


@pytest.mark.parametrize("key", list(MATRIX), ids=lambda k: f"{k[0]}-{k[1] or 'auto'}-{k[2]}")
def test_matrix(build, key):
    kind, impl, device = key
    want, want_warn = MATRIX[key]
    build(kind)
    got, msgs = _resolve(impl, device)
    assert got == want
    assert bool(msgs) == want_warn, msgs


def test_default_path_warning_names_both_ways_to_the_cpu_engine(build):
    build("default")
    got, msgs = _resolve("c", "cpu")
    assert got == "eager"
    (msg,) = msgs
    assert "no CPU engine" in msg
    assert "SWEEP_JIT_FULL=1" in msg and "SWEEP_BUILD_CUDA=1" in msg


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
def test_indexed_cuda_string_is_cuda(build, device):
    """``dev='cuda:0'`` is a CUDA device; a string compare against 'cuda' is not."""
    import torch
    build("default")
    assert _resolve(None, device) == ("c", [])
    assert _resolve(None, torch.device(device)) == ("c", [])


@pytest.mark.parametrize("kind", ["default", "jit_full", "aot"])
def test_other_devices_never_get_the_compiled_path(build, kind):
    """The CPU engine serves host tensors only, not mps/xla/..."""
    build(kind)
    assert _resolve(None, "meta")[0] == "eager"
    got, msgs = _resolve("c", "meta")
    assert got == "eager" and len(msgs) == 1


def test_no_device_keeps_the_compiled_path(build):
    build("default")
    assert _resolve(None, None) == ("c", [])


@pytest.mark.parametrize("kind", ["jit_full", "aot"])
def test_cuda_only_equation_on_cpu_is_eager_even_with_the_cpu_engine(build, kind):
    """The CPU engine has no kernel for it: 'c' would hand the CUDA core host
    tensors.  auto -> eager quietly; explicit c -> eager with the reason."""
    build(kind)
    assert _resolve(None, "cpu", CUDA_ONLY_EQ) == ("eager", [])
    got, msgs = _resolve("c", "cpu", CUDA_ONLY_EQ)
    assert got == "eager"
    (msg,) = msgs
    assert "no kernel in the compiled CPU engine" in msg and "elastic_tti_2nd2d" in msg
    # ...and on CUDA it is the compiled path as usual.
    assert _resolve("c", "cuda", CUDA_ONLY_EQ) == ("c", [])


@pytest.mark.parametrize("kind", ["default", "jit_full", "aot"])
def test_missing_binding_still_demotes_everywhere(build, monkeypatch, kind):
    build(kind)
    monkeypatch.setattr(ptorch, "_compiled_binding_available", lambda: False)
    for device in ("cpu", "cuda"):
        assert _resolve(None, device)[0] == "eager"
        got, msgs = _resolve("c", device)
        assert got == "eager" and len(msgs) == 1


def test_real_env_flag_reaches_the_gate(monkeypatch):
    """The JIT_FULL leg of the matrix patches ``jit.jit_full``; make sure the
    environment variable is what that function reads."""
    monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: False)
    monkeypatch.setenv("SWEEP_JIT_FULL", "1")
    assert ptorch._cpu_engine_available() is True
    monkeypatch.delenv("SWEEP_JIT_FULL")
    assert ptorch._cpu_engine_available() is False


def _cpu_dispatched_c_names():
    """C_NAMEs whose binding routes host tensors to the CPU engine
    (``dispatch_forward`` in bindings/module.cpp)."""
    src = (jit._CSRC / "bindings" / "module.cpp").read_text()
    return set(re.findall(r'm\.def\("([a-z0-9_]+)_forward",\s*wrap_forward\(dispatch_forward\(', src))


def _all_c_names():
    import inspect
    import sweep.equations as eqs
    names = set()
    for _, cls in inspect.getmembers(eqs, inspect.isclass):
        name = getattr(cls, "C_NAME", None)
        if isinstance(name, str) and name:
            names.add(name)
    return names


def test_cpu_kernel_lookup_matches_the_binding():
    """``_equation_has_cpu_kernel`` reads csrc/cpu/equations/<C_NAME>/; the
    binding decides with dispatch_forward.  They must name the same set."""
    dispatched = _cpu_dispatched_c_names()
    assert "acoustic2d" in dispatched and "elastic_tti_2nd2d" not in dispatched
    names = _all_c_names()
    assert dispatched <= names, dispatched - names
    has_cpu = {n for n in names if ptorch._equation_has_cpu_kernel(_Eq(n))}
    assert has_cpu == dispatched
    assert names - has_cpu, "expected some CUDA-only equations"
