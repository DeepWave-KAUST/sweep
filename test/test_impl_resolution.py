"""Which impl ``PropTorch`` resolves to, per device and per build.

The compiled backend runs on CUDA only, in every build: the default ctypes
layer over the prebuilt core, the ``SWEEP_JIT_FULL=1`` pybind shim and an
AOT-built extension all reach the same CUDA core.  (The compiled CPU engine the
last two used to carry is gone: its gradients disagreed with eager and with the
CUDA core.)  So a CPU propagator resolves ``impl='auto'`` to eager quietly and
an explicit ``impl='c'`` to eager with a warning, whatever the build; a CUDA
propagator gets the compiled path.

No kernel runs here: the build flags and the binding probe are monkeypatched,
and only the resolver is called.
"""
import warnings

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


EQ = _Eq("acoustic2d")
BUILDS = ("default", "jit_full", "aot")


@pytest.fixture
def build(monkeypatch):
    """Pin the build: binding available, then pick default / jit_full / aot."""
    monkeypatch.delenv("SWEEP_JIT_FULL", raising=False)
    monkeypatch.setattr(ptorch, "_compiled_binding_available", lambda: True)

    def _set(kind):
        assert kind in BUILDS
        monkeypatch.setattr(jit, "jit_full", lambda: kind == "jit_full")
        monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: kind == "aot")
    return _set


def _resolve(impl, device, equation=EQ):
    """Resolve like PropTorch does; return (impl, [warning messages])."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _, got = ptorch._normalize_backend_impl("torch", impl, equation=equation, device=device)
    return got, [str(w.message) for w in rec if issubclass(w.category, UserWarning)]


# (impl, device) -> (resolved impl, warned?), the same in every build
MATRIX = {
    (None, "cpu"): ("eager", False),
    ("c", "cpu"): ("eager", True),
    (None, "cuda"): ("c", False),
    ("c", "cuda"): ("c", False),
}


@pytest.mark.parametrize("kind", BUILDS)
@pytest.mark.parametrize("key", list(MATRIX), ids=lambda k: f"{k[0] or 'auto'}-{k[1]}")
def test_matrix(build, kind, key):
    impl, device = key
    want, want_warn = MATRIX[key]
    build(kind)
    got, msgs = _resolve(impl, device)
    assert got == want
    assert bool(msgs) == want_warn, msgs


@pytest.mark.parametrize("kind", BUILDS)
def test_explicit_c_on_cpu_says_cuda_only(build, kind):
    build(kind)
    got, msgs = _resolve("c", "cpu")
    assert got == "eager"
    (msg,) = msgs
    assert "runs on CUDA only" in msg and "cpu" in msg


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
def test_indexed_cuda_string_is_cuda(build, device):
    """``dev='cuda:0'`` is a CUDA device; a string compare against 'cuda' is not."""
    import torch
    build("default")
    assert _resolve(None, device) == ("c", [])
    assert _resolve(None, torch.device(device)) == ("c", [])


@pytest.mark.parametrize("kind", BUILDS)
def test_other_devices_never_get_the_compiled_path(build, kind):
    """Nothing but CUDA: not the host, not mps/xla/meta."""
    build(kind)
    assert _resolve(None, "meta")[0] == "eager"
    got, msgs = _resolve("c", "meta")
    assert got == "eager" and len(msgs) == 1


def test_no_device_keeps_the_compiled_path(build):
    build("default")
    assert _resolve(None, None) == ("c", [])


@pytest.mark.parametrize("kind", BUILDS)
def test_missing_binding_still_demotes_everywhere(build, monkeypatch, kind):
    build(kind)
    monkeypatch.setattr(ptorch, "_compiled_binding_available", lambda: False)
    for device in ("cpu", "cuda"):
        assert _resolve(None, device)[0] == "eager"
        got, msgs = _resolve("c", device)
        assert got == "eager" and len(msgs) == 1
