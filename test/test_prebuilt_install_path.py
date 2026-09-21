"""The ahead-of-time install path, which nothing else exercises.

`SWEEP_BUILD_CUDA=1 pip install -v ".[cuda]" --no-build-isolation` is what
README.md and docs/getting-started/installation.md tell people to run from
source, and it produces an artifact no test had ever looked at:

  * the arch list it targets -- a fixed default silently overrode torch's own
    detection, so the build on a non-Volta machine carried sm_70 code only;
  * what `sweep.precompile()` does afterwards -- the command the same documents
    publish -- when `sweep._C` is that compiled extension rather than the JIT
    shim.

Both are checked here without compiling anything.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import build_config  # noqa: E402


@pytest.fixture()
def clean_arch_env(monkeypatch):
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.delenv("SWEEP_CUDA_ARCH_LIST", raising=False)


def _fake_torch(monkeypatch, *, available: bool, count: int = 1):
    mod = types.ModuleType("torch")
    mod.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: count,
    )
    monkeypatch.setitem(sys.modules, "torch", mod)


# --------------------------------------------------------------------------- #
# arch list
# --------------------------------------------------------------------------- #

def test_a_visible_gpu_is_left_to_torch(clean_arch_env, monkeypatch):
    """The regression: a fixed default here beat torch's detection, so a build
    on an Ada card produced an sm_70-only .so that would not run on it."""
    _fake_torch(monkeypatch, available=True, count=1)
    build_config.configure_cuda_arch_list()
    assert "TORCH_CUDA_ARCH_LIST" not in os.environ


def test_no_gpu_falls_back_and_the_fallback_carries_ptx(clean_arch_env, monkeypatch):
    """A login node cannot be detected from, so a target is needed -- but one
    without PTX cannot run anywhere except the exact architecture named."""
    _fake_torch(monkeypatch, available=False, count=0)
    build_config.configure_cuda_arch_list()
    got = os.environ["TORCH_CUDA_ARCH_LIST"]
    assert got == build_config.FALLBACK_CUDA_ARCH_LIST
    assert got.endswith("+PTX"), got


def test_no_torch_at_all_still_falls_back(clean_arch_env, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)   # import torch -> ImportError
    build_config.configure_cuda_arch_list()
    assert os.environ["TORCH_CUDA_ARCH_LIST"].endswith("+PTX")


def test_an_explicit_request_always_wins(clean_arch_env, monkeypatch):
    _fake_torch(monkeypatch, available=True, count=1)
    monkeypatch.setenv("SWEEP_CUDA_ARCH_LIST", "8.9")
    build_config.configure_cuda_arch_list()
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "8.9"


def test_a_preset_torch_variable_is_not_touched(clean_arch_env, monkeypatch):
    _fake_torch(monkeypatch, available=False, count=0)
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0")
    build_config.configure_cuda_arch_list()
    assert os.environ["TORCH_CUDA_ARCH_LIST"] == "9.0"


# --------------------------------------------------------------------------- #
# precompile() against a prebuilt extension
# --------------------------------------------------------------------------- #

def _fake_compiled_C(monkeypatch):
    """`sweep._C` as the COMPILED module: real symbols, and no _load."""
    import sweep

    mod = types.ModuleType("sweep._C")
    mod.acoustic2d_forward = object()
    assert not hasattr(mod, "_load")
    monkeypatch.setitem(sys.modules, "sweep._C", mod)
    # sys.modules alone is not enough: `import sweep._C` binds the attribute on
    # the package, and once it is cached the fake is bypassed.
    monkeypatch.setattr(sweep, "_C", mod, raising=False)
    return mod


def test_precompile_is_a_noop_on_a_prebuilt_extension(monkeypatch):
    """The command the docs publish must not raise on the install the docs
    publish. A compiled sweep._C has no _load, and calling one is an
    AttributeError."""
    import sweep

    _fake_compiled_C(monkeypatch)
    assert sweep.precompile() is True


def test_precompile_still_compiles_when_the_jit_shim_is_in_place(monkeypatch):
    """The other half: with the JIT shim, precompile must actually call it --
    otherwise this fix would turn the documented warm-up into a no-op."""
    import sweep

    called = []
    mod = types.ModuleType("sweep._C")
    mod._load = lambda **kw: called.append(kw)
    monkeypatch.setitem(sys.modules, "sweep._C", mod)
    monkeypatch.setattr(sweep, "_C", mod, raising=False)

    assert sweep.precompile() is True
    assert called == [{"compile_only": False}], called

    # ...and that require_gpu actually reaches the loader, rather than being
    # accepted and dropped.
    called.clear()
    assert sweep.precompile(require_gpu=False) is True
    assert called == [{"compile_only": True}], called
