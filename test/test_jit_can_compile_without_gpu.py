"""Compiling the backend must not require a card.

``can_build()`` gated the COMPILE on a RUNTIME predicate: no visible device, no
build -- even with ``TORCH_CUDA_ARCH_LIST`` naming the target explicitly, which
is exactly how wheels are cross-built. The cost is not theoretical: pre-building
sweep on a cluster then has to occupy a GPU allocation to run nvcc, and on ibex
that was a 4.6-hour GPU-partition queue for work that needs no GPU.

Split: ``can_compile()`` = torch + nvcc + an architecture to target;
``can_build()`` = that, plus a device. ``is_torch_binding_available()`` keeps
asking ``can_build()``, because a machine that can only cross-compile cannot
serve ``impl='c'``.

Every test here is about the nvcc path, so the shipped-core lookup is pinned
away: on a tree where ``python -m sweep.build --core`` has run, ``src/sweep/lib``
holds a core that fits and ``can_compile()`` would say "ok" without ever
looking at nvcc or the arch.
"""
import sys

import pytest

from sweep import _jit


@pytest.fixture(autouse=True)
def _no_compile(monkeypatch):
    """Nothing here may compile: every test is about the *answer* of the nvcc
    path, never the path itself.  Staging is the first step of any core build
    and the shim compile is torch's ``cpp_extension.load``, so both are
    refused, together with the build entries behind them."""
    from torch.utils import cpp_extension

    def refuse(*_a, **_k):
        raise AssertionError("a compile was attempted; these tests only ask, never build")

    monkeypatch.setattr(cpp_extension, "load", refuse)
    for entry in ("_stage", "_build_core"):        # _jit.load itself stays: a test inspects its signature
        monkeypatch.setattr(_jit, entry, refuse)
    yield


@pytest.fixture
def no_shipped_core(monkeypatch, tmp_path):
    monkeypatch.setattr(_jit, "_LIB", tmp_path / "lib")
    monkeypatch.delenv("SWEEP_CORE", raising=False)
    # ...and no cached local core either: the host's ~/.cache/torch_extensions
    # must not answer "a core is at hand" for tests about the nvcc path.
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "ext"))
    return monkeypatch


@pytest.fixture
def no_gpu(no_shipped_core):
    import torch
    monkeypatch = no_shipped_core
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(_jit, "_find_cuda_home", lambda: "/usr/local/cuda")
    return monkeypatch


def test_can_compile_without_a_device_when_the_arch_is_named(no_gpu):
    no_gpu.setenv("TORCH_CUDA_ARCH_LIST", "7.0")
    ok, why = _jit.can_compile()
    assert ok, why


def test_can_compile_refuses_when_there_is_no_target_arch(no_gpu):
    no_gpu.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    ok, why = _jit.can_compile()
    assert not ok
    assert "TORCH_CUDA_ARCH_LIST" in why, why


def test_can_build_still_requires_a_device(no_gpu):
    """Cross-compiling does not make impl='c' runnable here."""
    no_gpu.setenv("TORCH_CUDA_ARCH_LIST", "7.0")
    ok, why = _jit.can_build()
    assert not ok
    assert "no CUDA GPU is visible" in why


def test_binding_availability_still_follows_can_build(no_gpu, monkeypatch):
    import sweep
    monkeypatch.setattr(sweep, "_prebuilt_binding_present", lambda: False)
    no_gpu.setenv("TORCH_CUDA_ARCH_LIST", "7.0")
    assert sweep.is_torch_binding_available() is False


def test_can_build_reports_the_toolkit_problem_when_there_is_a_device(no_shipped_core):
    import torch
    monkeypatch = no_shipped_core
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)
    ok, why = _jit.can_build()
    assert not ok and "CUDA toolkit" in why


@pytest.mark.parametrize("cuda, major", [("12.8", "12"), ("13.0", "13")])
def test_toolkit_message_names_the_torch_cuda_major(no_gpu, cuda, major):
    """Two shipped cores, cu12 and cu13, and a local build wants the nvcc of
    the same major as torch: the refusal says which one, not a bare
    "matching your torch"."""
    import torch
    no_gpu.setattr(torch.version, "cuda", cuda)
    no_gpu.setenv("TORCH_CUDA_ARCH_LIST", "8.9")
    no_gpu.setattr(_jit, "_find_cuda_home", lambda: None)
    ok, why = _jit.can_compile()
    assert not ok and "CUDA toolkit" in why
    assert f"an nvcc of CUDA {major}" in why, why


def test_build_cli_refuses_without_an_arch(no_shipped_core, capsys):
    """The arch gate exists for the nvcc path; a fitting shipped core would
    skip it, hence the pin."""
    from sweep import build
    monkeypatch = no_shipped_core
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    assert build.main(["--no-gpu-required"]) == 2
    assert "TORCH_CUDA_ARCH_LIST" in capsys.readouterr().err


def test_load_signature_carries_the_flag():
    import inspect
    assert "compile_only" in inspect.signature(_jit.load).parameters
    import sweep
    assert "require_gpu" in inspect.signature(sweep.precompile).parameters
