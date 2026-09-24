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
"""
import sys

import pytest

from sweep import _jit


@pytest.fixture
def no_gpu(monkeypatch):
    import torch
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


def test_can_build_reports_the_toolkit_problem_when_there_is_a_device(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(_jit, "_find_cuda_home", lambda: None)
    ok, why = _jit.can_build()
    assert not ok and "CUDA toolkit" in why


def test_build_cli_refuses_without_an_arch(monkeypatch, capsys):
    from sweep import build
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    assert build.main(["--no-gpu-required"]) == 2
    assert "TORCH_CUDA_ARCH_LIST" in capsys.readouterr().err


def test_load_signature_carries_the_flag():
    import inspect
    assert "compile_only" in inspect.signature(_jit.load).parameters
    import sweep
    assert "require_gpu" in inspect.signature(sweep.precompile).parameters
