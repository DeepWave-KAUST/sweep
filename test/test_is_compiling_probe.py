"""The compile-only paths hang on one probe, which must work on older torch too.

The slice stencils and the source scatter switch on ``_is_compiling()``. It was
``torch.compiler.is_compiling`` or nothing; torch 2.1 has ``torch.compiler``
without it, so there both paths silently stayed off under ``torch.compile``.
Older torch is reached through ``torch._dynamo.is_compiling``.
"""
import sys
import types

import torch

from conftest import requires_compile
from sweep.operators.torch import _is_compiling, _resolve_is_compiling


@requires_compile
def test_the_probe_is_true_while_tracing_only():
    def f(x):
        return x + 1 if _is_compiling() else x - 1

    x = torch.zeros(3)
    assert torch.equal(f(x), x - 1)
    try:
        assert torch.equal(torch.compile(f, backend="eager", fullgraph=True)(x), x + 1)
    finally:
        torch._dynamo.reset()


def _module(name, monkeypatch, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def test_torch_compiler_is_used_when_it_has_the_probe(monkeypatch):
    def probe():
        return True

    fake = _module("fake_torch_new", monkeypatch, compiler=types.SimpleNamespace(is_compiling=probe))
    assert _resolve_is_compiling(fake) is probe


def test_older_torch_falls_back_to_dynamo(monkeypatch):
    def probe():
        return True

    fake = _module("fake_torch_21", monkeypatch, compiler=types.SimpleNamespace())   # as in 2.1
    _module("fake_torch_21._dynamo", monkeypatch, is_compiling=probe)
    assert _resolve_is_compiling(fake) is probe


def test_without_either_nothing_is_compiling(monkeypatch):
    fake = _module("fake_torch_none", monkeypatch)
    assert _resolve_is_compiling(fake)() is False
