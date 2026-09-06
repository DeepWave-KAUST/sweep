"""``requires_binding`` must not turn a build failure into a green skip.

Three files used to probe the extension as
``try: import sweep._C as _C; return hasattr(_C, "sym") except Exception:
return False``. ``sweep._C`` is a lazy shim whose attribute access triggers the
JIT, so when nvcc failed torch raised, the ``except`` swallowed it, the
module-level ``skipif`` fired, and 38 collected items reported green on a build
that does not exist.
"""
import sys
import types

import pytest

from conftest import binding_status, requires_binding


@pytest.fixture()
def cuda_present(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def _fake_binding(monkeypatch, *, available=True, symbols=()):
    import sweep
    monkeypatch.setattr(sweep, "is_torch_binding_available", lambda: available)
    module = types.ModuleType("sweep._C")
    for s in symbols:
        setattr(module, s, object())
    monkeypatch.setitem(sys.modules, "sweep._C", module)


def test_a_missing_binding_skips(cuda_present, monkeypatch):
    _fake_binding(monkeypatch, available=False)
    ok, reason = binding_status("anything")
    assert not ok and "compiled sweep._C" in reason


def test_a_present_binding_runs(cuda_present, monkeypatch):
    _fake_binding(monkeypatch, symbols=("acoustic2d_forward",))
    assert binding_status("acoustic2d_forward") == (True, "")


def test_a_stale_build_raises_instead_of_skipping(cuda_present, monkeypatch):
    """The extension is there but the symbol is not: that is a stale or
    partial build, which a skip would hide."""
    _fake_binding(monkeypatch, symbols=("acoustic2d_forward",))
    with pytest.raises(AssertionError) as exc:
        binding_status("elastic_tti_sg3d_forward")
    assert "stale" in str(exc.value) and "elastic_tti_sg3d_forward" in str(exc.value)


def test_no_cuda_skips_with_a_reason(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    mark = requires_binding("anything")
    assert mark.kwargs["reason"] == "CUDA device required"


def test_the_strictness_switch_turns_the_skip_into_an_error(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("SWEEP_TEST_REQUIRE_CUDA", "1")
    with pytest.raises(RuntimeError, match="SWEEP_TEST_REQUIRE_CUDA=1"):
        requires_binding("anything")
