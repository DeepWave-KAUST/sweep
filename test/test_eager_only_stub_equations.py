"""An equation that defines ``_C`` only to refuse impl='c' is eager-only.

``AcousticCurvilinear`` and ``ElasticCurvilinear`` have no CUDA kernels. Their
``_C()`` exists to refuse an explicit ``impl='c'`` with a clear message, but
``supports_torch_binding()`` was ``callable(cls._C)``, so the stub counted as a
compiled binding. The CLI and ``torch_binding_supported_equations()`` then
listed both as compiled, and ``impl=None`` on a CUDA device resolved to ``'c'``
and raised ``NotImplementedError`` at construction, instead of taking the eager
path every other eager-only equation gets.
"""
import pytest

import sweep.equations as E
from sweep.equations import Acoustic, AcousticCurvilinear, ElasticCurvilinear
from sweep.propagator import torch as ptorch

STUBS = (AcousticCurvilinear, ElasticCurvilinear)


@pytest.fixture()
def binding(monkeypatch):
    """Pretend the compiled core is usable, so only the equation decides."""
    monkeypatch.setattr(ptorch, "_compiled_binding_available", lambda: True)


@pytest.mark.parametrize("cls", STUBS, ids=lambda c: c.__name__)
def test_a_refusing_stub_is_not_a_compiled_binding(cls):
    assert not cls.supports_torch_binding()
    assert cls.__name__ not in E.torch_binding_supported_equations()


def test_every_class_with_kernels_still_reports_its_binding():
    """The control: C_NAME is what gives a class kernels, and all of those keep
    reporting a binding."""
    compiled = [name for name, cls in E.equation_classes().items() if cls.C_NAME]
    assert compiled
    assert all(E.equation_classes()[name].supports_torch_binding() for name in compiled)


@pytest.mark.parametrize("cls", STUBS, ids=lambda c: c.__name__)
def test_auto_on_cuda_picks_eager_for_a_stub(binding, cls):
    eq = cls(device="cpu")
    assert ptorch._resolve_impl_with_fallback(
        "auto", explicit=False, equation=eq, device="cuda") == "eager"


def test_auto_on_cuda_still_picks_c_for_a_compiled_equation(binding):
    assert ptorch._resolve_impl_with_fallback(
        "auto", explicit=False, equation=Acoustic(device="cpu"), device="cuda") == "c"


@pytest.mark.parametrize("cls", STUBS, ids=lambda c: c.__name__)
def test_an_explicit_c_for_a_stub_warns_and_runs_eager(binding, cls):
    """Like any equation without kernels: the slowdown is visible, not fatal."""
    with pytest.warns(UserWarning):
        impl = ptorch._resolve_impl_with_fallback(
            "c", explicit=True, equation=cls(device="cpu"), device="cuda")
    assert impl == "eager"
