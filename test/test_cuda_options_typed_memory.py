"""A strategy inside ``cuda_options`` must reach the impl='c' backend.

``CUDAOptions(memory=...)`` is one of the two documented places for a memory
strategy. The typed spellings (``Full()`` / ``BoundarySaving()`` / ``Ckpt()``)
did not arrive from there: they keep ``strategy`` in a ClassVar, which
``options_to_dict()`` drops, so the layers below read no strategy at all.
``Ckpt()`` and ``Full()`` then clashed with the ``'boundary'`` default
(ValueError at construction), and ``BoundarySaving(storage='cpu',
pinned_memory=True)`` quietly built the default GPU ring. The same request
passed as ``memory=`` always worked.

The CPU tests replace the compiled backend with a recorder: the question is
which kwargs reach it, not what the kernels do with them.
"""
import pytest
import torch

from conftest import requires_binding
from sweep.core import arguments as _args
from sweep.equations import Acoustic
from sweep.propagator import _c, torch as ptorch
from sweep.propagator.options import (
    BoundaryOptions, BoundarySaving, CUDAOptions, Ckpt, Full, MemoryOptions)
from sweep.propagator.torch import PropTorch


class _Recorder(torch.nn.Module):
    """Stands in for ``_CompiledPropagator``: keeps the init kwargs and exposes
    the two flags PropTorch checks the build against."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.use_ckpt = kwargs.get("use_ckpt")
        self.boundary_saving_config = kwargs.get("boundary_saving_config")


@pytest.fixture()
def compiled(monkeypatch):
    """Keep impl='c' on a CPU box and record what the backend is built with."""
    monkeypatch.setattr(ptorch, "_resolve_impl_with_fallback",
                        lambda impl, **kw: "c" if impl in ("c", "auto") else impl)
    monkeypatch.setattr(_c, "_CompiledPropagator", _Recorder)


@pytest.fixture()
def fresh_warnings():
    """Deprecation warnings fire once per process per spelling (see
    test_memory_legacy_spellings.py); reset so this test sees its own."""
    _args._WARNED_SPELLINGS.clear()
    yield
    _args._WARNED_SPELLINGS.clear()


def _prop(**kw):
    return PropTorch(Acoustic(spatial_order=4, device="cpu"), backend="torch",
                     impl="c", shape=(24, 28), dh=10.0, dt=1e-3, nt=8, abcn=6, **kw)


def _memory_kwargs(prop):
    kw = prop._backend_impl.kwargs
    return {k: v for k, v in kw.items()
            if k in ("use_ckpt", "boundary_saving_config") or k.startswith("ckpt_")}


STRATEGIES = [
    BoundarySaving(storage="cpu", pinned_memory=True, transfer_interval=4),
    BoundarySaving(storage="gpu", storage_dtype="bf16"),
    Ckpt(mode="chunk", chunks=7),
    Full(),
]


@pytest.mark.parametrize("strategy", STRATEGIES, ids=lambda s: repr(s)[:40])
def test_cuda_options_memory_builds_what_memory_builds(compiled, strategy):
    via_memory = _memory_kwargs(_prop(memory=strategy))
    via_options = _memory_kwargs(_prop(cuda_options=CUDAOptions(memory=strategy)))
    assert via_options == via_memory


def test_cpu_storage_is_not_turned_into_the_gpu_ring(compiled):
    cfg = _prop(cuda_options=CUDAOptions(
        memory=BoundarySaving(storage="cpu", pinned_memory=True)))._backend_impl.kwargs[
        "boundary_saving_config"]
    assert cfg["enabled"] is True
    assert cfg["storage"] == "cpu"
    assert cfg["pinned_memory"] is True


@pytest.mark.parametrize("strategy, expected", [(Ckpt(), "ckpt"), (Full(), "full")])
def test_ckpt_and_full_inside_cuda_options_construct(compiled, strategy, expected):
    assert _prop(cuda_options=CUDAOptions(memory=strategy)).memory_strategy == expected


def test_both_places_at_once_is_still_refused(compiled):
    with pytest.raises(ValueError, match="not both"):
        _prop(memory=Full(), cuda_options=CUDAOptions(memory=Full()))


def test_the_legacy_spelling_inside_cuda_options_still_reads_and_warns_once(compiled, fresh_warnings):
    legacy = MemoryOptions(strategy="boundary", boundary=BoundaryOptions(storage="cpu"))
    with pytest.warns(DeprecationWarning) as record:
        prop = _prop(cuda_options=CUDAOptions(memory=legacy))
    assert sum(issubclass(w.category, DeprecationWarning) for w in record) == 1
    assert prop._backend_impl.kwargs["boundary_saving_config"]["storage"] == "cpu"


def test_cuda_options_with_an_explicit_eager_impl_is_still_an_error():
    with pytest.raises(ValueError, match="cuda_options can only be used with impl='c'"):
        PropTorch(Acoustic(spatial_order=4, device="cpu"), backend="torch", impl="eager",
                  shape=(24, 28), dh=10.0, dt=1e-3, nt=8, abcn=6,
                  cuda_options=CUDAOptions(memory=Full()))


@requires_binding()
def test_the_real_compiled_backend_gets_the_cpu_ring():
    dev = torch.device("cuda")
    prop = PropTorch(Acoustic(spatial_order=4, device=dev), backend="torch", impl="c",
                     shape=(24, 28), dh=10.0, dt=1e-3, nt=8, abcn=6, device=dev,
                     cuda_options=CUDAOptions(
                         memory=BoundarySaving(storage="cpu", pinned_memory=True)))
    assert prop.impl == "c"
    cfg = prop._backend_impl.boundary_saving_config
    assert (cfg["enabled"], cfg["storage"], cfg["pinned_memory"]) == (True, "cpu", True)
