"""A demoted ``impl='c'`` must not silently lose the memory strategy.

When the compiled binding is unavailable (or the equation has no ``_C``),
``impl='c'`` falls back to eager. The cuda-only knobs are dropped with it --
correctly -- but the gradient-memory STRATEGY is not a cuda-only knob: it just
travels inside ``cuda_options``. Dropping the object took the request with it,
so a caller who asked for boundary saving got the eager default, checkpointing,
and nothing said so.

CPU + eager: this is about which request survives construction.
"""
import pytest

from sweep.equations import Acoustic
from sweep.propagator import torch as ptorch
from sweep.propagator.options import BoundarySaving, CUDAOptions, Ckpt, Full
from sweep.propagator.torch import PropTorch


@pytest.fixture()
def demoted(monkeypatch):
    """Force the fallback: asked for 'c', gets 'eager'."""
    monkeypatch.setattr(ptorch, "_compiled_binding_available", lambda: False)


def _prop(**kw):
    return PropTorch(Acoustic(spatial_order=4, device="cpu"), backend="torch",
                     impl="c", shape=(24, 28), dh=10.0, dt=1e-3, nt=8, abcn=6,
                     B=1, dev="cpu", source_type=["h1"], receiver_type=["h1"], **kw)


def test_the_fallback_really_happens(demoted):
    with pytest.warns(UserWarning, match="falling back"):
        prop = _prop()
    assert prop.impl == "eager"


@pytest.mark.parametrize("strategy,expected", [
    (BoundarySaving(), "boundary"),
    (Ckpt(), "ckpt"),
    (Full(), "full"),
])
def test_a_strategy_inside_cuda_options_survives(demoted, strategy, expected):
    """All three survive, including the one the eager backend implements
    differently: the caller asked for a strategy, not for a backend."""
    with pytest.warns(UserWarning):
        prop = _prop(cuda_options=CUDAOptions(memory=strategy))
    assert prop._backend_impl.memory_strategy == expected, (
        "the request travelled inside cuda_options and was dropped with it"
    )


def test_the_resolved_strategy_and_the_built_backend_agree(demoted):
    """PropTorch asserts these match at construction; carrying the request into
    only one of the two would trip that assertion rather than silently differ."""
    with pytest.warns(UserWarning):
        prop = _prop(cuda_options=CUDAOptions(memory=Ckpt()))
    assert prop._backend_impl.memory_strategy == "ckpt"
    assert prop._backend_impl.use_ckpt is True


def test_an_explicit_memory_still_wins(demoted):
    """``memory=`` is the caller's more direct statement of intent."""
    with pytest.warns(UserWarning):
        prop = _prop(memory=Full(), cuda_options=CUDAOptions(memory=Ckpt()))
    assert prop._backend_impl.memory_strategy == "full"


def test_no_strategy_anywhere_leaves_the_eager_default(demoted):
    with pytest.warns(UserWarning):
        prop = _prop(cuda_options=CUDAOptions())
    assert prop._backend_impl.memory_strategy == "ckpt"
