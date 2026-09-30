"""``use_ckpt`` reports the gradient-memory strategy; it does not set it.

It used to be a writable flag, and six places flipped it after construction
while the boundary configuration stayed as it was, so a propagator could hold
two answers to one question. Making it a getter-only property closed that, but
Python's own message for a getter-only property ("property has no setter") does
not say what to write instead, and code predating the change was still
assigning to it.
"""
import pytest

from sweep.equations import Acoustic
from sweep.propagator.options import BoundarySaving
from sweep.propagator.torch import PropTorch


def _prop(**kw):
    return PropTorch(Acoustic(spatial_order=4, device="cpu"), backend="torch",
                     impl="eager", shape=(24, 28), dh=10.0, dt=1e-3, nt=8, abcn=6,
                     B=1, dev="cpu", source_type=["h1"], receiver_type=["h1"], **kw)


def test_assigning_use_ckpt_names_the_replacement():
    prop = _prop()._backend_impl
    with pytest.raises(AttributeError) as exc:
        prop.use_ckpt = True
    message = str(exc.value)
    assert "read-only" in message
    assert "memory_strategy" in message, "the error must say what to write instead"


@pytest.mark.parametrize("strategy", ["full", "boundary", "ckpt"])
def test_memory_strategy_is_the_writable_one(strategy):
    prop = _prop()._backend_impl
    prop.memory_strategy = strategy
    assert prop.memory_strategy == strategy
    assert prop.use_ckpt is (strategy == "ckpt")


def test_a_contradiction_is_refused_where_it_is_written():
    """Two spellings of the mode that disagree fail at construction, so no
    propagator can carry the contradiction to call time."""
    with pytest.raises(ValueError, match="Conflicting gradient-memory"):
        _prop(memory=BoundarySaving(storage="cpu"), use_ckpt=True)
