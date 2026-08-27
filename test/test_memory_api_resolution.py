"""One resolution decides the gradient-memory strategy, for both impls.

Before this, `impl='eager'` called `resolve_memory_strategy` before building the
backend, while `impl='c'` did not call it at all: it constructed the backend and
then read the strategy back out of whichever flags the constructor happened to
default to. The two agreed, but they were two decisions, and downstream code
still carries a runtime `inspect.signature` probe and a comment recording what
happens when they stop agreeing ("thought I measured bs, actually ran ckpt").
"""
from __future__ import annotations

import pytest
import torch

from sweep.propagator.options import (
    BoundaryOptions,
    BoundarySaving,
    Ckpt,
    Full,
    MemoryOptions,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _build(**kw):
    from sweep.equations.acoustic import Acoustic
    from sweep.propagator.torch import PropTorch
    dev = torch.device("cuda")
    eq = Acoustic(spatial_order=4, device=dev, backend="torch")
    return PropTorch(eq, backend="torch", shape=(48, 56), dev=dev, dh=10.0, dt=1e-3,
                     source_type=["p"], receiver_type=["p"], abcn=20, nt=50, B=1,
                     allow_growth=True, **kw)


def _bs_for(impl):
    # disk staging is impl='c' only; eager keeps the ring on gpu or host
    return (BoundarySaving(storage="cpu", transfer_interval=10) if impl == "c"
            else BoundarySaving(storage="gpu"))


@pytest.mark.parametrize("impl", ["c", "eager"])
@pytest.mark.parametrize("name,want", [("full", "full"), ("boundary", "boundary"),
                                       ("ckpt", "ckpt")])
def test_the_typed_spelling_resolves_on_both_impls(impl, name, want):
    memory = {"full": Full(), "boundary": _bs_for(impl),
              "ckpt": Ckpt(mode="chunk", chunks=10)}[name]
    assert _build(impl=impl, memory=memory).memory_strategy == want


@pytest.mark.parametrize("spelling", ["typed", "legacy_object", "flat_dict", "nested_dict"])
def test_every_spelling_of_the_same_request_agrees(spelling):
    """The nested dict is the shape already written into stored experiment
    YAML, so it has to keep reading."""
    memory = {
        "typed": BoundarySaving(storage="cpu", transfer_interval=10),
        "legacy_object": MemoryOptions(strategy="boundary",
                                       boundary=BoundaryOptions(storage="cpu",
                                                                transfer_interval=10)),
        "flat_dict": {"kind": "boundary", "storage": "cpu", "transfer_interval": 10},
        "nested_dict": {"strategy": "boundary",
                        "boundary": {"storage": "cpu", "transfer_interval": 10}},
    }[spelling]
    prop = _build(impl="c", memory=memory)
    assert prop.memory_strategy == "boundary"
    # Resolving the STRATEGY correctly while dropping the PARAMETERS would be a
    # silent regression, so check one that has to travel the whole way down.
    assert prop.transfer_interval == 10


@pytest.mark.parametrize("impl,want", [("c", "boundary"), ("eager", "ckpt")])
def test_the_backend_default_applies_when_nothing_is_asked(impl, want):
    assert _build(impl=impl).memory_strategy == want


@pytest.mark.parametrize("impl", ["c", "eager"])
def test_an_explicit_off_switch_selects_full(impl):
    """`use_ckpt=False` has always meant "no memory trick", not "use the other
    one" -- it is not a vote for boundary saving."""
    assert _build(impl=impl, use_ckpt=False).memory_strategy == "full"


def test_conflicting_requests_still_raise():
    with pytest.raises(ValueError, match="Conflicting"):
        _build(impl="c", memory=Ckpt(mode="chunk", chunks=10),
               boundary_saving_config={"enabled": True})


def test_the_built_state_is_asserted_not_assumed():
    """The plumbing runs through several translations; a disagreement between
    what was decided and what was built is silent. Force one and check it is
    caught rather than shipped."""
    from sweep.propagator import torch as ptorch

    class _Fake:
        use_ckpt = True
        boundary_saving_config = {"enabled": False}

    with pytest.raises(RuntimeError, match="resolved to 'boundary' but the backend"):
        ptorch._assert_backend_matches_strategy(_Fake(), "boundary")
    # and stays quiet when they agree
    ptorch._assert_backend_matches_strategy(_Fake(), "ckpt")
