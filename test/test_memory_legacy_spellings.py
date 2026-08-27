"""Everything the deprecated memory spellings still accept.

THIS FILE IS A DELETION CHECKLIST. It exists only to pin what the legacy layer
reads, so that removing that layer is one deletion with a written criterion
rather than a judgement call. See the fenced section in `core/arguments.py` for
the criterion; when it is met, delete that section, the legacy branches of
`as_memory_strategy`, and this file together.

Two properties matter and are easy to get backwards:

* the old spellings must still WORK, not merely warn -- stored experiment YAML
  uses the nested dict, and those files are a record of what was run;
* the new spellings must not warn. A deprecation warning fires at the boundary
  the caller crosses, never on an internal parameter -- `boundary_saving_config`
  IS the internal wire format that `PropTorch` synthesises from
  `memory=BoundarySaving(...)`, so warning where it is consumed would tell a
  caller who used the new API that they used the old one. That mistake was made
  twice while writing this, in both directions.
"""
from __future__ import annotations

import warnings

import pytest
import torch

import sweep.core.arguments as _args
from sweep.propagator.options import (
    BoundaryOptions,
    BoundarySaving,
    Ckpt,
    CkptOptions,
    Full,
    MemoryOptions,
    as_memory_strategy,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(autouse=True)
def _reset_warned():
    """The warning fires once per process per spelling, so tests must reset it
    or the second test in a run sees nothing."""
    _args._WARNED_SPELLINGS.clear()
    yield
    _args._WARNED_SPELLINGS.clear()


def _build(**kw):
    from sweep.equations.acoustic import Acoustic
    from sweep.propagator.torch import PropTorch
    return PropTorch(Acoustic(spatial_order=4, device="cuda", backend="torch"),
                     backend="torch", impl="c", shape=(48, 56), dev="cuda",
                     dh=10.0, dt=1e-3, source_type=["p"], receiver_type=["p"],
                     abcn=20, nt=50, B=1, allow_growth=True, **kw)


def _deprecations(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fn()
    return result, [str(x.message).split(" is deprecated")[0]
                    for x in w if issubclass(x.category, DeprecationWarning)]


# --------------------------------------------------------------------------- #
# what the legacy layer accepts -- the checklist
# --------------------------------------------------------------------------- #
LEGACY_SPELLINGS = [
    ("boundary_saving_config dict",
     dict(boundary_saving_config={"enabled": True, "storage": "gpu"}), "boundary"),
    ("boundary_saving_config disabled",
     dict(boundary_saving_config={"enabled": False}), "full"),
    ("loose transfer_interval",
     dict(boundary_saving_config={"enabled": True}, transfer_interval=4), "boundary"),
    ("loose boundary_on_cpu",
     dict(boundary_saving_config={"enabled": True}, boundary_on_cpu=True,
          transfer_interval=4), "boundary"),
    ("MemoryOptions boundary",
     dict(memory=MemoryOptions(strategy="boundary",
                               boundary=BoundaryOptions(storage="gpu"))), "boundary"),
    ("MemoryOptions ckpt",
     dict(memory=MemoryOptions(strategy="ckpt",
                               ckpt=CkptOptions(mode="chunk", chunks=10))), "ckpt"),
    ("MemoryOptions full", dict(memory=MemoryOptions(strategy="full")), "full"),
    ("nested dict",
     dict(memory={"strategy": "boundary", "boundary": {"storage": "gpu"}}), "boundary"),
]


@pytest.mark.parametrize("label,kwargs,want", LEGACY_SPELLINGS,
                         ids=[s[0] for s in LEGACY_SPELLINGS])
def test_legacy_spelling_still_works_and_warns(label, kwargs, want):
    prop, warned = _deprecations(lambda: _build(**kwargs))
    assert prop.memory_strategy == want, f"{label} resolved to {prop.memory_strategy!r}"
    assert warned, f"{label} produced no DeprecationWarning"


# --------------------------------------------------------------------------- #
# and the new spellings must stay silent
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("memory,want", [
    (Full(), "full"),
    (BoundarySaving(storage="gpu"), "boundary"),
    (Ckpt(mode="chunk", chunks=10), "ckpt"),
    ({"kind": "boundary", "storage": "gpu"}, "boundary"),
])
def test_the_current_spelling_does_not_warn(memory, want):
    prop, warned = _deprecations(lambda: _build(memory=memory))
    assert prop.memory_strategy == want
    assert not warned, f"the current API warned about itself: {warned}"


def test_the_default_construction_does_not_warn():
    """No memory argument at all must be silent -- otherwise every existing
    script warns and the warning stops meaning anything."""
    _, warned = _deprecations(lambda: _build())
    assert not warned


def test_a_warning_fires_once_per_process_per_spelling():
    """Bounded noise: a line per propagator construction would get the whole
    category filtered out, taking the warnings that matter with it."""
    _, first = _deprecations(
        lambda: as_memory_strategy({"strategy": "ckpt", "ckpt": {"mode": "chunk"}}))
    _, second = _deprecations(
        lambda: as_memory_strategy({"strategy": "ckpt", "ckpt": {"mode": "chunk"}}))
    assert first and not second
