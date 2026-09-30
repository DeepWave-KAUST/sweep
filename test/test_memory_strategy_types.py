"""The gradient-memory strategy as a type.

`MemoryOptions(strategy=..., boundary=..., ckpt=...)` can express states that are
not valid -- a strategy with the wrong bag filled, a bag with no strategy -- so
it carries six `__post_init__` checks whose only job is to reject them. Carrying
the strategy in the TYPE makes those states unrepresentable, which moves the
error from our runtime check to Python's own call-site TypeError.

These tests pin the three properties that make the migration safe: the legacy
shapes still read, nothing is silently dropped in translation, and the
distinction between "no request" and "explicitly full" survives.
"""
from __future__ import annotations

from dataclasses import fields

import pytest

from sweep.propagator.options import (
    BoundaryOptions,
    BoundarySaving,
    Ckpt,
    CkptOptions,
    Full,
    MemoryOptions,
    as_memory_strategy,
)


# --------------------------------------------------------------------------- #
# every spelling reads, including the one already written into stored YAML
# --------------------------------------------------------------------------- #
def test_all_three_spellings_agree():
    flat = as_memory_strategy({"kind": "boundary", "storage": "cpu", "transfer_interval": 10})
    nested = as_memory_strategy({"strategy": "boundary",
                                 "boundary": {"storage": "cpu", "transfer_interval": 10}})
    legacy = as_memory_strategy(MemoryOptions(
        strategy="boundary",
        boundary=BoundaryOptions(storage="cpu", transfer_interval=10)))
    assert flat == nested == legacy
    assert isinstance(flat, BoundarySaving)


def test_the_nested_shape_is_accepted_because_it_is_already_on_disk():
    """~22 stored experiment YAMLs use `strategy:` + a same-named sub-block.
    Those files are the record of what was run; the reader accommodates them
    rather than the files being rewritten."""
    got = as_memory_strategy({"strategy": "ckpt", "ckpt": {"mode": "recursive", "count": 8}})
    assert got == Ckpt(mode="recursive", count=8)


def test_passthrough_and_none():
    assert as_memory_strategy(None) is None
    c = Ckpt(mode="recursive", count=4)
    assert as_memory_strategy(c) is c


def test_none_is_not_full():
    """`None` means "the caller said nothing", which lets the backend default
    apply. `Full()` means "the caller asked for no reconstruction". Collapsing
    them is how a default silently overrides an explicit request."""
    assert as_memory_strategy(None) is None
    assert as_memory_strategy({"kind": "full"}) == Full()
    assert as_memory_strategy(None) != Full()


# --------------------------------------------------------------------------- #
# nothing is dropped in translation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy,options_cls,sub_kw", [
    ("boundary", BoundaryOptions, dict(storage="disk", transfer_interval=8,
                                       disk_dir="/tmp/bs", ring_buffers=3,
                                       disk_async_read=True, storage_dtype="int8",
                                       tail_steps=50)),
    ("ckpt", CkptOptions, dict(mode="recursive", count=8, storage="cpu",
                               pinned_memory=True)),
])
def test_every_field_survives_the_legacy_conversion(strategy, options_cls, sub_kw):
    """Derived from `fields()`, not a hand-written list: a new option added to
    BoundaryOptions/CkptOptions and forgotten in the converter fails here rather
    than being silently defaulted on the way through."""
    sub = options_cls(**sub_kw)
    legacy = MemoryOptions(strategy=strategy, **{strategy: sub})
    got = as_memory_strategy(legacy)
    for f in fields(options_cls):
        assert getattr(got, f.name) == getattr(sub, f.name), f"{f.name} was not carried across"


# --------------------------------------------------------------------------- #
# the states that used to need runtime checks
# --------------------------------------------------------------------------- #
def test_wrong_parameter_for_the_strategy_is_a_call_site_TypeError():
    """Previously `MemoryOptions(strategy='ckpt', boundary=...)` had to be
    caught by __post_init__. Now it cannot be written."""
    with pytest.raises(TypeError, match="transfer_interval"):
        Ckpt(transfer_interval=4)
    with pytest.raises(TypeError, match="mode"):
        BoundarySaving(mode="recursive")
    with pytest.raises(TypeError):
        Full(storage="cpu")


def test_inherited_validation_still_fires():
    """The new types subclass the option dataclasses, so their rules come along
    rather than being reimplemented."""
    with pytest.raises(ValueError, match="count must be"):
        Ckpt(mode="recursive", count=0)
    with pytest.raises(ValueError, match="pinned_memory"):
        Ckpt(storage="gpu", pinned_memory=True)


def test_isinstance_compatibility_is_kept():
    """Code written against the old option classes keeps working during the
    migration."""
    assert isinstance(BoundarySaving(), BoundaryOptions)
    assert isinstance(Ckpt(), CkptOptions)


def test_an_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="must be 'full', 'boundary' or 'ckpt'"):
        as_memory_strategy({"kind": "bs"})
    with pytest.raises(ValueError, match="needs 'kind'"):
        as_memory_strategy({"storage": "cpu"})
    with pytest.raises(TypeError, match="cannot read a memory strategy"):
        as_memory_strategy(42)
