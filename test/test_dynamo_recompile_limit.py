"""The Dynamo recompile cap must be raised on old torch too.

Dynamo specializes the eager step on per-tensor metadata, and each wavefield's
``requires_grad`` flips False->True the first time it absorbs a contribution
from the models. The many-wavefield equations exhaust the default cap of 8
before Dynamo settles, and it then falls back to eager silently -- which is why
the propagator raises the cap before compiling.

The knob was renamed: ``cache_size_limit`` up to torch 2.5, ``recompile_limit``
from 2.6. torch is unpinned, so keying on the new name alone made the bump a
no-op for every user on an older torch: exactly the users who then hit the
fallback the bump exists to prevent.
"""
import types

import pytest
import torch

from sweep.propagator import _torch_eager as te


class _Cfg:
    """A stand-in for torch._dynamo.config with a chosen set of knobs."""

    def __init__(self, **knobs):
        for k, v in knobs.items():
            setattr(self, k, v)


@pytest.fixture()
def fake_dynamo(monkeypatch):
    def install(cfg):
        monkeypatch.setattr(torch, "_dynamo", types.SimpleNamespace(config=cfg),
                            raising=False)
        return cfg

    return install


def test_new_torch_spelling(fake_dynamo):
    cfg = fake_dynamo(_Cfg(recompile_limit=8, accumulated_recompile_limit=8))
    te._raise_dynamo_recompile_limit()
    assert cfg.recompile_limit == 32
    assert cfg.accumulated_recompile_limit == 32


def test_old_torch_spelling(fake_dynamo):
    """torch <= 2.5 has cache_size_limit and no recompile_limit at all."""
    cfg = fake_dynamo(_Cfg(cache_size_limit=8, accumulated_cache_size_limit=64))
    te._raise_dynamo_recompile_limit()
    assert cfg.cache_size_limit == 32, "the bump was a no-op on the old spelling"
    assert cfg.accumulated_cache_size_limit == 64, "already above the target, left alone"


def test_both_spellings_present(fake_dynamo):
    """torch 2.6+ keeps the old name as an alias; move both."""
    cfg = fake_dynamo(_Cfg(recompile_limit=8, cache_size_limit=8))
    te._raise_dynamo_recompile_limit()
    assert (cfg.recompile_limit, cfg.cache_size_limit) == (32, 32)


def test_a_limit_already_high_is_left_alone(fake_dynamo):
    cfg = fake_dynamo(_Cfg(recompile_limit=64))
    te._raise_dynamo_recompile_limit()
    assert cfg.recompile_limit == 64


def test_an_unknown_dynamo_warns_once(fake_dynamo, monkeypatch):
    monkeypatch.setattr(te, "_DYNAMO_LIMIT_WARNED", False)
    fake_dynamo(_Cfg(some_other_knob=1))
    with pytest.warns(RuntimeWarning, match="recompile_limit nor cache_size_limit"):
        te._raise_dynamo_recompile_limit()
    te._raise_dynamo_recompile_limit()          # second call is silent


def test_no_dynamo_at_all_is_not_an_error(monkeypatch):
    monkeypatch.delattr(torch, "_dynamo", raising=False)
    te._raise_dynamo_recompile_limit()
