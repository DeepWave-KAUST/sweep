"""Constructor-argument resolution: the parts a numeric gate cannot see.

Every one of these is a decision about what the caller MEANT, and a
forward-record comparison is blind to all of them -- it only ever runs one
spelling of the arguments, so a resolution rule that changed would simply
produce a different (still self-consistent) run.
"""
from __future__ import annotations

import warnings

import pytest

from sweep.core.arguments import (
    merge_legacy_boundary_kwargs,
    normalise_spacing,
    resolve_device,
)


class _Eq:
    def __init__(self, device=None):
        self.device = device


# --------------------------------------------------------------------------- #
# resolve_device
# --------------------------------------------------------------------------- #
def test_device_wins_over_the_deprecated_dev():
    assert resolve_device("cuda:1", "cuda:0", _Eq()) == "cuda:1"


def test_passing_the_same_device_twice_is_not_a_warning():
    """Redundant is not wrong. Warning here would train callers to filter the
    warning, which is how the real conflict then goes unnoticed."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning becomes a failure
        assert resolve_device("cuda:0", "cuda:0", _Eq()) == "cuda:0"


def test_conflicting_device_and_dev_warns():
    with pytest.warns(DeprecationWarning):
        resolve_device("cuda:1", "cuda:0", _Eq())


def test_falls_back_to_the_equations_device():
    """The equation is the source of truth -- its operators were already built
    there, so disagreeing means a silent cross-device copy every step."""
    assert resolve_device(None, None, _Eq("cuda:3")) == "cuda:3"
    assert resolve_device(None, None, _Eq()) is None


# --------------------------------------------------------------------------- #
# normalise_spacing
# --------------------------------------------------------------------------- #
def test_scalar_spacing_fills_every_axis():
    assert normalise_spacing(12.5, 3) == (12.5, (12.5, 12.5, 12.5))


def test_scalar_of_a_nonuniform_grid_is_the_LAST_axis():
    """Not the first, not the mean. Every stencil that still reads the scalar
    reads x, so any other choice silently rescales those."""
    dh, grid = normalise_spacing((5.0, 10.0, 20.0), 3)
    assert grid == (5.0, 10.0, 20.0)
    assert dh == 20.0


def test_a_numeric_string_is_absorbed_by_the_scalar_branch():
    """Pins PRE-EXISTING behaviour, not desired behaviour.

    `np.isscalar("10.0")` is True, so the str/bytes guard below it never runs
    and `dh="10.0"` has always meant `dh=10.0`. Asserted here so that a future
    tightening is a deliberate, visible change rather than a silent one.
    """
    assert normalise_spacing("10.0", 4) == (10.0, (10.0, 10.0, 10.0, 10.0))
    with pytest.raises(ValueError):
        normalise_spacing("not-a-number", 2)


def test_a_non_sequence_non_scalar_is_rejected():
    with pytest.raises(TypeError):
        normalise_spacing({"dz": 5.0, "dx": 10.0}, 2)


def test_wrong_length_is_rejected():
    with pytest.raises(ValueError):
        normalise_spacing((5.0, 10.0), 3)


# --------------------------------------------------------------------------- #
# merge_legacy_boundary_kwargs
# --------------------------------------------------------------------------- #
def test_legacy_keywords_are_consumed_not_copied():
    """They must be popped: a leftover would reach **kwargs and be reported as
    an unexpected keyword, which is the behaviour that catches typos."""
    kw = {"transfer_interval": 4, "unrelated": 1}
    merge_legacy_boundary_kwargs(kw, None)
    assert "transfer_interval" not in kw
    assert kw == {"unrelated": 1}


def test_boundary_on_cpu_false_means_gpu_not_absent():
    """A bool became a two-valued enum; dropping the False case would leave the
    key unset and fall back to whatever the default happens to be."""
    assert merge_legacy_boundary_kwargs({"boundary_on_cpu": False}, None) == {"storage": "gpu"}
    assert merge_legacy_boundary_kwargs({"boundary_on_cpu": True}, None) == {"storage": "cpu"}


def test_an_explicit_config_beats_a_legacy_keyword():
    """The caller who passed a config is stating intent; a stray legacy keyword
    is usually left over from older code. Reversing this would let a deprecated
    spelling silently override the current one."""
    out = merge_legacy_boundary_kwargs({"transfer_interval": 4},
                                       {"transfer_interval": 9})
    assert out["transfer_interval"] == 9


def test_neither_form_yields_none():
    assert merge_legacy_boundary_kwargs({}, None) is None


# --------------------------------------------------------------------------- #
# validate_memory_strategy -- both refusals prevent a SILENT wrong answer
# --------------------------------------------------------------------------- #
def test_checkpoint_plus_boundary_saving_is_refused():
    """The compiled wrapper picks the checkpoint backward when both are set, so
    the boundary-saving request would be ignored rather than honoured."""
    from sweep.core.arguments import validate_memory_strategy
    with pytest.raises(ValueError, match="three-way choice"):
        validate_memory_strategy(use_checkpoint=True, use_boundary_saving=True,
                                 boundary_tail_steps=0)


def test_tail_steps_with_checkpointing_is_refused():
    """Truncated backward exists only in the boundary-saving backward. The
    checkpoint path would ignore the truncation and return a full-length
    gradient that looks entirely plausible."""
    from sweep.core.arguments import validate_memory_strategy
    with pytest.raises(NotImplementedError, match="tail_steps"):
        validate_memory_strategy(use_checkpoint=True, use_boundary_saving=False,
                                 boundary_tail_steps=50)


@pytest.mark.parametrize("ckpt,bs,tail", [
    (False, False, 0), (True, False, 0), (False, True, 0), (False, True, 50),
])
def test_the_valid_combinations_pass(ckpt, bs, tail):
    from sweep.core.arguments import validate_memory_strategy
    validate_memory_strategy(use_checkpoint=ckpt, use_boundary_saving=bs,
                             boundary_tail_steps=tail)
