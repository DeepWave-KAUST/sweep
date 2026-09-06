"""Domain decomposition must refuse a geometry it cannot answer.

A 3-D coords array means source encoding, whose leading axis is 1. Anything
longer is what a caller writes when they mean "several shots", and the
single-domain propagator refuses it with a ValueError. `ModelParallel` did not:
`_prepare_call` boolean-indexes the owned sources of EVERY leading entry into
one flat array and reads the receivers -- and their ownership, which
`own_receiver_indices` publishes -- from index 0 alone. The result was one
fused supershot recorded at the first entry's receivers, returned without an
error and looking plausible.

No GPU and no process group: the guard runs on the coordinate shapes.
"""
import numpy as np
import pytest
import torch

from sweep.parallel.dd_propagator import ModelParallel


class _Stub:
    """The attributes `_prepare_call` touches before the guard."""
    ndim = 2

    def _slice_tile(self, m):
        return m


def _call(sources, receivers):
    return ModelParallel._prepare_call(
        _Stub(), wavelet=torch.zeros(8),
        sources_global=sources, receivers_global=receivers, models=[torch.zeros(4, 4)])


ONE_SHOT_ENCODED = np.array([[[10, 3], [20, 3]]], np.int64)      # (1, nsrc, ndim)
RECEIVERS = np.array([[[4, 3], [8, 3]]], np.int64)               # (1, nrec, ndim)


@pytest.mark.parametrize("n", [2, 3])
def test_several_leading_entries_are_refused(n):
    sources = np.repeat(ONE_SHOT_ENCODED, n, axis=0)
    with pytest.raises(NotImplementedError) as exc:
        _call(sources, RECEIVERS)
    message = str(exc.value)
    assert "sources_global" in message and str(n) in message
    assert "shot_groups" in message, "the refusal must name a way to run several shots"


def test_a_multi_entry_receiver_array_is_refused_too():
    """Receivers were read from index 0, so a caller's second entry was
    silently ignored rather than rejected."""
    with pytest.raises(NotImplementedError, match="receivers_global"):
        _call(ONE_SHOT_ENCODED, np.repeat(RECEIVERS, 2, axis=0))


def test_the_encoded_supershot_still_passes_the_guard():
    """Leading axis 1 is the legal shape; the guard must not touch it.

    It fails later, on a stub that has none of the mesh state, but it gets
    PAST the guard -- which is what this asserts.
    """
    with pytest.raises(Exception) as exc:
        _call(ONE_SHOT_ENCODED, RECEIVERS)
    assert not isinstance(exc.value, NotImplementedError) or \
        "leading entries" not in str(exc.value)
