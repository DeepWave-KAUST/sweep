"""``decimate`` must not hand back a view of the full-resolution array.

Basic slicing yields a view, so the decimated model kept the array it came from
alive through ``.base`` for as long as the caller held it: a downsampled 3-D
benchmark is a few megabytes pinning a few hundred.
"""
import numpy as np

from sweep.datasets._formats import decimate


def test_the_result_does_not_pin_its_source():
    big = np.zeros((200, 400), np.float32)
    small, factor = decimate(big, 4)
    assert factor == (4, 4)
    assert small.base is None, "the decimated array still holds the full-size one alive"
    assert small.flags["C_CONTIGUOUS"]


def test_the_values_are_the_strided_ones():
    a = np.arange(24, dtype=np.float32).reshape(4, 6)
    small, _ = decimate(a, (2, 3))
    assert np.array_equal(small, a[::2, ::3])


def test_no_downsample_is_still_zero_copy():
    """factor 1 everywhere must not copy: it is the common path."""
    a = np.zeros((8, 8), np.float32)
    out, factor = decimate(a, 1)
    assert out is a and factor == (1, 1)


def test_a_per_axis_factor_only_copies_what_it_keeps():
    a = np.zeros((100, 100), np.float32)
    small, _ = decimate(a, (10, 1))
    assert small.shape == (10, 100)
    assert small.nbytes == 10 * 100 * 4
    assert small.base is None
