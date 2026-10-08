"""Receivers sample a field with a flat ``index_select`` when they know its shape.

Advanced indexing's backward is an ``index_put(accumulate=True)``, which sorts
its indices -- about 0.25 ms per backward step on a GPU, every step -- and a
multi-component record concatenated its fields (a full copy of each) before
gathering a few cells. The flat gather must sample, and pass gradients back,
exactly as the indexing it replaces.
"""
import numpy as np
import pytest
import torch

from sweep.receivers.torch import ReceiverTorch

SHAPE = (2, 1, 30, 36)
COORDS = torch.tensor([[[3, 4], [20, 9], [35, 29]], [[0, 0], [17, 12], [3, 4]]])
BINOMIAL = np.outer([1.0, 2.0, 1.0], [1.0, 2.0, 1.0]).astype(np.float32) / 16.0


def _field(shape, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), requires_grad=True)


def _node_names(t):
    names, stack = set(), [t.grad_fn]
    while stack:
        node = stack.pop()
        if node is not None and type(node).__name__ not in names:
            names.add(type(node).__name__)
            stack.extend(n for n, _ in node.next_functions)
    return names


@pytest.mark.parametrize("kernel", [None, BINOMIAL], ids=["point", "binomial"])
def test_flat_gather_samples_and_differentiates_like_indexing(kernel):
    coords = COORDS if kernel is None else COORDS.clamp(1, 28)      # keep the 3x3 gather on the grid
    a, b = _field(SHAPE, 0), _field(SHAPE, 0)
    g = torch.randn(SHAPE[0] * coords.shape[1], generator=torch.Generator().manual_seed(1))
    flat = ReceiverTorch(coords, gather_kernel=kernel, shape=SHAPE)(a).reshape(-1)
    ref = ReceiverTorch(coords, gather_kernel=kernel)(b).reshape(-1)
    assert torch.equal(flat, ref)
    (flat * g).sum().backward()
    (ref * g).sum().backward()
    assert torch.equal(a.grad, b.grad)
    names = _node_names(flat)
    assert "IndexSelectBackward0" in names and "IndexBackward0" not in names, names


def test_3d():
    shape = (1, 1, 9, 10, 11)
    coords = torch.tensor([[[3, 4, 5], [10, 0, 8]]])                # (x, y, z)
    f = torch.randn(shape)
    out = ReceiverTorch(coords, shape=shape)(f)
    assert torch.equal(out, torch.stack([f[0, 0, 5, 4, 3], f[0, 0, 8, 0, 10]]))


def test_several_fields_are_gathered_without_concatenating_them():
    fields = [_field(SHAPE, s) for s in range(3)]
    flat = ReceiverTorch(COORDS, shape=SHAPE).sample_fields(fields)
    ref = ReceiverTorch(COORDS).sample_fields([f.detach() for f in fields])
    assert flat.shape == (SHAPE[0], COORDS.shape[1], 3)
    assert torch.equal(flat.detach(), ref)
    names = _node_names(flat)
    assert not any("Cat" in n for n in names), names


@pytest.mark.parametrize("coords", [[[[36, 4]]], [[[3, -1]]]], ids=["x-past-the-edge", "negative-z"])
def test_a_receiver_off_the_grid_is_refused(coords):
    """A flat index would wrap a receiver past the right edge into the next row."""
    with pytest.raises(ValueError, match="outside"):
        ReceiverTorch(torch.tensor(coords), shape=(1, 1, 30, 36))
