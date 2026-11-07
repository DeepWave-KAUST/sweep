"""The compiled slice stencil ``_apply_taps`` differentiates as its transpose.

``_apply_taps`` is a zero-padded correlation; ``_Taps`` gives it the same
correlation with the offsets negated as its derivative instead of autograd's
``slice_backward`` per tap.  These tests pin that the derivative is the exact
transpose (float64 gradcheck and gradgradcheck, and against autograd through
the slices), that vmap still works, and that the compiled op
differentiates as the uncompiled one.
"""
import pytest
import torch

from conftest import requires_compile
from sweep.operators.torch import _apply_taps, _correlate, _mirrored


def _random_taps(nd, reach, n, seed):
    g = torch.Generator().manual_seed(seed)
    taps = set()
    while len(taps) < n:
        taps.add(tuple(int(v) for v in torch.randint(-reach, reach + 1, (nd,), generator=g)))
    return tuple((off, float(torch.randn((), generator=g))) for off in sorted(taps))


CASES = {
    "2d_symmetric": (2, ((( -2, 0), 1 / 12), ((-1, 0), -4 / 3), ((0, 0), 5 / 2), ((1, 0), -4 / 3),
                         ((2, 0), 1 / 12))),
    "2d_one_sided": (2, (((0, 0), -1.125), ((0, 1), 1.125), ((0, -1), 0.04), ((0, 2), -0.0417))),
    "2d_random": (2, _random_taps(2, 3, 9, 0)),
    "3d_random": (3, _random_taps(3, 2, 11, 1)),
}


def _field(nd, batch=(2, 1), seed=0):
    shape = batch + ((7, 9) if nd == 2 else (5, 6, 7))
    return torch.randn(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("name", list(CASES))
def test_the_derivative_is_the_transpose(name):
    nd, taps = CASES[name]
    u = _field(nd).requires_grad_(True)
    assert torch.autograd.gradcheck(lambda x: _apply_taps(x, taps), (u,))
    assert torch.autograd.gradgradcheck(lambda x: _apply_taps(x, taps), (u,))


@pytest.mark.parametrize("name", list(CASES))
def test_the_derivative_matches_autograd_through_the_slices(name):
    nd, taps = CASES[name]
    u = _field(nd).requires_grad_(True)
    g = _field(nd, seed=1)
    (ours,) = torch.autograd.grad(_apply_taps(u, taps), u, g)
    (slices,) = torch.autograd.grad(_correlate(u, taps), u, g)
    torch.testing.assert_close(ours, slices, rtol=0, atol=1e-13)
    assert _mirrored(_mirrored(taps)) == taps


@pytest.mark.parametrize("name", ["2d_random", "3d_random"])
def test_vmap(name):
    nd, taps = CASES[name]
    stack = torch.stack([_field(nd, seed=s) for s in range(3)])
    batched = torch.func.vmap(lambda x: _apply_taps(x, taps))(stack)
    torch.testing.assert_close(batched, torch.stack([_correlate(x, taps) for x in stack]), rtol=0, atol=0)


@requires_compile
@pytest.mark.parametrize("name", ["2d_random", "3d_random"])
def test_compiled_the_same_derivative(name):
    nd, taps = CASES[name]
    u = _field(nd).float().requires_grad_(True)
    g = _field(nd, seed=1).float()
    loss = lambda x: (_apply_taps(x, taps) * g).sum()  # noqa: E731
    (ref,) = torch.autograd.grad(loss(u), u)
    (out,) = torch.autograd.grad(torch.compile(loss, backend="aot_eager", fullgraph=True)(u), u)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
