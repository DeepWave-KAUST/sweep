"""Source injection is a scatter onto the source cells, not a full-grid mask.

A step adds the wavelet sample to a handful of cells, but ``wavefield + mask *
wavelet`` read and wrote the whole padded grid twice per step -- about a fifth
to a quarter of a compiled acoustic forward's GPU time. The scatter must inject
exactly what the mask did: the references below rebuild the mask (and, with a
spread kernel, its convolution) the way ``SourceTorch`` used to.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sweep.sources.torch import SourceTorch

SHAPE = (2, 1, 40, 48)          # (B, 1, nz, nx)
COORDS = torch.tensor([[20, 10], [22, 30]], dtype=torch.long)
BINOMIAL = np.outer([1.0, 2.0, 1.0], [1.0, 2.0, 1.0]).astype(np.float32) / 16.0
SKEW = np.arange(9, dtype=np.float32).reshape(3, 3) / 7.0      # not symmetric: catches a flip


def _mask(coords, shape, kernel=None):
    """The old mask: 1 at each shot's source cell, convolved with ``kernel``."""
    mask = torch.zeros(shape)
    flipped = torch.flip(coords, [-1])
    mask[(torch.arange(coords.shape[0]), slice(None), *flipped.unbind(-1))] = 1.0
    if kernel is not None:
        k = torch.as_tensor(kernel).view(1, 1, *kernel.shape)
        mask = F.conv2d(mask, k, padding=kernel.shape[-1] // 2)
    return mask


def _field(shape, seed=0):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("kernel", [None, BINOMIAL, SKEW], ids=["point", "binomial", "skew"])
@pytest.mark.parametrize("wavelet", [torch.tensor(1.7), torch.tensor([2.0, -3.0])],
                         ids=["shared", "per-shot"])
def test_the_scatter_injects_what_the_mask_did(kernel, wavelet):
    field = _field(SHAPE)
    out = SourceTorch(COORDS, SHAPE, torch.device("cpu"), spread_kernel=kernel)(field, wavelet)
    w = wavelet.reshape(wavelet.shape + (1,) * (4 - wavelet.ndim))
    ref = field + _mask(COORDS, SHAPE, kernel) * w
    if kernel is None:
        assert torch.equal(out, ref)
    else:   # the convolution may round the mask's weights differently
        torch.testing.assert_close(out, ref, rtol=0, atol=1e-6)


def test_spread_cells_off_the_grid_are_dropped():
    coords = torch.tensor([[0, 0], [47, 39]], dtype=torch.long)    # opposite corners
    out = SourceTorch(coords, SHAPE, torch.device("cpu"), spread_kernel=SKEW)(
        torch.zeros(SHAPE), torch.tensor(1.0))
    torch.testing.assert_close(out, _mask(coords, SHAPE, SKEW), rtol=0, atol=1e-7)


def test_3d():
    shape = (2, 1, 9, 10, 11)                                       # (B, 1, nz, ny, nx)
    coords = torch.tensor([[3, 4, 5], [10, 0, 8]], dtype=torch.long)   # (x, y, z)
    field = _field(shape)
    out = SourceTorch(coords, shape, torch.device("cpu"))(field, torch.tensor([1.5, -2.5]))
    ref = field.clone()
    ref[0, 0, 5, 4, 3] += 1.5
    ref[1, 0, 8, 0, 10] += -2.5
    assert torch.equal(out, ref)


@pytest.mark.parametrize("wavelet", [torch.tensor(2.0), torch.tensor([2.0, 3.0, -1.0])],
                         ids=["shared", "per-source"])
def test_encoded_sources_share_batch_zero_and_accumulate(wavelet):
    coords = torch.tensor([[[20, 10], [22, 30], [20, 10]]], dtype=torch.long)   # 1st == 3rd
    out = SourceTorch(coords, (1, 1, 40, 48), torch.device("cpu"), source_encoding=True)(
        torch.zeros(1, 1, 40, 48), wavelet)
    w = wavelet.expand(3)
    assert float(out[0, 0, 10, 20]) == float(w[0] + w[2])
    assert float(out[0, 0, 30, 22]) == float(w[1])
    assert float(out.abs().sum()) == float(abs(w[0] + w[2]) + abs(w[1]))


def test_in_place_add_matches_forward():
    src = SourceTorch(COORDS, SHAPE, torch.device("cpu"), spread_kernel=BINOMIAL)
    field, w = _field(SHAPE), torch.tensor([2.0, -3.0])
    expected = src(field, w)
    assert src.add_(field, w) is field
    assert torch.equal(field, expected)


@pytest.mark.parametrize("kernel", [None, SKEW], ids=["point", "skew"])
@pytest.mark.parametrize("n", [1, 2, 6], ids=["shared", "per-shot", "per-source"])
def test_value_grad_is_the_adjoint_of_the_injection(kernel, n):
    """<inject(w), g> == <w, value_grad(g)>: what the boundary-saving backward
    hands the wavelet."""
    coords = torch.tensor([[[5, 3], [20, 10], [40, 30]], [[7, 9], [22, 30], [1, 38]]])
    src = SourceTorch(coords, SHAPE, torch.device("cpu"), spread_kernel=kernel)
    w, g = torch.randn(n, dtype=torch.float64), _field(SHAPE).double()
    lhs = float((src(torch.zeros(SHAPE, dtype=torch.float64), w) * g).sum())
    rhs = float((w * src.value_grad(g, w.shape)).sum())
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_no_full_grid_buffer_is_built():
    src = SourceTorch(COORDS, SHAPE, torch.device("cpu"), spread_kernel=BINOMIAL)
    big = [k for k, v in vars(src).items()
           if isinstance(v, torch.Tensor) and v.numel() >= np.prod(SHAPE[2:])]
    assert not big, f"full-grid buffers {big} on a source of {COORDS.shape[0]} points"


@pytest.mark.parametrize("coords", [[[48, 10], [22, 30]], [[20, -1], [22, 30]]],
                         ids=["x-past-the-edge", "negative-z"])
def test_a_source_off_the_grid_is_refused(coords):
    """A flat cell index would wrap a source past the right edge into the next
    row, silently."""
    with pytest.raises(ValueError, match="outside"):
        SourceTorch(torch.tensor(coords), SHAPE, torch.device("cpu"))


def test_a_wavelet_sample_of_the_wrong_size_is_refused():
    src = SourceTorch(COORDS, SHAPE, torch.device("cpu"))
    with pytest.raises(ValueError, match="expected 1, 2, or 2"):
        src(torch.zeros(SHAPE), torch.ones(3))
