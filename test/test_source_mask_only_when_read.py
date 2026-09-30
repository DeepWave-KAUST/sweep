"""The source injection mask is built only on the path that reads it.

`self.mask` is a whole padded wavefield, filled with an `index_put_` and, with
a spread kernel, convolved. Exactly one place reads it -- `SourceBase.forward`
-- and `SourceTorch.forward` reaches that only when neither source encoding nor
adjoint modelling is active; both of those inject through
`_add_indexed_sources`. It was built on every construction regardless.
"""
import numpy as np
import pytest
import torch

from sweep.sources.torch import SourceTorch

SHAPE = (2, 1, 40, 48)          # (B, 1, nz, nx)
COORDS = torch.tensor([[20, 10], [22, 30]], dtype=torch.long)


def _src(**kw):
    return SourceTorch(COORDS, SHAPE, dev=torch.device("cpu"), **kw)


def test_the_plain_path_still_has_its_mask():
    src = _src()
    assert src.mask is not None
    assert src.mask.shape == SHAPE
    assert float(src.mask.sum()) == len(COORDS), "one cell per source"


@pytest.mark.parametrize("kw", [
    dict(source_encoding=True),
    dict(adj=True),
    dict(source_encoding=True, adj=True),
])
def test_the_indexed_paths_build_no_mask(kw):
    assert _src(**kw).mask is None, (
        "a whole padded wavefield was allocated for a buffer this path never reads"
    )


def test_injection_is_unchanged_on_the_plain_path():
    src = _src()
    wavefield = torch.zeros(SHAPE)
    out = src(wavefield, torch.tensor([2.0, 3.0]))
    assert float(out[0, 0, 10, 20]) == 2.0
    assert float(out[1, 0, 30, 22]) == 3.0
    assert float(out.abs().sum()) == 5.0, "nothing else was touched"


def test_a_spread_kernel_still_widens_the_plain_mask():
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]], np.float32)
    src = _src(spread_kernel=kernel)
    assert src.mask is not None
    assert float(src.mask.sum()) == pytest.approx(5.0 * len(COORDS))


def test_an_encoded_source_still_injects():
    """The mask is gone; the indexed path must be untouched.

    That path takes ``(B, nsrc, ndim)`` coords -- it reads all three -- while
    the mask path takes the flat ``(nsrc, ndim)`` form.
    """
    coords = torch.tensor([[[20, 10], [22, 30]]], dtype=torch.long)
    src = SourceTorch(coords, (1, 1, 40, 48), dev=torch.device("cpu"),
                      source_encoding=True)
    assert src.mask is None
    out = src(torch.zeros(1, 1, 40, 48), torch.tensor([2.0, 3.0]))
    assert float(out[0, 0, 10, 20]) == 2.0
    assert float(out[0, 0, 30, 22]) == 3.0
