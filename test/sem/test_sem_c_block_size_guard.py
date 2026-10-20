"""impl='c' refuses an order whose element does not fit in one CUDA block.

Every SEM element kernel launches one block per element and one THREAD per
element node, so ``ngll ** ndim`` IS the block size and CUDA's 1024-thread
limit bounds the order: n_order <= 9 in 3-D, <= 31 in 2-D. Without the guard
the launch returns cudaErrorInvalidConfiguration, and because the C entries
check ``cudaGetLastError`` only after the whole nt loop, the user sees a bare
"invalid configuration argument" attributed to nothing, nt steps after the
cause -- measured before the guard existed.

The guard is at construction, not at the first forward: the 3-D C-PML route
runs through ``dd.SEMModelParallel`` and never touches ``_check_forward``,
which is where it first went and why it did not fire.
"""
from __future__ import annotations

import pytest
import torch

from sweep.sem import PropSEM
from sweep.sem.equations import AcousticPotentialSEM2D, AcousticSEM3D
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")


def _build(equation, ndim, n_order):
    n = 3 * n_order + 1
    return PropSEM(equation, shape=(n,) * ndim, dh=40.0, dt=2e-4, nt=4,
                   n_order=n_order, free_surface=("top",),
                   abcn=n_order, f_dominant=10.0, dev="cuda", impl="c")


@pytest.mark.parametrize("equation,ndim,n_order,m", [
    (AcousticSEM3D, 3, 10, 1331),
    (AcousticPotentialSEM2D, 2, 40, 1681),
])
def test_an_order_too_large_for_a_block_is_refused_at_construction(
        equation, ndim, n_order, m):
    with pytest.raises(ValueError, match=str(m)):
        _build(equation, ndim, n_order)


@pytest.mark.parametrize("equation,ndim,n_order", [
    (AcousticSEM3D, 3, 9),              # m = 1000, the largest 3-D order
    (AcousticPotentialSEM2D, 2, 10),    # m = 121
])
def test_the_largest_orders_that_do_fit_are_not_refused(equation, ndim, n_order):
    """The guard must not cost a legal configuration: 9 is the real 3-D
    ceiling and it runs."""
    p = _build(equation, ndim, n_order)
    assert p.mesh.ngll ** ndim <= 1024
