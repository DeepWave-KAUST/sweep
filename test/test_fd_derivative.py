"""``sweep.operators.Derivative``: finite-difference derivatives of any order.

Pins: the Fornberg weights against the coefficient tables the equations use
and textbook stencils; the operator bitwise against the second derivative of
``separable_d2_2d/3d`` and the staggered first derivative of
``StaggeredDerivative`` (the two derivatives it generalises); the measured
order of accuracy for orders 1-4 at accuracies 2-8, centred and staggered; where
a staggered result lands; the sawtooth null mode of a centred odd stencil; the
transpose (gradcheck) and the slice form used under torch.compile.
"""
import math

import numpy as np
import pytest
import torch

import sweep.operators.torch as OPS
from sweep.equations.utils import to_backend
from sweep.operators import Derivative, StaggeredDerivative
from sweep.scalars import (fd_coefficients, fd_stencil, fd_weights, generate_convolution_kernel,
                           staggered_grid_coes)


@pytest.mark.parametrize("accuracy", [2, 4, 6, 8, 12])
def test_weights_match_the_equations_tables(accuracy):
    m = accuracy // 2
    _, w2 = fd_stencil(2, accuracy)
    # fd_coefficients solves a Vandermonde system (4e-13 off at accuracy 12); fd_stencil is exact
    np.testing.assert_allclose(w2[m + 1:], fd_coefficients(2, accuracy), rtol=1e-11, atol=0)
    _, w1 = fd_stencil(1, accuracy, "forward")
    np.testing.assert_allclose(w1[m:], staggered_grid_coes(m), rtol=1e-6)     # float32 table


def test_textbook_stencils():
    cases = [
        ((1, 2, None), [-0.5, 0, 0.5]),
        ((2, 2, None), [1, -2, 1]),
        ((3, 2, None), [-0.5, 1, 0, -1, 0.5]),
        ((4, 2, None), [1, -4, 6, -4, 1]),
        ((3, 2, "forward"), [-1, 3, -3, 1]),
        ((1, 2, "forward"), [-1, 1]),
        ((1, 4, "forward"), [1 / 24, -9 / 8, 9 / 8, -1 / 24]),
    ]
    for args, want in cases:
        np.testing.assert_allclose(fd_stencil(*args)[1], want, rtol=1e-15, atol=1e-15, err_msg=str(args))
    assert list(fd_stencil(3, 2, "forward")[0]) == [-1, 0, 1, 2]
    assert list(fd_stencil(3, 2, "backward")[0]) == [-2, -1, 0, 1]
    np.testing.assert_allclose(fd_weights(1, [0, 1, 2]), [-1.5, 2, -0.5], rtol=1e-15)   # one-sided


def test_bad_arguments_raise():
    with pytest.raises(ValueError, match="odd derivative order"):
        fd_stencil(2, 4, "forward")
    with pytest.raises(ValueError, match="even"):
        fd_stencil(2, 3)
    with pytest.raises(ValueError, match="stagger"):
        fd_stencil(1, 4, "left")
    with pytest.raises(ValueError, match="distinct"):
        fd_weights(1, [0, 1, 1])
    with pytest.raises(ValueError, match="more than"):
        fd_weights(2, [0, 1])
    d = Derivative(3, 4)
    with pytest.raises(ValueError, match="axis"):
        d(torch.zeros(1, 1, 8, 8), "y")
    with pytest.raises(ValueError, match="shape"):
        d(torch.zeros(1, 8, 8), "x")
    with pytest.raises(ValueError, match="ndim"):
        Derivative(2, 4, ndim=1)


@pytest.mark.parametrize("accuracy", [2, 4, 8])
def test_order_two_is_bitwise_separable_d2(accuracy):
    g = torch.Generator().manual_seed(0)
    k1d = torch.tensor(generate_convolution_kernel(accuracy, mode="x")[0, 0][accuracy // 2, :])
    u = torch.randn(2, 1, 30, 41, generator=g)
    dz2, dx2 = OPS.separable_d2_2d(u, k1d, 0.7, 1.3)
    d = Derivative(2, accuracy)
    assert torch.equal(d.dz(u, 0.7), dz2) and torch.equal(d.dx(u, 1.3), dx2)
    u3 = torch.randn(1, 1, 12, 13, 14, generator=g)
    ref = OPS.separable_d2_3d(u3, k1d, 1.0, 1.0, 1.0)
    d3 = Derivative(2, accuracy, ndim=3)
    assert all(torch.equal(d3(u3, ax), r) for ax, r in zip("zyx", ref))


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("accuracy", [2, 4, 8])
def test_staggered_order_one_is_bitwise_staggered_derivative(ndim, accuracy):
    pd = StaggeredDerivative(accuracy, ndim=ndim)
    pd.to_backend(to_backend)
    shape = (2, 1, 20, 23) if ndim == 2 else (1, 1, 10, 11, 12)
    u = torch.randn(*shape, generator=torch.Generator().manual_seed(1))
    for stagger in ("forward", "backward"):
        d = Derivative(1, accuracy, ndim=ndim, stagger=stagger)
        for ax in ("zx" if ndim == 2 else "zyx"):
            assert torch.equal(d(u, ax), getattr(pd, f"{ax}_{stagger}")(u)), (stagger, ax)


def _measured_order(order, accuracy, stagger):
    """Slope of the error over two grids, on sin(x), at the points whose stencil
    lies inside the grid.  The grids are coarse enough that the error on the
    finer one stays well above float64 round-off (which grows as h**-order)."""
    errs = []
    for n in ((20, 40) if order >= 5 else (24, 48)):
        h = 2 * math.pi / n
        x = torch.arange(n, dtype=torch.float64) * h
        u = torch.sin(x).view(1, 1, 1, n)
        d = Derivative(order, accuracy, stagger=stagger, dtype=torch.float64)
        got = d.dx(u, h)[0, 0, 0]
        shift = {None: 0.0, "forward": 0.5, "backward": -0.5}[stagger] * h
        want = torch.sin(x + shift + order * math.pi / 2)     # d^n sin = sin(x + n pi / 2)
        inner = slice(d.halo, n - d.halo)
        errs.append((got - want)[inner].abs().max().item())
    return math.log2(errs[0] / errs[1])


@pytest.mark.parametrize("order", [1, 2, 3, 4])
@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
def test_order_of_accuracy_centred(order, accuracy):
    assert _measured_order(order, accuracy, None) == pytest.approx(accuracy, abs=0.25)


@pytest.mark.parametrize("order", [1, 3, 5])
@pytest.mark.parametrize("accuracy", [2, 4, 6, 8])
@pytest.mark.parametrize("stagger", ["forward", "backward"])
def test_order_of_accuracy_staggered(order, accuracy, stagger):
    assert _measured_order(order, accuracy, stagger) == pytest.approx(accuracy, abs=0.25)


def test_axes_in_3d_and_mixed_derivative():
    n, h = 32, 2 * math.pi / 32
    c = torch.arange(n, dtype=torch.float64) * h
    z, y, x = torch.meshgrid(c, c, c, indexing="ij")
    u = (torch.sin(z) * torch.cos(2 * y) * torch.sin(3 * x)).view(1, 1, n, n, n)
    inner = (0, 0) + (slice(8, 24),) * 3
    d3 = Derivative(3, 8, ndim=3, dtype=torch.float64)
    want = {"z": -torch.cos(z) * torch.cos(2 * y) * torch.sin(3 * x),
            "y": 8 * torch.sin(z) * torch.sin(2 * y) * torch.sin(3 * x),
            "x": -27 * torch.sin(z) * torch.cos(2 * y) * torch.cos(3 * x)}
    for ax, w in want.items():
        err = (d3(u, ax, h) - w.view(1, 1, n, n, n))[inner].abs().max().item()
        assert err < 1e-3 * w.abs().max().item(), ax
    d1 = Derivative(1, 8, ndim=3, dtype=torch.float64)
    mixed = d1.dz(d1.dx(u, h), h)                               # d2u / dz dx
    w = (3 * torch.cos(z) * torch.cos(2 * y) * torch.cos(3 * x)).view(1, 1, n, n, n)
    assert (mixed - w)[inner].abs().max().item() < 1e-4


def test_centred_odd_stencil_misses_the_sawtooth():
    saw = ((-1.0) ** torch.arange(32, dtype=torch.float64)).view(1, 1, 1, 32)
    inner = (0, 0, 0, slice(8, 24))
    for order in (1, 3):
        centred = Derivative(order, 4, dtype=torch.float64).dx(saw)
        staggered = Derivative(order, 4, stagger="forward", dtype=torch.float64).dx(saw)
        assert centred[inner].abs().max().item() < 1e-12
        assert staggered[inner].abs().min().item() > 1.0


@pytest.mark.parametrize("order,stagger", [(2, None), (3, None), (3, "forward"), (4, None)])
def test_gradcheck(order, stagger):
    d = Derivative(order, 4, stagger=stagger, dtype=torch.float64)
    u = torch.randn(2, 1, 9, 11, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: d.dz(d.dx(v, 0.5), 2.0), (u,))


def test_slice_form_matches_the_conv():
    """Under torch.compile the kernel runs as its registered taps."""
    u = torch.randn(2, 1, 17, 19, dtype=torch.float64)
    for order, stagger in [(3, None), (3, "backward"), (4, None), (5, "forward")]:
        d = Derivative(order, 6, stagger=stagger, dtype=torch.float64)
        for ax in "zx":
            taps = d.kernels[ax]._sweep_taps
            assert taps is not None
            torch.testing.assert_close(OPS._apply_taps(u, taps), d(u, ax), rtol=1e-12, atol=1e-12)


def test_to_moves_and_casts():
    d = Derivative(3, 4).to(dtype=torch.float64)
    assert all(k.dtype == torch.float64 for k in d.kernels.values())
    assert repr(Derivative(3, 4, stagger="forward")) == "Derivative(order=3, accuracy=4, ndim=2, stagger='forward')"
