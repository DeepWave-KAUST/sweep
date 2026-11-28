"""The second-derivative stencil must vanish on a constant field in float32,
and impl='c' must take it in difference form (but the forward at order 4).

Rounding each coefficient to float32 on its own left c0 + 2 sum(c_k) = 7.45e-8
at 4th order: a term v^2 (sum / h^2) u in the wave equation, whose relative
effect grows like 1 / (k h)^2, so records on fine grids stopped converging
(2-D, 300 m, 10 Hz: 6.4e-4 at dh 1.25 m and 2.6e-3 at 0.625 m against 1.8e-4
at 2.5 m).  These tests pin the zero sum on every copy of the stencil -- the
coefficient helper, eager's kernels, the coefficients impl='c' is handed and
(on a GPU) the constants compiled into its order 4/6/8 kernels -- and that
the fine-grid record converges to the analytic solution again.

Zero-sum, impl='c' still summed c_k (u_k + u_-k) to ~c0 u0 before taking the
centre off, and the float32 rounding of those partial sums set a floor below
~1 m (dh 0.625 m: 4.3e-5, 5.7e-5 and 2.7e-4 at orders 6, 8 and 10; eager
1.3e-5 .. 2.0e-5); the adjoint kernels, which form every tap as a product,
were worse.  The kernels now take c_k ((u_k - u_0) + (u_-k - u_0)) (operators/
laplace.cuh), except the forward at order 4, where the summed form stays the
better one at small Courant numbers: the last two tests pin the forward at
0.625 m, order 4 included, and the adjoint, through the wavelet gradient, at
1.25 m.
"""
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sweep.equations import Acoustic, Acoustic3D
from sweep.propagator._c import _CompiledPropagator
from sweep.scalars import fd_coefficients, generate_convolution_kernel, second_derivative_coefficients_fp32

ORDERS = (2, 4, 6, 8, 10, 12, 14, 16)


def _exact_sum(values):
    return sum(Fraction(float(v)) for v in np.asarray(values, dtype=np.float32).ravel())


@pytest.mark.parametrize("order", ORDERS)
def test_coefficients_sum_to_zero_and_stay_close(order):
    c0, c = second_derivative_coefficients_fp32(order)
    assert c0.dtype == np.float32 and c.dtype == np.float32
    assert Fraction(float(c0)) + 2 * _exact_sum(c) == 0
    exact = fd_coefficients(2, order)
    assert abs(float(c0) + 2 * exact.sum()) <= abs(np.spacing(c0))
    assert np.all(np.abs(c.astype(np.float64) - exact) <= 6e-8)


@pytest.mark.parametrize("order", ORDERS)
def test_eager_kernels_sum_to_zero(order):
    assert _exact_sum(Acoustic(spatial_order=order).kernel.numpy()) == 0             # the 1-D stack
    assert _exact_sum(generate_convolution_kernel(order)) == 0                        # 2-D, centre 4 c0
    assert _exact_sum(Acoustic3D(spatial_order=order).kernel.numpy()) == 0


@pytest.mark.parametrize("order", ORDERS)
def test_compiled_coefficients_sum_to_zero(order):
    host = SimpleNamespace(dev="cpu", equation=Acoustic(spatial_order=order))
    lap, _ = _CompiledPropagator._build_fd_coefficients(host, order // 2)
    lap = lap.numpy()
    assert Fraction(float(lap[0])) == 2 * _exact_sum(lap[1:])
    c0, c = second_derivative_coefficients_fp32(order)
    assert lap[0] == -c0 and np.array_equal(lap[1:], c)                             # eager's stencil


def test_cuda_constants_are_the_helper_values():
    """The order 4/6/8 kernels compute their last coefficient as the fp32 remainder of
    the others (operators/laplace.cuh); every step is exact, so it is the helper's."""
    f32 = np.float32
    fractions = {4: (f32(5) / f32(2), [f32(4) / f32(3)]),
                 6: (f32(49) / f32(18), [f32(3) / f32(2), f32(-3) / f32(20)]),
                 8: (f32(205) / f32(72), [f32(8) / f32(5), f32(-1) / f32(5), f32(8) / f32(315)])}
    for order, (c0, inner) in fractions.items():
        last = f32(f32(0.5) * c0)
        for x in inner:
            last = f32(last - x)
        h0, h = second_derivative_coefficients_fp32(order)
        assert -h0 == c0 and np.array_equal(h[:-1], inner) and h[-1] == last


V, F0, T, L, OFF = 2000.0, 10.0, 0.45, 1600.0, 300.0


def _ricker(t):
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


def _analytic(times, n=4001):
    """2-D Green's function of p_tt = V^2 lap p + w(t) H(t) delta(x), convolved with w,
    at OFF: with tau = OFF cosh(u) / V, p = 1/(2 pi V^2) int_0^acosh(V t/OFF) w(t - tau) du."""
    out = np.zeros(len(times))
    for k, t in enumerate(times):
        if V * t > OFF:
            u = np.linspace(0.0, np.arccosh(V * t / OFF), n)
            y = _ricker(t - OFF * np.cosh(u) / V)
            out[k] = (y.sum() - 0.5 * (y[0] + y[-1])) * (u[1] - u[0]) / (2 * np.pi * V ** 2)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("order", (4, 6, 8, 10))
def test_fine_grid_record_converges_to_the_analytic_one(order):
    """dh 1.25 m (160 points per wavelength at 10 Hz): the zero-sum stencil gives
    4e-5; the rounded one gave 6.4e-4 (order 4) to 5.7e-4 (order 8)."""
    from sweep.propagator.torch import PropTorch
    dh = 1.25
    dt = 1e-4 * dh
    nt, n = int(round(T / dt)), int(round(L / dh)) + 1
    s = (n - 1) // 2
    prop = PropTorch(Acoustic(spatial_order=order, device="cuda"), shape=(n, n), dh=dh, dt=dt, nt=nt, abcn=30,
                     dev="cuda", impl="c")
    w = torch.tensor(_ricker(np.arange(nt) * dt), dtype=torch.float32, device="cuda")
    with torch.no_grad():
        d = prop(w, np.array([[s, s]]), np.array([[[s + int(round(OFF / dh)), s]]]),
                 models=[torch.full((n, n), V, device="cuda")])
    # the source adds w[k] after step k: p_tt = V^2 lap p + (dh / dt)^2 w delta; column k is p at (k + 1) dt
    trace = d[0, :, 0, 0].double().cpu().numpy() * dt ** 2 / dh ** 2
    ref = _analytic((np.arange(nt) + 1) * dt)
    assert np.linalg.norm(trace - ref) / np.linalg.norm(ref) < 1.5e-4


def _fine_grid(order, dh, memory=None):
    """The forward_check geometry on a grid of spacing dh, impl='c' (Courant 0.2)."""
    from sweep.propagator.torch import PropTorch
    dt = 1e-4 * dh
    nt, n = int(round(T / dt)), int(round(L / dh)) + 1
    s = (n - 1) // 2
    kw = {} if memory is None else dict(memory=memory)
    prop = PropTorch(Acoustic(spatial_order=order, device="cuda"), shape=(n, n), dh=dh, dt=dt, nt=nt, abcn=30,
                     dev="cuda", impl="c", **kw)
    run = lambda w: prop(w, np.array([[s, s]]), np.array([[[s + int(round(OFF / dh)), s]]]),
                         models=[torch.full((n, n), V, device="cuda")])
    return run, dt, nt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("order", (4, 6, 8, 10))
def test_finest_grid_record_reaches_the_float32_floor(order):
    """dh 0.625 m (320 points per wavelength): the record reaches eager's float32
    floor, 1.7e-5 .. 1.8e-5 (eager 1.3e-5 .. 2.0e-5).  Summed, orders 6 / 8 / 10
    stopped at 4.3e-5 / 5.7e-5 / 2.7e-4; in difference form order 4 gives 3.2e-5
    (Courant 0.2), which is why it keeps the summed form."""
    dh = 0.625
    run, dt, nt = _fine_grid(order, dh)
    w = torch.tensor(_ricker(np.arange(nt) * dt), dtype=torch.float32, device="cuda")
    with torch.no_grad():
        trace = run(w)[0, :, 0, 0].double().cpu().numpy() * dt ** 2 / dh ** 2
    ref = _analytic((np.arange(nt) + 1) * dt)
    assert np.linalg.norm(trace - ref) / np.linalg.norm(ref) < 2.5e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize("order", (4, 10))
def test_fine_grid_adjoint_record_converges_to_the_analytic_one(order):
    """The adjoint kernels: with the residual a Ricker reversed in time, the wavelet
    gradient read backwards is the Ricker record, carried receiver -> source by the
    adjoint step (grad_w[j] = sum_i S[i, j] r[i] = (G * w)[nt - 1 - j]).  dh 1.25 m:
    4.8e-5 (order 4) and 5.1e-5 (order 10), as the forward; summed, with the tap products
    fused against a rounded centre, 1.7e-4 and 2.0e-4."""
    from sweep.propagator.options import BoundarySaving   # no history: wavelet gradient only
    dh = 1.25
    run, dt, nt = _fine_grid(order, dh, BoundarySaving(storage="gpu"))
    w = torch.tensor(_ricker(np.arange(nt) * dt), dtype=torch.float32, device="cuda", requires_grad=True)
    r = torch.tensor(_ricker((nt - 1 - np.arange(nt)) * dt), dtype=torch.float32, device="cuda")
    (run(w)[0, :, 0, 0] * r).sum().backward()
    trace = w.grad.flip(0).double().cpu().numpy() * dt ** 2 / dh ** 2
    ref = _analytic((np.arange(nt) + 1) * dt)
    assert np.linalg.norm(trace - ref) / np.linalg.norm(ref) < 7e-5
