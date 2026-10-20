"""Unit tests for the asymmetric collocated first-derivative operators.

Reference: Bartana, Xu & Rickett (SLB), *Solving the elastodynamic
equations in curvilinear coordinates using asymmetric operators*.
"""

import numpy as np
import pytest
import torch

from sweep.operators.asymmetric import (
    CollocatedAsymmetricOperator,
    asymmetric_coefficients,
    symbol,
)

KMAX = 0.56


def test_interior_symbol_matches_paper_figure_1():
    """Im part follows -k in band, Re part is zero in band and -2 at Nyquist."""
    op = CollocatedAsymmetricOperator(half_width=8, kmax_frac=KMAX,
                                      nyquist_real=-2.0)
    theta = np.linspace(0.0, np.pi, 4001)
    s = symbol(op.coef, op.offsets, theta)
    band = theta <= KMAX * np.pi
    assert np.abs(s[band].imag + theta[band]).max() < 1e-4
    assert np.abs(s[band].real).max() < 1e-3
    assert s[-1].real == pytest.approx(-2.0, abs=1e-8)
    # The whole point: the symbol does not vanish at high wavenumbers, so
    # the collocated grid has no odd-even null mode.  A central operator
    # would give |symbol| -> 0 at Nyquist.
    assert np.abs(s[theta >= 0.7 * np.pi]).min() > 1.5


def test_operator_is_consistent_to_machine_precision():
    op = CollocatedAsymmetricOperator(half_width=8)
    assert abs(op.coef.sum()) < 1e-12
    assert np.dot(op.coef, op.offsets) == pytest.approx(1.0, abs=1e-12)
    # Divergence stencil is the negative transpose of the gradient one.
    assert np.allclose(op.coef_adj, -op.coef[::-1])
    assert abs(op.coef_adj.sum()) < 1e-12
    assert np.dot(op.coef_adj, op.offsets) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("adjoint", [False, True])
def test_boundary_stencils_are_consistent_and_not_null(adjoint):
    op = CollocatedAsymmetricOperator(half_width=8)
    for row in op.boundary_report(adjoint=adjoint):
        assert abs(row["sum"]) < 1e-10, row
        assert row["moment1"] == pytest.approx(1.0, abs=1e-10), row
        # A vanishing symbol at Nyquist would trap the vertical odd-even
        # mode in the first L rows, where nothing damps it.
        assert row["nyquist_symbol"] > 0.4, row
        assert row["offsets"][0] == -row["row"], row      # never above the surface
        assert row["max_abs_coef"] < 1.5, row


def test_interior_derivative_accuracy():
    op = CollocatedAsymmetricOperator(half_width=8)
    n, lam = 200, 20.0
    z = np.arange(n, dtype=np.float64)
    f = np.sin(2 * np.pi * z / lam)
    ref = (2 * np.pi / lam) * np.cos(2 * np.pi * z / lam)
    u = torch.tensor(f, dtype=torch.float64)[None, None, :, None].repeat(1, 1, 1, 3)
    inner = slice(op.L + 2, n - op.L - 2)
    for fn in (op.d_z, op.d_z_adj):
        d = fn(u)[0, 0, :, 1].numpy()
        assert np.abs(d[inner] - ref[inner]).max() / np.abs(ref).max() < 1e-4
    ux = torch.tensor(np.tile(f[None, :], (5, 1)), dtype=torch.float64)[None, None]
    dx = op.d_x(ux)[0, 0, 2].numpy()
    assert np.abs(dx[inner] - ref[inner]).max() / np.abs(ref).max() < 1e-4


def test_free_surface_closure_never_reads_above_the_surface():
    op = CollocatedAsymmetricOperator(half_width=4)
    surf, n = op.L, 60
    for fn in (op.d_z, op.d_z_adj):
        mat = np.zeros((n, n))
        for j in range(n):
            e = torch.zeros(1, 1, n, 1, dtype=torch.float64)
            e[0, 0, j, 0] = 1.0
            mat[:, j] = fn(e, surf)[0, 0, :, 0].numpy()
        # Output rows inside the medium must not touch the void rows.
        assert np.abs(mat[surf:n - op.L, :surf]).max() == 0.0


def test_free_surface_closure_is_spectrally_neutral():
    """The closure must not add energy: the coupled semi-discrete system
    (sigma pinned on the surface row) has a zero spectral abscissa."""
    op = CollocatedAsymmetricOperator(half_width=4)
    surf, n = op.L, 80

    def dense(fn):
        mat = np.zeros((n, n))
        for j in range(n):
            e = torch.zeros(1, 1, n, 1, dtype=torch.float64)
            e[0, 0, j, 0] = 1.0
            mat[:, j] = fn(e, surf)[0, 0, :, 0].numpy()
        return mat[surf:, surf:]

    grad, div = dense(op.d_z), dense(op.d_z_adj)
    m = grad.shape[0]
    grad = grad.copy()
    grad[0, :] = 0.0                      # free surface: sigma rate from the BC
    a = np.zeros((2 * m, 2 * m))
    a[:m, m:] = div
    a[m:, :m] = grad
    damp = np.zeros(m)
    damp[-20:] = np.linspace(0.0, 3.0, 20)          # absorb at the base
    a[:m, :m] -= np.diag(damp)
    a[m:, m:] -= np.diag(damp)
    assert np.linalg.eigvals(a).real.max() < 1e-9


def test_zero_nyquist_recovers_a_central_operator():
    """Control: without the Nyquist constraint the symmetric part of the
    stencil vanishes and the odd-even null at Nyquist comes back."""
    op = CollocatedAsymmetricOperator(half_width=8, nyquist_real=0.0)
    assert np.allclose(op.coef, -op.coef[::-1], atol=1e-8)
    assert abs(symbol(op.coef, op.offsets, [np.pi])[0]) < 1e-8


def test_cfl_dt_scales_as_expected():
    op = CollocatedAsymmetricOperator(half_width=8)
    assert 2.0 < op.max_symbol() < 2.5
    dt1 = op.cfl_dt(2000.0, (10.0, 10.0))
    assert dt1 == pytest.approx(op.cfl_dt(4000.0, (10.0, 10.0)) * 2, rel=1e-9)
    assert dt1 == pytest.approx(op.cfl_dt(2000.0, (5.0, 5.0)) * 2, rel=1e-9)


def test_zero_closure_is_inconsistent_at_the_surface():
    """Control for the paper's literal "stencil points above the free
    surface contribute zero": applied to a flux that vanishes on the
    surface row it recovers only ~40% of the derivative, whereas the
    truncated (no-ghost) stencils are accurate to a couple of percent."""
    op = CollocatedAsymmetricOperator(half_width=8)
    surf, n, lam = op.L, 80, 20.0
    z = np.arange(n, dtype=np.float64) - surf
    f = np.where(z >= 0, np.sin(2 * np.pi * z / lam)
                 + 0.4 * (1.0 - np.cos(2 * np.pi * z / lam)), 0.0)
    ref = (2 * np.pi / lam) * (np.cos(2 * np.pi * z / lam)
                               + 0.4 * np.sin(2 * np.pi * z / lam))
    u = torch.tensor(f, dtype=torch.float64)[None, None, :, None].repeat(1, 1, 1, 3)
    scale = np.abs(ref[surf:surf + 2 * op.L]).max()

    def err(mode):
        d = op.d_z_adj(u, surf, mode)[0, 0, :, 1].numpy()
        return np.abs(d[surf] - ref[surf]) / scale

    assert err("zero") > 0.3
    assert err("shifted") < 0.05
    assert err("mirror_odd") < 0.05
