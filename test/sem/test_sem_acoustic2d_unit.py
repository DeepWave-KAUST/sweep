"""Unit tests for sweep.sem acoustic 2-D operators.

These do not exercise the time loop — that's covered by the smoketest /
comparison scripts. Here we check the building blocks: GLL exactness,
mass-vector integral, stiffness SPSD and consistency with the Laplacian.
"""

import pytest
import torch

from sweep.sem import (
    AcousticSEM2D,
    build_quad4_mesh,
    gll_nodes_weights,
    lagrange_derivative_matrix,
)


@pytest.fixture
def small_mesh():
    return build_quad4_mesh(nex=5, nez=4, x_max=500.0, z_max=400.0, n_order=4, dtype=torch.float64)


def test_gll_quadrature_exact_up_to_2nminus1():
    """∫_{-1}^1 x^k dx exact for k = 0..2N-1."""
    n = 4
    xi, w = gll_nodes_weights(n, dtype=torch.float64)
    for k in range(2 * n):  # up to 2N-1 = 7
        exact = 0.0 if k % 2 else 2.0 / (k + 1)
        approx = (w * xi**k).sum().item()
        assert abs(approx - exact) < 1e-12, f"k={k}"


def test_lagrange_derivative_polynomial():
    """D · x^k = k · x^{k-1} exact for k ≤ N."""
    n = 4
    xi, _ = gll_nodes_weights(n, dtype=torch.float64)
    D = lagrange_derivative_matrix(xi)
    for k in range(1, n + 1):
        ref = k * xi**(k - 1)
        got = D @ (xi**k)
        assert torch.allclose(got, ref, atol=1e-12), f"k={k}"


def test_mass_integrates_domain(small_mesh):
    """Sum of mass-vector entries equals area of the physical domain (ρ=c=1)."""
    vp_e = torch.ones(small_mesh.n_elements, small_mesh.ngll, small_mesh.ngll,
                      dtype=torch.float64)
    eq = AcousticSEM2D.build(small_mesh, vp_e)
    area = small_mesh.x_max * small_mesh.z_max
    assert abs(eq.mass.sum().item() - area) < 1e-9


def test_stiffness_zero_on_constants(small_mesh):
    """K applied to a constant field returns zero (rigid-body mode).

    Float64 roundoff floor ~ c² · u · w · ε_mach ≈ 7e6 · 5e-16 · few ≈ 1e-8.
    """
    vp_e = 1500.0 * torch.ones(
        small_mesh.n_elements, small_mesh.ngll, small_mesh.ngll, dtype=torch.float64
    )
    eq = AcousticSEM2D.build(small_mesh, vp_e)
    u_const = torch.full((small_mesh.n_global,), 3.14, dtype=torch.float64)
    Ku = eq.stiffness_apply(u_const)
    assert Ku.abs().max().item() < 1e-7, Ku.abs().max().item()


def test_stiffness_symmetric(small_mesh):
    """uᵀ K v ≈ vᵀ K u (K is symmetric — discretely up to floating point)."""
    torch.manual_seed(0)
    vp_e = (1500.0 + 100.0 * torch.randn(
        small_mesh.n_elements, small_mesh.ngll, small_mesh.ngll, dtype=torch.float64
    )).abs()
    eq = AcousticSEM2D.build(small_mesh, vp_e)
    u = torch.randn(small_mesh.n_global, dtype=torch.float64)
    v = torch.randn(small_mesh.n_global, dtype=torch.float64)
    a = (u * eq.stiffness_apply(v)).sum().item()
    b = (v * eq.stiffness_apply(u)).sum().item()
    assert abs(a - b) / max(abs(a), abs(b), 1e-30) < 1e-11


def test_stiffness_psd(small_mesh):
    """uᵀ K u ≥ 0 for any u (K is positive semi-definite for vp > 0)."""
    torch.manual_seed(1)
    vp_e = 2000.0 * torch.ones(
        small_mesh.n_elements, small_mesh.ngll, small_mesh.ngll, dtype=torch.float64
    )
    eq = AcousticSEM2D.build(small_mesh, vp_e)
    u = torch.randn(small_mesh.n_global, dtype=torch.float64)
    quad = (u * eq.stiffness_apply(u)).sum().item()
    assert quad >= -1e-9, quad


def test_stiffness_matches_laplacian_smooth(small_mesh):
    """For a smooth field u, M⁻¹ K u ≈ -c² ∇²u (verifies the weak form).

    Use a doubly-periodic-on-the-boundary mode  u = sin(kx x) sin(kz z)
    with kx, kz chosen so u and ∂u/∂n vanish on the four faces — that way
    the natural BC the weak form imposes (free / Neumann at corners,
    Dirichlet on the test space restricted to satisfy u=0) is consistent
    with the analytical ∇²u and the comparison is clean.
    """
    m = small_mesh
    c = 2000.0
    vp_e = c * torch.ones(m.n_elements, m.ngll, m.ngll, dtype=torch.float64)
    eq = AcousticSEM2D.build(m, vp_e)

    # u = sin(2π x / Lx) sin(2π z / Lz) — vanishes on the four faces.
    Lx, Lz = m.x_max, m.z_max
    kx = 2 * torch.pi / Lx
    kz = 2 * torch.pi / Lz
    u = torch.sin(kx * m.x_global) * torch.sin(kz * m.z_global)

    Ku = eq.stiffness_apply(u)
    lap_proxy = Ku / eq.mass
    ref = (c * c) * (kx * kx + kz * kz) * u

    # Restrict to a clean interior (away from boundary where lumped-mass
    # projection has its largest artefact).
    interior = (
        (m.x_global > 0.15 * Lx) & (m.x_global < 0.85 * Lx) &
        (m.z_global > 0.15 * Lz) & (m.z_global < 0.85 * Lz)
    )
    rel = (lap_proxy[interior] - ref[interior]).norm() / ref[interior].norm()
    assert rel < 1e-2, rel.item()
