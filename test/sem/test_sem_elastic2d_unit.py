"""Unit tests for sweep.sem.ElasticSEM2D — 2-D isotropic P-SV elastic.

Covers the math building blocks before any time-stepping is exercised:
mass integral, rigid-body null space, K-symmetry, K-positive-semi-
definite, and the plane-wave P/S eigenmode test (the headline check
that the stiffness picks up the right wavespeeds).
"""

import pytest
import torch

from sweep.sem import ElasticSEM2D, build_quad4_mesh


@pytest.fixture
def small_mesh():
    return build_quad4_mesh(nex=6, nez=5, x_max=600.0, z_max=500.0, n_order=4, dtype=torch.float64)


def _uniform_model(mesh, vp=2200.0, vs=1100.0, rho=2000.0,
                   dtype=torch.float64):
    shape = (mesh.n_elements, mesh.ngll, mesh.ngll)
    return (torch.full(shape, vp,  dtype=dtype),
            torch.full(shape, vs,  dtype=dtype),
            torch.full(shape, rho, dtype=dtype))


def test_mass_integrates_rho_times_area(small_mesh):
    vp, vs, rho = _uniform_model(small_mesh)
    eq = ElasticSEM2D.build(small_mesh, vp, vs, rho)
    area = small_mesh.x_max * small_mesh.z_max
    expected = float(rho[0, 0, 0]) * area
    assert abs(eq.mass.sum().item() - expected) < 1e-8 * expected


def test_stiffness_zero_on_constant_translation(small_mesh):
    """K · (c_x, c_z) ≈ 0 for any constant translation."""
    vp, vs, rho = _uniform_model(small_mesh)
    eq = ElasticSEM2D.build(small_mesh, vp, vs, rho)
    u_x = torch.full((small_mesh.n_global,),  3.14, dtype=torch.float64)
    u_z = torch.full((small_mesh.n_global,), -1.59, dtype=torch.float64)
    Kx, Kz = eq.stiffness_apply(u_x, u_z)
    # Float64 roundoff: K element values scale as (λ+2μ)·u·w·ε_mach
    #                ≈ 2e10 · 3 · 0.5 · 5e-16 ≈ 1e-5  per element,
    # accumulated across ~6 elements per shared node → ~1e-4.
    tol = 1e-3
    assert Kx.abs().max().item() < tol, Kx.abs().max().item()
    assert Kz.abs().max().item() < tol, Kz.abs().max().item()


def test_stiffness_symmetric(small_mesh):
    """⟨(u_x, u_z), K(v_x, v_z)⟩ ≈ ⟨(v_x, v_z), K(u_x, u_z)⟩."""
    vp, vs, rho = _uniform_model(small_mesh)
    eq = ElasticSEM2D.build(small_mesh, vp, vs, rho)
    torch.manual_seed(0)
    u_x = torch.randn(small_mesh.n_global, dtype=torch.float64)
    u_z = torch.randn(small_mesh.n_global, dtype=torch.float64)
    v_x = torch.randn(small_mesh.n_global, dtype=torch.float64)
    v_z = torch.randn(small_mesh.n_global, dtype=torch.float64)
    Kv_x, Kv_z = eq.stiffness_apply(v_x, v_z)
    Ku_x, Ku_z = eq.stiffness_apply(u_x, u_z)
    a = ((u_x * Kv_x).sum() + (u_z * Kv_z).sum()).item()
    b = ((v_x * Ku_x).sum() + (v_z * Ku_z).sum()).item()
    rel = abs(a - b) / max(abs(a), abs(b), 1e-30)
    assert rel < 1e-10, rel


def test_stiffness_psd(small_mesh):
    """⟨u, K u⟩ ≥ 0 for any displacement field."""
    vp, vs, rho = _uniform_model(small_mesh)
    eq = ElasticSEM2D.build(small_mesh, vp, vs, rho)
    torch.manual_seed(1)
    u_x = torch.randn(small_mesh.n_global, dtype=torch.float64)
    u_z = torch.randn(small_mesh.n_global, dtype=torch.float64)
    Kx, Kz = eq.stiffness_apply(u_x, u_z)
    quad = ((u_x * Kx).sum() + (u_z * Kz).sum()).item()
    assert quad >= -1e-9, quad


def _interior_mask(mesh, *, pad=0.15):
    return (
        (mesh.x_global > pad * mesh.x_max)
        & (mesh.x_global < (1 - pad) * mesh.x_max)
        & (mesh.z_global > pad * mesh.z_max)
        & (mesh.z_global < (1 - pad) * mesh.z_max)
    )


def test_p_wave_eigenmode(small_mesh):
    """Plane P-wave u_P = (cos(kx), 0): K u / M ≈ +v_p² k² · u in interior.

    Boundary nodes pick up the natural-BC integral ∮ v·(σ·n) dΓ which is
    non-zero for a 1-D mode that doesn't satisfy traction-free on the
    test box edges, so we only check the deep interior.
    """
    vp_v, vs_v, rho_v = 2200.0, 1100.0, 2000.0
    vp, vs, rho = _uniform_model(small_mesh, vp_v, vs_v, rho_v)
    eq = ElasticSEM2D.build(small_mesh, vp, vs, rho)

    Lx = small_mesh.x_max
    kx = 2 * torch.pi / Lx
    u_x = torch.cos(kx * small_mesh.x_global)
    u_z = torch.zeros_like(u_x)
    Kx, Kz = eq.stiffness_apply(u_x, u_z)
    lap_x_proxy = Kx / eq.mass
    ref_x = (vp_v ** 2) * (kx ** 2) * u_x

    interior = _interior_mask(small_mesh)
    rel = (lap_x_proxy[interior] - ref_x[interior]).norm() / ref_x[interior].norm()
    assert rel < 1e-2, rel.item()
    # In the interior, K_z (which would represent shear coupling) should
    # be near zero; the P-wave generates no shear at interior nodes.
    assert Kz[interior].abs().max().item() < 1e-6 * Kx[interior].abs().max().item()


def test_s_wave_eigenmode(small_mesh):
    """Plane S-wave u_S = (0, cos(kx)): K_z u / M ≈ +v_s² k² · u_z in interior."""
    vp_v, vs_v, rho_v = 2200.0, 1100.0, 2000.0
    vp, vs, rho = _uniform_model(small_mesh, vp_v, vs_v, rho_v)
    eq = ElasticSEM2D.build(small_mesh, vp, vs, rho)

    Lx = small_mesh.x_max
    kx = 2 * torch.pi / Lx
    u_x = torch.zeros(small_mesh.n_global, dtype=torch.float64)
    u_z = torch.cos(kx * small_mesh.x_global)
    Kx, Kz = eq.stiffness_apply(u_x, u_z)
    lap_z_proxy = Kz / eq.mass
    ref_z = (vs_v ** 2) * (kx ** 2) * u_z

    interior = _interior_mask(small_mesh)
    rel = (lap_z_proxy[interior] - ref_z[interior]).norm() / ref_z[interior].norm()
    assert rel < 1e-2, rel.item()
    assert Kx[interior].abs().max().item() < 1e-6 * Kz[interior].abs().max().item()
