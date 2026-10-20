"""Degeneracy test: ElasticSEM2DDeformed on an axis-aligned Quad9 mesh
must reproduce ElasticSEM2D (Quad4) bit-for-bit (modulo floating-point).

This is the headline correctness check for the deformed kernel — if
the per-node Jacobian / inv_J indexing is wrong anywhere, this test
catches it immediately.
"""

import pytest
import torch

from sweep.sem import (
    ElasticSEM2D,
    ElasticSEM2DDeformed,
    build_axis_aligned_quad9,
    build_quad4_mesh,
)


def _uniform_model(mesh, vp=2200.0, vs=1100.0, rho=2000.0,
                   dtype=torch.float64):
    shape = (mesh.n_elements, mesh.ngll, mesh.ngll)
    return (torch.full(shape, vp,  dtype=dtype),
            torch.full(shape, vs,  dtype=dtype),
            torch.full(shape, rho, dtype=dtype))


def test_mass_matches_quad4():
    mq4 = build_quad4_mesh(nex=6, nez=5, x_max=600.0, z_max=500.0, n_order=4, dtype=torch.float64)
    mq9 = build_axis_aligned_quad9(nex=6, nez=5, x_max=600.0, z_max=500.0,
                                   n_order=4, dtype=torch.float64)
    vp4, vs4, rho4 = _uniform_model(mq4)
    vp9, vs9, rho9 = _uniform_model(mq9)
    eq4 = ElasticSEM2D.build(mq4, vp4, vs4, rho4)
    eq9 = ElasticSEM2DDeformed.build(mq9, vp9, vs9, rho9)
    assert torch.allclose(eq4.mass, eq9.mass, rtol=1e-12, atol=1e-12), \
        (eq4.mass - eq9.mass).abs().max().item()


def test_stiffness_matches_quad4_on_random_field():
    """K · u — exact agreement with the Quad4 operator on the same field."""
    nex, nez = 6, 5
    mq4 = build_quad4_mesh(nex=nex, nez=nez, x_max=600.0, z_max=500.0, n_order=4, dtype=torch.float64)
    mq9 = build_axis_aligned_quad9(nex=nex, nez=nez, x_max=600.0, z_max=500.0,
                                   n_order=4, dtype=torch.float64)
    vp4, vs4, rho4 = _uniform_model(mq4)
    vp9, vs9, rho9 = _uniform_model(mq9)
    eq4 = ElasticSEM2D.build(mq4, vp4, vs4, rho4)
    eq9 = ElasticSEM2DDeformed.build(mq9, vp9, vs9, rho9)

    torch.manual_seed(0)
    u_x = torch.randn(mq4.n_global, dtype=torch.float64)
    u_z = torch.randn(mq4.n_global, dtype=torch.float64)
    K4x, K4z = eq4.stiffness_apply(u_x, u_z)
    K9x, K9z = eq9.stiffness_apply(u_x, u_z)
    rel_x = (K4x - K9x).norm() / max(K4x.norm().item(), 1e-30)
    rel_z = (K4z - K9z).norm() / max(K4z.norm().item(), 1e-30)
    assert rel_x < 1e-10, rel_x.item()
    assert rel_z < 1e-10, rel_z.item()


def test_stiffness_matches_on_p_wave_eigenmode():
    """Tighter check on a smooth field: K · u_P should be identical."""
    mq4 = build_quad4_mesh(nex=8, nez=6, x_max=800.0, z_max=600.0, n_order=4, dtype=torch.float64)
    mq9 = build_axis_aligned_quad9(nex=8, nez=6, x_max=800.0, z_max=600.0,
                                   n_order=4, dtype=torch.float64)
    vp4, vs4, rho4 = _uniform_model(mq4)
    vp9, vs9, rho9 = _uniform_model(mq9)
    eq4 = ElasticSEM2D.build(mq4, vp4, vs4, rho4)
    eq9 = ElasticSEM2DDeformed.build(mq9, vp9, vs9, rho9)

    kx = 2 * torch.pi / mq4.x_max
    u_x4 = torch.cos(kx * mq4.x_global)
    u_z4 = torch.zeros_like(u_x4)
    u_x9 = torch.cos(kx * mq9.x_global)
    u_z9 = torch.zeros_like(u_x9)
    K4x, _ = eq4.stiffness_apply(u_x4, u_z4)
    K9x, _ = eq9.stiffness_apply(u_x9, u_z9)
    rel = (K4x - K9x).norm() / max(K4x.norm().item(), 1e-30)
    assert rel < 1e-10, rel.item()
