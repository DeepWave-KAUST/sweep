"""Degeneracy: AcousticSEM2DDeformed on an axis-aligned Quad9 mesh must
reproduce AcousticSEM2D (Quad4) to machine precision."""
import torch

from sweep.sem import (
    AcousticSEM2D,
    AcousticSEM2DDeformed,
    build_axis_aligned_quad9,
    build_quad4_mesh,
)

DT = torch.float64


def _meshes(nex, nez):
    mq4 = build_quad4_mesh(nex=nex, nez=nez, x_max=100.0 * nex, z_max=100.0 * nez,
                           n_order=4, dtype=DT)
    mq9 = build_axis_aligned_quad9(nex=nex, nez=nez, x_max=100.0 * nex,
                                   z_max=100.0 * nez, n_order=4, dtype=DT)
    return mq4, mq9


def _eqs(mq4, mq9, vp=2200.0):
    vp4 = torch.full((mq4.n_elements, mq4.ngll, mq4.ngll), vp, dtype=DT)
    vp9 = torch.full((mq9.n_elements, mq9.ngll, mq9.ngll), vp, dtype=DT)
    eq4 = AcousticSEM2D.build(mq4, vp4, dirichlet=())
    eq9 = AcousticSEM2DDeformed.build(mq9, vp9, dirichlet=())
    return eq4, eq9


def test_mass_matches_quad4():
    mq4, mq9 = _meshes(6, 5)
    eq4, eq9 = _eqs(mq4, mq9)
    assert torch.allclose(eq4.mass, eq9.mass, rtol=1e-12, atol=1e-12), \
        (eq4.mass - eq9.mass).abs().max().item()


def test_stiffness_matches_on_random_field():
    mq4, mq9 = _meshes(6, 5)
    eq4, eq9 = _eqs(mq4, mq9)
    torch.manual_seed(0)
    u = torch.randn(mq4.n_global, dtype=DT)
    K4 = eq4.stiffness_apply(u)
    K9 = eq9.stiffness_apply(u)
    rel = (K4 - K9).norm() / max(K4.norm().item(), 1e-30)
    assert rel < 1e-10, rel.item()


def test_stiffness_matches_on_smooth_field():
    mq4, mq9 = _meshes(8, 6)
    eq4, eq9 = _eqs(mq4, mq9)
    kx = 2 * torch.pi / mq4.x_max
    u4 = torch.cos(kx * mq4.x_global)
    u9 = torch.cos(kx * mq9.x_global)
    rel = (eq4.stiffness_apply(u4) - eq9.stiffness_apply(u9)).norm() \
        / max(eq4.stiffness_apply(u4).norm().item(), 1e-30)
    assert rel < 1e-10, rel.item()
