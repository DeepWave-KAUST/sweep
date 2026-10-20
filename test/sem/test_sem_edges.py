"""Stage-0 gate for the fluid-solid coupling port: the Quad9 edge metric.

Surface integrals (free-surface traction, fluid-solid coupling) need the 1-D
edge Jacobian and a correctly oriented normal. These gates need no SPECFEM:
they pin the metric against (a) exact flat-mesh geometry, (b) direct SEM
differentiation of the node coordinates, (c) a densely-sampled Q2 arclength,
and (d) the geometric orientation rule that replaces SPECFEM's z-up edge table.
"""
import math

import numpy as np
import torch

from sweep.sem.basis.quad9 import q2_basis_at
from sweep.sem.mesh.edges import edge_metric, orient_normal
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9, build_topography_quad9

DT = torch.float64
NEX, NEZ, XM, ZM, N = 4, 3, 400.0, 300.0, 4
DXE, DZE = XM / NEX, ZM / NEZ


def _flat():
    return build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=XM, z_max=ZM,
                                    n_order=N, dtype=DT)


def _topo(amp=40.0):
    z_top = lambda x: amp * torch.sin(2 * math.pi * x / XM)
    return build_topography_quad9(nex=NEX, nez=NEZ, x_max=XM, z_bottom=ZM,
                                  z_top_fn=z_top, n_order=N, dtype=DT)


# --------------------------------------------------------------------------- #
# (a) flat mesh: exact geometry
# --------------------------------------------------------------------------- #
def test_flat_edge_metric_exact():
    m = _flat()
    e = torch.arange(m.n_elements)
    for side, jac_exp, size in (("top", DXE / 2, DXE), ("bottom", DXE / 2, DXE),
                                ("left", DZE / 2, DZE), ("right", DZE / 2, DZE)):
        em = edge_metric(m, e, side)
        assert torch.allclose(em.jac1d, torch.full_like(em.jac1d, jac_exp), atol=1e-12)
        # ∮ dΓ over one face == the element's edge length, to roundoff
        assert torch.allclose(em.weight.sum(1),
                              torch.full((m.n_elements,), size, dtype=DT), atol=1e-12)
        # unit tangent / normal
        assert torch.allclose(em.tx**2 + em.tz**2, torch.ones_like(em.tx), atol=1e-12)
        assert torch.allclose(em.nx**2 + em.nz**2, torch.ones_like(em.nx), atol=1e-12)
        assert torch.allclose(em.tx * em.nx + em.tz * em.nz,
                              torch.zeros_like(em.tx), atol=1e-12)   # t ⟂ n


def test_raw_normal_is_unoriented():
    """The raw (tangent rotated −90°) normal is the SAME for opposite faces —
    which is exactly why callers must orient it geometrically, and why
    transcribing SPECFEM's z-up edge→sign table would silently flip it."""
    m = _flat()
    e = torch.arange(m.n_elements)
    top, bot = edge_metric(m, e, "top"), edge_metric(m, e, "bottom")
    for em in (top, bot):                       # both give (0, -1) on a flat mesh
        assert torch.allclose(em.nx, torch.zeros_like(em.nx), atol=1e-12)
        assert torch.allclose(em.nz, -torch.ones_like(em.nz), atol=1e-12)
    left, right = edge_metric(m, e, "left"), edge_metric(m, e, "right")
    for em in (left, right):                    # both give (1, 0)
        assert torch.allclose(em.nx, torch.ones_like(em.nx), atol=1e-12)
        assert torch.allclose(em.nz, torch.zeros_like(em.nz), atol=1e-12)


# --------------------------------------------------------------------------- #
# (b) curved mesh: metric == direct SEM differentiation of the coordinates
# --------------------------------------------------------------------------- #
def test_curved_metric_matches_sem_differentiation():
    m = _topo()
    e = torch.arange(m.n_elements)
    D = m.deriv
    # top face (η = const): ∂x/∂ξ at node jx = Σ_k D[jx,k] x_e[0,k]
    em = edge_metric(m, e, "top")
    dx_ds, dz_ds = em.tx * em.jac1d, em.tz * em.jac1d
    dx_ref = torch.einsum("xk,ek->ex", D, m.x_e[e, 0, :])
    dz_ref = torch.einsum("xk,ek->ex", D, m.z_e[e, 0, :])
    assert (dx_ds - dx_ref).abs().max() < 1e-9
    assert (dz_ds - dz_ref).abs().max() < 1e-9
    # left face (ξ = const): ∂x/∂η at node jz = Σ_k D[jz,k] x_e[k,0]
    em = edge_metric(m, e, "left")
    dx_ds, dz_ds = em.tx * em.jac1d, em.tz * em.jac1d
    dx_ref = torch.einsum("zk,ek->ez", D, m.x_e[e, :, 0])
    dz_ref = torch.einsum("zk,ek->ez", D, m.z_e[e, :, 0])
    assert (dx_ds - dx_ref).abs().max() < 1e-9
    assert (dz_ds - dz_ref).abs().max() < 1e-9


# --------------------------------------------------------------------------- #
# (c) curved mesh: GLL surface quadrature == dense-sampled Q2 arclength
# --------------------------------------------------------------------------- #
def test_curved_arclength_matches_dense_q2():
    m = _topo()
    top_e = torch.arange(NEX)                       # jz = 0 row of elements
    em = edge_metric(m, top_e, "top")
    gll_len = em.weight.sum(1)                      # Σ w·jac1d per element

    # dense arclength of the Q2 edge curve: at η = -1 the Q2 η-basis is [1,0,0],
    # so the face curve is the Q2 curve through control_xy[e, 0, :, :].
    s = torch.linspace(-1, 1, 20001, dtype=DT)
    psi = q2_basis_at(s)                            # (n_s, 3)
    ctrl = m.control_xy[top_e, 0, :, :]             # (n_e, 3, 2)
    xy = torch.einsum("sj,ejd->esd", psi, ctrl)     # (n_e, n_s, 2)
    dense_len = (xy[:, 1:, :] - xy[:, :-1, :]).norm(dim=-1).sum(1)

    rel = ((gll_len - dense_len).abs() / dense_len).max()
    assert rel < 1e-6, f"GLL surface quadrature vs dense arclength rel={rel:.2e}"
    assert (gll_len > DXE).all()                    # curved edge is longer than flat


# --------------------------------------------------------------------------- #
# (d) the orientation rule that replaces SPECFEM's z-up edge table
# --------------------------------------------------------------------------- #
def test_orient_normal_points_toward_target():
    m = _flat()
    # vertical neighbours: element (jz, jx) has id jz*nex + jx
    upper = torch.arange(NEX)                       # top row
    lower = upper + NEX                             # the row below (larger z)
    em = edge_metric(m, upper, "bottom")
    assert torch.allclose(em.nz, -torch.ones_like(em.nz), atol=1e-12)   # raw: wrong way
    oz = orient_normal(em, m, upper, lower)
    assert torch.allclose(oz.nz, torch.ones_like(oz.nz), atol=1e-12)    # now toward lower
    # flipping the target flips the sign
    ou = orient_normal(edge_metric(m, lower, "top"), m, lower, upper)
    assert torch.allclose(ou.nz, -torch.ones_like(ou.nz), atol=1e-12)
    # jac1d / weight are untouched by orientation
    assert torch.allclose(oz.jac1d, em.jac1d)


def test_orient_normal_on_curved_interface():
    m = _topo()
    upper, lower = torch.arange(NEX), torch.arange(NEX) + NEX
    oz = orient_normal(edge_metric(m, upper, "bottom"), m, upper, lower)
    cz = m.z_e.flatten(1).mean(1)
    dz = (cz[lower] - cz[upper]).reshape(-1, 1)
    # every node's normal points toward the lower element
    assert ((oz.nx * 0.0 + oz.nz * dz) > 0).all()
    assert torch.allclose(oz.nx**2 + oz.nz**2, torch.ones_like(oz.nx), atol=1e-12)
