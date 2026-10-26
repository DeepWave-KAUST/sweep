"""Stage-3 gates: the fluid-solid interface operators.

THE gate is the **discrete transpose** <C u, chi> == <u, C^T chi> to machine
precision on random vectors. It needs no SPECFEM, and it is exactly the property
that makes the FWI kernel exact — autograd then differentiates the fused step and
reproduces the adjoint coupling for free, which is why (unlike SPECFEM) we never
hand-code an adjoint coupling at all.

Also gated: the normal really is fluid-outward (the sign trap that a transcribed
z-up edge table would silently flip), and detection finds the right faces.
"""
import pytest
import torch

from sweep.sem.coupling import FluidSolidInterface
from sweep.sem.mesh.domains import DomainMask
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9, build_topography_quad9

DT = torch.float64
NEX, NEZ, DE = 4, 6, 100.0
Z_IF = 3 * DE                       # fluid above, solid below


def _flat():
    return build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=NEX * DE,
                                    z_max=NEZ * DE, n_order=4, dtype=DT)


def _layered(m, z_if=Z_IF):
    zc = m.z_e.flatten(1).mean(1)
    return DomainMask.from_element_flags(m, zc < z_if)


def _nodes_z(m):
    zg = torch.zeros(m.n_global, dtype=DT); zg[m.ibool] = m.z_e
    return zg


# --------------------------------------------------------------------------- #
# detection + orientation
# --------------------------------------------------------------------------- #
def test_detect_flat_interface():
    m = _flat()
    ifc = FluidSolidInterface.detect(m, _layered(m))
    assert ifc.n_face == NEX                       # one face per column
    # every interface node sits on z = Z_IF
    zg = _nodes_z(m)
    assert torch.allclose(zg[ifc.node_idx], torch.full_like(ifc.weight, Z_IF),
                          atol=1e-9)
    # total interface length == the model width
    assert abs(float(ifc.weight.sum()) - NEX * DE) < 1e-9


def test_normal_is_outward_from_the_fluid():
    """Fluid on top (z-down mesh) => the fluid-outward normal points DOWN (+z).
    This is the sign SPECFEM's z-up IBOTTOM/ITOP table would have flipped."""
    m = _flat()
    ifc = FluidSolidInterface.detect(m, _layered(m))
    assert torch.allclose(ifc.nx, torch.zeros_like(ifc.nx), atol=1e-12)
    assert torch.allclose(ifc.nz, torch.ones_like(ifc.nz), atol=1e-12)   # +z = down

    # flip the domains: fluid BELOW => the normal must flip to -z (up)
    zc = m.z_e.flatten(1).mean(1)
    ifc2 = FluidSolidInterface.detect(m, DomainMask.from_element_flags(m, zc > Z_IF))
    assert torch.allclose(ifc2.nz, -torch.ones_like(ifc2.nz), atol=1e-12)


def test_detect_vertical_interface():
    """A left/right domain split exercises the xi=const face branch."""
    m = _flat()
    xc = m.x_e.flatten(1).mean(1)
    ifc = FluidSolidInterface.detect(m, DomainMask.from_element_flags(m, xc < 2 * DE))
    assert ifc.n_face == NEZ
    assert torch.allclose(ifc.nx, torch.ones_like(ifc.nx), atol=1e-12)   # +x
    assert torch.allclose(ifc.nz, torch.zeros_like(ifc.nz), atol=1e-12)
    assert abs(float(ifc.weight.sum()) - NEZ * DE) < 1e-9


def test_no_interface_when_single_domain():
    m = _flat()
    allac = DomainMask.from_element_flags(m, torch.ones(m.n_elements, dtype=torch.bool))
    assert FluidSolidInterface.detect(m, allac).n_face == 0


# --------------------------------------------------------------------------- #
# THE gate: exact discrete transpose
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("curved", [False, True])
def test_discrete_transpose(curved):
    """<apply_ac(u), chi> == <u, apply_el(chi)> to machine precision.

    Both sides use the same weights/nodes/normal, so C_ac<-el and C_el<-ac are
    exact transposes. This is what makes the FWI kernel exact; it must hold on a
    curved interface too.
    """
    if curved:
        z_top = lambda x: 30.0 * torch.sin(3.14159 * x / (NEX * DE))
        m = build_topography_quad9(nex=NEX, nez=NEZ, x_max=NEX * DE,
                                   z_bottom=NEZ * DE, z_top_fn=z_top,
                                   n_order=4, dtype=DT)
    else:
        m = _flat()
    ifc = FluidSolidInterface.detect(m, _layered(m))
    assert ifc.n_face > 0

    torch.manual_seed(0)
    like = torch.zeros(m.n_global, dtype=DT)
    for _ in range(5):
        u_x = torch.randn(m.n_global, dtype=DT)
        u_z = torch.randn(m.n_global, dtype=DT)
        chi = torch.randn(m.n_global, dtype=DT)

        lhs = (ifc.apply_ac(u_x, u_z, like) * chi).sum()          # <C u, chi>
        sx, sz = ifc.apply_el(chi, like)
        rhs = (u_x * sx + u_z * sz).sum()                          # <u, C^T chi>
        scale = max(abs(float(lhs)), 1e-30)
        assert abs(float(lhs - rhs)) / scale < 1e-13, f"{lhs} vs {rhs}"


def test_apply_ac_is_the_normal_flux():
    """A uniform solid displacement u = (0, 1) through a flat, fluid-on-top
    interface must give exactly ∮ (u.n) phi = ∮ 1 phi = the interface length."""
    m = _flat()
    ifc = FluidSolidInterface.detect(m, _layered(m))
    like = torch.zeros(m.n_global, dtype=DT)
    u_x = torch.zeros(m.n_global, dtype=DT)
    u_z = torch.ones(m.n_global, dtype=DT)              # unit displacement, +z (down)
    s = ifc.apply_ac(u_x, u_z, like)
    assert abs(float(s.sum()) - NEX * DE) < 1e-9       # n = +z here, so u.n = 1
    # only interface nodes are touched
    touched = torch.zeros(m.n_global, dtype=torch.bool)
    touched[ifc.node_idx.reshape(-1)] = True
    assert bool((s[~touched] == 0).all())
