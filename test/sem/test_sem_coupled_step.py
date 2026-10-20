"""Stage-4a gates: the fused fluid-solid step.

The plan originally called for an energy-conservation gate to catch a wrong
interface normal. **Measured, that gate has no teeth** — see
``test_normal_sign_is_invisible_to_the_fluid``: flipping n leaves the fluid
BIT-IDENTICAL and merely negates the solid, because S^ac = ∮(u·n)φ is invariant
under (u, n) -> (-u, -n). So the fluid, its energy, and any hydrophone cannot see
the sign at all; only the solid's direction can. (The naive
½χ̇ᵀMχ̇ + ½χᵀKχ is not this system's energy either — in the potential form the
fluid's *kinetic* energy is ½χᵀK_acχ and its potential energy involves p = −χ̈,
not χ̇.) These gates replace it:

1. a compression arriving from the fluid must push the solid AWAY from the fluid,
2. flipping n must push it the wrong way (proves the gate discriminates),
3. the coupling must actually be connected (guards a silently-dead interface),
4. the run stays bounded.
"""
import numpy as np
import pytest
import torch

from sweep.sem.coupling import FluidSolidInterface
from sweep.sem.equations.acoustic_elastic2d import AcousticElasticSEM2D
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9

DT = torch.float64
NEX, NEZ, DE = 8, 8, 100.0
Z_IF = 4 * DE                      # fluid above, solid below


def _build(flip=False):
    m = build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=NEX * DE, z_max=NEZ * DE,
                                 n_order=4, dtype=DT)
    shp = (m.n_elements, m.ngll, m.ngll)
    zc = m.z_e.flatten(1).mean(1)
    fluid = (zc < Z_IF).reshape(-1, 1, 1).expand(shp)
    vp = torch.where(fluid, 1500.0, 3000.0).to(DT).contiguous()
    vs = torch.where(fluid, 0.0, 1732.0).to(DT).contiguous()
    rho = torch.where(fluid, 1000.0, 2200.0).to(DT).contiguous()
    eq = AcousticElasticSEM2D.build(m, vp, vs, rho, dirichlet=("top",))
    if flip:                       # sabotage: reverse the interface normal
        i = eq.ifc
        eq.ifc = FluidSolidInterface(i.node_idx, i.weight, -i.nx, -i.nz, i.n_global)
    return m, eq


def _coords(m):
    xg = torch.zeros(m.n_global, dtype=DT); xg[m.ibool] = m.x_e
    zg = torch.zeros(m.n_global, dtype=DT); zg[m.ibool] = m.z_e
    return xg, zg


def _chi_bump_in_fluid(m, eq):
    """A χ bump in the water just above Γ, zero on the solid's dead DOFs."""
    xg, zg = _coords(m)
    b = torch.exp(-(((xg - 400.0) ** 2 + (zg - 300.0) ** 2) / (2 * 90.0 ** 2)))
    return b * eq.domain.node_in_acoustic.to(DT)


def _solid_push_sign(flip):
    """Single step from rest: with u = 0 the solid's own stiffness contributes
    nothing, so ü_z = M_el⁻¹ ∮ p n_z φ EXACTLY, and the scatter returns
    weight·p[k]·n_z to the same node k — so p[k] factors out and
    sign(ü_z[k]) == sign(p[k])·sign(n_z), with no dependence on source polarity.
    """
    m, eq = _build(flip)
    z = torch.zeros(m.n_global, dtype=DT)
    chi = _chi_bump_in_fluid(m, eq)
    chi_dd, _, uz_dd = eq.accelerations(chi, z, z, z, z, z)
    p = -chi_dd                                     # pressure
    inode = eq.ifc.node_idx.reshape(-1).unique()
    k = inode[p[inode].abs().argmax()]              # most-excited interface node
    assert p[k].abs() > 1e-9, "no pressure reached the interface"
    assert uz_dd[k].abs() > 1e-12, "the solid was not pushed at all"
    return torch.sign(p[k]).item(), torch.sign(uz_dd[k]).item()


# --------------------------------------------------------------------------- #
def test_compression_pushes_solid_away_from_the_fluid():
    """n is fluid-outward (+z here, fluid on top of a z-down mesh), so a
    compression (p > 0) must accelerate the solid in +z — away from the water."""
    sp, su = _solid_push_sign(flip=False)
    assert sp == su, f"sign(p)={sp} but sign(u_z_ddot)={su} — solid pushed the wrong way"


def test_flipped_normal_pushes_the_wrong_way():
    """The gate above must actually discriminate: with n reversed the solid is
    pushed INTO the fluid."""
    sp, su = _solid_push_sign(flip=True)
    assert sp == -su, "flipping the normal did not flip the solid's push — the "\
                      "sign gate has no teeth"


def test_coupling_is_actually_connected():
    """Guards the failure my first diagnostic nearly hid: a silently dead
    interface looks exactly like a passing 'stability' run."""
    m, eq = _build()
    assert eq.ifc.n_face == NEX
    assert abs(float(eq.ifc.weight.sum()) - NEX * DE) < 1e-9
    assert torch.allclose(eq.ifc.nz, torch.ones_like(eq.ifc.nz))      # fluid-outward
    # a fluid-only excitation must move the solid
    z = torch.zeros(m.n_global, dtype=DT)
    _, _, uz_dd = eq.accelerations(_chi_bump_in_fluid(m, eq), z, z, z, z, z)
    assert uz_dd.abs().max() > 1e-9


def test_normal_sign_is_invisible_to_the_fluid():
    """Documents the symmetry that killed the planned energy gate:
    S^ac = ∮(u·n)φ is invariant under (u, n) -> (-u, -n), so reversing n leaves χ
    BIT-IDENTICAL and exactly negates u. Any fluid-side observable (hydrophone,
    energy) is therefore blind to the sign — only the solid's direction sees it.
    """
    dt, nt = 2e-4, 400
    outs = []
    for flip in (False, True):
        m, eq = _build(flip)
        xg, zg = _coords(m)
        s = int(((xg - 400.0) ** 2 + (zg - 200.0) ** 2).argmin())
        pat = torch.zeros(m.n_global, dtype=DT); pat[s] = 1.0
        zero = torch.zeros(m.n_global, dtype=DT)
        t = np.arange(nt) * dt
        a = (np.pi * 12.0 * (t - 0.12)) ** 2
        wav = torch.tensor((1 - 2 * a) * np.exp(-a), dtype=DT)
        f = eq.init_state(dt, "cpu", DT)
        for n in range(nt):
            f = eq.step(f, [pat * wav[n], zero, zero], dt)
        outs.append(f)
    a, b = outs
    assert (a[0] - b[0]).abs().max() == 0.0        # chi bit-identical
    assert a[3].abs().max() > 1e-9                 # the solid did move
    assert (a[2] + b[2]).abs().max() == 0.0        # u_x -> -u_x exactly
    assert (a[3] + b[3]).abs().max() == 0.0        # u_z -> -u_z exactly


@pytest.mark.slow
def test_coupled_run_stays_bounded():
    """Closed box, source off after 0.3 s: no exponential growth."""
    dt, nt = 2e-4, 6000
    m, eq = _build()
    xg, zg = _coords(m)
    s = int(((xg - 400.0) ** 2 + (zg - 200.0) ** 2).argmin())
    pat = torch.zeros(m.n_global, dtype=DT); pat[s] = 1.0
    zero = torch.zeros(m.n_global, dtype=DT)
    t = np.arange(nt) * dt
    a = (np.pi * 12.0 * (t - 0.12)) ** 2
    wav = torch.tensor((1 - 2 * a) * np.exp(-a), dtype=DT)
    wav[t > 0.30] = 0.0
    f = eq.init_state(dt, "cpu", DT)
    peaks = []
    for n in range(nt):
        f = eq.step(f, [pat * wav[n], zero, zero], dt)
        if n % 500 == 0 and n * dt > 0.35:
            peaks.append(float(f[0].abs().max()))
    assert all(np.isfinite(peaks))
    assert max(peaks) / min(peaks) < 2.0, f"field is growing: {peaks}"
