"""Stage-2 gates: per-element domain flags, masked assembly, dead DOFs.

The load-bearing claim is that SPECFEM's "one mesh, one global numbering, flag
each element, assemble each operator over its own elements" needs **no** mesh
surgery in our data model. The gate: a fluid sub-domain embedded in a bigger
mesh must be **bit-identical** to the same problem meshed on its own — the
masked interface must behave exactly like the natural boundary it replaces —
and the dead DOFs must be **exactly** 0, not 1e-12.
"""
import numpy as np
import pytest
import torch

from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.mesh.domains import DomainMask, masked_inv_mass
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9

DT = torch.float64
NEX, DE = 4, 100.0
NEZ_FLUID, NEZ_FULL = 4, 8
Z_IF = NEZ_FLUID * DE                      # interface depth = 400


def _mesh(nez):
    return build_axis_aligned_quad9(nex=NEX, nez=nez, x_max=NEX * DE,
                                    z_max=nez * DE, n_order=4, dtype=DT)


def _node_coords(m):
    xg = torch.zeros(m.n_global, dtype=DT); xg[m.ibool] = m.x_e
    zg = torch.zeros(m.n_global, dtype=DT); zg[m.ibool] = m.z_e
    return xg, zg


def _ricker(nt, dt, f0=12.0, t0=0.12):
    t = np.arange(nt) * dt - t0
    a = (np.pi * f0 * t) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=DT)


# --------------------------------------------------------------------------- #
# DomainMask bookkeeping
# --------------------------------------------------------------------------- #
def test_domain_mask_from_vs_and_exclusivity():
    m = _mesh(NEZ_FULL)
    shp = (m.n_elements, m.ngll, m.ngll)
    zc = m.z_e.flatten(1).mean(1)
    vs = torch.where(zc < Z_IF, 0.0, 1200.0).reshape(-1, 1, 1).expand(shp).contiguous()
    d = DomainMask.from_vs(m, vs)
    assert d.n_acoustic == NEX * NEZ_FLUID
    assert d.n_elastic == NEX * (NEZ_FULL - NEZ_FLUID)
    assert bool((d.is_acoustic ^ d.is_elastic).all())          # exclusive+complete
    # interface nodes belong to BOTH domains — that is the point
    _, zg = _node_coords(m)
    on_if = (zg - Z_IF).abs() < 1e-6
    assert bool((d.node_in_acoustic & d.node_in_elastic)[on_if].all())
    # and strictly-interior nodes belong to exactly one
    deep = zg > Z_IF + 1e-6
    assert bool((~d.node_in_acoustic[deep]).all())


def test_domain_mask_rejects_mixed_element():
    m = _mesh(2)
    vs = torch.zeros((m.n_elements, m.ngll, m.ngll), dtype=DT)
    vs[0, 0, 0] = 1200.0                                   # one node solid
    with pytest.raises(ValueError, match="wholly acoustic or wholly elastic"):
        DomainMask.from_vs(m, vs)


def test_masked_inv_mass_is_static_and_exact():
    mass = torch.tensor([2.0, 0.0, 4.0], dtype=DT)
    alive = torch.tensor([True, False, True])
    inv = masked_inv_mass(mass, alive)
    assert torch.equal(inv, torch.tensor([0.5, 0.0, 0.25], dtype=DT))
    assert torch.isfinite(inv).all()                       # no 1/0 leaked in
    # gradient must NOT flow into a dead DOF (static branch, not clamp(min=eps))
    mg = mass.clone().requires_grad_(True)
    masked_inv_mass(mg, alive).sum().backward()
    assert mg.grad[1] == 0.0


# --------------------------------------------------------------------------- #
# THE gate: embedded fluid sub-domain == fluid-only mesh, bit-for-bit
# --------------------------------------------------------------------------- #
def test_embedded_fluid_subdomain_is_bit_identical_to_fluid_only_mesh():
    big, small = _mesh(NEZ_FULL), _mesh(NEZ_FLUID)
    shp_b = (big.n_elements, big.ngll, big.ngll)
    shp_s = (small.n_elements, small.ngll, small.ngll)

    zc = big.z_e.flatten(1).mean(1)
    dom = DomainMask.from_element_flags(big, zc < Z_IF)
    vp_b = torch.full(shp_b, 1500.0, dtype=DT)
    rho_b = torch.full(shp_b, 1000.0, dtype=DT)
    eq_b = AcousticPotentialSEM2D.build(big, vp_b, rho_b, dirichlet=("top",),
                                        domain=dom)
    eq_s = AcousticPotentialSEM2D.build(small, torch.full(shp_s, 1500.0, dtype=DT),
                                        torch.full(shp_s, 1000.0, dtype=DT),
                                        dirichlet=("top",), domain=None)

    # node correspondence by coordinate
    xb, zb = _node_coords(big)
    xs, zs = _node_coords(small)
    key = lambda x, z: [(round(float(a), 6), round(float(b), 6)) for a, b in zip(x, z)]
    bmap = {k: i for i, k in enumerate(key(xb, zb))}
    s2b = torch.tensor([bmap[k] for k in key(xs, zs)], dtype=torch.long)

    # same physical point source, inside the fluid
    src_x, src_z = 2 * DE, 1.5 * DE
    pick = lambda m: int(((_node_coords(m)[0] - src_x) ** 2
                          + (_node_coords(m)[1] - src_z) ** 2).argmin())
    pat_b = torch.zeros(big.n_global, dtype=DT); pat_b[pick(big)] = 1.0
    pat_s = torch.zeros(small.n_global, dtype=DT); pat_s[pick(small)] = 1.0

    dt, nt = 4e-4, 400
    wav = _ricker(nt, dt)
    fb, fs = eq_b.init_state(dt, "cpu", DT), eq_s.init_state(dt, "cpu", DT)
    for n in range(nt):
        fb = eq_b.step(fb, [pat_b * wav[n]], dt)
        fs = eq_s.step(fs, [pat_s * wav[n]], dt)

    assert fs[0].abs().max() > 1e-6, "the fluid-only reference never propagated"
    # (a) the same numbers on the shared sub-domain, to float64 round-off.  Not
    # bit-identical: the two meshes give arrays of different lengths, and that
    # is not bitwise-stable on every CPU -- an AMD EPYC 9655 differs in the last
    # bits (4.3e-14 on a field of 128: 3e-16), single-threaded, before the CUDA
    # port as after.  One fluid element 1% off is 1.2e-3.
    ref = fs[0].abs().max()
    diff = (fb[0][s2b] - fs[0]).abs().max()
    assert diff <= 1e-12 * ref, f"embedded != standalone, max|diff|={diff:.3e} of {ref:.3e}"
    # (b) dead DOFs are EXACTLY zero, not merely small
    dead = ~dom.node_in_acoustic
    assert int(dead.sum()) > 0
    assert bool((fb[0][dead] == 0.0).all()), "chi leaked into the solid interior"
