"""Absorption gates for every C-PML in the package -- 2-D and 3-D.

Big-domain SUBTRACTION with receivers strictly outside the PML band. This is
not decay-to-floor: measured on this suite, decay-to-floor passes with two
orders of magnitude to spare at a boundary reflection coefficient of 0.4-0.5
and SATURATES below R~0.2 -- a worse PML scores better -- so it certifies
stability, nothing else. Until this file, the SPECFEM2D-ported displacement
C-PML (cpml/elastic.py) had never had its absorption measured at all.

Geometry shared by every case:
* the C-PML sits on the LATERAL sides only, so the bottom echo is common mode
  between the small box and the reference and cancels in the subtraction;
* the reference shares the small box's depth and differs only laterally, wide
  enough that its own echo cannot arrive inside the record;
* the comparison window starts at the first possible lateral echo, Ricker
  delay included -- a first version of the 3-D gate forgot the delay, ended
  the record before any echo arrived, and the bare box "leaked" 1%: the
  discrimination assert below is what caught it.

Every gate ships with a discrimination check: the SAME configuration fed a
nearly-undamped profile (reflection_target=0.7) must fail, and the bare box
must leak >= 3x the gate. A gate that a broken PML can pass proves nothing.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import SEMPropagator
from sweep.sem.cpml.acoustic import build_cpml
from sweep.sem.cpml.acoustic3d import build_cpml3d
from sweep.sem.cpml.elastic import build_elastic_cpml
from sweep.sem.cpml.elastic_vs import build_simple_elastic_cpml
from sweep.sem.cpml.elastic_vs3d import build_simple_elastic_cpml_3d
from sweep.sem.equations.acoustic3d import AcousticSEM3D
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.equations.elastic2d_deformed import ElasticSEM2DDeformed
from sweep.sem.equations.elastic_vs3d import ElasticVSSEM3D
from sweep.sem.equations.elastic_vs_deformed import ElasticVSSEM2DDeformed
from sweep.sem.mesh.hex8 import build_hex8_mesh
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="needs CUDA")

NO, DH = 4, 25.0
EL = NO * DH                     # 100 m element
VP, VS, RHO, F0 = 2400.0, 1250.0, 2000.0, 8.0
DT = 0.30 * (0.172 * EL) / (VP * 1.2)
DELAY = 1.5 / F0
GATE = 0.05


def _wav(nt):
    t = np.arange(nt) * DT
    a = (np.pi * F0 * (t - DELAY)) ** 2
    return torch.tensor((((1 - 2 * a) * np.exp(-a)) * 1e8).astype(np.float32),
                        device="cuda")


def _leak(small, ref, i0):
    res = np.abs(small[i0:] - ref[i0:]).max(axis=0)
    return float((res / np.abs(ref).max(axis=0)).max())


# ------------------------------- 2-D cases --------------------------------- #
# small 12x12 elements (1200 m); reference 28 wide, same depth. Lateral echo
# path (600 + 400) m: P 0.417 s, S 0.80 s (+ delay). Reference echo
# (1400 + 1000)/vp + delay = 1.19 s > the 1.10 s record.
_NE2, _BIG2, _PML2 = 12, 28, 3
_NT2 = int(1.10 / DT)
_I0_2 = int(((600.0 + 400.0) / VP + DELAY) / DT)


def _mesh2(ne):
    L = ne * EL
    return build_axis_aligned_quad9(nex=ne, nez=_NE2, x_max=L, z_max=_NE2 * EL,
                                    n_order=NO, dtype=torch.float32,
                                    device="cuda"), L


def _full2(m, v):
    return torch.full((m.n_elements, m.ngll, m.ngll), v, dtype=torch.float32,
                      device="cuda")


_CASES_2D = {
    "acoustic_potential": dict(
        builder=build_cpml,
        make=lambda m, cp: AcousticPotentialSEM2D.build(
            m, _full2(m, VP), _full2(m, RHO), dirichlet=("top",), cpml=cp),
        kind="scalar", payload=(1e6,), cmax=VP),
    "elastic_displacement": dict(          # the SPECFEM2D port, never gated
        builder=build_elastic_cpml,
        make=lambda m, cp: ElasticSEM2DDeformed.build(
            m, _full2(m, VP), _full2(m, VS), _full2(m, RHO), cpml=cp),
        kind="force", payload=(0.0, 1.0), cmax=VP * 1.2),
    "elastic_velocity_stress": dict(
        builder=build_simple_elastic_cpml,
        make=lambda m, cp: ElasticVSSEM2DDeformed.build(
            m, _full2(m, VP), _full2(m, VS), _full2(m, RHO), cpml=cp),
        kind="force", payload=(0.0, 1.0), cmax=VP * 1.2),
}


def _run2(case, ne, reflection_target):
    c = _CASES_2D[case]
    m, L = _mesh2(ne)
    cp = None
    if reflection_target is not None:
        cp = c["builder"](m, dt=DT, sides=("left", "right"),
                          pml_thickness_elements=_PML2, c_max=c["cmax"],
                          f_dominant=F0, reflection_target=reflection_target)
    eq = c["make"](m, cp)
    assert (reflection_target is None) == (eq.cpml is None)
    ctr = L / 2.0
    rec = [(ctr + o, 150.0) for o in (-200.0, -100.0, 0.0, 100.0, 200.0)]
    P = SEMPropagator.build(eq, dt=DT, nt=_NT2, source_xz=[(ctr, 150.0)],
                            receiver_xz=rec, source_kinds=[c["kind"]],
                            source_payloads=[c["payload"]], pml_guard="off")
    with torch.no_grad():
        s = P.forward(_wav(_NT2))
    out = s[..., -1].double().cpu().numpy()      # last component (u/χ or u_z)
    del P, eq, m
    torch.cuda.empty_cache()
    return out


@pytest.mark.parametrize("case", sorted(_CASES_2D))
def test_2d_cpml_absorbs(case):
    ref = _run2(case, _BIG2, None)
    box = _run2(case, _NE2, None)
    pml = _run2(case, _NE2, 1e-3)
    leak_box, leak_pml = _leak(box, ref, _I0_2), _leak(pml, ref, _I0_2)
    assert leak_box > 3 * GATE, (
        f"{case}: bare box leaks only {leak_box:.3f} -- the window contains "
        f"no echo and the gate below proves nothing")
    assert leak_pml < GATE, f"{case}: C-PML leak {leak_pml:.3f} >= {GATE}"


@pytest.mark.parametrize("case", sorted(_CASES_2D))
def test_2d_gate_discriminates(case):
    """reflection_target=0.7 scales sigma_0 by ln(0.7)/ln(1e-3) ~ 5%: a nearly
    undamped layer. If the gate passes it, the gate measures nothing."""
    ref = _run2(case, _BIG2, None)
    bad = _run2(case, _NE2, 0.7)
    assert _leak(bad, ref, _I0_2) > GATE, (
        f"{case}: a nearly-undamped profile passed the absorption gate")


# ------------------------------- 3-D cases --------------------------------- #
_NE3, _BIG3, _PML3 = 10, 24, 3
_NT3 = int(1.10 / DT)
_I0_3 = int(((500.0 + 350.0) / VP + DELAY) / DT)


def _mesh3(ne):
    L = ne * EL
    return build_hex8_mesh(nex=ne, ney=ne, nez=_NE3, x_max=L, y_max=L,
                           z_max=_NE3 * EL, n_order=NO, dtype=torch.float32,
                           device="cuda"), L


def _full3(m, v):
    return torch.full((m.n_elements,) + (m.ngll,) * 3, v, dtype=torch.float32,
                      device="cuda")


_CASES_3D = {
    "acoustic": dict(
        builder=build_cpml3d,
        make=lambda m, cp: AcousticSEM3D.build(
            m, _full3(m, VP), dirichlet=("top",), cpml=cp),
        kind="scalar", payload=(1e6,), cmax=VP),
    "elastic_velocity_stress": dict(
        builder=build_simple_elastic_cpml_3d,
        make=lambda m, cp: ElasticVSSEM3D.build(
            m, _full3(m, VP), _full3(m, VS), _full3(m, RHO), cpml=cp),
        kind="force", payload=(0.0, 0.0, 1.0), cmax=VP * 1.2),
}


def _run3(case, ne, reflection_target):
    c = _CASES_3D[case]
    m, L = _mesh3(ne)
    cp = None
    if reflection_target is not None:
        cp = c["builder"](m, dt=DT, sides=("left", "right", "front", "back"),
                          pml_thickness_elements=_PML3, c_max=c["cmax"],
                          f_dominant=F0, reflection_target=reflection_target)
    eq = c["make"](m, cp)
    assert (reflection_target is None) == (eq.cpml is None)
    ctr = L / 2.0
    rec = [(ctr + o, ctr, 150.0) for o in (-150.0, -75.0, 0.0, 75.0, 150.0)]
    P = SEMPropagator.build(eq, dt=DT, nt=_NT3, source_xz=[(ctr, ctr, 100.0)],
                            receiver_xz=rec, source_kinds=[c["kind"]],
                            source_payloads=[c["payload"]], pml_guard="off")
    with torch.no_grad():
        s = P.forward(_wav(_NT3))
    out = s[..., -1].double().cpu().numpy()
    del P, eq, m
    torch.cuda.empty_cache()
    return out


@pytest.mark.parametrize("case", sorted(_CASES_3D))
def test_3d_cpml_absorbs(case):
    ref = _run3(case, _BIG3, None)
    box = _run3(case, _NE3, None)
    pml = _run3(case, _NE3, 1e-3)
    leak_box, leak_pml = _leak(box, ref, _I0_3), _leak(pml, ref, _I0_3)
    # 2x, not the 2-D file's 3x: a 3-D point-source echo spreads as 1/r and
    # arrives relatively weaker -- measured bare-box leaks are 0.145 (acoustic)
    # and 0.109 (elastic v-s) against the 0.05 gate, which still discriminates.
    assert leak_box > 2 * GATE, (
        f"{case}: bare box leaks only {leak_box:.3f}; no echo in the window")
    assert leak_pml < GATE, f"{case}: C-PML leak {leak_pml:.3f} >= {GATE}"


@pytest.mark.parametrize("case", sorted(_CASES_3D))
def test_3d_gate_discriminates(case):
    ref = _run3(case, _BIG3, None)
    bad = _run3(case, _NE3, 0.7)
    assert _leak(bad, ref, _I0_3) > GATE, (
        f"{case}: a nearly-undamped profile passed the absorption gate")


# ----------------------------- grazing incidence --------------------------- #
def _run2_grazing(case, ne, reflection_target, f_dom=F0):
    """Source close to the LEFT band, receivers strung parallel to it: the
    band sees the wave at near-grazing angles, which the normal-incidence gate
    is measurably blind to (f_dominant=0 deletes the CFS frequency shift and
    the normal gate moves 1.70% -> 1.67%).

    The reference places the source at the DOMAIN CENTRE with the receivers at
    the same offsets from it: the medium is homogeneous and the top/bottom sit
    at the same depths, so the true field depends only on the relative
    geometry. A first version kept the absolute source position -- 400 m from
    the reference's own reflecting left wall -- and the reference was polluted
    by that wall from 0.52 s: the "grazing leak" read 0.91 and measured
    nothing but the broken reference.
    """
    c = _CASES_2D[case]
    m, L = _mesh2(ne)
    cp = None
    if reflection_target is not None:
        cp = c["builder"](m, dt=DT, sides=("left", "right"),
                          pml_thickness_elements=_PML2, c_max=c["cmax"],
                          f_dominant=f_dom, reflection_target=reflection_target)
    eq = c["make"](m, cp)
    x_src = (_PML2 * EL + 100.0) if reflection_target is not None else L / 2.0
    rec = [(x_src, z) for z in (350.0, 500.0, 650.0, 800.0, 950.0)]
    P = SEMPropagator.build(eq, dt=DT, nt=_NT2, source_xz=[(x_src, 200.0)],
                            receiver_xz=rec, source_kinds=[c["kind"]],
                            source_payloads=[c["payload"]], pml_guard="off")
    with torch.no_grad():
        s = P.forward(_wav(_NT2))
    out = s[..., -1].double().cpu().numpy()
    del P, eq, m
    torch.cuda.empty_cache()
    return out


_I0_G = int((200.0 / VP + DELAY) / DT)          # first band interaction


@pytest.mark.parametrize("case", ["acoustic_potential", "elastic_displacement"])
def test_2d_grazing_incidence(case):
    ref = _run2_grazing(case, _BIG2, None)
    pml = _run2_grazing(case, _NE2, 1e-3)
    leak = _leak(pml, ref, _I0_G)
    assert leak < 2 * GATE, f"{case}: grazing leak {leak:.3f} >= {2*GATE}"
    if case == "elastic_displacement":
        # the discriminator the normal-incidence gate is blind to: killing the
        # CFS frequency shift must hurt at grazing
        nocfs = _run2_grazing(case, _NE2, 1e-3, f_dom=1e-6)
        leak_nocfs = _leak(nocfs, ref, _I0_G)
        assert leak_nocfs > 1.5 * leak, (
            f"{case}: f_dominant->0 changed the grazing leak only "
            f"{leak:.4f} -> {leak_nocfs:.4f}; this geometry does not "
            f"exercise grazing incidence")
    else:
        # MEASURED: the acoustic double-convolution C-PML's grazing leak is
        # insensitive to the CFS shift here -- f_dominant->0 moves it
        # 0.0280 -> 0.0276 (and 0.0366 -> 0.0362 with the source 50 m off the
        # band). That is a property of this profile at these angles and this
        # bandwidth, not a broken geometry: the ELASTIC port's grazing leak
        # does depend on it in the same setup. Discriminate on the damping
        # itself instead.
        bad = _run2_grazing(case, _NE2, 0.7)
        assert _leak(bad, ref, _I0_G) > 2 * GATE, (
            f"{case}: a nearly-undamped profile passed the grazing gate")
