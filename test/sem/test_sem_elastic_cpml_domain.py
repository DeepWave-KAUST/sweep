"""Stage-6/S2 gates: elastic CPML under a domain mask + the corner-PML rule.

1. bit-parity: an all-elastic DomainMask must not change the CPML run at all
   (the mask multiplies coefficients by 1.0 — exact).
2. mixed domain: on a fluid-over-solid split, the CPML memory recursions must
   stay EXACTLY zero on fluid elements (frozen by the masked coefficients)
   while still accumulating on solid band elements — and the run stays finite.
3. corner rule: an interface crossing an X-only band builds fine; an interface
   inside a corner region (both directions stretched) must raise at build time
   (SPECFEM2D makes the same configuration a fatal error at run time).
"""
import numpy as np
import pytest
import torch

from sweep.sem.coupling import (FluidSolidInterface,
                                assert_no_corner_pml_on_interface)
from sweep.sem.cpml.acoustic import build_cpml
from sweep.sem.cpml.elastic import build_elastic_cpml
from sweep.sem.equations.elastic2d_deformed import ElasticSEM2DDeformed
from sweep.sem.mesh.domains import DomainMask
from sweep.sem.mesh.quad9 import build_axis_aligned_quad9

DT = torch.float64
NEX, NEZ, DE = 16, 12, 50.0


def _mesh():
    return build_axis_aligned_quad9(nex=NEX, nez=NEZ, x_max=NEX * DE,
                                    z_max=NEZ * DE, n_order=4, dtype=DT)


def _mats(m):
    shp = (m.n_elements, m.ngll, m.ngll)
    return (torch.full(shp, 3000.0, dtype=DT), torch.full(shp, 1732.0, dtype=DT),
            torch.full(shp, 2200.0, dtype=DT))


def _run(eq, m, nt=200, dt=6e-4, src_xz=(400.0, 450.0)):
    xg = torch.zeros(m.n_global, dtype=DT); xg[m.ibool] = m.x_e
    zg = torch.zeros(m.n_global, dtype=DT); zg[m.ibool] = m.z_e
    s = int(((xg - src_xz[0]) ** 2 + (zg - src_xz[1]) ** 2).argmin())
    pat = torch.zeros(m.n_global, dtype=DT); pat[s] = 1.0
    zero = torch.zeros(m.n_global, dtype=DT)
    t = np.arange(nt) * dt
    a = (np.pi * 14.0 * (t - 0.09)) ** 2
    wav = torch.tensor((1 - 2 * a) * np.exp(-a), dtype=DT)
    f = eq.init_state(dt, "cpu", DT)
    for n in range(nt):
        f = eq.step(f, [zero, pat * wav[n]], dt)
    return f


def test_all_elastic_domain_is_bit_exact():
    m = _mesh()
    vp, vs, rho = _mats(m)
    cp = build_elastic_cpml(m, dt=6e-4, sides=("left", "right", "bottom"),
                            pml_thickness_elements=4, c_max=3000.0,
                            f_dominant=14.0)
    dom = DomainMask.from_element_flags(
        m, torch.zeros(m.n_elements, dtype=torch.bool))     # nothing acoustic
    eq_plain = ElasticSEM2DDeformed.build(m, vp, vs, rho, cpml=cp)
    eq_dom = ElasticSEM2DDeformed.build(m, vp, vs, rho, cpml=cp, domain=dom)
    f_p = _run(eq_plain, m)
    f_d = _run(eq_dom, m)
    assert f_p[0].abs().max() > 0
    for a, b in zip(f_p, f_d):
        assert torch.equal(a, b), "all-elastic domain changed the CPML run"


def test_mixed_domain_freezes_fluid_memory():
    m = _mesh()
    vp, vs, rho = _mats(m)
    zc = m.z_e.flatten(1).mean(1)
    dom = DomainMask.from_element_flags(m, zc < 200.0)       # top 4 rows fluid
    cp = build_elastic_cpml(m, dt=6e-4, sides=("left", "right", "bottom"),
                            pml_thickness_elements=4, c_max=3000.0,
                            f_dominant=14.0)
    eq = ElasticSEM2DDeformed.build(m, vp, vs, rho, cpml=cp, domain=dom)
    f = _run(eq, m, src_xz=(400.0, 450.0))                   # source in solid
    # the memory fields live on the C-PML band rows (band2): fluid flags of those
    fl = dom.is_acoustic[eq.band_elements]
    # finite + the solid actually moved
    assert all(torch.isfinite(x).all() for x in f)
    assert f[0].abs().max() > 0
    # every element-local CPML memory field is EXACTLY zero on fluid elements
    for k in range(4, 10):
        r = f[k]                                             # (E_band, ngll, ngll, 2)
        assert bool((r[fl] == 0).all()), f"memory field {k} leaked into fluid"
    # ... and genuinely nonzero somewhere in the solid bands (mask didn't
    # kill the live recursions)
    live = torch.stack([f[k][~fl].abs().max() for k in range(4, 10)])
    assert bool((live > 0).any()), "masking froze the SOLID memory too"


def test_corner_rule():
    m = _mesh()
    zc = m.z_e.flatten(1).mean(1)
    # interface at z=300: crosses the left/right X-only bands at mid-height
    dom_mid = DomainMask.from_element_flags(m, zc < 300.0)
    ifc_mid = FluidSolidInterface.detect(m, dom_mid)
    prof = build_cpml(m, dt=6e-4, sides=("left", "right", "bottom"),
                      pml_thickness_elements=4, c_max=3000.0, f_dominant=14.0)
    assert_no_corner_pml_on_interface(ifc_mid, prof)         # X-only: fine

    # interface one element above the bottom: inside the bottom band, so its
    # ends sit where BOTH directions are stretched -> must raise
    z_low = (NEZ - 1) * DE                                    # 550 m
    dom_low = DomainMask.from_element_flags(m, zc < z_low)
    ifc_low = FluidSolidInterface.detect(m, dom_low)
    with pytest.raises(ValueError, match="CORNER absorbing region"):
        assert_no_corner_pml_on_interface(ifc_low, prof)
