"""SEMPropagator.build should catch sources / receivers placed inside the C-PML
absorbing band, where the recorded / injected field is artificially damped
(non-physical) — a silent seismogram corruption. See lesson: the SEM-vs-SPECFEM
"vz residual" was entirely receivers sitting inside the small-domain PML.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from sweep.sem import ElasticSEM2DDeformed, SEMPropagator, build_axis_aligned_quad9
from sweep.sem.cpml.elastic import build_elastic_cpml

DT_T = torch.float64
SIDES = ("left", "right", "bottom")


def _eq(cpml=True):
    m = build_axis_aligned_quad9(nex=64, nez=44, x_max=1270.0, z_max=890.0,
                                 n_order=4, dtype=DT_T)
    sh = (m.n_elements, m.ngll, m.ngll)
    full = lambda v: torch.full(sh, v, dtype=DT_T)
    cp = build_elastic_cpml(m, dt=1e-3, sides=SIDES, pml_thickness_elements=8,
                            c_max=2200.0, f_dominant=12.0) if cpml else None
    return ElasticSEM2DDeformed.build(m, full(2200.0), full(1100.0), full(2000.0), cpml=cp), cp


def _build(eq, *, src, rec, guard="warn"):
    return SEMPropagator.build(eq, dt=1e-3, nt=10, source_xz=src, receiver_xz=rec,
                               source_kinds=["moment_tensor"] * len(src),
                               source_payloads=[(1.0, 1.0, 0.0)] * len(src),
                               pml_guard=guard)


def test_receiver_in_pml_warns():
    eq, cp = _eq()
    assert 60.0 < cp.pml_thickness            # x=60 is inside the left band
    with pytest.warns(UserWarning, match="C-PML absorbing band"):
        _build(eq, src=[(640.0, 30.0)], rec=[(60.0, 30.0), (640.0, 30.0)])


def test_interior_points_do_not_warn():
    eq, _ = _eq()
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # any warning fails the test
        _build(eq, src=[(640.0, 30.0)], rec=[(300.0, 30.0), (640.0, 30.0), (900.0, 30.0)])


def test_source_in_pml_error_mode_raises():
    eq, _ = _eq()
    with pytest.raises(ValueError, match="source.*C-PML absorbing band"):
        _build(eq, src=[(100.0, 30.0)], rec=[(640.0, 30.0)], guard="error")


def test_guard_off_silent():
    eq, _ = _eq()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _build(eq, src=[(640.0, 30.0)], rec=[(60.0, 30.0)], guard="off")


def test_no_cpml_no_check():
    eq, _ = _eq(cpml=False)                    # bare box: nothing to guard against
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _build(eq, src=[(640.0, 30.0)], rec=[(60.0, 30.0)])


def test_bottom_band_flagged():
    eq, cp = _eq()
    z_bot = 890.0 - cp.pml_thickness + 1.0     # just inside the bottom band
    with pytest.warns(UserWarning, match="bottom band"):
        _build(eq, src=[(640.0, 30.0)], rec=[(640.0, z_bot)])


def test_degenerate_all_pml_domain_no_warn():
    """A tiny box whose opposite PML bands overlap (2·thickness ≥ extent) has no
    interior at all — every point is 'in the PML', so per-point flagging is noise
    (it's a domain-sizing issue, not a placement one). The guard skips such axes."""
    m = build_axis_aligned_quad9(nex=16, nez=16, x_max=200.0, z_max=200.0,
                                 n_order=4, dtype=DT_T)
    sh = (m.n_elements, m.ngll, m.ngll)
    full = lambda v: torch.full(sh, v, dtype=DT_T)
    cp = build_elastic_cpml(m, dt=5e-4, sides=SIDES, pml_thickness_elements=8,
                            c_max=2200.0, f_dominant=12.0)   # L=8*12.5=100, 2L=200≥extent
    assert 2.0 * cp.pml_thickness >= 200.0
    eq = ElasticSEM2DDeformed.build(m, full(2200.0), full(1100.0), full(2000.0), cpml=cp)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _build(eq, src=[(50.0, 100.0)], rec=[(100.0, 100.0)])
