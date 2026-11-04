"""SEMPropagator.build should catch sources / receivers placed inside the C-PML
absorbing band, where the recorded / injected field is artificially damped
(non-physical) — a silent seismogram corruption. See lesson: the SEM-vs-SPECFEM
"vz residual" was entirely receivers sitting inside the small-domain PML.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from sweep.sem import (ElasticSEM2DDeformed, PropSEM, SEMPropagator,
                       build_axis_aligned_quad9)
from sweep.sem.equations import AcousticSEM2DDeformed
from sweep.sem.equations.acoustic3d_deformed import AcousticSEM3DDeformed
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



def test_point_on_the_band_edge_is_interior():
    """The band's inner edge is the interior's first line (PropSEM puts the
    model's edge column there), and the band's thickness comes out of float32
    mesh coordinates (200.00002 for a 200 m band in PropSEM). A point on the
    edge, up to that round-off, must not be flagged; a metre in still is."""
    eq, cp = _eq()
    L = cp.pml_thickness
    for eps in (-2e-4, 0.0, 2e-4):            # round-off of a ~1e3 m extent
        edges = [(L + eps, 30.0), (1270.0 - L - eps, 30.0), (640.0, 890.0 - L - eps)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _build(eq, src=[(640.0, 30.0)], rec=edges)
    with pytest.warns(UserWarning, match="left band"):
        _build(eq, src=[(640.0, 30.0)], rec=[(L - 1.0, 30.0)])


# --------------------------------------------------------------------------- #
# PropSEM: the layer is added outside the model, so the model's edge column
# (ix = 0 or nx-1) is the band's inner edge -- in the model. It was flagged
# "in left band (x<200.0)" on a topography mesh, the thickness a float32
# round-off above 200. The warning that does fire names PropSEM's options.
# --------------------------------------------------------------------------- #
NZ, NY, NX, DH = 17, 13, 21, 25.0
NT = 8
TOPO2 = 1.0 + 0.5 * np.sin(np.arange(NX) / 4.0)
TOPO3 = TOPO2[None, :] * np.ones((NY, 1))
EAGER = dict(impl="eager", eager_options=dict(use_compile=False))


def _wav():
    w = torch.zeros(NT)
    w[2] = 1.0
    return w


def _p3():
    return PropSEM(AcousticSEM3DDeformed, shape=(NZ, NY, NX), dh=DH, dt=2e-3,
                   n_order=4, abcn=8, free_surface=True, topography=TOPO3, **EAGER)


def _p2(**kw):
    return PropSEM(AcousticSEM2DDeformed, shape=(NZ, NX), dh=DH, dt=2e-3,
                   n_order=4, abcn=8, topography=TOPO2, **EAGER, **kw)


def _run(p, src, rec, **kw):
    vp = torch.full(p.shape, 2000.0)
    with torch.no_grad():
        return p(_wav(), np.asarray(src), np.asarray(rec), models=[vp], **kw)


def test_edge_receivers_3d_topography_do_not_warn():
    p = _p3()
    rec = [[[0, 6, 6], [NX - 1, 6, 6], [10, 6, NZ - 1], [0, 6, NZ - 1]]]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _run(p, [[10, 6, 6]], rec)


def test_edge_receivers_2d_topography_do_not_warn():
    p = _p2()
    rec = [[[0, 6], [NX - 1, 6], [10, NZ - 1], [NX - 1, NZ - 1]]]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _run(p, [[10, 6]], rec)


def test_receiver_outside_the_model_warns_with_propsem_options():
    p = _p2()
    with pytest.warns(UserWarning, match="left band") as rec:
        _run(p, [[10, 6]], [[[-1, 6]]])
    msg = str(rec[0].message)
    assert "abcn" in msg and "outside the model grid" in msg
    assert "reduce pml_thickness_elements" not in msg
    # the option the warning names is one PropSEM's forward takes
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _run(p, [[10, 6]], [[[-1, 6]]], pml_guard="off")
