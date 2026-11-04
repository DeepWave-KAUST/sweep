"""The rigid outer edge of the velocity-stress SEMs' C-PML (``pin_outer_edge``).

Left traction-free, the outer edge of the layer let the field grow without bound at late
times -- in 2-D and 3-D, elastic and Biot alike, with or without a free surface. In 2-D
elastic a record decayed to about 1e-3 of its direct wave and then rose some 30 times per
second from about 3 s on. Pinned at zero velocity, the rigid wall a C-PML is closed by (as the
displacement form's is, ``sweep.sem.cpml.elastic.outer_pml_edge_nodes``), it decays; the
absorption is unchanged (test_sem_absorption_gates.py; the records differ by 6e-5). PropSEM
refuses a layer between two free faces (``refuse_free_waveguide``).
"""

import math

import numpy as np
import pytest
import torch

import sweep.sem.equations.elastic_vs_deformed as elastic_vs_deformed
from sweep.equations.biot import biot_from_biot_constants
from sweep.equations.biot_params import biot_fluid
from sweep.propagator import Full
from sweep.sem import (BiotAnisoPermSEM2DDeformed, BiotVSSEM2DDeformed, BiotVSSEM3DDeformed,
                       ElasticVSSEM2DDeformed, ElasticVSSEM3D, ElasticVSSEM3DDeformed,
                       ElasticVTISEM2DDeformed, PropSEM, ViscoBiotVSSEM2DDeformed,
                       ViscoElasticVSSEM2DDeformed)
from sweep.sem.mesh.boundaries import free_surface_node_idx
from test_biot import DAI1

F64 = torch.float64
ELASTIC = (3000.0, 1700.0, 2200.0)
ROCK = tuple(biot_from_biot_constants(**DAI1, b=7e4, phi=0.3))


@pytest.mark.slow                   # 16000 f64 CPU steps, 31 s each
@pytest.mark.parametrize("pinned", [True, False])
def test_the_layer_stays_bounded_past_the_late_growth(pinned, monkeypatch):
    # 8 s, more than twice the onset; unpinned, the control that the run is long enough
    if not pinned:
        monkeypatch.setattr(elastic_vs_deformed, "pin_outer_edge", lambda inv_M, mesh, cpml: inv_M)
    shape, dh, dt, nt = (21, 21), 10.0, 5e-4, 16000
    prop = PropSEM(ElasticVSSEM2DDeformed, shape=shape, dh=dh, dt=dt, n_order=4,
                   f_dominant=15.0, dev="cpu", impl="eager", dtype=F64, abcn=12,
                   free_surface=True, source_type=["vz"], receiver_type=["vz"], memory=Full())
    t = np.arange(nt) * dt
    a = (math.pi * 15.0 * (t - 0.1)) ** 2
    w = torch.from_numpy((1 - 2 * a) * np.exp(-a))
    models = [torch.full(shape, v, dtype=F64) for v in (3210.0, 1792.0, 2192.0)]
    with torch.no_grad():
        d = prop(w, np.array([[10.3, 6.0]]), np.array([[[2.0, 9.0], [10.0, 15.0]]]),
                 models=models)[0, :, :, 0]
    peak, late = float(d[:2000].abs().max()), float(d[-2000:].abs().max())
    if pinned:
        assert late < 1e-4 * peak, late / peak
    else:
        assert late > peak, late / peak


def _prop(eq, ndim, free_surface=True):
    return PropSEM(eq, shape=(9,) * ndim, dh=10.0, dt=5e-4, n_order=4, f_dominant=15.0,
                   dev="cpu", impl="eager", dtype=F64, abcn=4, free_surface=free_surface)


def _models(ndim, coef):
    return [torch.full((9,) * ndim, float(c), dtype=F64) for c in coef]


def _built(eq, ndim, coef, free_surface=True):
    return _prop(eq, ndim, free_surface)._build_eq(_models(ndim, coef))


@pytest.mark.parametrize("eq, ndim, coef", [
    (ElasticVSSEM2DDeformed, 2, ELASTIC),
    (ViscoElasticVSSEM2DDeformed, 2, ELASTIC + (30.0, 20.0)),
    (ElasticVTISEM2DDeformed, 2, ELASTIC + (0.1, 0.05)),
    (BiotVSSEM2DDeformed, 2, ROCK),
    (ViscoBiotVSSEM2DDeformed, 2, ROCK + (30.0, 20.0)),
    (BiotAnisoPermSEM2DDeformed, 2, ROCK + (0.0, ROCK[-1])),
    (ElasticVSSEM3D, 3, ELASTIC),
    (ElasticVSSEM3DDeformed, 3, ELASTIC),
    (BiotVSSEM3DDeformed, 3, ROCK)])
def test_every_velocity_stress_layer_pins_its_outer_edge(eq, ndim, coef):
    e = _built(eq, ndim, coef)
    wall = free_surface_node_idx(e.mesh, tuple(e.cpml.sides), "cpu")
    assert "top" not in e.cpml.sides and wall.numel() > 0
    inside = torch.ones(e.mesh.n_global, dtype=torch.bool)
    inside[wall] = False
    assert not e._inv_M[wall].any() and bool((e._inv_M[inside] > 0).all())


BOTTOM_ONLY_2D = {"top": True, "left": True, "right": True, "bottom": False}
BOTTOM_ONLY_3D = {"top": True, "left": True, "right": True, "front": True, "back": True,
                  "bottom": False}


@pytest.mark.parametrize("eq, ndim, coef, free_surface", [
    (ElasticVSSEM2DDeformed, 2, ELASTIC, BOTTOM_ONLY_2D),
    (ElasticVSSEM2DDeformed, 2, ELASTIC, {"top": True, "bottom": True, "left": False,
                                         "right": False}),
    (BiotVSSEM2DDeformed, 2, ROCK, BOTTOM_ONLY_2D),
    (ElasticVSSEM3D, 3, ELASTIC, {"top": True, "left": False, "right": False, "front": True,
                                  "back": True, "bottom": False}),
    (ElasticVSSEM3DDeformed, 3, ELASTIC, BOTTOM_ONLY_3D),
    (BiotVSSEM3DDeformed, 3, ROCK, BOTTOM_ONLY_3D)])
def test_a_layer_between_two_free_faces_is_refused(eq, ndim, coef, free_surface):
    # a plate's backward Lamb modes grow in any PML (see refuse_free_waveguide); PropSEM
    # refuses it, the equation's build() leaves it to its caller (the absorption gates)
    prop = _prop(eq, ndim, free_surface)
    with pytest.raises(ValueError, match="plate"):
        prop._build_eq(_models(ndim, coef))
    assert prop._build_eq_raw(_models(ndim, coef)).cpml is not None


@pytest.mark.parametrize("eq, ndim, coef, free_surface", [
    (ElasticVSSEM2DDeformed, 2, ELASTIC, {"top": True, "left": True, "right": False,
                                         "bottom": False}),
    (BiotVSSEM2DDeformed, 2, biot_fluid(2.25e9, 1000.0), BOTTOM_ONLY_2D),
    (BiotVSSEM3DDeformed, 3, biot_fluid(2.25e9, 1000.0), BOTTOM_ONLY_3D)])
def test_one_free_face_per_axis_or_a_fluid_alone_is_allowed(eq, ndim, coef, free_surface):
    assert _built(eq, ndim, coef, free_surface).cpml is not None
