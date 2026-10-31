"""PropSEM(interfaces=...): a Quad9 mesh whose element rows follow curved internal interfaces
(SPECFEM2D's internal mesher), and the per-layer models that step at them.

test/sem/specfem_compare/run_compare_box.py checks it against SPECFEM2D (a curved sea floor,
a sinusoidal interface between rocks); here: one layer is the topography mesh, an interface's
control nodes lie on it, a flat interface on element rows gives the flat mesh's records, the
layers of the elements and of the absorbing pads, the per-layer values' gradients, and what is
refused.
"""
import math

import numpy as np
import pytest
import torch

from sweep.sem import (AcousticSEM2D, ElasticVSSEM2DDeformed, PropSEM, build_layered_quad9,
                       build_topography_quad9)

F64 = torch.float64
SHAPE, DH = (41, 81), 10.0


def _prop(**kw):
    kw = {"abcn": 8, "free_surface": True, **kw}
    return PropSEM(ElasticVSSEM2DDeformed, shape=SHAPE, dh=DH, dt=1e-3, n_order=4,
                   f_dominant=10.0, dev="cpu", dtype=F64, impl="eager",
                   source_type=["sxx", "szz"], receiver_type=["vx", "vz"], **kw)


def _shot(prop, models):
    t = np.arange(200) * 1e-3 - 0.1
    a = (math.pi * 10.0 * t) ** 2
    w = torch.from_numpy((1 - 2 * a) * np.exp(-a))
    rx = np.arange(5.0, 76.0, 10.0)
    return prop(w, np.array([[40.3, 12.6]]), np.stack([rx, np.full_like(rx, 30.0)], 1)[None],
                models=models)


def test_one_layer_is_the_topography_mesh():
    top = lambda x: 20.0 * torch.exp(-((x - 400.0) / 120.0) ** 2)
    a = build_layered_quad9(20, 800.0, 400.0, [top], [10], dtype=F64)
    b = build_topography_quad9(20, 10, 800.0, 400.0, top, dtype=F64)
    assert torch.equal(a.x_e, b.x_e) and torch.equal(a.z_e, b.z_e)


def test_an_interface_runs_through_its_element_rows():
    top = lambda x: torch.zeros_like(x)
    floor = lambda x: 200.0 + 40.0 * torch.cos(2.0 * math.pi * x / 800.0)
    m = build_layered_quad9(20, 800.0, 400.0, [top, floor], [6, 9], dtype=F64)
    c = m.control_xy.reshape(15, 20, 3, 3, 2)[6, :, 0]          # the top control row of layer 1
    assert torch.equal(c[..., 1], floor(c[..., 0]))
    assert bool((m.det_J > 0).all())
    with pytest.raises(ValueError, match="cross"):
        build_layered_quad9(20, 800.0, 400.0, [top, lambda x: 450.0 + 0.0 * x], [6, 9])


def test_a_flat_interface_on_element_rows_is_the_flat_mesh():
    layered = _prop(interfaces=[np.full(SHAPE[1], 20.0)])
    flat = _prop()
    vals = ((2400.0, 3000.0), (1300.0, 1700.0), (2000.0, 2300.0))
    ours = _shot(layered, [layered.layered_model(v) for v in vals])
    # the same models by element centre on the mesh without interfaces
    zc = flat.mesh.z_e.mean((1, 2))
    deep = zc > 20 * DH
    ref = [torch.where(deep, v[1], v[0]).view(-1, 1, 1).expand_as(flat.mesh.z_e).to(F64)
           for v in vals]
    assert (ours - _shot(flat, ref)).abs().max() <= 1e-12 * ours.abs().max()


def test_the_layers_take_the_pads_at_their_sides():
    rows = 15.0 + 4.0 * np.sin(np.arange(SHAPE[1]) * DH * 2.0 * math.pi / 800.0)
    p = _prop(interfaces=[rows], interface_elements=[4, 6])
    assert p._layer_rows == [4, 6 + 2]                    # the bottom pad joins the last layer
    lay = p.element_layer.view(-1, p._elems_pad[-1])
    assert bool((lay[:4] == 0).all()) and bool((lay[4:] == 1).all())
    # an absorbing top: its pad joins the first layer, the interface moves down with it
    q = _prop(interfaces=[rows], interface_elements=[4, 6], free_surface=False)
    assert q._layer_rows == [4 + 2, 6 + 2]
    nez, nex = q._elems_pad
    z = q.mesh.control_xy.reshape(nez, nex, 3, 3, 2)[6, 2:-2, 0, :, 1]     # the model's columns
    # control node j of model element e sits on grid column 4 e + 2 j, 8 rows (the pad) down
    want = np.array([[rows[4 * e + 2 * j] for j in range(3)] for e in range(nex - 4)]) + 8.0
    assert torch.allclose(z, torch.as_tensor(want * DH, dtype=F64), rtol=0, atol=1e-9)


def test_the_layer_values_are_differentiable():
    p = _prop(interfaces=[np.full(SHAPE[1], 20.0)])
    vp = torch.tensor([2400.0, 3000.0], dtype=F64, requires_grad=True)
    seis = _shot(p, [p.layered_model([vp[0], vp[1]]), p.layered_model([1300.0, 1700.0]),
                     p.layered_model([2000.0, 2300.0])])
    (seis ** 2).sum().backward()
    assert torch.isfinite(vp.grad).all() and bool((vp.grad != 0).all())


def test_what_is_refused():
    x = np.zeros(SHAPE[1])
    with pytest.raises(ValueError, match="below the surface"):
        _prop(interfaces=[x])                               # on the surface itself
    with pytest.raises(ValueError, match="below the interface 0"):
        _prop(interfaces=[x + 20.0, x + 10.0])
    with pytest.raises(ValueError, match="interface_elements"):
        _prop(interfaces=[x + 20.0], interface_elements=[3, 3])
    with pytest.raises(ValueError, match="one value per layer"):
        _prop(interfaces=[x + 20.0]).layered_model([1.0])
    with pytest.raises(ValueError, match="curved-element"):
        PropSEM(AcousticSEM2D, shape=SHAPE, dh=DH, dt=1e-3, dev="cpu", interfaces=[x + 20.0])
