"""Curved internal interfaces on the 3-D Hex27 mesh: ``build_layered_hex27`` and
``PropSEM(interfaces=...)`` in 3-D.

One layer is the topography mesh; flat interfaces on element-layer boundaries are the plain
mesh, node for node, and give the same records; a curved interface carries the control
nodes of its layer exactly; the elements' layers and ``layered_model`` follow the layer
counts; a flat Hex8 equation is refused.
"""
import math

import numpy as np
import pytest
import torch

from sweep.propagator import EagerOptions, Full
from sweep.sem import (ElasticVSSEM3D, ElasticVSSEM3DDeformed, PropSEM, build_layered_hex27,
                       build_topography_hex27)

F64 = torch.float64
SHAPE, DH, DT, NT = (17, 13, 17), 10.0, 1e-3, 100


def _hill(x, y):
    return 20.0 + 10.0 * torch.exp(-((x - 80.0) ** 2 + (y - 60.0) ** 2) / 900.0)


def _bowl(x, y):
    return 100.0 + 15.0 * torch.cos(math.pi * x / 160.0) * torch.cos(math.pi * y / 120.0)


def test_one_layer_is_the_topography_mesh():
    kw = dict(nex=4, ney=3, x_max=160.0, y_max=120.0, z_bottom=200.0, dtype=F64)
    a = build_layered_hex27(surfaces=[_hill], rows=[5], **kw)
    b = build_topography_hex27(nez=5, z_top_fn=_hill, **kw)
    for c in ("x_e", "y_e", "z_e"):
        assert torch.equal(getattr(a, c), getattr(b, c))


def test_a_curved_interface_carries_its_control_nodes():
    """The control nodes of the interface layer lie on the interface at their columns: the
    element faces between the layers follow it."""
    mesh = build_layered_hex27(4, 3, 160.0, 120.0, 200.0, [_hill, _bowl], [2, 3], dtype=F64)
    n = mesh.ngll - 1
    ez = 1                                          # the last element layer above the interface
    E = torch.arange(mesh.n_elements).view(mesh.nez, mesh.ney, mesh.nex)[ez].reshape(-1)
    for k in (0, n // 2, n):                        # corner and mid control columns of a face
        for j in (0, n // 2, n):
            x, y, z = (getattr(mesh, c)[E, n, j, k] for c in ("x_e", "y_e", "z_e"))
            torch.testing.assert_close(z, _bowl(x, y), rtol=1e-13, atol=1e-10)


def test_crossing_interfaces_are_refused():
    with pytest.raises(ValueError, match="must not touch or cross"):
        build_layered_hex27(4, 3, 160.0, 120.0, 200.0, [_bowl, _hill], [2, 3], dtype=F64)


def _prop(equation=ElasticVSSEM3DDeformed, **kw):
    kw = {"memory": Full(), "eager_options": EagerOptions(use_compile=False), "abcn": 8,
          "free_surface": True, "dh": DH, "f_dominant": 15.0, "source_type": ["vz"],
          "receiver_type": ["vx", "vy", "vz"], **kw}
    return PropSEM(equation, shape=SHAPE, dt=DT, n_order=4, dev="cpu", dtype=F64,
                   impl="eager", **kw)


def _call(prop, models):
    t = torch.arange(NT, dtype=F64) * DT - 0.06
    w = (1 - 2 * (math.pi * 15.0 * t) ** 2) * torch.exp(-(math.pi * 15.0 * t) ** 2)
    rx = np.arange(2.0, 15.0, 4.0)
    rec = np.stack([rx, np.full_like(rx, 6.0), np.full_like(rx, 2.0)], 1)[None]
    return prop(w, np.array([[8.0, 6.0, 10.0]]), rec, models=models)


def test_flat_interfaces_on_element_layers_are_the_plain_mesh():
    """Interfaces at rows 4 and 12 (element-layer boundaries, n_order 4) with one and two
    element layers between them: the mesh is the one without interfaces, and so are the
    records."""
    ny, nx = SHAPE[1:]
    flat = [np.full((ny, nx), 4.0), np.full((ny, nx), 12.0)]
    a = _prop(interfaces=flat, interface_elements=[1, 2, 1])
    b = _prop(topography=np.zeros((ny, nx)))
    for c in ("x_e", "y_e", "z_e"):
        torch.testing.assert_close(getattr(a.mesh, c), getattr(b.mesh, c), rtol=0, atol=1e-12)
    ms = [torch.full(SHAPE, v, dtype=F64) for v in (2400.0, 1300.0, 2000.0)]
    ra, rb = _call(a, ms), _call(b, ms)
    assert float((ra - rb).norm() / rb.norm()) < 1e-12


def test_element_layers_and_layered_model():
    ny, nx = SHAPE[1:]
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    curved = 7.0 + 1.5 * np.sin(2 * np.pi * xx / 16.0) * np.cos(np.pi * yy / 12.0)
    p = _prop(interfaces=[curved], interface_elements=[2, 2])
    layer = p.element_layer
    per = p.mesh.nex * p.mesh.ney
    counts = torch.bincount(layer).tolist()
    assert counts == [p._layer_rows[0] * per, p._layer_rows[1] * per]
    m = p.layered_model([1.0, 2.0])
    assert m.shape == (p.mesh.n_elements,) + (p.mesh.ngll,) * 3
    assert torch.equal(m[:, 0, 0, 0], layer.to(F64) + 1.0)
    vp, vs, rho = (p.layered_model(v) for v in ((2000.0, 3000.0), (1000.0, 1700.0),
                                                 (2000.0, 2300.0)))
    out = _call(p, [vp, vs, rho])
    assert torch.isfinite(out).all() and float(out.abs().max()) > 0.0


def test_a_flat_hex8_equation_is_refused():
    ny, nx = SHAPE[1:]
    with pytest.raises(ValueError, match="curved-element mesh"):
        _prop(ElasticVSSEM3D, interfaces=[np.full((ny, nx), 8.0)])


def test_a_curved_sea_floor_for_the_biot_class():
    """Water over dry rock across a curved sea floor: the layered mesh lets each element be
    wholly one medium, as BiotVSSEM3DDeformed needs; its fluid-solid faces are the sea floor's
    and their GLL nodes lie on it at the control columns."""
    from sweep.equations.biot_params import biot_elastic, biot_fluid
    from sweep.sem import BiotVSSEM3DDeformed
    ny, nx = SHAPE[1:]
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    floor = 6.0 - 2.0 * np.exp(-((xx - 8.0) ** 2 + (yy - 6.0) ** 2) / 10.0)   # a seamount
    p = _prop(BiotVSSEM3DDeformed, interfaces=[floor], interface_elements=[2, 2],
              source_type=["p"], receiver_type=["vx", "vy", "vz", "p"])
    water, rock = biot_fluid(1020.0 * 1500.0 ** 2, 1020.0), biot_elastic(2400.0, 1300.0, 2000.0)
    models = [p.layered_model([w, r]) for w, r in zip(water, rock)]
    eq = p._build_eq(models)
    ifc = eq.ifc
    per = p.mesh.nex * p.mesh.ney
    assert ifc.n_face == per                        # one face per column of elements
    g = ifc.node_idx.reshape(-1)
    x, y, z = (getattr(p.mesh, c).reshape(-1) for c in ("x_e", "y_e", "z_e"))
    first = torch.zeros(p.mesh.n_global, dtype=torch.long)
    first[p.mesh.ibool.reshape(-1)] = torch.arange(p.mesh.ibool.numel())
    xs, ys, zs = (v[first[g]] for v in (x, y, z))
    pz, py, px = p._pad0                            # the mesh starts at the absorbing pads
    on_grid = ((xs / DH - xs.div(DH).round()).abs() < 1e-9) & ((ys / DH - ys.div(DH).round()).abs() < 1e-9)
    ix = xs[on_grid].div(DH).round().long() - px
    iy = ys[on_grid].div(DH).round().long() - py
    keep = ((ix % 2 == 0) & (iy % 2 == 0) & (ix >= 0) & (ix < nx) & (iy >= 0)
            & (iy < ny))                            # the model's control columns (n_order 4)
    assert int(keep.sum()) > 0
    want = (torch.as_tensor(floor, dtype=F64)[iy[keep], ix[keep]] + pz) * DH
    torch.testing.assert_close(zs[on_grid][keep], want, rtol=0, atol=1e-9)
    t = torch.arange(NT, dtype=F64) * DT - 0.06
    w = (1 - 2 * (math.pi * 15.0 * t) ** 2) * torch.exp(-(math.pi * 15.0 * t) ** 2)
    rec = np.array([[[4.0, 6.0, 1.0], [12.0, 6.0, 12.0]]])
    out = p(w, np.array([[8.0, 6.0, 2.0]]), rec, models=models)
    assert torch.isfinite(out).all() and float(out.abs().max()) > 0.0
