"""3-D moment-tensor sources: the stamp ``-M : grad(phi)`` on Hex8 and Hex27 meshes, its
sharing among the elements that meet at a face, an edge or a corner, and the 3-D elastic
equations' stress source names."""

import math

import numpy as np
import pytest
import torch

from sweep.propagator import EagerOptions
from sweep.sem import (ElasticSEM3D, ElasticSEM3DDeformed, ElasticVSSEM3D,
                       ElasticVSSEM3DDeformed, PropSEM)
from sweep.sem.mesh.hex8 import build_hex8_mesh
from sweep.sem.mesh.hex27 import build_topography_hex27
from sweep.sem.propagator import (EdgeStamp, _stamp_components, build_moment_stamp,
                                  build_point_stamp)

F64 = torch.float64
NO = 4
M6 = (1.3, -0.4, 0.7, 0.25, -0.6, 0.45)          # (Mxx, Myy, Mzz, Mxy, Mxz, Myz)


def _curved():
    topo = lambda x, y: 40.0 + 30.0 * torch.sin(2 * math.pi * x / 300.0) * torch.cos(
        2 * math.pi * y / 300.0)
    return build_topography_hex27(nex=3, ney=3, nez=3, x_max=300.0, y_max=300.0,
                                  z_bottom=300.0, z_top_fn=topo, n_order=NO, dtype=F64)


@pytest.mark.parametrize("curved", [False, True])
def test_the_stamp_is_the_derivative_of_the_point_load(curved):
    # -M : grad(phi_a)(x_s) against central differences of the Lagrange weights a force
    # takes at x_s +- eps e_j, inside an element; on the curved mesh the inverse Jacobian at
    # the point has every cross term
    mesh = _curved() if curved else build_hex8_mesh(nex=3, ney=3, nez=3, x_max=300.0,
                                                    y_max=300.0, z_max=300.0, n_order=NO,
                                                    dtype=F64)
    x0, eps = np.array([137.0, 161.0, 183.0]), 1e-3
    stamp = build_point_stamp(mesh, *x0)
    load = stamp.moment_tensor_stamp(*M6)
    grad = []
    for d in range(3):
        shift = np.zeros(3)
        shift[d] = eps
        plus, minus = build_point_stamp(mesh, *(x0 + shift)), build_point_stamp(mesh, *(x0 - shift))
        assert plus.elem_id == minus.elem_id == stamp.elem_id
        grad.append((plus.scalar_stamp(1.0) - minus.scalar_stamp(1.0)) / (2 * eps))
    M = ((M6[0], M6[3], M6[4]), (M6[3], M6[1], M6[5]), (M6[4], M6[5], M6[2]))
    for i in range(3):
        fd = -sum(M[i][j] * grad[j] for j in range(3))
        assert float((load[i] - fd).abs().max()) < 1e-7 * float(fd.abs().max()), i


@pytest.mark.parametrize("curved, where, n", [
    (False, (100.0, 150.0, 150.0), 2), (False, (100.0, 100.0, 150.0), 4),
    (False, (100.0, 100.0, 100.0), 8), (True, None, 8)])
def test_a_point_on_a_face_an_edge_or_a_corner_is_shared(curved, where, n):
    # every element whose closure holds the point takes a share, the solid angle it holds:
    # 1/2, 1/4, 1/8 on the axis-aligned mesh, about that on the curved one, summing to 1
    mesh = _curved() if curved else build_hex8_mesh(nex=3, ney=3, nez=3, x_max=300.0,
                                                    y_max=300.0, z_max=300.0, n_order=NO,
                                                    dtype=F64)
    if where is None:                     # the corner element (1, 1, 1) shares with seven
        node = int(mesh.ibool[(1 * 3 + 1) * 3 + 1, 0, 0, 0])
        where = tuple(float(c[node]) for c in (mesh.x_global, mesh.y_global, mesh.z_global))
    stamp = build_moment_stamp(mesh, *where)
    assert isinstance(stamp, EdgeStamp) and len(stamp.parts) == n
    assert sum(stamp.weights) == pytest.approx(1.0, abs=1e-14)
    if curved:
        assert max(abs(w - 1.0 / n) for w in stamp.weights) < 0.05
        assert len(set(round(w, 12) for w in stamp.weights)) > 1      # not all alike
    else:
        assert all(w == pytest.approx(1.0 / n, abs=1e-14) for w in stamp.weights)
    # an explosion's load sums to zero over the shared nodes (a moment has no net force)
    loads = [_stamp_components(p, "explosion", (1.0,)) for p in stamp.parts]
    for c in range(3):
        total = sum(w * float(l[c].sum()) for w, l in zip(stamp.weights, loads))
        assert abs(total) < 1e-12 * max(float(l[c].abs().max()) for l in loads)


def test_an_explosion_on_a_corner_is_mirror_symmetric():
    # one-sided, a moment on an element corner radiates lopsided (see EdgeStamp); shared by
    # the eight elements there it is symmetric about the point to round-off
    shape, dh, dt, nt = (17, 17, 17), 10.0, 5e-4, 140
    t = np.arange(nt) * dt
    a = (math.pi * 15.0 * (t - 0.06)) ** 2
    w = torch.from_numpy((1 - 2 * a) * np.exp(-a))
    prop = PropSEM(ElasticVSSEM3DDeformed, shape=shape, dh=dh, dt=dt, n_order=NO,
                   f_dominant=15.0, dev="cpu", impl="eager", dtype=F64, abcn=8,
                   free_surface=False, source_type=["sxx", "syy", "szz"],
                   receiver_type=["vx", "vy", "vz"],
                   eager_options=EagerOptions(use_compile=False))
    rec = np.array([[[11.0, 8.0, 8.0], [5.0, 8.0, 8.0], [8.0, 11.0, 8.0], [8.0, 8.0, 11.0]]])
    models = [torch.full(shape, v, dtype=F64) for v in (3000.0, 1700.0, 2200.0)]
    with torch.no_grad():
        d = prop(w, np.array([[8.0, 8.0, 8.0]]), rec, models=models)[0]
    scale = float(d.abs().max())
    assert float((d[:, 0, 0] + d[:, 1, 0]).abs().max()) < 1e-12 * scale     # vx odd in x
    assert float((d[:, 0, 0] - d[:, 2, 1]).abs().max()) < 1e-12 * scale     # x, y, z alike
    assert float((d[:, 0, 0] - d[:, 3, 2]).abs().max()) < 1e-12 * scale


@pytest.mark.parametrize("names", [["sxx", "syy", "szz"], ["sxy"], ["syz", "sxx"]])
def test_hex27_moment_is_hex8s_on_a_flat_mesh(names):
    shape, dh, dt, nt = (13, 13, 13), 10.0, 5e-4, 80
    t = np.arange(nt) * dt
    a = (math.pi * 15.0 * (t - 0.04)) ** 2
    w = torch.from_numpy((1 - 2 * a) * np.exp(-a))
    kw = dict(shape=shape, dh=dh, dt=dt, n_order=NO, f_dominant=15.0, dev="cpu",
              impl="eager", dtype=F64, abcn=4, free_surface=True, source_type=names,
              receiver_type=["vx", "vy", "vz"], eager_options=EagerOptions(use_compile=False))
    models = [torch.full(shape, v, dtype=F64) for v in (3000.0, 1700.0, 2200.0)]
    src, rec = np.array([[6.3, 5.6, 4.0]]), np.array([[[3.0, 7.0, 8.0], [9.0, 4.0, 2.0]]])
    with torch.no_grad():
        a8 = PropSEM(ElasticVSSEM3D, **kw)(w, src, rec, models=models)
        a27 = PropSEM(ElasticVSSEM3DDeformed, **kw)(w, src, rec, models=models)
    assert float((a27 - a8).abs().max()) < 1e-12 * float(a8.abs().max())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("eq", [ElasticVSSEM3D, ElasticSEM3D, ElasticSEM3DDeformed])
def test_the_c_backend_takes_the_moment_tensor(eq):
    # the kernels see the assembled per-field patterns, a moment tensor's as a force's (the
    # displacement forms take no C-PML: a reflecting box)
    shape, dh, dt, nt = (25, 25, 25), 10.0, 5e-4, 200
    t = np.arange(nt) * dt
    a = (math.pi * 15.0 * (t - 0.06)) ** 2
    w = torch.from_numpy((1 - 2 * a) * np.exp(-a)).float()
    models = [torch.full(shape, v) for v in (3000.0, 1700.0, 2200.0)]
    src = np.array([[12.0, 12.0, 8.0], [10.3, 13.6, 9.1]])          # a corner, inside
    rec = np.array([[[4.0, 12.0, 14.0], [20.0, 9.0, 3.0]]] * 2)
    out = []
    for impl in ("eager", "c"):
        prop = PropSEM(eq, shape=shape, dh=dh, dt=dt, n_order=NO, f_dominant=15.0, dev="cuda",
                       impl=impl, abcn=8 if eq is ElasticVSSEM3D else 0, free_surface=True,
                       source_type=["sxx", "syy", "szz"],
                       receiver_type=["vx", "vy", "vz"])
        with torch.no_grad():
            out.append(prop(w.cuda(), src, rec, models=models).double().cpu())
    assert float((out[1] - out[0]).abs().max()) < 1e-4 * float(out[0].abs().max())
