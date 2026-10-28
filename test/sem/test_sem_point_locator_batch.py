"""Source / receiver points are located together and remembered per point.

Locating a point on a deformed mesh is a Newton solve on the Q2 map. It ran
one point at a time, in torch scalars -- ~0.1 s a point on a 3 km line
meshed at 6 m -- and PropSEM(impl='c') kept its stamps keyed on the whole geometry of
a call, so a new source position re-located every receiver as well: 37
receivers cost 4.7 s against 0.11 s of propagation. The points of a call are
now located in one batched Newton, stamped in one copy to the device, and
each point's location and stamp are kept on the mesh (least recently used
out first), so a shot at the same receivers locates only its source.

These tests pin: the batch recovers points of known element and local
coordinates; its stamps are the per-point formulas; a repeated point is not
located again; the record stays bounded; and records do not depend on
whether a stamp came from the record.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import sweep.sem.propagator as P
from sweep.sem import (PropSEM, ElasticSEM2DDeformed, build_point_stamps,
                       build_topography_hex27, build_topography_quad9, locate_points)
from sweep.sem.basis.quad9 import evaluate_q2_jacobian, q2_basis_at
from sweep.sem.propagator import lagrange_basis_at, lagrange_basis_deriv_at


def _quad9(dtype):
    def z_top(x):
        return 40.0 * (1.0 - torch.exp(-((x - 300.0) ** 2) / (2.0 * 90.0 ** 2)))
    return build_topography_quad9(nex=24, nez=12, x_max=600.0, z_bottom=300.0,
                                  z_top_fn=z_top, n_order=4, dtype=dtype)


def _hex27():
    def z_top(x, y):
        return 30.0 * torch.exp(-((x - 100.0) ** 2 + (y - 80.0) ** 2) / (2.0 * 40.0 ** 2))
    return build_topography_hex27(nex=6, ney=5, nez=4, x_max=200.0, y_max=160.0, z_bottom=120.0,
                                  z_top_fn=z_top, n_order=4, dtype=torch.float64)


def _known_points(control, n, seed, ndim):
    """``n`` points of known element and local coordinates, through the Q2 map."""
    rng = np.random.default_rng(seed)
    e = rng.integers(0, control.shape[0], n)
    loc = rng.uniform(-0.98, 0.98, (n, ndim))
    psi = [q2_basis_at(torch.tensor(loc[:, m])) for m in range(ndim)]
    c = control.double()[torch.tensor(e)]
    if ndim == 2:
        xyz = torch.einsum("ni,nj,nijk->nk", psi[1], psi[0], c)
    else:
        xyz = torch.einsum("nk,ni,nj,nkijd->nd", psi[2], psi[1], psi[0], c)
    return e, loc, xyz.numpy()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
def test_batched_quad9_locate_recovers_known_points(dtype):
    """Element and (ξ, η) to round-off, float32 meshes included: the Newton
    runs in float64 whatever the mesh dtype."""
    mesh = _quad9(dtype)
    e, loc, xz = _known_points(mesh.control_xy, 300, 0, 2)
    got_e, xi, eta = P._locate_in_quad9_many(mesh, xz)
    assert np.array_equal(got_e, e)
    assert np.abs(np.stack([xi, eta], -1) - loc).max() < 1e-11
    one = P._locate_in_quad9(mesh, *xz[7])
    assert one[0] == e[7] and abs(one[1] - xi[7]) < 1e-12 and abs(one[2] - eta[7]) < 1e-12


def test_batched_hex27_locate_recovers_known_points():
    mesh = _hex27()
    e, loc, xyz = _known_points(mesh.control_xyz, 200, 1, 3)
    got_e, xi, eta, gam = P._locate_in_hex27_many(mesh, xyz)
    assert np.array_equal(got_e, e)
    assert np.abs(np.stack([xi, eta, gam], -1) - loc).max() < 1e-11


def test_points_on_shared_edges_and_the_surface_locate():
    """Every GLL node -- element edges, corners, the curved surface -- lands in
    an element whose closure holds it, at its own position."""
    mesh = _quad9(torch.float64)
    nodes = np.unique(np.round(np.stack([mesh.x_e.numpy().ravel(), mesh.z_e.numpy().ravel()], -1), 9), axis=0)
    e, xi, eta = P._locate_in_quad9_many(mesh, nodes)
    assert np.abs(xi).max() <= 1.0 and np.abs(eta).max() <= 1.0
    back = np.stack([P._q2_eval_one(mesh.control_xy[int(k)], float(a), float(b)).numpy()
                     for k, a, b in zip(e, xi, eta)])
    assert np.abs(back - nodes).max() < 1e-9


def test_batched_stamps_are_the_per_point_stamps():
    """The vectorised weights are the scalar formulas' products in their order
    (the same numbers); the inverse Jacobian is that of the Q2 map."""
    mesh = _quad9(torch.float64)
    _, _, xz = _known_points(mesh.control_xy, 40, 2, 2)
    stamps = build_point_stamps(mesh, xz)
    for (x, z), s in zip(xz, stamps):
        e, xi, eta = P._locate_in_quad9(mesh, x, z)
        assert s.elem_id == e
        assert torch.equal(s.w_eta, lagrange_basis_at(mesh.xi, eta))
        assert torch.equal(s.w_xi, lagrange_basis_at(mesh.xi, xi))
        assert torch.equal(s.dw_eta, lagrange_basis_deriv_at(mesh.xi, eta))
        assert torch.equal(s.dw_xi, lagrange_basis_deriv_at(mesh.xi, xi))
        _, inv_J, _ = evaluate_q2_jacobian(mesh.control_xy[e:e + 1], torch.tensor([xi], dtype=torch.float64),
                                           torch.tensor([eta], dtype=torch.float64))
        want = inv_J[0, 0, 0].numpy()                 # [[∂ξ/∂x, ∂ξ/∂z], [∂η/∂x, ∂η/∂z]]
        np.testing.assert_allclose([s.ixx, s.ixz, s.iex, s.iez], want.ravel(), rtol=1e-12, atol=1e-15)


def test_a_repeated_receiver_is_not_located_again(monkeypatch):
    """A second call at the same receivers locates only its new source, and
    hands back the same stamps."""
    mesh = _quad9(torch.float64)
    located = []
    real = P._newton_q2
    monkeypatch.setattr(P, "_newton_q2", lambda c, t, *a, **k: (located.append(len(t)), real(c, t, *a, **k))[1])
    rec = [(30.0 + 20.0 * j, 120.0) for j in range(25)]
    first = build_point_stamps(mesh, rec + [(55.5, 150.0)])
    n_first = sum(located)
    second = build_point_stamps(mesh, rec + [(75.5, 150.0)])
    assert all(a is b for a, b in zip(first[:25], second[:25]))
    assert 0 < sum(located) - n_first <= 8              # one point, at most 8 candidate rows
    located.clear()
    assert locate_points(mesh, rec)[3] == locate_points(mesh, [rec[3]])[0]
    assert sum(located) == 0


def test_the_point_record_is_bounded(monkeypatch):
    """Least recently used out: a streamer's ever-new receivers do not grow it
    without end, and an evicted point locates again, to the same answer."""
    monkeypatch.setattr(P, "POINT_RECORD_SIZE", 8)
    mesh = _quad9(torch.float64)
    _, _, xz = _known_points(mesh.control_xy, 20, 3, 2)
    got = locate_points(mesh, xz)
    assert len(mesh.__dict__["_located"]) == 8
    stamps = build_point_stamps(mesh, xz)
    assert len(mesh.__dict__["_stamps"]) == 8 and len(stamps) == 20
    assert locate_points(mesh, xz[:3]) == got[:3]                  # evicted, located again


def _shot(impl, device, src):
    nz, nx = 25, 41
    x = np.arange(nx)
    topo = 3.0 - 3.0 * np.exp(-((x - 20) / 7.0) ** 2)
    dtype = torch.float64 if device == "cpu" else torch.float32
    prop = PropSEM(ElasticSEM2DDeformed, shape=(nz, nx), dh=5.0, dt=4e-4, n_order=4, abcn=8,
                   free_surface=True, topography=topo, f_dominant=15.0, impl=impl, device=device,
                   dtype=dtype, source_type=["sxx", "szz"], receiver_type=["vx", "vz"])
    nt = 150
    t = np.arange(nt) * 4e-4 - 0.08
    a = (np.pi * 15 * t) ** 2
    w = torch.tensor((1 - 2 * a) * np.exp(-a), dtype=dtype, device=device)
    rx = np.arange(3, 38, 3.0)
    rec = np.stack([rx, np.interp(rx, x, topo)], -1)[None]
    ms = [torch.full((nz, nx), v, dtype=dtype, device=device) for v in (2400.0, 1300.0, 2000.0)]

    def run(p, s):
        if impl == "eager":
            s, r = torch.tensor(s, dtype=torch.float64), torch.tensor(rec.repeat(len(s), 0))
        else:
            r = rec.repeat(len(s), 0)
        with torch.no_grad():
            out = p(w, s, r, models=ms)
        return out[0] if isinstance(out, (tuple, list)) else out
    return prop, run


@pytest.mark.parametrize("impl,device", [
    ("eager", "cpu"),
    pytest.param("c", "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")),
])
def test_records_do_not_depend_on_the_record(impl, device):
    """Shots run after others (their receivers' and one source's stamps from
    the record) give the records the same shots give on a fresh propagator.
    Bit for bit on the CPU; the CUDA kernels assemble with atomics, so there
    to their run-to-run noise (~1e-5 of the peak: a wrong stamp is O(1))."""
    shots = np.array([[8.0, 7.0], [20.0, 8.5], [31.0, 7.0]])
    prop, run = _shot(impl, device, shots)
    run(prop, shots[:2])
    later = run(prop, shots[1:])                                    # shot 1 again, shot 2 new
    fresh_prop, fresh = _shot(impl, device, shots)
    again = fresh(fresh_prop, shots[1:])
    if device == "cpu":
        assert torch.equal(later, again)
    else:
        torch.testing.assert_close(later, again, rtol=0, atol=1e-4 * float(again.abs().max()))
