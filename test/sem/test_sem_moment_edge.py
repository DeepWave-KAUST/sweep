"""A moment-tensor source on an element edge is stamped in every element there.

The load ``-M : ∇φ_a(x_s)`` differentiates the basis, whose gradient jumps
across C0 element edges; stamped in the one element the locator returned, an
explosion on a corner radiated 7e-3 (v_z) and 2e-2 (v_x) off mirror symmetry,
and integer grid sources sit on edges every ``n_order`` cells. ``EdgeStamp``
gives each element meeting at the point the share of a small disk it holds.
Pinned here: which elements and what shares (Quad4, Quad9, float32), that this
is the limit of a symmetric source shrunk onto the point, that the per-source
(source-encoding) layout carries it, and that the explosion is symmetric again.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from sweep.sem import (EdgeStamp, PointStamp, PropSEM, ElasticVSSEM2DDeformed,
                       build_moment_stamp, build_point_stamp, build_quad4_mesh,
                       build_topography_quad9)
from sweep.sem.propagator import _stamp_loads

NO = 4
MT = (1.0, 1.0, 0.0)                     # explosion


def _quad4(dtype=torch.float64):
    # 8 x 6 elements of 40 x 30 m: edges at x = 40 k, z = 30 k
    return build_quad4_mesh(nex=8, nez=6, x_max=320.0, z_max=180.0, n_order=NO,
                            dtype=dtype, device="cpu")


def _quad9(dtype=torch.float64):
    hill = lambda x: 20.0 * torch.exp(-((x - 160.0) / 60.0) ** 2)
    return build_topography_quad9(nex=8, nez=6, x_max=320.0, z_bottom=200.0,
                                  z_top_fn=hill, n_order=NO, dtype=dtype,
                                  device="cpu")


def _pattern(mesh, stamp, kind="moment_tensor", payload=MT):
    """The global per-field load of one stamp, as _build_source_patterns
    scatters it."""
    pats = [torch.zeros(mesh.n_global, dtype=torch.float64) for _ in range(2)]
    for e, comps in _stamp_loads(stamp, kind, payload):
        g = mesh.ibool[e].reshape(-1)
        for p, c in zip(pats, comps):
            p.index_add_(0, g, c.reshape(-1).double())
    return torch.stack(pats)


# ----------------------------- which elements ------------------------------- #
def test_inside_an_element_it_is_the_plain_stamp():
    m = _quad4()
    s = build_moment_stamp(m, 55.0, 47.0)
    assert isinstance(s, PointStamp)
    ref = build_point_stamp(m, 55.0, 47.0)
    assert s.elem_id == ref.elem_id
    assert torch.equal(_pattern(m, s), _pattern(m, ref))


@pytest.mark.parametrize("x,z,n,shares", [
    (80.0, 47.0, 2, [0.5, 0.5]),               # a vertical edge
    (55.0, 60.0, 2, [0.5, 0.5]),               # a horizontal edge
    (80.0, 60.0, 4, [0.25] * 4),               # an interior corner
    (0.0, 60.0, 2, [0.5, 0.5]),                # a corner on the outer boundary
    (0.0, 47.0, 1, None),                      # an outer edge: one element
])
def test_quad4_edges_and_corners(x, z, n, shares):
    m = _quad4()
    s = build_moment_stamp(m, x, z)
    if n == 1:
        assert isinstance(s, PointStamp)
        return
    assert isinstance(s, EdgeStamp) and len(s.parts) == n
    assert len({p.elem_id for p in s.parts}) == n
    assert s.weights == pytest.approx(shares, abs=1e-12)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_quad9_corner_shares_are_its_sector_angles(dtype):
    """On the hill flank the row lines slope, so the four sectors at a corner
    are near 90 deg -/+ the slope angle rather than all 90: the shares follow
    the angles. float32 geometry is still found on the edge."""
    m = _quad9(dtype)
    x = 120.0                                   # a column edge, on the flank
    e = 2 * m.nex + int(round(x / 40.0))        # row 2, the element right of x
    z = float(m.z_e[e][0, 0])                   # one of its corner nodes at x
    s = build_moment_stamp(m, x, z)
    assert isinstance(s, EdgeStamp) and len(s.parts) == 4
    assert len({p.elem_id for p in s.parts}) == 4
    assert sum(s.weights) == pytest.approx(1.0, abs=1e-12)
    assert all(abs(w - 0.25) < 0.05 for w in s.weights), s.weights
    assert max(s.weights) - min(s.weights) > 1e-3, (
        "the flank's slope should split the corner unevenly")


def test_quad9_flat_edge_is_split_in_half():
    m = _quad9()
    x = 40.0
    e = 5 * m.nex + 1                           # column 1 of the bottom row
    z = float(m.z_e[e][2, 0])                   # the mid node of its x = 40 edge
    s = build_moment_stamp(m, x, z)
    assert isinstance(s, EdgeStamp) and len(s.parts) == 2
    assert s.weights == pytest.approx([0.5, 0.5], abs=1e-12)


# ------------------------------ what load ----------------------------------- #
@pytest.mark.parametrize("x,z,offsets", [
    (80.0, 47.0, [(-1, 0), (1, 0)]),
    (80.0, 60.0, [(-1, -1), (-1, 1), (1, -1), (1, 1)]),
])
def test_edge_load_is_the_limit_of_a_symmetric_source(x, z, offsets):
    """Average the one-sided stamps of points nudged symmetrically off the
    edge; as the nudge shrinks it tends to the EdgeStamp's load."""
    m = _quad4()
    edge = _pattern(m, build_moment_stamp(m, x, z))
    rel = {}
    for d in (1e-2, 1e-4):
        avg = sum(_pattern(m, build_point_stamp(m, x + dx * d, z + dz * d))
                  for dx, dz in offsets) / len(offsets)
        rel[d] = float((avg - edge).norm() / edge.norm())
    assert rel[1e-2] < 1e-2 and rel[1e-4] < 1e-4, rel
    assert rel[1e-4] < rel[1e-2] / 50, f"not converging linearly: {rel}"
    # and the one-sided stamp is what it replaced: far from it
    one = _pattern(m, build_point_stamp(m, x, z))
    assert float((one - edge).norm() / edge.norm()) > 0.1


def test_force_sources_keep_their_single_element_stamp():
    """A force has no derivative: the Lagrange weights already agree on the
    shared nodes, so a force on a corner is stamped as before, bit for bit."""
    m = _quad4()
    edge = build_moment_stamp(m, 80.0, 60.0)
    one = build_point_stamp(m, 80.0, 60.0)
    f_edge = _pattern(m, edge, "force", (0.0, 1.0))
    f_one = _pattern(m, one, "force", (0.0, 1.0))
    assert torch.allclose(f_edge, f_one, rtol=0, atol=1e-15)


# --------------------------- through PropSEM -------------------------------- #
NZ, NX, DH, F0 = 41, 41, 12.5, 8.0
VP, VS, RHO = 2400.0, 1250.0, 2000.0
DT = 0.30 * (0.172 * NO * DH) / (VP * 1.2)
NT = 160


def _prop(**kw):
    return PropSEM(ElasticVSSEM2DDeformed, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                   n_order=NO, free_surface=("top",),
                   abcn=4 * NO, f_dominant=F0, dev="cpu", impl="eager",
                   dtype=torch.float32, **kw)


def _wav():
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32))


def _models():
    return [torch.full((NZ, NX), v) for v in (VP, VS, RHO)]


def test_explosion_on_a_corner_is_symmetric():
    """x = 20 and z = 12 are both element boundaries (the pad is 16 cells):
    the case that was 7e-3 / 2e-2 off."""
    p = _prop(source_type=["sxx", "szz"])
    src = np.array([[NX // 2, 12]])
    rix = NX // 2 + np.array([-12, -9, -6, -3, 3, 6, 9, 12])
    rec = np.stack([rix, np.full_like(rix, 2)], -1)[None]
    with torch.no_grad():
        s = p(_wav(), src, rec, models=_models())
    vx, vz = s[0, :, :, 0], s[0, :, :, 1]
    vz_defect = float((vz - vz.flip(-1)).norm() / vz.norm())
    vx_defect = float((vx + vx.flip(-1)).norm() / vx.norm())
    assert vz_defect < 1e-4 and vx_defect < 1e-4, (vz_defect, vx_defect)


def test_per_source_wavelets_carry_edge_stamps():
    """Source encoding with a per-source wavelet keeps each source's stamp on
    its own row; an edge source is wider than an interior one, and the rows are
    padded. With the same wavelet on every source it must reproduce the shared
    wavelet run."""
    p = _prop(source_type=["sxx", "szz"])
    src = np.array([[[NX // 2, 12], [NX // 2 + 1, 13]]])     # a corner, an interior point
    rix = np.arange(6, NX - 6, 4)
    rec = np.stack([rix, np.full_like(rix, 2)], -1)[None]
    w = _wav()
    with torch.no_grad():
        shared = p(w, src, rec, models=_models())
        per = p(torch.stack([w, w]), src, rec, models=_models())
    scale = float(shared.abs().max())
    assert scale > 0
    assert float((per - shared).abs().max()) <= 1e-5 * scale
