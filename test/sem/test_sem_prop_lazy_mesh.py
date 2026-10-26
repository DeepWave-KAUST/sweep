"""`PropSEM.mesh` is built on first access, not in the constructor.

The DD compact path reads the propagator's `Hex8Layout` and never the mesh,
so a rank that only runs DD must never pay for the mesh's arrays. Everything
else keeps reading `prop.mesh` and must get the same object, just later --
with two deliberate exceptions pinned here: a topography mesh is still built
in the constructor (an inverted element is refused where it always was), and
a mesh the caller assigns is what BOTH paths then describe.

CPU only, no forward: this is about construction semantics.
"""
from __future__ import annotations

import dataclasses
import pickle

import pytest
import torch

import sweep.sem as S
from sweep.sem.equations import (AcousticSEM2D, AcousticSEM2DDeformed,
                                 AcousticSEM3D, AcousticSEM3DDeformed)
from sweep.sem.mesh.hex8 import Hex8Layout, build_hex8_mesh

NO, DH = 4, 10.0


def _bump2d(x):                      # the surface in metres, z_top(x)
    return 20.0 * torch.exp(-((x - 60.0) / 25.0) ** 2)


def _bump3d(x, y):
    return 20.0 * torch.exp(-(((x - 60.0) / 25.0) ** 2 + ((y - 40.0) / 25.0) ** 2))


# ``topography`` is the surface row of every model column: the bumps sampled
# on the (9, 13) / (9, 9, 13) models below, (nx,) in 2-D and (ny, nx) in 3-D
TOPO2D = (_bump2d(torch.arange(13, dtype=torch.float64) * DH) / DH).numpy()
_Y, _X = torch.meshgrid(torch.arange(9, dtype=torch.float64) * DH,
                        torch.arange(13, dtype=torch.float64) * DH, indexing="ij")
TOPO3D = (_bump3d(_X, _Y) / DH).numpy()


def _prop(equation, shape, abcn=2 * NO, **kw):
    return S.PropSEM(equation, shape=shape, dh=DH, dt=1e-3, nt=4, n_order=NO,
                     abcn=abcn, dev="cpu", impl="eager", **kw)


def test_flat_mesh_is_built_on_first_access_and_then_shared():
    p = _prop(AcousticSEM3D, (9, 9, 13), free_surface=("top",))
    assert p._mesh is None, "the constructor built the mesh"
    m = p.mesh
    assert p._mesh is m and p.mesh is m, "the property must cache one object"


@pytest.mark.parametrize("equation,shape,topo", [
    (AcousticSEM2D, (9, 13), None),
    (AcousticSEM2DDeformed, (9, 13), TOPO2D),
    (AcousticSEM3D, (9, 9, 13), None),
    (AcousticSEM3DDeformed, (9, 9, 13), TOPO3D),
])
@pytest.mark.parametrize("abcn", [0, 2 * NO])
def test_analytic_gll_shape_equals_the_meshes(equation, shape, topo, abcn):
    """`_gll_shape` used to be read off `mesh.x_e.shape`; now it is computed
    from the padded element counts and must agree for every family, with
    and without an absorbing layer (padding is per SIDE: the free surface is
    never padded)."""
    p = _prop(equation, shape, abcn=abcn, free_surface=("top",),
              topography=topo)
    assert p._gll_shape == tuple(p.mesh.x_e.shape)


def test_topography_mesh_is_still_built_in_the_constructor():
    """An inverted element must be refused at construction, not at the first
    forward -- so the topography families keep building eagerly."""
    # valid rows (inside the model) that the mesh cannot follow: a cliff the
    # model's full depth after column 0, with no absorbing layer below to
    # grade into, folds the first element over
    cliff = torch.full((13,), 7.9, dtype=torch.float64)
    cliff[0] = 0.0
    with pytest.raises(ValueError, match="inverted or degenerate"):
        _prop(AcousticSEM2DDeformed, (9, 13), abcn=0, free_surface=("top",),
              topography=cliff)
    p = _prop(AcousticSEM2DDeformed, (9, 13), free_surface=("top",),
              topography=TOPO2D)
    assert p._mesh is not None, "topography mesh was deferred"


@pytest.mark.parametrize("equation,shape,topo", [
    (AcousticSEM3D, (9, 9, 13), None),
    (AcousticSEM2DDeformed, (9, 13), TOPO2D),
    (AcousticSEM3DDeformed, (9, 9, 13), TOPO3D),
])
def test_propagator_pickles_and_rebuilds_the_same_mesh(equation, shape, topo):
    """The deferred builder lives on the instance now. A nested closure there
    would have made every topography propagator unpicklable (it did, in the
    first cut of this change -- caught in review); the surface object it
    became must round-trip, and a round-tripped propagator must rebuild the
    same mesh."""
    p = _prop(equation, shape, free_surface=("top",), topography=topo)
    q = pickle.loads(pickle.dumps(p))
    if topo is None:
        assert q._mesh is None, "pickling must not force the flat mesh"
    else:
        q.mesh = None                          # force a rebuild from the builder
    assert torch.equal(q.mesh.ibool, p.mesh.ibool)
    assert torch.equal(q.mesh.x_e, p.mesh.x_e)
    assert torch.equal(q.mesh.z_e, p.mesh.z_e)


def test_layout_follows_an_assigned_mesh():
    """`prop.mesh = other` is honoured by the compact DD path: the layout is
    the assigned mesh's own, not the constructor's, so the two paths can
    never describe different geometries."""
    p = _prop(AcousticSEM3D, (9, 9, 13), free_surface=("top",))
    lay0 = p.hex8_layout()                     # derived from the build args
    assert isinstance(lay0, Hex8Layout) and p._mesh is None
    m = p.mesh
    assert torch.equal(p.hex8_layout().x_axis, m.layout.x_axis)
    other = build_hex8_mesh(nex=m.nex, ney=m.ney, nez=m.nez,
                            x_max=2.0 * m.x_max, y_max=m.y_max, z_max=m.z_max,
                            n_order=m.n_order)
    p.mesh = other
    assert p.hex8_layout() is other.layout
    assert p.hex8_layout().x_max == 2.0 * m.x_max
    # a hand-assembled mesh carries no layout: refuse, do not fall back to
    # the constructor's geometry silently
    p.mesh = dataclasses.replace(other, layout=None)
    with pytest.raises(TypeError, match="carries no Hex8Layout"):
        p.hex8_layout()


def test_hex8_layout_is_refused_for_other_families():
    p = _prop(AcousticSEM2D, (9, 13), free_surface=("top",))
    with pytest.raises(TypeError, match="flat hex8 family"):
        p.hex8_layout()
