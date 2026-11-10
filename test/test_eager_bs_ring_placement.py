"""Eager boundary saving's ring: where it sits and what it holds.

The reverse pass rebuilds each step's input backwards in time with the C-PML
memory zeroed and puts the saved boundary ring back over it last; the ring
keeps the reconstruction exact inside it.  Two ways it went wrong, each a
gradient far off the full tape's:

* The ring was saved before the source went in and put back over the source
  the reverse drivers re-inject, so a source on a ring cell was dropped.  A
  free-surface face has no PML, and its ring lies inside the model at depth
  ``abcn - halo .. abcn - 1``: a source there gave a vp gradient 17-21 % off
  (relative L2), against 1e-3 a few cells away.
* Every face had its ring at ``max(abcn) + halo``.  With per-face PML widths a
  thinner face had its ring inside the model too, and the strip between the
  ring and the thin PML, outside the ring, came back wrong: 1.8-22 % off, the
  error in the columns next to that face.

Eager boundary saving matches the full tape to round-off except on the
outermost ``M + 1`` cells next to an absorbing face, where it keeps an error by
design (the step is re-run from a state without C-PML memory), 2e-5 to 1e-3
relative on these small grids and larger next to a thinner, steeper ramp.  So
a source on the ring is held against a control, the same run with the source a
few cells deeper, and a thinner face is held to round-off away from those
cells.
"""
import numpy as np
import pytest
import torch

from conftest import requires_compile
from sweep.equations import Acoustic
from sweep.equations.elastic import Elastic
from sweep.propagator.options import BoundarySaving, EagerOptions, Full
from sweep.propagator.torch import PropTorch
from sweep.propagator._ring_geometry import _ring_offsets
from sweep.propagator._ring_geometry import _interior_index_tuple, _ring_index_tuples

SHAPE, NT, ABCN = (40, 44), 160, 8


def _wavelet():
    t = np.arange(NT) * 1e-3 - 0.03
    a = (np.pi * 25.0 * t) ** 2
    return ((1 - 2 * a) * np.exp(-a)).astype(np.float32)


def _models(eq):
    g = torch.Generator().manual_seed(1)
    vp = 2000 + 300 * torch.rand(SHAPE, generator=g)
    return [vp] + ([vp / 1.8, torch.full(SHAPE, 2000.0)] if eq is Elastic else [])


def _vp_gradient(eq, memory, src, abcn=ABCN, free_surface=False, k=1, adj=False):
    eager = EagerOptions(use_compile=k > 1, compile_backend="aot_eager", **({"steps_per_call": k} if k > 1 else {}))
    kw = dict(source_type=["szz"], receiver_type=["vz"]) if eq is Elastic else {}
    prop = PropTorch(eq(spatial_order=4, device="cpu"), shape=SHAPE, dh=10.0, dt=1e-3, nt=NT, dev="cpu",
                     impl="eager", abcn=abcn, free_surface=free_surface, memory=memory, eager_options=eager, **kw)
    models = [m.clone().requires_grad_(i == 0) for i, m in enumerate(_models(eq))]
    rec = np.array([[[ix, 12] for ix in range(2, SHAPE[1] - 2, 3)]])
    out = prop(_wavelet(), np.array([src]), rec, models=models, adj=adj)
    out.square().sum().backward()
    return models[0].grad.detach()


def _bs_error(eq, src, storage_dtype="fp32", inner=False, **kw):
    """Relative L2 of boundary saving's vp gradient against the full tape's,
    over the box or, with ``inner``, without the outermost ``EDGE`` cells next
    to an absorbing face."""
    full = _vp_gradient(eq, Full(), src, **kw)
    bs = _vp_gradient(eq, BoundarySaving(storage="gpu", storage_dtype=storage_dtype), src, **kw)
    if inner:
        top = 0 if kw.get("free_surface") else EDGE
        full, bs = full[top:-EDGE, EDGE:-EDGE], bs[top:-EDGE, EDGE:-EDGE]
    return float((bs - full).norm() / full.norm())


HALO = 2                       # spatial_order 4
EDGE = HALO + 1                # the cells next to an absorbing face that keep an error
FLOOR = 1e-6                   # float32 round-off of the relative gradient error here


@pytest.mark.parametrize("eq, k, storage_dtype, adj", [
    (Acoustic, 1, "fp32", False),
    (Acoustic, 1, "fp32", True),
    (Acoustic, 1, "int8", False),
    pytest.param(Acoustic, 4, "fp32", False, marks=requires_compile),
    (Elastic, 1, "fp32", False),
    pytest.param(Elastic, 4, "fp32", False, marks=requires_compile),
], ids=["acoustic", "acoustic_adj", "acoustic_int8", "acoustic_k4", "elastic", "elastic_k4"])
def test_a_source_on_the_ring_of_a_free_surface_face(eq, k, storage_dtype, adj):
    """The free-surface face's ring lies inside the model: a source on it is
    reconstructed as one a few cells deeper is.  int8 keeps one scale per 256
    ring cells, and a source on the ring sets its block's scale, which
    quantizes the cells next to it coarser: 8x the control's error (3.5e-3
    against 4.4e-4), where a dropped source cost 17-21 %.  Below FLOOR both
    errors are float32 round-off (1e-7 to 5e-7 in adjoint mode), and their
    ratio says nothing."""
    factor = 20 if storage_dtype == "int8" else 3
    control = _bs_error(eq, (22, ABCN + 3), storage_dtype, free_surface=True, k=k, adj=adj)
    for z in (ABCN - HALO, ABCN - 1):
        err = _bs_error(eq, (22, z), storage_dtype, free_surface=True, k=k, adj=adj)
        assert err < max(factor * control, FLOOR), (z, err, control)


@pytest.mark.parametrize("eq", [Acoustic, Elastic], ids=["acoustic", "elastic"])
@pytest.mark.parametrize("abcn", [(8, 8, 4, 8), (12, 8, 8, 8)], ids=["thin_x_lo", "thick_z_lo"])
def test_a_thinner_pml_face_keeps_its_ring_out_of_the_model(eq, abcn):
    """Each face's ring sits in its own PML: with a face thinner than the
    others the gradient matches the full tape's to round-off away from the
    outermost cells next to an absorbing face (3e-6 at most), as with all
    faces alike.  With every ring at ``max(abcn) + halo`` the thinner face's
    ring lay inside the model, and the cells next to it read the wrongly
    rebuilt strip outside it: 5e-3 off there.  On the outermost cells the
    error of the design is 3-4x larger next to the 4-cell face."""
    src = (6, 20)                                  # next to the x_lo face
    assert _bs_error(eq, src, abcn=abcn, inner=True) < 2e-5


def test_ring_offsets_follow_each_faces_ramp():
    # every face alike: the former max(abcn) + halo
    assert _ring_offsets((8, 8, 8, 8), 8, 2) == (10, 10, 10, 10)
    # a face without a ramp (free surface) keeps the former depth, inside the model
    assert _ring_offsets((0, 8, 8, 8), 8, 2) == (10, 10, 10, 10)
    # a thinner face has its ring in its own ramp
    assert _ring_offsets((12, 8, 4, 8), 12, 2) == (14, 10, 6, 10)


def test_rings_and_interior_take_offsets_per_face():
    shape = (1, 1, 30, 40)
    rings = _ring_index_tuples(shape, 2, (14, 10, 6, 10), 2)
    assert [r[2] if r[3] == slice(None) else r[3] for r in rings] == [
        slice(12, 14), slice(20, 22), slice(4, 6), slice(30, 32)]
    assert _interior_index_tuple(shape, 2, (14, 10, 6, 10))[2:] == (slice(14, 20), slice(6, 30))
    # one offset per axis: both faces alike, as before
    assert _ring_index_tuples(shape, 2, (10, 10), 2) == _ring_index_tuples(shape, 2, (10, 10, 10, 10), 2)
