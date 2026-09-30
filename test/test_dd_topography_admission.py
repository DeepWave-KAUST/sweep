"""``ModelParallel`` must refuse a propagator built with ``topography=``.

The tiles are built from the wrapped propagator's spec, and ``topography=`` was
never part of it: a model with a surface was decomposed as a flat-top problem,
with no error. Carrying it across would not help either. DD's only gradient
path is boundary saving, and boundary saving under a per-column surface gives a
wrong gradient -- the single-card path refuses that pair at run time
(``_guard_boundary_saving_topography``), but the tiles never see a surface, so
that guard could not fire under DD.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic
from sweep.propagator.options import BoundarySaving
from sweep.propagator.torch import PropTorch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="ModelParallel builds CUDA tile solvers")

SHAPE, DH, DT, NT, ABCN = (48, 51), 10.0, 6e-4, 16, 8


def _hill(nx=SHAPE[1], height=4.0, width=8.0):
    x = np.arange(nx, dtype=np.float32)
    return (height * np.exp(-((x - nx / 2) ** 2) / (2.0 * width ** 2))).round().astype(np.int64)


def _prop(**kw):
    dev = torch.device("cuda:0")
    return PropTorch(Acoustic(spatial_order=4, device=dev, backend="torch"),
                     backend="torch", impl="c", shape=SHAPE, dh=DH, dt=DT,
                     nt=NT, abcn=ABCN, device=dev, memory=BoundarySaving(), **kw)


def _model_parallel(prop):
    from sweep.parallel import MeshTopology
    from sweep.parallel.dd_propagator import ModelParallel
    return ModelParallel(prop, MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0))


def test_dd_refuses_topography():
    prop = _prop(topography=_hill(), topo_method="image")
    assert prop.topography is not None
    with pytest.raises(NotImplementedError, match="topography"):
        _model_parallel(prop)


def test_dd_still_accepts_a_flat_top():
    """The control: the same propagator without a surface still decomposes."""
    assert _model_parallel(_prop()) is not None
