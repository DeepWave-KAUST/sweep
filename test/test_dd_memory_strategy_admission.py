"""``ModelParallel`` must refuse a full-storage or checkpoint propagator.

DD reconstructs each tile's forward wavefield from saved boundaries; there is
no decomposed full-storage or checkpoint backward, and ``eq_driver.cuh``'s
"domain-decomposed backward (cut_face_mask) is boundary-saving only" says the
same from the C side.

Until this check existed the request was silently REPLACED. ``__init__`` reads
six keys off ``boundary_saving_config`` and never reads ``enabled`` or
``use_ckpt``, and ``_tile_memory_strategy`` returns ``BoundarySaving``
unconditionally, so a caller who passed ``Full()`` got reconstruction anyway.
Measured: ``dd=full`` against a ``full`` single-card reference differed by rel
1.600e-07 and ``dd=ckpt`` against a ``ckpt`` reference by the SAME 1.600e-07 --
the same number twice being the tell -- while both were BIT-EXACT against a
``bs-gpu`` reference, i.e. the DD side had been running boundary saving all
along. The C-side check could never fire, because Python had already swapped
the strategy out from under it.
"""
import numpy as np
import pytest
import torch

from sweep.equations import Acoustic
from sweep.propagator.options import BoundarySaving, Ckpt, Full
from sweep.propagator.torch import PropTorch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="ModelParallel builds CUDA tile solvers")

SHAPE, DH, DT, NT, ABCN = (48, 51), 10.0, 6e-4, 16, 8


def _prop(memory):
    dev = torch.device("cuda:0")
    return PropTorch(Acoustic(spatial_order=4, device=dev, backend="torch"),
                     backend="torch", impl="c", shape=SHAPE, dh=DH, dt=DT,
                     nt=NT, abcn=ABCN, dev=dev, memory=memory)


def _mesh():
    from sweep.parallel import MeshTopology
    return MeshTopology(py=1, px=1, shot_groups=1, world_size=1, rank=0)


def _model_parallel(memory):
    from sweep.parallel.dd_propagator import ModelParallel
    return ModelParallel(_prop(memory), _mesh())


@pytest.mark.parametrize("memory, expected", [(Full(), "full"), (Ckpt(), "ckpt")])
def test_dd_refuses_non_boundary_strategies(memory, expected):
    with pytest.raises(NotImplementedError) as excinfo:
        _model_parallel(memory)
    message = str(excinfo.value)
    assert "boundary saving" in message
    # the message must name what was actually asked for, not just complain
    assert repr(expected) in message
    # and say what to write instead
    assert "BoundarySaving" in message


def test_dd_accepts_boundary_saving():
    """The control: the strategy DD does support still constructs."""
    assert _model_parallel(BoundarySaving(storage="gpu")) is not None


def test_dd_accepts_the_default_propagator():
    """impl='c' with no memory knob resolves to 'boundary', so it must pass.

    Without this, tightening the check to something stricter than
    memory_strategy == 'boundary' would break every existing DD script and the
    parametrised test above would still be green.
    """
    dev = torch.device("cuda:0")
    prop = PropTorch(Acoustic(spatial_order=4, device=dev, backend="torch"),
                     backend="torch", impl="c", shape=SHAPE, dh=DH, dt=DT,
                     nt=NT, abcn=ABCN, dev=dev)
    assert prop.memory_strategy == "boundary"
    from sweep.parallel.dd_propagator import ModelParallel
    assert ModelParallel(prop, _mesh()) is not None
