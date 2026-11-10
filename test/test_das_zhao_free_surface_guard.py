"""DASZhao / DASElastic (2-D and 3-D) have no free-surface condition in their
step, eager or CUDA: a free surface used to be accepted and only dropped the
top PML (eager Rayleigh speed 8-17 % off; CUDA recorded zeros at the surface).
They must refuse it and point to DASMu, which has one."""
import numpy as np
import pytest

from sweep.equations import DASElastic, DASElastic3D, DASMu, DASZhao, DASZhao3D
from sweep.propagator.torch import PropTorch


def _build(cls, ndim, **extra):
    eq = cls(spatial_order=4, device="cpu", backend="torch")
    shape = (40, 48) if ndim == 2 else (16, 14, 12)
    return PropTorch(eq, shape, impl="eager", dev="cpu", dh=10.0, dt=1e-3,
                     abcn=8, nt=10, B=1, **extra)


@pytest.mark.parametrize("cls,ndim", [(DASZhao, 2), (DASElastic, 2), (DASZhao3D, 3), (DASElastic3D, 3)])
def test_free_surface_is_refused(cls, ndim):
    assert not cls.supports_free_surface
    with pytest.raises(NotImplementedError, match="DASMu"):
        _build(cls, ndim, free_surface=True)


def test_topography_is_refused():
    with pytest.raises(NotImplementedError):
        _build(DASZhao, 2, topography=np.zeros(48, dtype=np.int64))


def test_no_free_surface_still_builds():
    assert _build(DASZhao, 2, free_surface=False) is not None


def test_dasmu_keeps_its_free_surface():
    assert _build(DASMu, 2, free_surface=True).free_surface
