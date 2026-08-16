"""Anisotropic equations must REFUSE a free surface.

The isotropic image method (odd mirror of the normal stresses about the
surface row) does not satisfy the stress-free condition of an anisotropic
medium — there the surface traction couples through the stiffness tensor and
needs a Mittet/Robertsson-style anisotropic treatment.  Before this guard,
``PropTorch(eq, free_surface=True)`` silently ran the isotropic mirror on
every VTI/TTI solver (the equation-constructor guard in AcousticVTI1st was
bypassed by the propagator-level argument), producing wrong surface physics.

Covers acoustic (VTI 1st-order 2D/3D, qP-VTI, qP-TTI, Alkhalifah) and elastic
(TTI, TTI-SG) equations, for both ``free_surface=True`` and the per-edge form.
"""
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEV = "cuda"


def _build(cls, free_surface, ndim=2):
    from sweep.propagator.torch import PropTorch
    from sweep.propagator.options import EagerOptions

    eq = cls(spatial_order=4, device=DEV, backend="torch")
    shape = (48, 56) if ndim == 2 else (24, 20, 24)
    return PropTorch(eq, impl="eager", eager_options=EagerOptions(use_compile=False),
                     use_ckpt=False, dev=DEV, shape=shape, abcn=16, dh=10.0, dt=8e-4,
                     free_surface=free_surface, nt=100, B=1)


def _cases():
    from sweep.equations import (AcousticTTI, AcousticVTI, AcousticVTI1st,
                                 AcousticVTI1st3D, ElasticTTI, ElasticTTISG,
                                 ElasticTTISG3D, ElasticTTI2nd)
    from sweep.equations.qP_tariq import AcousticTariq

    return [(AcousticVTI1st, 2), (AcousticVTI1st3D, 3), (AcousticVTI, 2),
            (AcousticTTI, 2), (AcousticTariq, 2), (ElasticTTI, 2), (ElasticTTISG, 2),
            (ElasticTTISG3D, 3), (ElasticTTI2nd, 2)]


@pytest.mark.parametrize("cls,ndim", _cases(), ids=lambda c: getattr(c, "__name__", c))
def test_free_surface_true_raises(cls, ndim):
    with pytest.raises(NotImplementedError, match="anisotropic"):
        _build(cls, True, ndim)


def test_per_edge_top_also_raises():
    from sweep.equations import ElasticTTISG

    with pytest.raises(NotImplementedError, match="anisotropic"):
        _build(ElasticTTISG, ["top"])


@pytest.mark.parametrize("cls,ndim", _cases(), ids=lambda c: getattr(c, "__name__", c))
def test_no_free_surface_still_constructs(cls, ndim):
    _build(cls, False, ndim)


def test_isotropic_free_surface_unaffected():
    from sweep.equations import Acoustic, Elastic

    _build(Elastic, True)
    _build(Elastic, ["top", "left"])
    _build(Acoustic, True)
