"""Anisotropic equations must REFUSE a free surface.

The isotropic image method (odd mirror of the normal stresses about the
surface row) does not satisfy the stress-free condition of an anisotropic
medium — there the surface traction couples through the stiffness tensor and
needs a Mittet/Robertsson-style anisotropic treatment.  Before this guard,
``PropTorch(eq, free_surface=True)`` silently ran the isotropic mirror on
every VTI/TTI solver (the equation-constructor guard in AcousticVTI1st was
bypassed by the propagator-level argument), producing wrong surface physics.

The set of equations is read from the registry, not listed by hand: every
registered equation with an anisotropy parameter among its models is covered,
so a new VTI/TTI equation that forgets ``supports_free_surface = False`` fails
here.  ``EXPECTED`` pins that set, so the model-name rule cannot drop one
silently either.  Each is checked for ``free_surface=True``, the per-edge
form and an implied (``topography=``) free surface, on impl='eager' and, where
the equation has a compiled backend, impl='c'.
"""
import pytest

torch = pytest.importorskip("torch")

DEV = "cuda"
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# Thomsen / tilt / Alkhalifah parameters: any of them makes the medium anisotropic.
ANISO_PARAMS = {"epsilon", "delta", "gamma", "eta", "theta", "phi"}

EXPECTED = {"AcousticVTI1st", "AcousticVTI1st3D", "AcousticVTI", "AcousticTTI", "AcousticVTIBorn",
            "AcousticTTIBorn", "AcousticTariq", "ElasticTTI", "ElasticTTISG", "ElasticTTISG3D",
            "ElasticTTI2nd"}


def _anisotropic_classes():
    from sweep.equations import equation_classes

    out = {}
    for cls in set(equation_classes().values()):
        # the class table where there is one; an equation without it is built
        # (and a failure to build is an error here, not a silent skip)
        names = {s.name for s in cls.MODEL_SPECS} or set(cls(spatial_order=4, device="cpu").models)
        if ANISO_PARAMS & names:
            out[cls.__name__] = (cls, int(cls(spatial_order=4, device="cpu").ndim))
    return out


ANISO = _anisotropic_classes()
CASES = [ANISO[n] for n in sorted(ANISO)]
IDS = sorted(ANISO)


def test_registry_rule_finds_every_anisotropic_equation():
    assert set(ANISO) == EXPECTED, (
        f"new: {sorted(set(ANISO) - EXPECTED)}, missing: {sorted(EXPECTED - set(ANISO))} -- "
        "a new anisotropic equation needs supports_free_surface = False and an entry in EXPECTED")


@pytest.mark.parametrize("cls,ndim", CASES, ids=IDS)
def test_class_declares_no_free_surface(cls, ndim):
    assert cls.supports_free_surface is False


def _impls(cls):
    from sweep.equations import supports_torch_binding

    return ["eager", "c"] if supports_torch_binding(cls) else ["eager"]


IMPL_CASES = [(cls, ndim, impl) for cls, ndim in CASES for impl in _impls(cls)]
IMPL_IDS = [f"{cls.__name__}-{impl}" for cls, _, impl in IMPL_CASES]


def _build(cls, free_surface, ndim=2, impl="eager", **extra):
    from sweep.propagator.options import EagerOptions
    from sweep.propagator.torch import PropTorch

    eq = cls(spatial_order=4, device=DEV, backend="torch")
    shape = (48, 56) if ndim == 2 else (24, 20, 24)
    kw = dict(eager_options=EagerOptions(use_compile=False)) if impl == "eager" else {}
    return PropTorch(eq, impl=impl, dev=DEV, shape=shape, abcn=16, dh=10.0, dt=8e-4,
                     free_surface=free_surface, nt=100, B=1, **kw, **extra)


@cuda
@pytest.mark.parametrize("cls,ndim,impl", IMPL_CASES, ids=IMPL_IDS)
def test_free_surface_true_raises(cls, ndim, impl):
    with pytest.raises(NotImplementedError, match="anisotropic"):
        _build(cls, True, ndim, impl)


@cuda
@pytest.mark.parametrize("cls,ndim,impl", IMPL_CASES, ids=IMPL_IDS)
def test_per_edge_top_also_raises(cls, ndim, impl):
    with pytest.raises(NotImplementedError, match="anisotropic"):
        _build(cls, ["top"], ndim, impl)


@cuda
@pytest.mark.parametrize("cls,ndim,impl", [c for c in IMPL_CASES if c[1] == 2],
                         ids=[i for c, i in zip(IMPL_CASES, IMPL_IDS) if c[1] == 2])
def test_topography_also_raises(cls, ndim, impl):
    """``topography=`` implies a free surface even with free_surface=False, so
    it has to hit the same guard.  It used to slip through: the guard ran on
    fs_faces before _resolve_topo_method folded the topography-implied top face
    in, and the isotropic image mirror then ran on the anisotropic solver --
    measured 29% wavefield difference against the no-topography run."""
    import numpy as np

    with pytest.raises(NotImplementedError, match="anisotropic"):
        _build(cls, False, ndim, impl, topography=np.zeros(56, dtype=np.int64))


@cuda
@pytest.mark.parametrize("cls,ndim,impl", IMPL_CASES, ids=IMPL_IDS)
def test_no_free_surface_still_constructs(cls, ndim, impl):
    _build(cls, False, ndim, impl)


@cuda
def test_isotropic_free_surface_unaffected():
    from sweep.equations import Acoustic, Elastic

    import numpy as np

    _build(Elastic, True)
    _build(Elastic, ["top", "left"])
    _build(Acoustic, True)
    _build(Elastic, False, topography=np.zeros(56, dtype=np.int64))
