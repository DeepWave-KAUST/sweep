from .base import WaveEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from .acoustic_lsrtm import AcousticLSRTM
from .acoustic_lsrtm3d import AcousticLSRTM3D
from .acoustic1st import Acoustic1st
# from .elasticz import ElasticZ
from .acoustic_vrz import AcousticVRZ, AcousticVRZ3D
# from .aec_lsrtm import AECLSRTM
# from .elastic_lsrtm import ElasticLSRTM
from .elastic import Elastic
from .elastic_tti import ElasticTTI
from .elastic_tti_sg import ElasticTTISG
from .elastic3d import Elastic as Elastic3D
# DAS family lives in `das.py` which imports torch at module top. Guard the
# import so jax-only environments (no torch installed) can still
# ``import sweep.equations`` and use jax-only equations / propagators.
# Mirrors the ``qP_tti`` pattern below.
try:
    from .das import (
        DAS,
        DASElastic,
        DASElastic3D,
        DASModeler,    # back-compat alias of `DAS`
        DASMu,
        DASMu3D,
        DASZhao,
        DASZhao3D,
        gauge_average,
        helical_das_response,
    )
except ModuleNotFoundError as _das_import_err:
    if _das_import_err.name != "torch":
        raise
    DAS = DASElastic = DASElastic3D = DASModeler = None
    DASMu = DASMu3D = DASZhao = DASZhao3D = None
    gauge_average = helical_das_response = None

from .acoustic_vrr import AcousticVRR
try:
    from .qP_tti import AcousticTTI
except ModuleNotFoundError:
    AcousticTTI = None
# from .aec import AEC
from .visco_acoustic import ViscoAcoustic
from .qP_vti import AcousticVTI
# from .elasticP import ElasticP
from .acoustic import Acoustic
from .qP_tariq import AcousticTariq
from .acoustic3d import Acoustic3D
# TODO: re-run any codegen script if one exists for this module.
from .acoustic_vti_1st import AcousticVTI1st, AcousticVTI1st3D
from .acoustic_aniso import AcousticAniso
from .acoustic_curvilinear import AcousticCurvilinear
from .elastic_curvilinear import ElasticCurvilinear
from .elastic_apm import ElasticAPM
from .elastic_vrr import (
    ElasticVRR,
    compute_vector_reflectivity,
)


# ---------------------------------------------------------------------------
# Anisotropic entry points — author-named aliases (mirror the DAS pattern
# in `das.py`, where the canonical user-facing name aliases to the author's
# class: `DAS = DASZhao`, `DASElastic = DASZhao`, etc.).
#
# Originals stay exported above for back-compat.  Prefer the author-named
# alias in new code; it makes the citation-to-class mapping obvious.
# ---------------------------------------------------------------------------

#   Liang K. et al. (2022) — 2nd-order pseudo-acoustic, 10.1190/geo2022-0292.1
AcousticVTILiang        = AcousticVTI
AcousticTTILiang        = AcousticTTI

#   Tariq Alkhalifah (2000) — qP η-formulation, 10.1190/1.1444815
AcousticVTIAlkhalifah   = AcousticTariq
AcousticTTIAlkhalifah   = AcousticTariq   # same class; eta-acoustic covers VTI & TTI

#   Duveneck et al. (2008) — 1st-order velocity-stress acoustic VTI,
#                            10.1190/1.3059320
AcousticVTIDuveneck     = AcousticVTI1st
AcousticVTIDuveneck3D   = AcousticVTI1st3D


# Per-symmetry default entry: when the user just wants "the standard" VTI
# acoustic propagator without thinking about authors.  Matches the DAS
# convention `DAS = DASZhao` (sensible default points at the most-used
# class).  Today we route 2-D VTI to Liang's 2nd-order scalar (memory-
# light, the original SWEEP entry); 3-D VTI routes to Duveneck's
# first-order because that's the only one with a 3-D class.
AcousticVTIDefault     = AcousticVTI            # 2-D, 2nd-order
AcousticVTIDefault3D   = AcousticVTI1st3D       # 3-D, 1st-order


def _equation_classes():
    return {
        name: obj
        for name, obj in globals().items()
        if isinstance(obj, type) and issubclass(obj, WaveEquation) and obj is not WaveEquation
    }


def supports_torch_binding(equation):
    """Check whether an equation class or exported equation name supports ``sweep._C``."""
    if isinstance(equation, str):
        equation_cls = _equation_classes().get(equation)
        if equation_cls is None:
            raise KeyError(f"Unknown equation '{equation}'")
        return equation_cls.supports_torch_binding()

    if isinstance(equation, type) and issubclass(equation, WaveEquation):
        return equation.supports_torch_binding()

    if isinstance(equation, WaveEquation):
        return equation.__class__.supports_torch_binding()

    raise TypeError("equation must be an equation name, equation class, or equation instance")


def torch_binding_supported_equations():
    """Return the exported equation names that support the compiled PyTorch binding."""
    return sorted(
        name for name, equation_cls in _equation_classes().items()
        if equation_cls.supports_torch_binding()
    )

# __all__ = [
#     'AcousticLSRTM',
#     'Acoustic1st',
#     'ElasticZ',
#     'AcousticVRZ',
#     'AECLSRTM',
#     'ElasticLSRTM',
#     'Elastic',
#     'AcousticVRR',
#     'AcousticTTI',
#     'AEC',
#     'ViscoAcoustic',
#     'AcousticVTI',
#     'ElasticP',
#     'Acoustic',
#     'AcousticTariq',
#     'Acoustic3D',
# ]
