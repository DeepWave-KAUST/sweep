# # AUTO-GENERATED FILE - DO NOT EDIT MANUALLY

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
from .elastic3d import Elastic as Elastic3D

# from .acoustic_vrr import AcousticVRR
try:
    from .qP_tti import AcousticTTI
except ModuleNotFoundError:
    AcousticTTI = None
# from .aec import AEC
# from .visco_acoustic import ViscoAcoustic
from .qP_vti import AcousticVTI
# from .elasticP import ElasticP
from .acoustic import Acoustic
from .qP_tariq import AcousticTariq
from .acoustic3d import Acoustic3D


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
