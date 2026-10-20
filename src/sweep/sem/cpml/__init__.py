"""CPML absorbing-boundary profiles for the SEM equations.

``acoustic``   scalar acoustic (Quad4).
``acoustic3d`` scalar acoustic (Hex8, 3-D).
``elastic``    2nd-order displacement elastic (Quad9).
``elastic_vs`` 1st-order velocity-stress elastic (Quad9).
"""

from .acoustic import CPMLProfile, build_cpml
from .acoustic3d import CPML3DProfile, build_cpml3d
from .elastic import ElasticCPMLProfile, build_elastic_cpml
from .elastic_vs import SimpleElasticCPMLProfile, build_simple_elastic_cpml
from .elastic_vs3d import SimpleElasticCPML3DProfile, build_simple_elastic_cpml_3d

__all__ = [
    "CPMLProfile",
    "build_cpml",
    "CPML3DProfile",
    "build_cpml3d",
    "ElasticCPMLProfile",
    "build_elastic_cpml",
    "SimpleElasticCPMLProfile",
    "build_simple_elastic_cpml",
    "SimpleElasticCPML3DProfile",
    "build_simple_elastic_cpml_3d",
]
