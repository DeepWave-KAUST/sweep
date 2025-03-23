from .acoustic import Acoustic
from .elastic import Elastic
from .elasticz import ElasticZ
from .acoustic1st import Acoustic as Acoustic1st
from .acoustic_lsrtm import AcousticLSRTM

__all__ = ['Acoustic', 'Elastic', 'Acoustic1st', 'AcousticLSRTM', 'ElasticZ']
supported_equations = { 'acoustic': {'fwi': [Acoustic, Acoustic1st],
                                     'lsrtm': [AcousticLSRTM]},
                        'elastic': {'fwi': [Elastic, ElasticZ]}}