from .acoustic import Acoustic
from .elastic import Elastic
from .elasticz import ElasticZ
from .elasticP import ElasticP
from .acoustic1st import Acoustic as Acoustic1st
from .acoustic_lsrtm import AcousticLSRTM
from .aec import AcousticElasticCoupled as AEC

__all__ = ['Acoustic', 'Elastic', 'Acoustic1st', 'AcousticLSRTM', 'ElasticZ', 'ElasticP']
supported_equations = { 'acoustic': {'fwi': [Acoustic, Acoustic1st],
                                     'lsrtm': [AcousticLSRTM]},
                        'elastic': {'fwi': [Elastic, ElasticZ, ElasticP],
                                    'lsrtm': []}
                      }