from .acoustic import Acoustic
from .elastic import Elastic
from .elasticz import ElasticZ
from .elasticP import ElasticP
from .acoustic1st import Acoustic as Acoustic1st
from .acoustic_lsrtm import AcousticLSRTM
from .aec import AcousticElasticCoupled as AEC
from .aec_lsrtm import AcousticElasticCoupledLSRTM as AECLSRTM
from .elastic_lsrtm import ElasticLSRTM
from .qP_tti import AcousticTTI
from .qP_vti import AcousticVTI
from .qP_tariq import AcousticTariq
from .visco_acoustic import ViscoAcoustic
from .acoustic_vrr import Acoustic as AcousticVRR
from .acoustic_vrz import Acoustic as AcousticVRZ

__all__ = ['Acoustic', 'Elastic', 'Acoustic1st', 'AcousticLSRTM', 'ElasticZ', 'ElasticP', 'AEC', 'ElasticLSRTM']
supported_equations = { 'acoustic': {'fwi': [Acoustic, Acoustic1st],
                                     'lsrtm': [AcousticLSRTM]},
                        'elastic': {'fwi': [Elastic, ElasticZ, ElasticP],
                                    'lsrtm': []}
                      }