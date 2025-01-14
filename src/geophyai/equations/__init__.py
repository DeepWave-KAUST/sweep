from .acoustic import Acoustic
from .elastic import Elastic
from .acoustic1st import Acoustic as Acoustic1st

__all__ = ['Acoustic', 'Elastic']
supported_equations = { 'acoustic': Acoustic,
                        'acoustic1st': Acoustic1st,
                        'elastic': Elastic }