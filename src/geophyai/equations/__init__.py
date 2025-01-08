from .acoustic import Acoustic
from .elastic import Elastic

__all__ = ['Acoustic', 'Elastic']
supported_equations = { 'acoustic': Acoustic,
                        'elastic': Elastic }