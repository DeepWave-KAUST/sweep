"""Spatial-derivative operators shared by the equation classes.

Three operator families are exposed:

- :class:`OperatorBase` — backend-dispatch bundle providing ``laplace2d``,
  ``separable_d2_2d``, ``separable_d2_3d`` and ``gradient``. Inherited by
  :class:`sweep.equations.base.SecondOrderEquation`, so 2-D / 3-D second-order
  acoustic equations get these as ``self.separable_d2_2d(...)`` etc. for free.
- :class:`StaggeredDerivative` — staggered-grid first-order derivatives with
  ``x_forward / x_backward / z_forward / z_backward`` (and ``y_*`` in 3-D).
  Instantiated automatically by
  :class:`sweep.equations.base.FirstOrderEquation` as ``self.pd``.
- :class:`RSGDerivative` — rotated-staggered-grid 2-D first-order derivatives
  with ``lx_fwd / lx_bwd / lz_fwd / lz_bwd``. Used by elastic TTI equations.
"""

from .factory import OperatorBase
from .general import StaggeredDerivative
from .rsg import RSGDerivative

__all__ = ["OperatorBase", "StaggeredDerivative", "RSGDerivative"]
