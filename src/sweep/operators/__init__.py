"""Spatial-derivative operators shared by the equation classes.

Four operator families are exposed:

- :class:`LaplaceGradientOps` — backend-dispatch bundle providing
  ``separable_d2_2d`` / ``separable_d2_3d`` (per-axis second-derivative
  components for anisotropic equations), ``laplacian_2d`` /
  ``laplacian_3d`` (isotropic ``∇²u`` shortcut = sum of components),
  and ``gradient``. Inherited by
  :class:`sweep.equations.base.SecondOrderEquation`, so 2-D / 3-D
  second-order acoustic equations get these as ``self.laplacian_2d(...)``
  / ``self.separable_d2_2d(...)`` etc. for free.
- :class:`StaggeredDerivative` — staggered-grid first-order derivatives with
  ``x_forward / x_backward / z_forward / z_backward`` (and ``y_*`` in 3-D).
  Instantiated automatically by
  :class:`sweep.equations.base.FirstOrderEquation` as ``self.pd``.
- :class:`RSGDerivative` — rotated-staggered-grid 2-D first-order derivatives
  with ``lx_fwd / lx_bwd / lz_fwd / lz_bwd``. Used by elastic TTI equations.
- :class:`Derivative` — the general form of the first two: the derivative of
  any order along one axis, any even accuracy, centred or (odd orders)
  staggered, from Fornberg weights.
"""

from .factory import LaplaceGradientOps
from .general import StaggeredDerivative
from .rsg import RSGDerivative
from .derivative import Derivative

__all__ = ["LaplaceGradientOps", "StaggeredDerivative", "RSGDerivative", "Derivative"]
