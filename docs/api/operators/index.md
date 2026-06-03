# Operators

This section documents the three spatial-derivative operator classes in
`sweep.operators`. For the conceptual / how-to guide and decision tree see
[Operators](../../user-guide/operators.md).

```python
from sweep.operators import (
    LaplaceGradientOps,      # Laplacian + gradient bundle
    StaggeredDerivative,     # 1st-order staggered FD
    RSGDerivative,           # rotated staggered grid (2-D)
)
```

You usually do **not** instantiate these directly:

- `LaplaceGradientOps` is mixed into `SecondOrderEquation` — accessed as
  `self.laplacian_2d(...)`, `self.separable_d2_2d(...)`, `self.gradient(...)`.
- `StaggeredDerivative` is built by `FirstOrderEquation.__init__` and stored
  on `self.pd`.
- `RSGDerivative` is constructed explicitly by `ElasticTTI` / `ElasticTTISG`
  as `self.rsg`.

## LaplaceGradientOps

::: sweep.operators.LaplaceGradientOps

## StaggeredDerivative

::: sweep.operators.StaggeredDerivative

## RSGDerivative

::: sweep.operators.RSGDerivative
