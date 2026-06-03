class LaplaceGradientOps:
    """Backend-dispatched second-derivative and gradient operator bundle.

    On construction this class imports five backend-specific functions
    from either :mod:`sweep.operators.torch` or :mod:`sweep.operators.jax`
    and binds them as instance attributes:

    - :func:`separable_d2_2d` / :func:`separable_d2_3d` — return per-axis
      second derivatives ``(d2z, d2x[, d2y])`` for anisotropic equations.
    - :func:`laplacian_2d` / :func:`laplacian_3d` — return the scalar
      isotropic Laplacian ``∇²u`` (= sum of the per-axis components).
    - :func:`gradient` — first-order partial derivative along one axis.

    Mixed into :class:`sweep.equations.base.SecondOrderEquation` so every
    2-D / 3-D second-order acoustic equation can call
    ``self.laplacian_2d(...)`` or ``self.separable_d2_2d(...)`` etc.
    without an explicit operator object.
    """

    def __init__(self, backend: str = 'torch'):
        """Bind second-derivative + gradient operators for the chosen backend.

        Args:
            backend: ``'torch'`` (default) or ``'jax'``. The corresponding
                ``sweep.operators.<backend>`` module is imported lazily so
                installations that lack one backend stay importable.
        """
        if backend == 'jax':
            from sweep.operators.jax import (
                separable_d2_2d, separable_d2_3d,
                laplacian_2d, laplacian_3d, gradient,
            )
        elif backend == 'torch':
            from sweep.operators.torch import (
                separable_d2_2d, separable_d2_3d,
                laplacian_2d, laplacian_3d, gradient,
            )
        else:
            raise ValueError(
                f"Unsupported backend {backend!r}; expected 'torch' or 'jax'."
            )

        self.separable_d2_2d = separable_d2_2d
        self.separable_d2_3d = separable_d2_3d
        self.laplacian_2d = laplacian_2d
        self.laplacian_3d = laplacian_3d
        self.gradient = gradient
