class LaplaceGradientOps:
    """Backend-dispatched Laplacian and gradient operator bundle.

    On construction this class imports four backend-specific functions
    (``laplace2d``, ``separable_d2_2d``, ``separable_d2_3d``, ``gradient``)
    from either :mod:`sweep.operators.torch` or :mod:`sweep.operators.jax`
    and binds them as instance attributes. Mixed into
    :class:`sweep.equations.base.SecondOrderEquation` so every 2-D / 3-D
    second-order acoustic equation can call ``self.separable_d2_2d(...)``
    etc. without an explicit operator object.
    """

    def __init__(self, backend: str = 'torch'):
        """Bind Laplacian + gradient operators for the chosen backend.

        Args:
            backend: ``'torch'`` (default) or ``'jax'``. The corresponding
                ``sweep.operators.<backend>`` module is imported lazily so
                installations that lack one backend stay importable.
        """
        if backend == 'jax':
            from sweep.operators.jax import laplace2d as lap2d_jax
            from sweep.operators.jax import separable_d2_2d as lap1d_jax
            from sweep.operators.jax import separable_d2_3d as lap3d_jax
            from sweep.operators.jax import gradient as gradient_jax

            self.laplace2d = lap2d_jax
            self.separable_d2_2d = lap1d_jax
            self.separable_d2_3d = lap3d_jax
            self.gradient = gradient_jax

        elif backend == 'torch':
            from sweep.operators.torch import laplace2d as lap2d_torch
            from sweep.operators.torch import separable_d2_2d as lap1d_torch
            from sweep.operators.torch import gradient as gradient_torch
            from sweep.operators.torch import separable_d2_3d as lap3d_torch

            self.gradient = gradient_torch
            self.laplace2d = lap2d_torch
            self.separable_d2_2d = lap1d_torch
            self.separable_d2_3d = lap3d_torch  # 3-D = three 1-D convs applied per axis
