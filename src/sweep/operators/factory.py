class OperatorBase:
    """ Base class for operators.
    """

    def __init__(self, backend: str = 'torch'):
        """ Initialize Laplace operator.
         Args:
             backend (str, optional): Backend to use ('torch' or 'jax'). Defaults to 'torch'.
         """
        if backend == 'jax':
            from sweep.operators.jax import laplace2d as lap2d_jax
            from sweep.operators.jax import laplace1d_sep as lap1d_jax
            from sweep.operators.jax import laplace3d_sep as lap3d_jax
            from sweep.operators.jax import gradient as gradient_jax

            self.laplace2d = lap2d_jax
            self.laplace1d_sep = lap1d_jax
            self.laplace3d_sep = lap3d_jax
            self.gradient = gradient_jax

        elif backend == 'torch':
            from sweep.operators.torch import laplace2d as lap2d_torch
            from sweep.operators.torch import laplace1d_sep as lap1d_torch
            from sweep.operators.torch import gradient as gradient_torch
            
            self.gradient = gradient_torch
            self.laplace2d = lap2d_torch
            self.laplace1d_sep = lap1d_torch



