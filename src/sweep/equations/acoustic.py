from .base import SecondOrderEquation

def step_cpml(
        u_now, u_pre, psix, psiz, zetax, zetaz, 
        vp, dt, h, b, 
        lap_x, lap_z, 
        pml,
        grad_op,
        grad_kernels=None,
        ):

    az, bz, dbzdz, ax, bx, dbxdx = pml

    w_sum = 0.
    
    # Use fixed stencil convolutions when available; this is much cheaper than torch.gradient.
    dudz = grad_op(u_now, h, -2, kernels=grad_kernels)
    dudx = grad_op(u_now, h, -1, kernels=grad_kernels)
    
    # Z direction
    tmpz = ((1+bz)*lap_z + dbzdz * dudz) + grad_op(az*psiz, h, -2, kernels=grad_kernels)
    w_sum += (1+bz) * tmpz + az * zetaz

    psiyn = bz * dudz + az * psiz
    zetaz = bz * tmpz + az * zetaz

    # X direction
    tmpx = ((1+bx)*lap_x + dbxdx * dudx) + grad_op(ax*psix, h, -1, kernels=grad_kernels)
    w_sum += (1+bx) * tmpx + ax * zetax
    psixn = bx * dudx + ax * psix
    zetax = bx * tmpx + ax * zetax

    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * w_sum

    return u_next, u_now, psixn, psiyn, zetax, zetaz

class Acoustic(SecondOrderEquation):

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=2):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        use_fast_grad_kernels = backend == 'torch' and 'cuda' in str(device)
        super().__init__(spatial_order, device, backend, dim=dim, other_kernels=use_fast_grad_kernels)
        super().init_laplace(ltype='1dsep', backend=backend)
        self.grad_kernels = None
        if use_fast_grad_kernels:
            self.grad_kernels = {-2: self.gkernel_z, -1: self.gkernel_x}

    @property
    def models(self):
        return ['vp']
    
    @property
    def wavefields(self):
        return ['h1', 'h2', 'psix', 'psiz', 'zetax', 'zetaz']

    def func(self, *args, **kwargs):
        dh = args[8]
        lap_u_now_z, lap_u_now_x = self.laplace1d_sep(args[0], self.laplace_kernels, dh, dh)
        return step_cpml(*args, lap_u_now_x, lap_u_now_z, self.b, self.gradient, self.grad_kernels)

    def _C(self, ):
        # CUDA IMPLEMENTATION
        import torch
        from sweep._C import (
            acoustic2d_forward,
            acoustic2d_backward,
            acoustic2d_backward_bs,
            acoustic2d_backward_ckpt,
            acoustic2d_backward_recursive_ckpt,
        )
        return (
            acoustic2d_forward,
            acoustic2d_backward,
            acoustic2d_backward_bs,
            acoustic2d_backward_ckpt,
            acoustic2d_backward_recursive_ckpt,
        )

    def _C_rtm(self):
        import torch
        from sweep._C import acoustic2d_rtm

        return acoustic2d_rtm

    @property
    def base_nvar(self):
        return 3

    @property
    def pml_nvar(self):
        return 4

    @property
    def last_two_nvar(self):
        return 2

    @property
    def last_two_storage_nvar(self):
        return 1

    @property
    def checkpoint_nvar(self):
        return 6
