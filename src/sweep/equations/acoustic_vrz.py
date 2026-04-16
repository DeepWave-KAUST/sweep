from .base import SecondOrderEquation

def step_cpml(u_now, u_pre, psix, psiz, zetax, zetaz, 
              vp, z, dt, h, b, 
              lap_x, lap_z,
              pml, grad_op
              ):

    az, bz, dbzdz, ax, bx, dbxdx = pml

    w_sum = 0.

    # Calcualte gradients based on 2nd order central finite difference
    dvpdx = grad_op(vp, h, axis=-1)
    dvpdz = grad_op(vp, h, axis=-2)
    dpdx = grad_op(u_now, h, axis=-1)
    dpdz = grad_op(u_now, h, axis=-2)
    z1_x = grad_op(1/z, h, axis=-1)
    z1_z = grad_op(1/z, h, axis=-2)

    # Z direction
    tmpz = ((1+bz)*lap_z + dbzdz * dpdz) + grad_op(az*psiz, h, axis=-2)
    w_sum += (1+bz) * tmpz + az * zetaz

    psiyn = bz * dpdz + az * psiz
    zetaz = bz * tmpz + az * zetaz

    # X direction
    tmpx = ((1+bx)*lap_x + dbxdx * dpdx) + grad_op(ax*psix, h, axis=-1)
    w_sum += (1+bx) * tmpx + ax * zetax

    psixn = bx * dpdx + ax * psix
    zetax = bx * tmpx + ax * zetax

    dpdx_cpml = dpdx + psixn
    dpdz_cpml = dpdz + psiyn

    Ax = vp * dvpdx + vp**2 * z * z1_x
    Az = vp * dvpdz + vp**2 * z * z1_z

    grad_term = Ax * dpdx_cpml + Az * dpdz_cpml

    u_next = 2 * u_now - u_pre + dt**2 * ( vp**2 * w_sum + grad_term )

    return u_next, u_now, psixn, psiyn, zetax, zetaz


class AcousticVRZ(SecondOrderEquation):
    """
    Parameter order: vp, rx, rz

    Wavefields: (h1, h2)

    Reference: 10.3997/2214-4609.202010332
    """
    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=2):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        super().__init__(spatial_order, device, backend, other_kernels=True)
        super().init_laplace(ltype='1dsep', backend=backend)

    @property
    def models(self):
        return ['vp', 'z']

    @property
    def wavefields(self):
        return  ['h1', 'h2', 'psix', 'psiz', 'zetax', 'zetaz']

    def func(self, *args, **kwargs):
        dh = args[9]
        lap_u_now_z, lap_u_now_x = self.laplace1d_sep(args[0], self.kernel, dh, dh)
        return step_cpml(*args, lap_u_now_x, lap_u_now_z, self.b, self.gradient)

    def _C(self):
        from sweep._C import (
            acoustic_vrz2d_forward,
            acoustic_vrz2d_backward,
            acoustic_vrz2d_backward_bs,
            acoustic_vrz2d_backward_ckpt,
            acoustic_vrz2d_backward_recursive_ckpt,
        )

        return (
            acoustic_vrz2d_forward,
            acoustic_vrz2d_backward,
            acoustic_vrz2d_backward_bs,
            acoustic_vrz2d_backward_ckpt,
            acoustic_vrz2d_backward_recursive_ckpt,
        )

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
