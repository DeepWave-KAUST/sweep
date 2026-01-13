from .base import SecondOrderEquation
from .operator import gradientO2

def step_cpml(
        u_now, u_pre, psix, psiy, psiz, zetax, zetay, zetaz, 
        vp, dt, h, b, 
        lap_x, lap_y, lap_z, 
        dudx, dudy, dudz
        ):

    az, bz, dbzdz, ay, by, dbydy, ax, bx, dbxdx = b

    w_sum = 0.

    # Z direction
    tmpz = ((1+bz)*lap_z + dbzdz * dudz) + gradientO2(az*psiz, h, -3)
    w_sum += (1+bz) * tmpz + az * zetaz

    psizn = bz * dudz + az * psiz
    zetaz = bz * tmpz + az * zetaz

    # Y direction
    tmpy = ((1+by)*lap_y + dbydy * dudy) + gradientO2(ay*psiy, h, -2)
    w_sum += (1+by) * tmpy + ay * zetay
    psiyn = by * dudy + ay * psiy
    zetay = by * tmpy + ay * zetay

    # X direction
    tmpx = ((1+bx)*lap_x + dbxdx * dudx) + gradientO2(ax*psix, h, -1)
    w_sum += (1+bx) * tmpx + ax * zetax
    psixn = bx * dudx + ax * psix
    zetax = bx * tmpx + ax * zetax

    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * w_sum

    return u_next, u_now, psixn, psiyn, psizn, zetax, zetay, zetaz

class Acoustic3D(SecondOrderEquation):

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=2):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        super().__init__(spatial_order, device, backend, dim=dim)
        self._wavefields = []
        super().init_laplace(ltype='3dsep', backend=backend)

    @property
    def models(self):
        return ['vp']
    
    @property
    def wavefields(self):
        return ['h1', 'h2', 'psix', 'psiy', 'psiz', 'zetax', 'zetay', 'zetaz']

    def func(self, *args, **kwargs):
        print(
            'kernel.shape:', self.kernel.shape,
        )
        dh = args[10]
        dudz = gradientO2(args[0], dh, -3)
        dudy = gradientO2(args[0], dh, -2)
        dudx = gradientO2(args[0], dh, -1)
        lap_z, lap_y, lap_x = self.laplace(args[0], self.kernel, dh, dh, dh)
        return step_cpml(*args, lap_x, lap_y, lap_z, dudx, dudy, dudz)