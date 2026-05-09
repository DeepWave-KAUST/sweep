from .base import SecondOrderEquation
from .fields import ModelSpec

def step(u_now, u_pre, vp, rx, rz, dt, h, b, lap_u_now, dpdx, dpdz, dvpdx, dvpdz):
    a = 1 / (1 + b * dt)
    u_next = 2 * u_now - u_pre + vp**2*dt**2*lap_u_now + vp*(dvpdx*dpdx + dvpdz*dpdz)*dt**2 - 2*vp**2*(rx*dpdx + rz*dpdz)*dt**2
    u_next = a * u_next + (1 - a) * u_now
    return u_next, u_now

class AcousticVRR(SecondOrderEquation):
    """
    Parameter order: vp, rx, rz

    Wavefields: (h1, h2)

    Reference: 10.3997/2214-4609.202010332
    """
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("velocity",), description="Acoustic velocity model.", unit="m/s"),
        ModelSpec("rx", description="Auxiliary horizontal parameter used by the VRR formulation."),
        ModelSpec("rz", description="Auxiliary vertical parameter used by the VRR formulation."),
    )
    def __init__(self, spatial_order=4, device='cpu', backend = 'torch', dim=2):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        super().__init__(spatial_order, device, backend, other_kernels=True)
        super().init_laplace(ltype='1dsep', backend=backend)

    @property
    def models(self):
        return [spec.name for spec in self.MODEL_SPECS]
    
    @property
    def wavefields(self):
        return ['h1', 'h2']
    
    def func(self, wavefields, models, dt, h, b, **kwargs):
        u_now = wavefields[0]
        vp = models[0]
        hz, hx = self._spacings_2d(h)
        lap_u_now_z, lap_u_now_x = self.laplace1d_sep(u_now, self.laplace_kernels, hz, hx)
        lap_u_now = lap_u_now_x + lap_u_now_z
        dvpdx = self.gradient(vp, h, axis=-1)
        dvpdz = self.gradient(vp, h, axis=-2)
        dpdx = self.gradient(u_now, h, axis=-1)
        dpdz = self.gradient(u_now, h, axis=-2)
        return step(*wavefields, *models, dt, h, b, lap_u_now, dpdx, dpdz, dvpdx, dvpdz)
