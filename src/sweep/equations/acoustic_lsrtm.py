from .base import SecondOrderEquation

def step(u_now, u_pre, su_now, su_pre, vp, ref, dt, h, b, lap_u_now, lap_su_now):
    
    a = 1 / (1 + b * dt)

    # background wavefield
    vp2_nabla_p0 = vp**2*lap_u_now*dt**2
    u_next = 2 * u_now - u_pre + vp2_nabla_p0
    u_next = a * u_next + (1 - a) * u_now
    
    # scatter wavefield
    vp2_nabla_sh0 = vp**2*lap_su_now*dt**2
    su_next = 2 * su_now - su_pre + vp2_nabla_sh0 + ref*vp2_nabla_p0
    su_next = a * su_next + (1 - a) * su_now

    return u_next, u_now, su_next, su_now

class AcousticLSRTM(SecondOrderEquation):

    def __init__(self, spatial_order=4, device='cpu', backend='torch'):
        """Acoustic wave equation solver.

        Args:
            spatial_order (int, optional): The order of the taylor expansion(Must be even). Defaults to 4.
        """
        super().__init__(spatial_order, device, backend)
        super().init_laplace(ltype='1dsep')
    
    @property
    def models(self):
        return ['vp', 'mp']
    
    @property
    def wavefields(self):
        return ['h1', 'h2', 'sh1', 'sh2']
    
    def func(self, *args, **kwargs):
        lap_u_now = self.laplace(args[0], self.kernel, args[7], args[7])
        lap_su_now = self.laplace(args[2], self.kernel, args[7], args[7])
        return step(*args, lap_u_now, lap_su_now)
    
