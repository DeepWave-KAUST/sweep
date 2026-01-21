from .base import SecondOrderEquation

def step(u_now, u_pre, psix, psiz, zetax, zetaz, 
         su_now, su_pre, spsix, spsiz, szetax, szetaz, 
         vp, ref, dt, h, b, 
         lap_ux, lap_uz, lap_sux, lap_suz,
         pml,
         grad_op,
         ):
    
    az, bz, dbzdz, ax, bx, dbxdx = pml

    # Calcualte gradients based on 2nd order central finite difference
    dudz = grad_op(u_now, h, -2)
    dudx = grad_op(u_now, h, -1)
    dsudz = grad_op(su_now, h, -2)
    dsudx = grad_op(su_now, h, -1)

    # Background wavefield
    w_sum = 0.
    # Z direction
    tmpz = ((1+bz)*lap_uz + dbzdz * dudz) + grad_op(az*psiz, h, -2)
    w_sum += (1+bz) * tmpz + az * zetaz

    psiyn = bz * dudz + az * psiz
    zetaz = bz * tmpz + az * zetaz

    # X direction
    tmpx = ((1+bx)*lap_ux + dbxdx * dudx) + grad_op(ax*psix, h, -1)
    w_sum += (1+bx) * tmpx + ax * zetax
    psixn = bx * dudx + ax * psix
    zetax = bx * tmpx + ax * zetax

    u_next = 2 * u_now - u_pre + vp**2 * dt**2 * w_sum

    # Scatter wavefield
    w_sum_s = 0.
    # Z direction
    tmpsz = ((1+bz)*lap_suz + dbzdz * dsudz) + grad_op(az*spsiz, h, -2)
    w_sum_s += (1+bz) * tmpsz + az * szetaz
    spsiyn = bz * dsudz + az * spsiz
    szetaz = bz * tmpsz + az * szetaz   
    # X direction
    tmpx_s = ((1+bx)*lap_sux + dbxdx * dsudx) + grad_op(ax*spsix, h, -1)
    w_sum_s += (1+bx) * tmpx_s + ax * szetax
    spsixn = bx * dsudx + ax * spsix
    szetax = bx * tmpx_s + ax * szetax
    su_next = 2 * su_now - su_pre + vp**2 * dt**2 * w_sum_s + ref * vp**2 * dt**2 * w_sum

    # # background wavefield
    # vp2_nabla_p0 = vp**2*lap_u_now*dt**2
    # u_next = 2 * u_now - u_pre + vp2_nabla_p0
    # u_next = a * u_next + (1 - a) * u_now
    
    # # scatter wavefield
    # vp2_nabla_sh0 = vp**2*lap_su_now*dt**2
    # su_next = 2 * su_now - su_pre + vp2_nabla_sh0 + ref*vp2_nabla_p0
    # su_next = a * su_next + (1 - a) * su_now

    return u_next, u_now, psixn, psiyn, zetax, zetaz, \
            su_next, su_now, spsixn, spsiyn, szetax, szetaz

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
        return ['h1', 'h2', 'psix', 'psiz', 'zetax', 'zetaz',
                'sh1', 'sh2', 'spsix', 'spsiz', 'szetax', 'szetaz']
    
    def func(self, *args, **kwargs):
        dh = args[15]
        lap_uz, lap_ux = self.laplace1d_sep(args[0], self.kernel, dh, dh)
        lap_suz, lap_sux = self.laplace1d_sep(args[6], self.kernel, dh, dh)
        return step(*args, lap_ux, lap_uz, lap_sux, lap_suz, self.b, self.gradient)
    
