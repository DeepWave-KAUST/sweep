from .base import FirstOrderEquation

def step_cpml(p, vx, vz, phix, phiz, psix, psiz, vp, rho, dt, h, b, pd, pml=None):

    az, bz, azh, bzh, ax, bx, axh, bxh = pml
    # az, bz, dbzdz, ax, bx, dbxdx = b

    # Update vx and vz
    p_x = pd.x_backward(p)
    p_z = pd.z_backward(p)
 
    psi_z = azh * psiz + bzh * p_z
    p_z += psi_z
    vz -= dt / (rho * h) * p_z

    psi_x = axh * psix + bxh * p_x
    p_x += psi_x
    vx -= dt / (rho * h) * p_x

    # Update p
    vx_x = pd.x_forward(vx)
    vz_z = pd.z_forward(vz)
    div_v = 0.

    phiz = az * phiz + bz * vz_z
    vz_z += phiz
    div_v += vz_z

    phix = ax * phix + bx * vx_x
    vx_x += phix
    div_v += vx_x

    p -= vp**2 * rho * dt / h * div_v

    return p, vx, vz, phix, phiz, psi_x, psi_z

def step_spml(px, pz, vx, vz, vp, rho, dt, h, b, pd, pml=None):

    bx, bz = pml
    p = px + pz

    p_x = pd.x_backward(p)
    p_z = pd.z_backward(p)

    vx = ((1.-0.5*dt*bx)*vx+dt/(rho*h)*p_x)/(1.+0.5*dt*bx)
    vz = ((1.-0.5*dt*bz)*vz+dt/(rho*h)*p_z)/(1.+0.5*dt*bz)

    vx_x = pd.x_forward(vx)
    vz_z = pd.z_forward(vz)

    px = ((1.-0.5*dt*bx)*px+vp**2 * rho * dt / h * vx_x)/(1.+0.5*dt*bx)
    pz = ((1.-0.5*dt*bz)*pz+vp**2 * rho * dt / h * vz_z)/(1.+0.5*dt*bz)

    return px, pz, vx, vz


class Acoustic1st(FirstOrderEquation):
    """
    Parameter order: vp, rho

    Wavefields: (p, vx, vz).

    References: 10.1190/GEO2011-0345.1
    """
    def __init__(self, spatial_order=4, device='cpu', backend='jax', **kwargs):
        super().__init__(spatial_order, device, backend)

    def setup_pml(self, pml_type):
        self.wavefields = {
            'cpmls': self.wavefields_cpml(),
            'spml': self.wavefields_spml()
            }[pml_type]
        self.step = {
            'cpmls': step_cpml,
            'spml': step_spml
            }[pml_type]
    
    @property
    def models(self):
        return ['vp', 'rho']
    
    def wavefields_cpml(self):
        return ['p', 'vx', 'vz', 'phix', 'phiz', 'psix', 'psiz']

    def wavefields_spml(self):
        return ['px', 'pz', 'vx', 'vz']
    
    @property
    def supported_pml(self):
        return ['cpmls', 'spml']

    def func(self, *args, **kwargs):
        return self.step(*args, pd=self.pd, pml=self.b, **kwargs)