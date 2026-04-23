from .base import FirstOrderEquation
from .fields import ModelSpec

def step(p, vx, vz, txx, tzz, txz,
         vp, vs, rho, 
         dt, h, b, pd):

    lame_lambda = rho*(vp**2-2*vs**2)
    lame_mu = rho*vs**2
    c = 0.5*dt*b

    vx_x = pd.x_backward(vx)
    vz_z = pd.z_forward(vz)
    vx_z = pd.z_backward(vx)
    vz_x = pd.x_forward(vz)

    y_txx = (1+c)**-1*(dt*lame_mu*h**(-1)*(vx_x-vz_z)+(1-c)*txx)
    y_tzz = (1+c)**-1*(dt*lame_mu*h**(-1)*(vz_z-vx_x)+(1-c)*tzz)
    y_txz = (1+c)**-1*(dt*lame_mu*h**(-1)*(vz_x+vx_z)+(1-c)*txz)
    y_p = (1+c)**-1*(-dt*(lame_lambda+lame_mu)*h**(-1)*(vx_x+vz_z)+(1-c)*p)

    txx_x = pd.x_forward(y_txx)
    txz_z = pd.z_forward(y_txz)
    tzz_z = pd.z_backward(y_tzz)
    txz_x = pd.x_backward(y_txz)

    p_x = pd.x_forward(y_p)
    p_z = pd.z_backward(y_p)

    y_vx = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txx_x+txz_z-p_x)+(1-c)*vx)
    y_vz = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txz_x+tzz_z-p_z)+(1-c)*vz)


    return y_p, y_vx, y_vz, y_txx, y_tzz, y_txz

class AEC(FirstOrderEquation):
    """
    Parameter order:vp, vs, rho.

    Wavefields: (p, vx, vz, txx, tzz, txz)

    Reference: Yu Pengfei, 10.1190/geo2015-0535.1
    """
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("p_velocity",), description="Coupled acoustic-elastic P-wave velocity model.", unit="m/s"),
        ModelSpec("vs", aliases=("s_velocity",), description="Coupled acoustic-elastic S-wave velocity model.", unit="m/s"),
        ModelSpec("rho", aliases=("density",), description="Density model.", unit="kg/m^3"),
    )
    def __init__(self, spatial_order=4, device='cpu', backend = 'torch'):
        super().__init__(spatial_order, device, backend)
        
    @property
    def models(self):
        return [spec.name for spec in self.MODEL_SPECS]
    
    @property
    def wavefields(self):
        return ['p', 'vx', 'vz', 'txx', 'tzz', 'txz']
    
    def func(self, *args, **kwargs):
        return step(*args, pd=self.pd, **kwargs)
