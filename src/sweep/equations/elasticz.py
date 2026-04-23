from .base import FirstOrderEquation
from .fields import ModelSpec

def step(vx, vz, txx, tzz, txz,
         vpz, vsz, rho, 
         dt, h, b, pd=None):
    
    vp = vpz/rho
    vs = vsz/rho
    
    lame_lambda = rho*(vp**2-2*vs**2)
    lame_mu = rho*vs**2
    c = 0.5*dt*b

    vx_x = pd.x_forward(vx)
    vz_z = pd.z_backward(vz)
    vx_z = pd.z_forward(vx)
    vz_x = pd.x_backward(vz)

    y_txx  = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vx_x+lame_lambda*vz_z)+(1-c)*txx)
    y_tzz  = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vz_z+lame_lambda*vx_x)+(1-c)*tzz)
    y_txz = (1+c)**-1*(dt*lame_mu*h**(-1)*(vz_x+vx_z)+(1-c)*txz)

    txx_x = pd.x_backward(y_txx)
    txz_z = pd.z_backward(y_txz)
    tzz_z = pd.z_forward(y_tzz)
    txz_x = pd.x_forward(y_txz)

    y_vx = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txx_x+txz_z)+(1-c)*vx)
    y_vz = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txz_x+tzz_z)+(1-c)*vz)

    return y_vx, y_vz, y_txx, y_tzz, y_txz


class ElasticZ(FirstOrderEquation):
    MODEL_SPECS = (
        ModelSpec("vpz", aliases=("vp",), description="Depth-dependent P-wave velocity parameter.", unit="m/s"),
        ModelSpec("vsz", aliases=("vs",), description="Depth-dependent S-wave velocity parameter.", unit="m/s"),
        ModelSpec("rho", aliases=("density",), description="Density model.", unit="kg/m^3"),
    )

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch'):
        super().__init__(spatial_order, device, backend)

    @property
    def models(self):
        return [spec.name for spec in self.MODEL_SPECS]
    
    @property
    def wavefields(self):
        return ['vx', 'vz', 'txx', 'tzz', 'txz']
    
    def func(self, *args, **kwargs):
        return step(*args, pd=self.pd, **kwargs)
