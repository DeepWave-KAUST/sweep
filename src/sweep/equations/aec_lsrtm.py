from .base import FirstOrderEquation
from .fields import ModelSpec

def step(p, vx, vz, txx, tzz, txz,
         ps, vxs, vzs, txxs, tzzs, txzs,
         mp, ms,
         vp, vs, rho, 
         dt, h, b, pd):

    lame_lambda = rho*(vp**2-2*vs**2)
    lame_mu = rho*vs**2
    c = 0.5*dt*b

    # Background wavefields
    txx_x = pd.x_forward(txx)
    txz_z = pd.z_forward(txz)
    tzz_z = pd.z_backward(tzz)
    txz_x = pd.x_backward(txz)

    p_x = pd.x_forward(p)
    p_z = pd.z_backward(p)

    y_vx = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txx_x+txz_z-p_x)+(1-c)*vx)
    y_vz = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txz_x+tzz_z-p_z)+(1-c)*vz)

    vx_x = pd.x_backward(y_vx)
    vz_z = pd.z_forward(y_vz)
    vx_z = pd.z_backward(y_vx)
    vz_x = pd.x_forward(y_vz)

    y_txx = (1+c)**-1*(dt*lame_mu*h**(-1)*(vx_x-vz_z)+(1-c)*txx)
    y_tzz = (1+c)**-1*(dt*lame_mu*h**(-1)*(vz_z-vx_x)+(1-c)*tzz)
    y_txz = (1+c)**-1*(dt*lame_mu*h**(-1)*(vz_x+vx_z)+(1-c)*txz)
    y_p = (1+c)**-1*(-dt*(lame_lambda+lame_mu)*h**(-1)*(vx_x+vz_z)+(1-c)*p)

    # Perturbed wavefields
    Ip = rho*vp
    Is = rho*vs
    dlame_lambda = 2*(Ip**2*mp-2*Is**2*ms)/rho
    dlame_mu = 2*Is**2*ms/rho

    txxs_x = pd.x_forward(txxs)
    txzs_z = pd.z_forward(txzs)
    tzzs_z = pd.z_backward(tzzs)
    txzs_x = pd.x_backward(txzs)

    ps_x = pd.x_forward(ps)
    ps_z = pd.z_backward(ps)

    y_vxs = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txxs_x+txzs_z-ps_x)+(1-c)*vxs)
    y_vzs = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txzs_x+tzzs_z-ps_z)+(1-c)*vzs)

    vxs_x = pd.x_backward(y_vxs)
    vzs_z = pd.z_forward(y_vzs)
    vxs_z = pd.z_backward(y_vxs)
    vzs_x = pd.x_forward(y_vzs)

    y_txxs = (1+c)**-1*(dt*h**(-1)*(lame_mu*(vxs_x-vzs_z)+dlame_mu*(vx_x-vz_z))+(1-c)*txxs)
    y_tzzs = (1+c)**-1*(dt*h**(-1)*(lame_mu*(vzs_z-vxs_x)+dlame_mu*(vz_z-vx_x))+(1-c)*tzzs)
    y_txzs = (1+c)**-1*(dt*h**(-1)*(lame_mu*(vzs_x+vxs_z)+dlame_mu*(vz_x+vx_z))+(1-c)*txzs)
    y_ps = (1+c)**-1*(-dt*h**(-1)*((lame_lambda+lame_mu)*(vxs_x+vzs_z) \
                                   +(dlame_lambda+dlame_mu)*(vx_x+vz_z))+(1-c)*ps)

    return y_p, y_vx, y_vz, y_txx, y_tzz, y_txz, y_ps, y_vxs, y_vzs, y_txxs, y_tzzs, y_txzs

class AECLSRTM(FirstOrderEquation):
    """
    Parameter order: mp, ms, vp, vs, rho.

    Wavefields: (p, vx, vz, txx, tzz, txz), (ps, vxs, vzs, txxs, tzzs, txzs)

    Reference: Sun M.N., et. al, 10.1109/TGRS.2020.3047117
    """
    MODEL_SPECS = (
        ModelSpec("mp", aliases=("p_reflectivity",), description="P-wave reflectivity perturbation for AEC LSRTM."),
        ModelSpec("ms", aliases=("s_reflectivity",), description="S-wave reflectivity perturbation for AEC LSRTM."),
        ModelSpec("vp", aliases=("p_velocity",), description="Background P-wave velocity model.", unit="m/s"),
        ModelSpec("vs", aliases=("s_velocity",), description="Background S-wave velocity model.", unit="m/s"),
        ModelSpec("rho", aliases=("density",), description="Background density model.", unit="kg/m^3"),
    )

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch'):
        super().__init__(spatial_order, device, backend)

    @property
    def models(self):
        return [spec.name for spec in self.MODEL_SPECS]
    
    @property
    def wavefields(self):
        return ['p', 'vx', 'vz', 'txx', 'tzz', 'txz', 'ps', 'vxs', 'vzs', 'txxs', 'tzzs', 'txzs']
    
    def func(self, wavefields, models, dt, h, b, **kwargs):
        return step(*wavefields, *models, dt, h, b, pd=self.pd, **kwargs)
