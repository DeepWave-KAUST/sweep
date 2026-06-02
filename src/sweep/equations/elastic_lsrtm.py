from .base import FirstOrderEquation
from .fields import FieldSpec, ModelSpec

# 10.1190/GEO2016-0254.1
def step(vx, vz, txx, tzz, txz,
         vxs, vzs, txxs, tzzs, txzs,
         mp, ms, vp, vs, rho, 
         dt, h, b, pd):

    lame_lambda = rho*(vp**2-2*vs**2)
    lame_mu = rho*vs**2
    Ip = rho*vp
    Is = rho*vs
    c = 0.5*dt*b

    # Elastic wavefields
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

    # Perturbed wavefields

    dlame_lambda = 2*(Ip**2*mp-2*Is**2*ms)/rho
    dlame_mu = 2*Is**2*ms/rho

    vxs_x = pd.x_forward(vxs)
    vzs_z = pd.z_backward(vzs)
    vxs_z = pd.z_forward(vxs)
    vzs_x = pd.x_backward(vzs)

    y_txxs  = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vxs_x+lame_lambda*vzs_z \
                                     +dlame_lambda*(vx_x+vz_z)+2*dlame_mu*vx_x)+(1-c)*txxs)
    y_tzzs  = (1+c)**-1*(dt*h**(-1)*((lame_lambda+2*lame_mu)*vzs_z+lame_lambda*vxs_x \
                                     +dlame_lambda*(vx_x+vz_z)+2*dlame_mu*vz_z)+(1-c)*tzzs)
    y_txzs = (1+c)**-1*(dt*h**(-1)*(lame_mu*(vzs_x+vxs_z)+dlame_mu*(vz_x+vx_z))+(1-c)*txzs)

    txxs_x = pd.x_backward(y_txxs)
    txzs_z = pd.z_backward(y_txzs)
    tzzs_z = pd.z_forward(y_tzzs)
    txzs_x = pd.x_forward(y_txzs)

    y_vxs = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txxs_x+txzs_z)+(1-c)*vxs)
    y_vzs = (1+c)**-1*(dt*rho**(-1)*h**(-1)*(txzs_x+tzzs_z)+(1-c)*vzs)

    return y_vx, y_vz, y_txx, y_tzz, y_txz, y_vxs, y_vzs, y_txxs, y_tzzs, y_txzs

class ElasticLSRTM(FirstOrderEquation):
    """Parameter order: mp, ms, vp, vs, rho.
    
       Wavefields: (vx, vz, txx, tzz, txz), (vxs, vzs, txxs, tzzs, txzs)

       Reference: Feng & Schuster, 10.1190/geo2016-0254.1
    """
    MODEL_SPECS = (
        ModelSpec("mp", aliases=("p_reflectivity",), description="P-wave reflectivity perturbation for elastic LSRTM."),
        ModelSpec("ms", aliases=("s_reflectivity",), description="S-wave reflectivity perturbation for elastic LSRTM."),
        ModelSpec("vp", aliases=("p_velocity",), description="Background elastic P-wave velocity model.", unit="m/s"),
        ModelSpec("vs", aliases=("s_velocity",), description="Background elastic S-wave velocity model.", unit="m/s"),
        ModelSpec("rho", aliases=("density",), description="Background density model.", unit="kg/m^3"),
    )
    FIELD_SPECS = (
        FieldSpec("vx", aliases=("velocity_x", "background_vx"), description="Background particle velocity in the x direction.", internal=True),
        FieldSpec("vz", aliases=("velocity_z", "background_vz"), description="Background particle velocity in the z direction.", internal=True),
        FieldSpec("txx", aliases=("background_txx",), description="Background normal stress in the x direction.", supports_source=True),
        FieldSpec("tzz", aliases=("background_tzz",), description="Background normal stress in the z direction.", supports_source=True),
        FieldSpec("txz", aliases=("background_txz",), description="Background shear stress.", supports_source=True),
        FieldSpec("vxs", aliases=("scattered_vx",), description="Scattered particle velocity in the x direction.", supports_receiver=True),
        FieldSpec("vzs", aliases=("scattered_vz",), description="Scattered particle velocity in the z direction.", supports_receiver=True),
        FieldSpec("txxs", aliases=("scattered_txx",), description="Scattered normal stress in the x direction.", supports_receiver=True),
        FieldSpec("tzzs", aliases=("scattered_tzz",), description="Scattered normal stress in the z direction.", supports_receiver=True),
        FieldSpec("txzs", aliases=("scattered_txz",), description="Scattered shear stress.", supports_receiver=True),
    )
    def __init__(self, spatial_order=4, device='cpu', backend = 'torch'):
        super().__init__(spatial_order, device, backend)

    def func(self, wavefields, models, dt, h, b, **kwargs):
        return step(*wavefields, *models, dt, h, b, pd=self.pd, **kwargs)
