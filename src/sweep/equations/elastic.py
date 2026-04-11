import numpy as np
from .base import FirstOrderEquation

def step(vx, vz, sxx, szz, sxz, 
         m_vxx, m_vxz, m_vzx, m_vzz,
         m_txxx, m_txxz, m_tzzx, m_tzzz,
         m_txzx, m_txzz,
         vp, vs, rho, 
         dt, h, b, pd, 
         pml=None,
         ):

    az, bz, azh, bzh, ax, bx, axh, bxh = pml

    lame_lambda = rho*(vp**2-2*vs**2)
    lame_mu = rho*vs**2

    txx_x = pd.x_forward(sxx)
    txz_z = pd.z_backward(sxz)
    tzz_z = pd.z_forward(szz)
    txz_x = pd.x_backward(sxz)

    # Update Veclocity fields
    m_tzzz = azh * m_tzzz + bzh * tzz_z
    tzz_z = tzz_z + m_tzzz
    m_txzx = ax * m_txzx + bx * txz_x
    txz_x = txz_x + m_txzx
    vz = vz + dt / (rho * h) * (tzz_z + txz_x)

    m_txzz = az * m_txzz + bz * txz_z
    txz_z = txz_z + m_txzz
    m_txxx = axh * m_txxx + bxh * txx_x
    txx_x = txx_x + m_txxx
    vx = vx + dt / (rho * h) * (txx_x + txz_z)

    # Update Stress fields
    vx_x = pd.x_backward(vx)
    vz_z = pd.z_backward(vz)
    vx_z = pd.z_forward(vx)
    vz_x = pd.x_forward(vz)

    m_vzz = az * m_vzz + bz * vz_z
    vz_z = vz_z + m_vzz
    m_vxx = ax * m_vxx + bx * vx_x
    vx_x = vx_x + m_vxx

    szz = szz + dt * (lame_lambda + 2 * lame_mu) / h * vz_z + dt * lame_lambda / h * vx_x
    sxx = sxx + dt * (lame_lambda + 2 * lame_mu) / h * vx_x + dt * lame_lambda / h * vz_z

    m_vxz = azh * m_vxz + bzh * vx_z
    vx_z = vx_z + m_vxz
    m_vzx = axh * m_vzx + bxh * vz_x
    vz_x = vz_x + m_vzx
    sxz = sxz + dt * lame_mu / h * (vx_z + vz_x)


    return vx, vz, sxx, szz, sxz, \
           m_vxx, m_vxz, m_vzx, m_vzz, \
           m_txxx, m_txxz, m_tzzx, m_tzzz, \
           m_txzx, m_txzz

class Elastic(FirstOrderEquation):
    """Parameter order: vp, vs, rho.
    
       Wavefields: (vx, vz, sxx, szz, sxz)

       Reference: Jean Virieux, 10.1190/1.1442147
    """
    def __init__(self, spatial_order=4, device='cpu', backend = 'torch'):
        super().__init__(spatial_order, device, backend)

    @property
    def models(self):
        return ['vp', 'vs', 'rho']
    
    @property
    def wavefields(self):
        return ['vx', 'vz', 'sxx', 'szz', 'sxz', 
                'm_vxx', 'm_vxz', 'm_vzx', 'm_vzz', 'm_txxx', 'm_txxz', 'm_tzzx', 'm_tzzz', 'm_txzx', 'm_txzz']
    
    def func(self, *args, **kwargs):
        return step(*args, pd=self.pd, pml=self.b, **kwargs)
    
    def _C(self, ):
        # CUDA IMPLEMENTATION
        import sweep._C as _C
        return (
            _C.elastic2d_forward,
            _C.elastic2d_backward,
            _C.elastic2d_backward_bs,
            _C.elastic2d_backward_ckpt,
            _C.elastic2d_backward_recursive_ckpt,
        )

    @property
    def base_nvar(self,):
        return 5

    @property
    def pml_nvar(self,):
        return 10

    @property
    def last_two_nvar(self):
        return 1

    @property
    def last_two_storage_nvar(self):
        return self.base_nvar

    @property
    def backward_workspace_nvar(self):
        return 8
