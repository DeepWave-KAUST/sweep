import numpy as np
from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec

def step(vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
         m_vxx, m_vxy, m_vxz,
         m_vyx, m_vyy, m_vyz,
         m_vzx, m_vzy, m_vzz, 
         m_sxxx, m_szzz,
         m_sxyx, m_sxyy,
         m_sxzx, m_sxzz,
            m_syyy,
         m_syzy, m_syzz,
         vp, vs, rho, 
         dt, h, b, pd, 
         pml=None,
         ):
    
    lame_lambda = rho*(vp**2-2*vs**2)
    lame_mu = rho*vs**2
    
    az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh = pml

    dsxx_dx = pd.x_forward(sxx)
    dsxy_dy = pd.y_backward(sxy)
    dsxz_dz = pd.z_backward(sxz)

    dsxy_dx = pd.x_backward(sxy)
    dsyy_dy = pd.y_forward(syy)
    dsyz_dz = pd.z_backward(syz)

    dsxz_dx = pd.x_backward(sxz)
    dsyz_dy = pd.y_backward(syz)
    dszz_dz = pd.z_forward(szz)

    m_szzz = azh * m_szzz + bzh * dszz_dz
    dszz_dz = dszz_dz + m_szzz
    m_sxzx = ax * m_sxzx + bx * dsxz_dx
    dsxz_dx = dsxz_dx + m_sxzx

    m_sxzz = az * m_sxzz + bz * dsxz_dz
    dsxz_dz = dsxz_dz + m_sxzz
    m_sxxx = axh * m_sxxx + bxh * dsxx_dx
    dsxx_dx = dsxx_dx + m_sxxx

    m_sxyy = ay * m_sxyy + by * dsxy_dy
    dsxy_dy = dsxy_dy + m_sxyy

    m_sxyx = ax * m_sxyx + bx * dsxy_dx
    dsxy_dx = dsxy_dx + m_sxyx

    m_syyy = ayh * m_syyy + byh * dsyy_dy
    dsyy_dy = dsyy_dy + m_syyy
    m_syzz = az * m_syzz + bz * dsyz_dz
    dsyz_dz = dsyz_dz + m_syzz

    m_syzy = ay * m_syzy + by * dsyz_dy
    dsyz_dy = dsyz_dy + m_syzy

    vx = vx + dt / rho * (dsxx_dx + dsxy_dy + dsxz_dz)
    vy = vy + dt / rho * (dsxy_dx + dsyy_dy + dsyz_dz)
    vz = vz + dt / rho * (dsxz_dx + dsyz_dy + dszz_dz)

    dvx_dx = pd.x_backward(vx)
    dvx_dy = pd.y_forward(vx)
    dvx_dz = pd.z_forward(vx)

    dvy_dx = pd.x_forward(vy)
    dvy_dy = pd.y_backward(vy)
    dvy_dz = pd.z_forward(vy)

    dvz_dx = pd.x_forward(vz)
    dvz_dy = pd.y_forward(vz)
    dvz_dz = pd.z_backward(vz)

    m_vzz = az * m_vzz + bz * dvz_dz
    dvz_dz = dvz_dz + m_vzz
    m_vyy = ay * m_vyy + by * dvy_dy
    dvy_dy = dvy_dy + m_vyy
    m_vxx = ax * m_vxx + bx * dvx_dx
    dvx_dx = dvx_dx + m_vxx
    m_vxz = azh * m_vxz + bzh * dvx_dz
    dvx_dz = dvx_dz + m_vxz
    m_vzx = axh * m_vzx + bxh * dvz_dx
    dvz_dx = dvz_dx + m_vzx

    m_vxy = ayh * m_vxy + byh * dvx_dy
    dvx_dy = dvx_dy + m_vxy
    m_vyx = axh * m_vyx + bxh * dvy_dx
    dvy_dx = dvy_dx + m_vyx
    m_vyz = azh * m_vyz + bzh * dvy_dz
    dvy_dz = dvy_dz + m_vyz
    m_vzy = ayh * m_vzy + byh * dvz_dy
    dvz_dy = dvz_dy + m_vzy

    div_v = dvx_dx + dvy_dy + dvz_dz

    sxx = sxx + dt * (lame_lambda * div_v + 2 * lame_mu * dvx_dx)
    syy = syy + dt * (lame_lambda * div_v + 2 * lame_mu * dvy_dy)
    szz = szz + dt * (lame_lambda * div_v + 2 * lame_mu * dvz_dz)
    sxy = sxy + dt * lame_mu * (dvx_dy + dvy_dx)
    sxz = sxz + dt * lame_mu * (dvx_dz + dvz_dx)
    syz = syz + dt * lame_mu * (dvy_dz + dvz_dy)
    
    return vx, vy, vz, sxx, syy, szz, sxy, sxz, syz, \
           m_vxx, m_vxy, m_vxz, \
           m_vyx, m_vyy, m_vyz, \
           m_vzx, m_vzy, m_vzz, \
           m_sxxx, m_szzz, \
           m_sxyx, m_sxyy, \
           m_sxzx, m_sxzz, \
           m_syyy, \
           m_syzy, m_syzz

class Elastic(FirstOrderEquation):
    """Parameter order: vp, vs, rho.
    
       Wavefields: (vx, vz, txx, tzz, txz)

       Reference: Jean Virieux, 10.1190/1.1442147
    """
    FIELD_SPECS = (
        FieldSpec("vx", aliases=("velocity_x",), description="Particle velocity in the x direction.", supports_receiver=True),
        FieldSpec("vy", aliases=("velocity_y",), description="Particle velocity in the y direction.", supports_receiver=True),
        FieldSpec("vz", aliases=("velocity_z",), description="Particle velocity in the z direction.", supports_receiver=True),
        FieldSpec("sxx", aliases=("stress_xx",), description="Normal stress in the x direction.", supports_source=True),
        FieldSpec("syy", aliases=("stress_yy",), description="Normal stress in the y direction.", supports_source=True),
        FieldSpec("szz", aliases=("stress_zz",), description="Normal stress in the z direction.", supports_source=True),
        FieldSpec("sxy", aliases=("stress_xy", "shear_xy"), description="Shear stress component.", supports_source=True),
        FieldSpec("sxz", aliases=("stress_xz", "shear_xz"), description="Shear stress component.", supports_source=True),
        FieldSpec("syz", aliases=("stress_yz", "shear_yz"), description="Shear stress component.", supports_source=True),
        FieldSpec("m_vxx", description="CPML memory variable for dvx/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vxy", description="CPML memory variable for dvx/dy.", internal=True, boundary_related=True),
        FieldSpec("m_vxz", description="CPML memory variable for dvx/dz.", internal=True, boundary_related=True),
        FieldSpec("m_vyx", description="CPML memory variable for dvy/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vyy", description="CPML memory variable for dvy/dy.", internal=True, boundary_related=True),
        FieldSpec("m_vyz", description="CPML memory variable for dvy/dz.", internal=True, boundary_related=True),
        FieldSpec("m_vzx", description="CPML memory variable for dvz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vzy", description="CPML memory variable for dvz/dy.", internal=True, boundary_related=True),
        FieldSpec("m_vzz", description="CPML memory variable for dvz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_sxxx", description="CPML memory variable for dsxx/dx.", internal=True, boundary_related=True),
        FieldSpec("m_szzz", description="CPML memory variable for dszz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_sxyx", description="CPML memory variable for dsxy/dx.", internal=True, boundary_related=True),
        FieldSpec("m_sxyy", description="CPML memory variable for dsxy/dy.", internal=True, boundary_related=True),
        FieldSpec("m_sxzx", description="CPML memory variable for dsxz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_sxzz", description="CPML memory variable for dsxz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_syyy", description="CPML memory variable for dsyy/dy.", internal=True, boundary_related=True),
        FieldSpec("m_syzy", description="CPML memory variable for dsyz/dy.", internal=True, boundary_related=True),
        FieldSpec("m_syzz", description="CPML memory variable for dsyz/dz.", internal=True, boundary_related=True),
    )

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch'):
        super().__init__(spatial_order, device, backend, ndim=3)
    
    @property
    def models(self):
        return ['vp', 'vs', 'rho']
    
    @property
    def wavefields(self):
        return ['vx', 'vy', 'vz', 'sxx', 'syy', 'szz', 'sxy', 'sxz', 'syz', 
                'm_vxx', 'm_vxy', 'm_vxz',
                'm_vyx', 'm_vyy', 'm_vyz',
                'm_vzx', 'm_vzy', 'm_vzz',
                'm_sxxx', 'm_szzz',
                'm_sxyx', 'm_sxyy',
                'm_sxzx', 'm_sxzz',
                'm_syyy', 'm_syzy', 'm_syzz']

    @property
    def field_specs(self):
        return list(self.FIELD_SPECS)

    @property
    def default_source_fields(self):
        return ["sxx", "syy", "szz"]

    @property
    def default_receiver_fields(self):
        return ["vx", "vy", "vz"]
    
    def func(self, *args, **kwargs):
        return step(*args, pd=self.pd, pml=self.b, **kwargs)
    
    def _C(self, ):
        # CUDA IMPLEMENTATION
        import torch
        import sweep._C as _C
        return (
            _C.elastic3d_forward,
            _C.elastic3d_backward,
            _C.elastic3d_backward_bs,
            _C.elastic3d_backward_ckpt,
            _C.elastic3d_backward_recursive_ckpt,
        )
    
    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            base_nvar=9,
            pml_nvar=27,
            last_two_nvar=1,
            last_two_storage_nvar=9,
            backward_workspace_nvar=18,
        )
