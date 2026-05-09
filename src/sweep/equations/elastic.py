from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from ._free_surface import top_free_surface_derivative, zero_top_row


def _match_reference_shape(u, ref):
    if u.shape[-2:] == ref.shape[-2:]:
        return u

    target_z, target_x = ref.shape[-2:]
    size_z, size_x = u.shape[-2:]
    if size_z < target_z or size_x < target_x:
        raise ValueError(
            f"Cannot match tensor shape {u.shape[-2:]} to larger reference {ref.shape[-2:]}."
        )

    crop_z = size_z - target_z
    crop_x = size_x - target_x
    z0 = crop_z // 2
    x0 = crop_x // 2
    return u[..., z0 : z0 + target_z, x0 : x0 + target_x]


def step(vx, vz, sxx, szz, sxz, 
         m_vxx, m_vxz, m_vzx, m_vzz,
         m_txxx, m_txxz, m_tzzx, m_tzzz,
         m_txzx, m_txzz,
         vp, vs, rho, 
         lame_lambda, lame_mu,
         dt, h, b, pd, 
         pml=None,
         free_surface=False,
         lame_lambda_2mu=None,
         ):

    az, bz, azh, bzh, ax, bx, axh, bxh = pml
    top_halo = pd.coes.shape[0]
    lame_lambda_2mu = lame_lambda + 2 * lame_mu if lame_lambda_2mu is None else lame_lambda_2mu

    txx_x = pd.x_forward(sxx)
    if free_surface:
        txz_z = top_free_surface_derivative(sxz, pd.z_backward, top_halo, odd=True, axis=-2)
        tzz_z = top_free_surface_derivative(szz, pd.z_forward, top_halo, odd=True, axis=-2)
    else:
        txz_z = pd.z_backward(sxz)
        tzz_z = pd.z_forward(szz)
    txz_x = pd.x_backward(sxz)

    # Update Veclocity fields
    m_tzzz = azh * m_tzzz + bzh * tzz_z
    tzz_z = tzz_z + m_tzzz
    m_txzx = ax * m_txzx + bx * txz_x
    txz_x = txz_x + m_txzx
    vz = vz + dt / rho * (tzz_z + txz_x)

    m_txzz = az * m_txzz + bz * txz_z
    txz_z = txz_z + m_txzz
    m_txxx = axh * m_txxx + bxh * txx_x
    txx_x = txx_x + m_txxx
    vx = vx + dt / rho * (txx_x + txz_z)

    # Update Stress fields
    vx_x = pd.x_backward(vx)
    if free_surface:
        vz_z = top_free_surface_derivative(vz, pd.z_backward, top_halo, odd=True, axis=-2)
        vx_z = top_free_surface_derivative(vx, pd.z_forward, top_halo, odd=False, axis=-2)
    else:
        vz_z = pd.z_backward(vz)
        vx_z = pd.z_forward(vx)
    vz_x = pd.x_forward(vz)

    m_vzz = az * m_vzz + bz * vz_z
    vz_z = vz_z + m_vzz
    m_vxx = ax * m_vxx + bx * vx_x
    vx_x = vx_x + m_vxx

    szz = szz + dt * (lame_lambda_2mu * vz_z + lame_lambda * vx_x)
    sxx = sxx + dt * (lame_lambda_2mu * vx_x + lame_lambda * vz_z)

    m_vxz = azh * m_vxz + bzh * vx_z
    vx_z = vx_z + m_vxz
    m_vzx = axh * m_vzx + bxh * vz_x
    vz_x = vz_x + m_vzx
    sxz = sxz + dt * lame_mu * (vx_z + vz_x)

    if free_surface:
        szz = zero_top_row(szz, top_halo, axis=-2)
        sxz = zero_top_row(sxz, top_halo, axis=-2)


    return vx, vz, sxx, szz, sxz, \
           m_vxx, m_vxz, m_vzx, m_vzz, \
           m_txxx, m_txxz, m_tzzx, m_tzzz, \
           m_txzx, m_txzz

class Elastic(FirstOrderEquation):
    """Parameter order: vp, vs, rho.
    
       Wavefields: (vx, vz, sxx, szz, sxz)

       Reference: Jean Virieux, 10.1190/1.1442147
    """
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("p_velocity",), description="Elastic P-wave velocity model.", unit="m/s"),
        ModelSpec("vs", aliases=("s_velocity",), description="Elastic S-wave velocity model.", unit="m/s"),
        ModelSpec("rho", aliases=("density",), description="Density model.", unit="kg/m^3"),
    )
    FIELD_SPECS = (
        FieldSpec("vx", aliases=("velocity_x",), description="Particle velocity in the x direction.", supports_receiver=True),
        FieldSpec("vz", aliases=("velocity_z",), description="Particle velocity in the z direction.", supports_receiver=True),
        FieldSpec("sxx", aliases=("stress_xx",), description="Normal stress in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("szz", aliases=("stress_zz",), description="Normal stress in the z direction.", supports_source=True, supports_receiver=True),
        FieldSpec("sxz", aliases=("stress_xz", "shear_xz"), description="Shear stress component.", supports_source=True),
        FieldSpec("m_vxx", description="CPML memory variable for dvx/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vxz", description="CPML memory variable for dvx/dz.", internal=True, boundary_related=True),
        FieldSpec("m_vzx", description="CPML memory variable for dvz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vzz", description="CPML memory variable for dvz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_txxx", description="CPML memory variable for dsxx/dx.", internal=True, boundary_related=True),
        FieldSpec("m_txxz", description="Reserved elastic auxiliary field.", internal=True, boundary_related=True),
        FieldSpec("m_tzzx", description="Reserved elastic auxiliary field.", internal=True, boundary_related=True),
        FieldSpec("m_tzzz", description="CPML memory variable for dszz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_txzx", description="CPML memory variable for dsxz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_txzz", description="CPML memory variable for dsxz/dz.", internal=True, boundary_related=True),
    )

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch'):
        super().__init__(spatial_order, device, backend)

    @property
    def models(self):
        return [spec.name for spec in self.MODEL_SPECS]
    
    @property
    def wavefields(self):
        return [spec.name for spec in self.FIELD_SPECS]

    @property
    def field_specs(self):
        return list(self.FIELD_SPECS)

    @property
    def default_source_fields(self):
        return ["sxx", "szz"]

    @property
    def default_receiver_fields(self):
        return ["vx", "vz"]

    def prepare_models(self, models):
        vp, vs, rho = models
        lame_lambda = rho * (vp**2 - 2 * vs**2)
        lame_mu = rho * vs**2
        return [vp, vs, rho, lame_lambda, lame_mu, lame_lambda + 2 * lame_mu]
    
    def func(self, wavefields, models, dt, h, b, **kwargs):
        if len(models) == 6:
            vp, vs, rho, lame_lambda, lame_mu, lame_lambda_2mu = models
        elif len(models) == 5:
            vp, vs, rho, lame_lambda, lame_mu = models
            lame_lambda_2mu = lame_lambda + 2 * lame_mu
        elif len(models) == 3:
            vp, vs, rho = models
            lame_lambda = rho * (vp**2 - 2 * vs**2)
            lame_mu = rho * vs**2
            lame_lambda_2mu = lame_lambda + 2 * lame_mu
        else:
            raise ValueError(f"Elastic.func expected 3, 5, or 6 models, got {len(models)}")
        return step(
            *wavefields,
            vp,
            vs,
            rho,
            lame_lambda,
            lame_mu,
            dt,
            h,
            b,
            pd=self.pd,
            pml=self.b,
            free_surface=getattr(self, "free_surface", False),
            lame_lambda_2mu=lame_lambda_2mu,
            **kwargs,
        )
    
    def _C(self, ):
        # CUDA IMPLEMENTATION
        import torch
        import sweep._C as _C
        return (
            _C.elastic2d_forward,
            _C.elastic2d_backward,
            _C.elastic2d_backward_bs,
            _C.elastic2d_backward_ckpt,
            _C.elastic2d_backward_recursive_ckpt,
        )

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            base_nvar=5,
            pml_nvar=10,
            last_two_nvar=1,
            last_two_storage_nvar=5,
            backward_workspace_nvar=8,
        )
