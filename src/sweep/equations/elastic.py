import numpy as np
from .base import FirstOrderEquation


def _flip(u, axis):
    module = np
    if hasattr(u, "flip"):
        try:
            return u.flip((axis,))
        except TypeError:
            pass
    return module.flip(u, axis=axis)


def _concat(arrays, axis):
    first = arrays[0]
    if hasattr(first, "device") and hasattr(first, "dtype"):
        import torch

        return torch.cat(arrays, dim=axis)
    try:
        import jax.numpy as jnp

        if type(first).__module__.startswith("jax"):
            return jnp.concatenate(arrays, axis=axis)
    except Exception:
        pass
    return np.concatenate(arrays, axis=axis)


def _extend_top_free_surface(u, halo, odd):
    if halo <= 0:
        return u
    ghost = _flip(u[..., 1 : halo + 1, :], axis=-2)
    if odd:
        ghost = -ghost
    return _concat([ghost, u], axis=-2)


def _top_free_surface_derivative(u, deriv, halo, odd):
    out = deriv(_extend_top_free_surface(u, halo, odd))
    return out[..., halo:, :]


def _zero_top_row(u):
    return _concat([u[..., :1, :] * 0, u[..., 1:, :]], axis=-2)


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
         dt, h, b, pd, 
         pml=None,
         free_surface=False,
         ):

    az, bz, azh, bzh, ax, bx, axh, bxh = pml
    top_halo = pd.coes.shape[0]
    vp = _match_reference_shape(vp, vx)
    vs = _match_reference_shape(vs, vx)
    rho = _match_reference_shape(rho, vx)

    lame_lambda = rho*(vp**2-2*vs**2)
    lame_mu = rho*vs**2

    txx_x = pd.x_forward(sxx)
    if free_surface:
        txz_z = _top_free_surface_derivative(sxz, pd.z_backward, top_halo, odd=True)
        tzz_z = _top_free_surface_derivative(szz, pd.z_forward, top_halo, odd=True)
    else:
        txz_z = pd.z_backward(sxz)
        tzz_z = pd.z_forward(szz)
    txz_z = _match_reference_shape(txz_z, vx)
    tzz_z = _match_reference_shape(tzz_z, vx)
    txz_x = pd.x_backward(sxz)
    txz_x = _match_reference_shape(txz_x, vx)

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
    if free_surface:
        vz_z = _top_free_surface_derivative(vz, pd.z_backward, top_halo, odd=True)
        vx_z = _top_free_surface_derivative(vx, pd.z_forward, top_halo, odd=False)
    else:
        vz_z = pd.z_backward(vz)
        vx_z = pd.z_forward(vx)
    vz_z = _match_reference_shape(vz_z, vx)
    vx_z = _match_reference_shape(vx_z, vx)
    vz_x = pd.x_forward(vz)
    vz_x = _match_reference_shape(vz_x, vx)

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

    if free_surface:
        szz = _zero_top_row(szz)
        sxz = _zero_top_row(sxz)


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
        return step(*args, pd=self.pd, pml=self.b, free_surface=getattr(self, "free_surface", False), **kwargs)
    
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
