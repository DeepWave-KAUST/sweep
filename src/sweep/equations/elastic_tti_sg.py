"""2D three-component elastic TTI equation on an axis-aligned staggered grid."""

from __future__ import annotations

from ._free_surface import top_free_surface_derivative, zero_top_row
from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec
from .elastic_tti import STIFFNESS_KEYS, ElasticTTI


def _slice_axis(u, axis, start=None, stop=None):
    axis = axis if axis >= 0 else u.ndim + axis
    slices = [slice(None)] * u.ndim
    slices[axis] = slice(start, stop)
    return u[tuple(slices)]


def _replace_axis_slice(u, axis, index, value):
    axis = axis if axis >= 0 else u.ndim + axis
    slices = [slice(None)] * u.ndim
    slices[axis] = slice(index, index + 1)

    if hasattr(u, "clone") and hasattr(u, "device") and hasattr(u, "dtype"):
        out = u.clone()
        out[tuple(slices)] = value
        return out
    if hasattr(u, "at"):
        return u.at[tuple(slices)].set(value)

    out = u.copy()
    out[tuple(slices)] = value
    return out


def _apply_top_traction_free_gradients(
    dvx_dx,
    dvy_dx,
    dvz_dx,
    dvx_dz,
    dvy_dz,
    dvz_dz,
    C13,
    C14,
    C15,
    C33,
    C34,
    C35,
    C36,
    C44,
    C45,
    C46,
    C55,
    C56,
    top_halo,
):
    """Enforce sigma_zz=sigma_xz=sigma_yz=0 on the SG free-surface row.

    For anisotropic media the three normal-direction velocity gradients on the
    free surface are coupled. Solving the local 3x3 traction system prevents
    the horizontal stress updates from seeing inconsistent top-row gradients.
    """

    row = top_halo
    vx_x = _slice_axis(dvx_dx, -2, row, row + 1)
    vy_x = _slice_axis(dvy_dx, -2, row, row + 1)
    vz_x = _slice_axis(dvz_dx, -2, row, row + 1)

    c13 = _slice_axis(C13, -2, row, row + 1)
    c14 = _slice_axis(C14, -2, row, row + 1)
    c15 = _slice_axis(C15, -2, row, row + 1)
    c33 = _slice_axis(C33, -2, row, row + 1)
    c34 = _slice_axis(C34, -2, row, row + 1)
    c35 = _slice_axis(C35, -2, row, row + 1)
    c36 = _slice_axis(C36, -2, row, row + 1)
    c44 = _slice_axis(C44, -2, row, row + 1)
    c45 = _slice_axis(C45, -2, row, row + 1)
    c46 = _slice_axis(C46, -2, row, row + 1)
    c55 = _slice_axis(C55, -2, row, row + 1)
    c56 = _slice_axis(C56, -2, row, row + 1)

    rhs_zz = c13 * vx_x + c36 * vy_x + c35 * vz_x
    rhs_yz = c14 * vx_x + c46 * vy_x + c45 * vz_x
    rhs_xz = c15 * vx_x + c56 * vy_x + c55 * vz_x

    # Matrix rows are the coefficients of [dvx_dz, dvy_dz, dvz_dz] in
    # sigma_zz, sigma_yz, and sigma_xz.
    a, b, c = c35, c34, c33
    d, e, f = c45, c44, c34
    g, h, i = c55, c45, c35
    r1, r2, r3 = -rhs_zz, -rhs_yz, -rhs_xz

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    vx_z = ((e * i - f * h) * r1 + (c * h - b * i) * r2 + (b * f - c * e) * r3) / det
    vy_z = ((f * g - d * i) * r1 + (a * i - c * g) * r2 + (c * d - a * f) * r3) / det
    vz_z = ((d * h - e * g) * r1 + (b * g - a * h) * r2 + (a * e - b * d) * r3) / det

    return (
        _replace_axis_slice(dvx_dz, -2, row, vx_z),
        _replace_axis_slice(dvy_dz, -2, row, vy_z),
        _replace_axis_slice(dvz_dz, -2, row, vz_z),
    )


def step(
    vx,
    vy,
    vz,
    sxx,
    szz,
    syz,
    sxz,
    sxy,
    m_vxx,
    m_vxz,
    m_vyx,
    m_vyz,
    m_vzx,
    m_vzz,
    m_txxx,
    m_txzz,
    m_txyx,
    m_tyzz,
    m_txzx,
    m_tzzz,
    rho,
    C11,
    C13,
    C14,
    C15,
    C16,
    C33,
    C34,
    C35,
    C36,
    C44,
    C45,
    C46,
    C55,
    C56,
    C66,
    dt,
    h,
    b,
    pd,
    pml=None,
    free_surface=False,
):
    """One axis-aligned staggered-grid elastic-TTI step.

    The update intentionally uses the natural staggered derivatives directly.
    Mixed-location stiffness couplings are not interpolated; this keeps the
    solver as a clean no-interpolation SG reference for the RSG implementation.
    """

    del h, b
    pml = pml if pml is not None else ()
    if len(pml) != 8:
        raise ValueError("ElasticTTISG requires pml_type='cpmls', which provides eight staggered CPML profiles.")
    az, bz, azh, bzh, ax, bx, axh, bxh = pml
    top_halo = pd.coes.shape[0]

    dsxx_dx = pd.x_forward(sxx)
    dsxz_dz = pd.z_backward(sxz)
    dsxy_dx = pd.x_backward(sxy)
    dsyz_dz = pd.z_backward(syz)
    dsxz_dx = pd.x_backward(sxz)
    dszz_dz = pd.z_forward(szz)

    if free_surface:
        dsxz_dz = top_free_surface_derivative(sxz, pd.z_backward, top_halo, odd=True, axis=-2)
        dsyz_dz = top_free_surface_derivative(syz, pd.z_backward, top_halo, odd=True, axis=-2)
        dszz_dz = top_free_surface_derivative(szz, pd.z_forward, top_halo, odd=True, axis=-2)

    m_txxx = axh * m_txxx + bxh * dsxx_dx
    m_txzz = az * m_txzz + bz * dsxz_dz
    m_txyx = ax * m_txyx + bx * dsxy_dx
    m_tyzz = az * m_tyzz + bz * dsyz_dz
    m_txzx = ax * m_txzx + bx * dsxz_dx
    m_tzzz = azh * m_tzzz + bzh * dszz_dz

    dsxx_dx = dsxx_dx + m_txxx
    dsxz_dz = dsxz_dz + m_txzz
    dsxy_dx = dsxy_dx + m_txyx
    dsyz_dz = dsyz_dz + m_tyzz
    dsxz_dx = dsxz_dx + m_txzx
    dszz_dz = dszz_dz + m_tzzz

    vx = vx + (dt / rho) * (dsxx_dx + dsxz_dz)
    vy = vy + (dt / rho) * (dsxy_dx + dsyz_dz)
    vz = vz + (dt / rho) * (dsxz_dx + dszz_dz)

    dvx_dx = pd.x_backward(vx)
    dvy_dx = pd.x_forward(vy)
    dvz_dx = pd.x_forward(vz)

    if free_surface:
        dvz_dz = top_free_surface_derivative(vz, pd.z_backward, top_halo, odd=True, axis=-2)
        dvx_dz = top_free_surface_derivative(vx, pd.z_forward, top_halo, odd=False, axis=-2)
        dvy_dz = top_free_surface_derivative(vy, pd.z_forward, top_halo, odd=False, axis=-2)
    else:
        dvz_dz = pd.z_backward(vz)
        dvx_dz = pd.z_forward(vx)
        dvy_dz = pd.z_forward(vy)

    m_vxx = ax * m_vxx + bx * dvx_dx
    m_vxz = azh * m_vxz + bzh * dvx_dz
    m_vyx = axh * m_vyx + bxh * dvy_dx
    m_vyz = azh * m_vyz + bzh * dvy_dz
    m_vzx = axh * m_vzx + bxh * dvz_dx
    m_vzz = az * m_vzz + bz * dvz_dz

    dvx_dx = dvx_dx + m_vxx
    dvx_dz = dvx_dz + m_vxz
    dvy_dx = dvy_dx + m_vyx
    dvy_dz = dvy_dz + m_vyz
    dvz_dx = dvz_dx + m_vzx
    dvz_dz = dvz_dz + m_vzz

    if free_surface:
        dvx_dz, dvy_dz, dvz_dz = _apply_top_traction_free_gradients(
            dvx_dx,
            dvy_dx,
            dvz_dx,
            dvx_dz,
            dvy_dz,
            dvz_dz,
            C13,
            C14,
            C15,
            C33,
            C34,
            C35,
            C36,
            C44,
            C45,
            C46,
            C55,
            C56,
            top_halo,
        )

    shear_xz = dvz_dx + dvx_dz

    sxx = sxx + dt * (C11 * dvx_dx + C16 * dvy_dx + C15 * shear_xz + C14 * dvy_dz + C13 * dvz_dz)
    szz = szz + dt * (C13 * dvx_dx + C36 * dvy_dx + C35 * shear_xz + C34 * dvy_dz + C33 * dvz_dz)
    syz = syz + dt * (C14 * dvx_dx + C46 * dvy_dx + C45 * shear_xz + C44 * dvy_dz + C34 * dvz_dz)
    sxz = sxz + dt * (C15 * dvx_dx + C56 * dvy_dx + C55 * shear_xz + C45 * dvy_dz + C35 * dvz_dz)
    sxy = sxy + dt * (C16 * dvx_dx + C66 * dvy_dx + C56 * shear_xz + C46 * dvy_dz + C36 * dvz_dz)

    if free_surface:
        szz = zero_top_row(szz, top_halo, axis=-2)
        sxz = zero_top_row(sxz, top_halo, axis=-2)
        syz = zero_top_row(syz, top_halo, axis=-2)

    return (
        vx,
        vy,
        vz,
        sxx,
        szz,
        syz,
        sxz,
        sxy,
        m_vxx,
        m_vxz,
        m_vyx,
        m_vyz,
        m_vzx,
        m_vzz,
        m_txxx,
        m_txzz,
        m_txyx,
        m_tyzz,
        m_txzx,
        m_tzzz,
    )


class ElasticTTISG(ElasticTTI):
    """2D-3C elastic TTI equation with axis-aligned SG derivatives and CPML."""

    prepare_models_for_c = True

    def __init__(self, spatial_order=8, device="cpu", backend="torch"):
        FirstOrderEquation.__init__(self, spatial_order, device, backend, ndim=2)

    def func(self, wavefields, models, dt, h, b, **kwargs):
        if len(models) == len(self.MODEL_SPECS):
            models = self.prepare_models(models)
        elif len(models) != 1 + len(STIFFNESS_KEYS):
            raise ValueError(
                "ElasticTTISG.func expected 8 raw models or "
                f"{1 + len(STIFFNESS_KEYS)} prepared models, got {len(models)}."
            )
        return step(
            *wavefields,
            *models,
            dt,
            h,
            b,
            pd=self.pd,
            pml=self.b,
            free_surface=getattr(self, "free_surface", False),
            **kwargs,
        )

    def _C(self):
        import sweep._C as _C

        return (
            _C.elastic_tti_sg2d_forward,
            _C.elastic_tti_sg2d_backward,
            _C.elastic_tti_sg2d_backward_bs,
            _C.elastic_tti_sg2d_backward_ckpt,
            None,
        )

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            base_nvar=8,
            pml_nvar=12,
            last_two_nvar=1,
            last_two_storage_nvar=8,
        )
