"""2D three-component elastic TTI equation on an axis-aligned staggered grid."""

from __future__ import annotations

from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec, history_fields, record_multi

from .elastic_tti import STIFFNESS_KEYS, ElasticTTI
from ._registry import register_equation


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
    if free_surface:
        # The class refuses a free surface at construction (supports_free_surface
        # is False, inherited from ElasticTTI -- the anisotropic stress-free
        # condition is not the isotropic mirror this branch used to apply). A
        # True here can only mean the propagator was bypassed; fail loud rather
        # than run physics nothing has ever tested.
        raise NotImplementedError(
            "ElasticTTISG.step has no free-surface implementation; construct "
            "via PropTorch, which refuses free_surface=True for this equation.")

    dsxx_dx = pd.x_forward(sxx)
    dsxz_dz = pd.z_backward(sxz)
    dsxy_dx = pd.x_backward(sxy)
    dsyz_dz = pd.z_backward(syz)
    dsxz_dx = pd.x_backward(sxz)
    dszz_dz = pd.z_forward(szz)

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

    shear_xz = dvz_dx + dvx_dz

    sxx = sxx + dt * (C11 * dvx_dx + C16 * dvy_dx + C15 * shear_xz + C14 * dvy_dz + C13 * dvz_dz)
    szz = szz + dt * (C13 * dvx_dx + C36 * dvy_dx + C35 * shear_xz + C34 * dvy_dz + C33 * dvz_dz)
    syz = syz + dt * (C14 * dvx_dx + C46 * dvy_dx + C45 * shear_xz + C44 * dvy_dz + C34 * dvz_dz)
    sxz = sxz + dt * (C15 * dvx_dx + C56 * dvy_dx + C55 * shear_xz + C45 * dvy_dz + C35 * dvz_dz)
    sxy = sxy + dt * (C16 * dvx_dx + C66 * dvy_dx + C56 * shear_xz + C46 * dvy_dz + C36 * dvz_dz)

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


@register_equation()
class ElasticTTISG(ElasticTTI):
    """First-order 2-D three-component elastic TTI wave equation (axis-aligned SG).

    Same physics as :class:`ElasticTTI` (Bond-rotated TTI stiffness, 8
    raw model parameters) but with derivatives taken on an
    axis-aligned standard staggered grid (forward / backward FD pairs
    along x and z). Mixed-location stiffness couplings are used
    directly without interpolation — this is a clean no-interpolation
    SG reference companion to the RSG implementation. CPML follows the
    staggered-grid ``cpmls`` convention (8 profiles).

    
    """

    C_NAME = "elastic_tti_sg2d"
    C_HAS_RECURSIVE_CKPT = False

    prepare_models_for_c = True
    default_pml_type = "cpmls"  # SG variant uses 8 staggered CPML profiles, not 6.
    # Axis-aligned stencils have no checkerboard null space — keep plain
    # point sources/receivers (overrides the RSG parent's smoothing stencil).
    source_receiver_stencil = None

    def __init__(self, spatial_order=8, device="cpu", backend="torch"):
        """Build the 2-D-3C elastic TTI equation operator (axis-aligned SG).

        Args:
            spatial_order: FD accuracy order of the staggered first-derivative operator — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding). Must
                be an even integer (``2, 4, 6, 8, 10, …``).
                **Performance note (`impl='c'` on CUDA):** the compiled
                kernels ship template specialisations only for
                ``spatial_order ∈ {2, 4, 6, 8}``. Above 8 the dispatcher
                drops to a generic runtime path (``order = -1`` in
                ``src/sweep/csrc/cuda/equations/elastic_tti_sg2d/forward.cu``)
                which uses more registers and runs noticeably slower.
                The PyTorch eager path is unaffected. Defaults to 8.
            device: Device for the operator's static gradient kernels.
                Use ``'cuda'`` / a ``torch.device`` for GPU runs so the
                propagator can follow without a host↔device copy.
                Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or
                ``'jax'``. When you later want ``impl='c'``, leave this
                on ``'torch'``. Defaults to ``'torch'``.
        """
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

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            record_shape=record_multi(),
            save_all_shape=history_fields(8),   # all 8 physical fields
            base_nvar=8,
            pml_nvar=12,
            last_two_nvar=1,
            last_two_storage_nvar=8,
            # The six velocity-gradient scratch grids of the 2-D stress adjoint
            # (elastic_tti_sg2d/driver_traits.cuh WorkspaceSlot) in every mode;
            # ckpt (the only checkpoint mode this equation runs) adds the
            # velocity carriers behind them (WS_CARRIERS) -- v(t) at slots 6,
            # 7, 8 (vx, vy, vz) and v(t+1) at 9, 10, 11.  ElasticTTISG3D
            # declares its own 18 + 6.  The full mode keeps one read-only
            # zero grid at [6] (v(nt)).
            backward_workspace_shapes=lambda B, nt, shape, mode: [[B, 1, *shape]] * (
                6 + (6 if mode == "ckpt" else 1 if mode == "full" else 0)),
            # ckpt: the per-segment vx, vy, vz histories, one row per replayed
            # step plus the segment start (elastic_tti_sg2d/driver_traits.cuh
            # seg_buffers).
            checkpoint_replay_shapes=lambda B, nt, grid, seg, mode: (
                [(seg + 1, B, 1, *grid)] * 3 if mode == "ckpt" else []),
            bs_reconstruction_nvar=11,  # 8 physical fields + fvx/fvy/fvz_next carriers
        )
