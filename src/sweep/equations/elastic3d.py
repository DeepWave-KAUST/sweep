from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from ._free_surface import (
    top_free_surface_derivative,
    top_free_surface_derivative_topo,
    zero_top_row,
    zero_at_topo,
)


def _fs_z_deriv(field, deriv, top_halo, odd, topo_rows):
    """3-D free-surface z-derivative: flat ``top_halo`` row when
    ``topo_rows`` is None, per-(iy,ix) ``topo_rows[iy, ix]`` row otherwise."""
    if topo_rows is None:
        return top_free_surface_derivative(field, deriv, top_halo, odd=odd, axis=-3)
    return top_free_surface_derivative_topo(
        field, deriv, top_halo, odd=odd, axis=-3, iz_surf=topo_rows
    )


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
         lame_lambda, lame_mu,
         dt, h, b, pd,
         pml=None,
         free_surface=False,
         topo_rows=None,
         ):
    az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh = pml
    top_halo = pd.coes.shape[0]

    dsxx_dx = pd.x_forward(sxx)
    dsxy_dy = pd.y_backward(sxy)
    if free_surface:
        dsxz_dz = _fs_z_deriv(sxz, pd.z_backward, top_halo, True, topo_rows)
    else:
        dsxz_dz = pd.z_backward(sxz)

    dsxy_dx = pd.x_backward(sxy)
    dsyy_dy = pd.y_forward(syy)
    if free_surface:
        dsyz_dz = _fs_z_deriv(syz, pd.z_backward, top_halo, True, topo_rows)
    else:
        dsyz_dz = pd.z_backward(syz)

    dsxz_dx = pd.x_backward(sxz)
    dsyz_dy = pd.y_backward(syz)
    if free_surface:
        dszz_dz = _fs_z_deriv(szz, pd.z_forward, top_halo, True, topo_rows)
    else:
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
    if free_surface:
        dvx_dz = _fs_z_deriv(vx, pd.z_forward, top_halo, False, topo_rows)
    else:
        dvx_dz = pd.z_forward(vx)

    dvy_dx = pd.x_forward(vy)
    dvy_dy = pd.y_backward(vy)
    if free_surface:
        dvy_dz = _fs_z_deriv(vy, pd.z_forward, top_halo, False, topo_rows)
    else:
        dvy_dz = pd.z_forward(vy)

    dvz_dx = pd.x_forward(vz)
    dvz_dy = pd.y_forward(vz)
    if free_surface:
        dvz_dz = _fs_z_deriv(vz, pd.z_backward, top_halo, True, topo_rows)
    else:
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

    if free_surface:
        if topo_rows is not None:
            # Irregular surface: zero exactly the surface row per (iy, ix)
            # column (σ_zz = σ_xz = σ_yz = 0).
            szz = zero_at_topo(szz, topo_rows, axis=-3)
            sxz = zero_at_topo(sxz, topo_rows, axis=-3)
            syz = zero_at_topo(syz, topo_rows, axis=-3)
        else:
            szz = zero_top_row(szz, top_halo, axis=-3)
            sxz = zero_top_row(sxz, top_halo, axis=-3)
            syz = zero_top_row(syz, top_halo, axis=-3)
    
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
    """First-order 3-D elastic wave equation on a staggered grid (Virieux 1986).

    Three-dimensional velocity-stress formulation. The nine physical fields
    ``(vx, vy, vz, sxx, syy, szz, sxy, sxz, syz)`` are evolved together
    with eighteen CPML memory variables (one per first-derivative
    direction). Sources are typically an explosion (``['sxx', 'syy',
    'szz']`` — the default) or a directional body force; receivers
    usually read particle velocities or stresses.

    Reference: J. Virieux, 1986, *P-SV wave propagation in heterogeneous
    media: velocity-stress finite-difference method*, Geophysics 51(4),
    [10.1190/1.1442147](https://doi.org/10.1190/1.1442147).

    !!! info "Models (constructor input order)"

        - ``vp`` (m/s): 3D elastic P-wave velocity model.
        - ``vs`` (m/s): 3D elastic S-wave velocity model.
        - ``rho`` (kg/m^3): 3D density model.

    !!! info "Wavefields"

        - ``vx`` (aliases: ``velocity_x``): Particle velocity in the x direction; default receiver.
        - ``vy`` (aliases: ``velocity_y``): Particle velocity in the y direction; default receiver.
        - ``vz`` (aliases: ``velocity_z``): Particle velocity in the z direction; default receiver.
        - ``sxx`` (aliases: ``stress_xx``): Normal stress in the x direction; default source.
        - ``syy`` (aliases: ``stress_yy``): Normal stress in the y direction; default source.
        - ``szz`` (aliases: ``stress_zz``): Normal stress in the z direction; default source.
        - ``sxy`` (aliases: ``stress_xy``, ``shear_xy``): Shear stress component.
        - ``sxz`` (aliases: ``stress_xz``, ``shear_xz``): Shear stress component.
        - ``syz`` (aliases: ``stress_yz``, ``shear_yz``): Shear stress component.
        - ``m_vxx``: CPML memory variable for dvx/dx (internal).
        - ``m_vxy``: CPML memory variable for dvx/dy (internal).
        - ``m_vxz``: CPML memory variable for dvx/dz (internal).
        - ``m_vyx``: CPML memory variable for dvy/dx (internal).
        - ``m_vyy``: CPML memory variable for dvy/dy (internal).
        - ``m_vyz``: CPML memory variable for dvy/dz (internal).
        - ``m_vzx``: CPML memory variable for dvz/dx (internal).
        - ``m_vzy``: CPML memory variable for dvz/dy (internal).
        - ``m_vzz``: CPML memory variable for dvz/dz (internal).
        - ``m_sxxx``: CPML memory variable for dsxx/dx (internal).
        - ``m_szzz``: CPML memory variable for dszz/dz (internal).
        - ``m_sxyx``: CPML memory variable for dsxy/dx (internal).
        - ``m_sxyy``: CPML memory variable for dsxy/dy (internal).
        - ``m_sxzx``: CPML memory variable for dsxz/dx (internal).
        - ``m_sxzz``: CPML memory variable for dsxz/dz (internal).
        - ``m_syyy``: CPML memory variable for dsyy/dy (internal).
        - ``m_syzy``: CPML memory variable for dsyz/dy (internal).
        - ``m_syzz``: CPML memory variable for dsyz/dz (internal).

    !!! info "Defaults"

        - ``source_type``: ``['sxx', 'syy', 'szz']``
        - ``receiver_type``: ``['vx', 'vy', 'vz']``
        - ``pml_type``: ``'cpmls'``

    Note:
        The class is named ``Elastic`` in this module but is exposed as
        :class:`Elastic3D` from :mod:`sweep.equations`; use the 3-D name
        in user code.
    """
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("p_velocity",), description="3D elastic P-wave velocity model.", unit="m/s"),
        ModelSpec("vs", aliases=("s_velocity",), description="3D elastic S-wave velocity model.", unit="m/s"),
        ModelSpec("rho", aliases=("density",), description="3D density model.", unit="kg/m^3"),
    )
    FIELD_SPECS = (
        FieldSpec("vx", aliases=("velocity_x",), description="Particle velocity in the x direction.", supports_receiver=True),
        FieldSpec("vy", aliases=("velocity_y",), description="Particle velocity in the y direction.", supports_receiver=True),
        FieldSpec("vz", aliases=("velocity_z",), description="Particle velocity in the z direction.", supports_receiver=True),
        FieldSpec("sxx", aliases=("stress_xx",), description="Normal stress in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("syy", aliases=("stress_yy",), description="Normal stress in the y direction.", supports_source=True, supports_receiver=True),
        FieldSpec("szz", aliases=("stress_zz",), description="Normal stress in the z direction.", supports_source=True, supports_receiver=True),
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

    default_pml_type = "cpmls"  # staggered-grid CPML: step() unpacks 12 profiles

    def __init__(self, spatial_order=4, device='cpu', backend = 'torch'):
        """Build the 3-D elastic equation operator.

        Args:
            spatial_order: FD accuracy order of the staggered first-derivative operator — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding). Must
                be an even integer (``2, 4, 6, 8, 10, …``). Higher orders
                reduce grid dispersion — most visibly on the slower
                S-wave — at the cost of more compute per step and a
                wider PML halo.
                **Performance note (`impl='c'` on CUDA):** the compiled
                kernels are template-specialised only for
                ``spatial_order ∈ {2, 4, 6, 8}``. Above 8 the dispatcher
                falls through to the generic ``order = -1`` runtime path
                (see ``src/sweep/csrc/cuda/equations/elastic3d/forward.cu``)
                which is noticeably slower; the PyTorch eager path is
                unaffected. Defaults to 4.
            device: Device for the operator's static gradient kernels.
                Use ``'cuda'`` / a ``torch.device`` for GPU runs so the
                propagator can follow without a host↔device copy.
                Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or
                ``'jax'``. When you later want ``impl='c'``, leave this
                on ``'torch'``. Defaults to ``'torch'``.
        """
        super().__init__(spatial_order, device, backend, ndim=3)
    
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
        return ["sxx", "syy", "szz"]

    @property
    def default_receiver_fields(self):
        return ["vx", "vy", "vz"]

    def prepare_models(self, models):
        vp, vs, rho = models
        lame_lambda = rho * (vp**2 - 2 * vs**2)
        lame_mu = rho * vs**2
        return [vp, vs, rho, lame_lambda, lame_mu]
    
    def func(self, wavefields, models, dt, h, b, **kwargs):
        if len(models) == 5:
            vp, vs, rho, lame_lambda, lame_mu = models
        elif len(models) == 3:
            vp, vs, rho = models
            lame_lambda = rho * (vp**2 - 2 * vs**2)
            lame_mu = rho * vs**2
        else:
            raise ValueError(f"Elastic3D.func expected 3 or 5 models, got {len(models)}")
        # Irregular topography (image method): 2-D ``topo_rows`` shape
        # ``(ny, nx)`` on the runtime grid; ``None`` for flat / no-FS.
        topo_rows = getattr(self, "_topo_rows_runtime", None)
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
            topo_rows=topo_rows,
            **kwargs,
        )
    
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
