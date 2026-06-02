from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from ._free_surface import (
    top_free_surface_derivative,
    top_free_surface_derivative_topo,
    zero_top_row,
    zero_at_topo,
)
from ._topography import (
    classify_topography_3d,
    precompute_apm_moduli_3d,
    enforce_apm_traction_bc_3d,
    zero_at_air,
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
            # column (σ_zz = σ_xz = σ_yz = 0).  Cells above are zeroed by
            # the mirror in the next step's z-derivative read; here only
            # the surface row needs explicit BC enforcement.
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


def step_apm(vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
             m_vxx, m_vxy, m_vxz,
             m_vyx, m_vyy, m_vyz,
             m_vzx, m_vzy, m_vzz,
             m_sxxx, m_szzz,
             m_sxyx, m_sxyy,
             m_sxzx, m_sxzz,
             m_syyy,
             m_syzy, m_syzz,
             # APM precomputed effective moduli + masked inverse densities.
             alpha_xx, alpha_yy, alpha_zz,
             lam_xx_yy, lam_xx_zz,
             lam_yy_xx, lam_yy_zz,
             lam_zz_xx, lam_zz_yy,
             mu_xy_node, mu_xz_node, mu_yz_node,
             inv_rho_x, inv_rho_y, inv_rho_z,
             dt, h, b, pd,
             pml=None,
             ):
    """One Virieux 3-D leapfrog step with APM-modified moduli.

    Identical FD stencil to :func:`step`; only the modulus arrays
    multiplying the stencil outputs change.  ``free_surface`` /
    ``topo_rows`` are intentionally not parameters — APM implements the
    surface BC entirely via the per-cell effective moduli and a
    pointwise stress-zero applied by the caller (``_func_apm``).
    """
    az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh = pml

    # ---- Stress-gradient derivatives feeding the velocity update ----
    dsxx_dx = pd.x_forward(sxx)
    dsxy_dy = pd.y_backward(sxy)
    dsxz_dz = pd.z_backward(sxz)

    dsxy_dx = pd.x_backward(sxy)
    dsyy_dy = pd.y_forward(syy)
    dsyz_dz = pd.z_backward(syz)

    dsxz_dx = pd.x_backward(sxz)
    dsyz_dy = pd.y_backward(syz)
    dszz_dz = pd.z_forward(szz)

    # CPML memory for stress-gradient derivatives (same staggering as
    # ``step``).
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

    # Velocity update: per-component inverse density (already zeroed
    # where the velocity node sits in air, so no NaN even if rho was
    # logically "0" for those nodes).
    vx = vx + dt * inv_rho_x * (dsxx_dx + dsxy_dy + dsxz_dz)
    vy = vy + dt * inv_rho_y * (dsxy_dx + dsyy_dy + dsyz_dz)
    vz = vz + dt * inv_rho_z * (dsxz_dx + dsyz_dy + dszz_dz)

    # ---- Velocity-gradient derivatives feeding the stress update ----
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

    # Normal stresses: per-cell effective α (diagonal) + λ (off-diagonal).
    # Bulk: α = λ + 2μ on the diagonal and λ off-diagonal — identical to
    # ``step``.  Surface cells: see ``precompute_apm_moduli_3d``.
    sxx = sxx + dt * (alpha_xx * dvx_dx + lam_xx_yy * dvy_dy + lam_xx_zz * dvz_dz)
    syy = syy + dt * (lam_yy_xx * dvx_dx + alpha_yy * dvy_dy + lam_yy_zz * dvz_dz)
    szz = szz + dt * (lam_zz_xx * dvx_dx + lam_zz_yy * dvy_dy + alpha_zz * dvz_dz)

    # Shear stresses: per-node μ (μ at bulk, μ/2 at half-air surface nodes).
    sxy = sxy + dt * mu_xy_node * (dvx_dy + dvy_dx)
    sxz = sxz + dt * mu_xz_node * (dvx_dz + dvz_dx)
    syz = syz + dt * mu_yz_node * (dvy_dz + dvz_dy)

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

    
    """
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("p_velocity",), description="3D elastic P-wave velocity model.", unit="m/s"),
        ModelSpec("vs", aliases=("s_velocity",), description="3D elastic S-wave velocity model.", unit="m/s"),
        ModelSpec("rho", aliases=("density",), description="3D density model.", unit="kg/m^3"),
    )
    FIELD_SPECS = (
        FieldSpec("vx", aliases=("velocity_x",), description="Particle velocity in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("vy", aliases=("velocity_y",), description="Particle velocity in the y direction.", supports_source=True, supports_receiver=True),
        FieldSpec("vz", aliases=("velocity_z",), description="Particle velocity in the z direction.", supports_source=True, supports_receiver=True),
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
    supports_apm = True          # Cao & Chen 2018 3-D APM is implemented

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
        # APM (Cao & Chen 2018) cache: keyed by id() of the input tensors so it
        # hits on every subsequent timestep within a propagator run.
        self._apm_cache_key = None
        self._apm_cache = None

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

        # APM dispatch: if the propagator's ``topography=`` was a 3-D air
        # mask it stored a replicate-padded runtime version on the
        # equation.  Route through the parameter-modified step (Cao & Chen 2018).
        air_mask_rt = getattr(self, "_apm_air_mask_runtime", None)
        if air_mask_rt is not None:
            return self._func_apm(
                wavefields, lame_lambda, lame_mu, rho, air_mask_rt,
                dt, h, b, **kwargs,
            )

        # Standard (image-method / flat) dispatch.  ``topo_rows`` is 2-D
        # ``(ny, nx)`` on the runtime grid for irregular topo; ``None``
        # for flat / no-FS.
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

    def _func_apm(self, wavefields, lame_lambda, lame_mu, rho, air_mask_rt,
                  dt, h, b, **kwargs):
        """One leapfrog step with APM-modified moduli (Cao & Chen 2018, 3-D).

        Caches the per-category effective moduli by the id() of the input
        tensors — the same (air_mask, λ, μ, ρ) hit the cache on every
        subsequent time step within a propagator run.
        """
        cache_key = (id(air_mask_rt), id(lame_lambda), id(lame_mu), id(rho))
        if cache_key == self._apm_cache_key and self._apm_cache is not None:
            cat, eff = self._apm_cache
        else:
            cat = classify_topography_3d(air_mask_rt)
            eff = precompute_apm_moduli_3d(lame_lambda, lame_mu, rho, cat)
            self._apm_cache_key = cache_key
            self._apm_cache = (cat, eff)
        (alpha_xx, alpha_yy, alpha_zz,
         lam_xx_yy, lam_xx_zz,
         lam_yy_xx, lam_yy_zz,
         lam_zz_xx, lam_zz_yy,
         mu_xy, mu_xz, mu_yz,
         inv_rho_x, inv_rho_y, inv_rho_z) = eff

        out = step_apm(
            *wavefields,
            alpha_xx, alpha_yy, alpha_zz,
            lam_xx_yy, lam_xx_zz,
            lam_yy_xx, lam_yy_zz,
            lam_zz_xx, lam_zz_yy,
            mu_xy, mu_xz, mu_yz,
            inv_rho_x, inv_rho_y, inv_rho_z,
            dt, h, b,
            pd=self.pd,
            pml=self.b,
        )
        (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
         m_vxx, m_vxy, m_vxz,
         m_vyx, m_vyy, m_vyz,
         m_vzx, m_vzy, m_vzz,
         m_sxxx, m_szzz,
         m_sxyx, m_sxyy,
         m_sxzx, m_sxzz,
         m_syyy,
         m_syzy, m_syzz) = out

        # APM traction-free BC: pointwise stress zero per cell category.
        sxx, syy, szz, sxy, sxz, syz = enforce_apm_traction_bc_3d(
            sxx, syy, szz, sxy, sxz, syz, cat,
        )
        # Vacuum approximation: zero all 9 wavefield components in pure-air.
        vx = zero_at_air(vx, air_mask_rt)
        vy = zero_at_air(vy, air_mask_rt)
        vz = zero_at_air(vz, air_mask_rt)
        sxx = zero_at_air(sxx, air_mask_rt)
        syy = zero_at_air(syy, air_mask_rt)
        szz = zero_at_air(szz, air_mask_rt)
        sxy = zero_at_air(sxy, air_mask_rt)
        sxz = zero_at_air(sxz, air_mask_rt)
        syz = zero_at_air(syz, air_mask_rt)

        return (
            vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
            m_vxx, m_vxy, m_vxz,
            m_vyx, m_vyy, m_vyz,
            m_vzx, m_vzy, m_vzz,
            m_sxxx, m_szzz,
            m_sxyx, m_sxyy,
            m_sxzx, m_sxzz,
            m_syyy,
            m_syzy, m_syzz,
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

    def _C_apm(self):
        """APM CUDA bindings (Cao & Chen 2018, 3-D).
        Forward is fully implemented; backward is a stub pending Phase 3D
        — the _c.py dispatch handles the fallback to eager autograd for
        gradient computation."""
        import sweep._C as _C
        return (
            _C.elastic3d_apm_forward,
            _C.elastic3d_apm_backward,
            _C.elastic3d_apm_backward_bs,
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
