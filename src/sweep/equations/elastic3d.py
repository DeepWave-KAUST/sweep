from .base import FirstOrderEquation
from ._cpml import cpml_update
from .cuda_layout import CUDALayoutSpec, history_fields, record_multi

from . import slot_table
from .fields import FieldSpec, ModelSpec
from ._free_surface import (
    FS_MOD_SXX,
    top_free_surface_derivative,
    top_free_surface_cell_derivative,
    top_free_surface_derivative_topo,
    zero_top_row,
    zero_at_topo,
    overwrite_top_row,
    overwrite_at_topo,
)
from ._registry import register_equation


def _fs_z_deriv(field, deriv, top_halo, odd, topo_rows, half=False):
    """3-D free-surface z-derivative: flat ``top_halo`` row when
    ``topo_rows`` is None, per-(iy,ix) ``topo_rows[iy, ix]`` row otherwise.

    ``half=True`` marks the staggered fields at ``z=+h/2`` (``sxz``, ``syz``,
    ``vz``): their image reflects about ``top_halo-1/2``, giving
    ``tau_zx(-h/2) = -tau_zx(+h/2)`` (Kristek, Moczo & Archuleta 2002, Table 1).
    The integer-grid mirror would instead pair ``-h/2`` with ``+3h/2`` — masked
    while the surface row is force-zeroed, destabilising once it is not.
    Irregular topography reflects about ``iz_surf-1/2`` for the same fields, so
    a constant ``topo_rows`` reproduces the flat path exactly."""
    if topo_rows is None:
        deriv_fn = (top_free_surface_cell_derivative if half
                    else top_free_surface_derivative)
        return deriv_fn(field, deriv, top_halo, odd=odd, axis=-3)
    return top_free_surface_derivative_topo(
        field, deriv, top_halo, odd=odd, axis=-3, iz_surf=topo_rows, half=half
    )


def velocity_substep(vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
                     m_vxx, m_vxy, m_vxz,
                     m_vyx, m_vyy, m_vyz,
                     m_vzx, m_vzy, m_vzz,
                     m_sxxx, m_szzz,
                     m_sxyx, m_sxyy,
                     m_sxzx, m_sxzz,
                     m_syyy,
                     m_syzy, m_syzz,
                     *, rho, dt, pd, pml, free_surface=False, topo_rows=None):
    """First half of :func:`step`: the stress derivatives (and their CPML
    memories), then the velocities. Returns all 27 fields, in order."""
    az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh = pml
    top_halo = pd.coes.shape[0]

    dsxx_dx = pd.x_forward(sxx)
    dsxy_dy = pd.y_backward(sxy)
    if free_surface:
        dsxz_dz = _fs_z_deriv(sxz, pd.z_backward, top_halo, True, topo_rows, half=True)
    else:
        dsxz_dz = pd.z_backward(sxz)

    dsxy_dx = pd.x_backward(sxy)
    dsyy_dy = pd.y_forward(syy)
    if free_surface:
        dsyz_dz = _fs_z_deriv(syz, pd.z_backward, top_halo, True, topo_rows, half=True)
    else:
        dsyz_dz = pd.z_backward(syz)

    dsxz_dx = pd.x_backward(sxz)
    dsyz_dy = pd.y_backward(syz)
    if free_surface:
        dszz_dz = _fs_z_deriv(szz, pd.z_forward, top_halo, True, topo_rows)
    else:
        dszz_dz = pd.z_forward(szz)

    dszz_dz, m_szzz = cpml_update(dszz_dz, m_szzz, azh, bzh, (szz, pd.kzf, pd._spacing["z"]))
    dsxz_dx, m_sxzx = cpml_update(dsxz_dx, m_sxzx, ax, bx, (sxz, pd.kxb, pd._spacing["x"]))

    dsxz_dz, m_sxzz = cpml_update(dsxz_dz, m_sxzz, az, bz, (sxz, pd.kzb, pd._spacing["z"]))
    dsxx_dx, m_sxxx = cpml_update(dsxx_dx, m_sxxx, axh, bxh, (sxx, pd.kxf, pd._spacing["x"]))

    dsxy_dy, m_sxyy = cpml_update(dsxy_dy, m_sxyy, ay, by, (sxy, pd.kyb, pd._spacing["y"]))

    dsxy_dx, m_sxyx = cpml_update(dsxy_dx, m_sxyx, ax, bx, (sxy, pd.kxb, pd._spacing["x"]))

    dsyy_dy, m_syyy = cpml_update(dsyy_dy, m_syyy, ayh, byh, (syy, pd.kyf, pd._spacing["y"]))
    dsyz_dz, m_syzz = cpml_update(dsyz_dz, m_syzz, az, bz, (syz, pd.kzb, pd._spacing["z"]))

    dsyz_dy, m_syzy = cpml_update(dsyz_dy, m_syzy, ay, by, (syz, pd.kyb, pd._spacing["y"]))

    vx = vx + dt / rho * (dsxx_dx + dsxy_dy + dsxz_dz)
    vy = vy + dt / rho * (dsxy_dx + dsyy_dy + dsyz_dz)
    vz = vz + dt / rho * (dsxz_dx + dsyz_dy + dszz_dz)

    return (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
            m_vxx, m_vxy, m_vxz,
            m_vyx, m_vyy, m_vyz,
            m_vzx, m_vzy, m_vzz,
            m_sxxx, m_szzz,
            m_sxyx, m_sxyy,
            m_sxzx, m_sxzz,
            m_syyy,
            m_syzy, m_syzz)


def stress_substep(vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
                   m_vxx, m_vxy, m_vxz,
                   m_vyx, m_vyy, m_vyz,
                   m_vzx, m_vzy, m_vzz,
                   m_sxxx, m_szzz,
                   m_sxyx, m_sxyy,
                   m_sxzx, m_sxzz,
                   m_syyy,
                   m_syzy, m_syzz,
                   *, lame_lambda, lame_mu, dt, pd, pml, free_surface=False, topo_rows=None,
                   return_gradients=False):
    """Second half of :func:`step`: the velocity derivatives (and their CPML
    memories), then the stresses and the free-surface rows."""
    az, bz, azh, bzh, ay, by, ayh, byh, ax, bx, axh, bxh = pml
    top_halo = pd.coes.shape[0]

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
        dvz_dz = _fs_z_deriv(vz, pd.z_backward, top_halo, True, topo_rows, half=True)
    else:
        dvz_dz = pd.z_backward(vz)

    dvz_dz, m_vzz = cpml_update(dvz_dz, m_vzz, az, bz, (vz, pd.kzb, pd._spacing["z"]))
    dvy_dy, m_vyy = cpml_update(dvy_dy, m_vyy, ay, by, (vy, pd.kyb, pd._spacing["y"]))
    dvx_dx, m_vxx = cpml_update(dvx_dx, m_vxx, ax, bx, (vx, pd.kxb, pd._spacing["x"]))
    dvx_dz, m_vxz = cpml_update(dvx_dz, m_vxz, azh, bzh, (vx, pd.kzf, pd._spacing["z"]))
    dvz_dx, m_vzx = cpml_update(dvz_dx, m_vzx, axh, bxh, (vz, pd.kxf, pd._spacing["x"]))

    dvx_dy, m_vxy = cpml_update(dvx_dy, m_vxy, ayh, byh, (vx, pd.kyf, pd._spacing["y"]))
    dvy_dx, m_vyx = cpml_update(dvy_dx, m_vyx, axh, bxh, (vy, pd.kxf, pd._spacing["x"]))
    dvy_dz, m_vyz = cpml_update(dvy_dz, m_vyz, azh, bzh, (vy, pd.kzf, pd._spacing["z"]))
    dvz_dy, m_vzy = cpml_update(dvz_dy, m_vzy, ayh, byh, (vz, pd.kyf, pd._spacing["y"]))

    div_v = dvx_dx + dvy_dy + dvz_dz

    sxx_pre_fs = sxx
    syy_pre_fs = syy
    sxx = sxx + dt * (lame_lambda * div_v + 2 * lame_mu * dvx_dx)
    syy = syy + dt * (lame_lambda * div_v + 2 * lame_mu * dvy_dy)
    szz = szz + dt * (lame_lambda * div_v + 2 * lame_mu * dvz_dz)
    sxy = sxy + dt * lame_mu * (dvx_dy + dvy_dx)
    sxz = sxz + dt * lame_mu * (dvx_dz + dvz_dx)
    syz = syz + dt * lame_mu * (dvy_dz + dvz_dy)

    if free_surface:
        if FS_MOD_SXX:
            # Robertsson tangential FS correction (3-D): at a z-low free surface
            # sigma_zz=0 => dvz_dz = -lambda/(lam+2mu) (dvx_dx + dvy_dy), so the
            # surface-row tangential normal stresses reduce to
            #   sigma_xx = coef*dvx_dx + coef2*dvy_dy   (x<->y for sigma_yy),
            # coef = 4 mu (lam+mu)/(lam+2mu), coef2 = 2 lam mu/(lam+2mu).  Without
            # it the image-method mirror alone leaves surface sigma_xx/yy too
            # large -> Rayleigh ~15% slow (2-D analogue: _elastic_step_core /
            # das.py DASMu-2D).
            _l2m = lame_lambda + 2.0 * lame_mu
            _coef = 4.0 * lame_mu * (lame_lambda + lame_mu) / _l2m
            _coef2 = 2.0 * lame_lambda * lame_mu / _l2m
            _sxx_surf = sxx_pre_fs + dt * (_coef * dvx_dx + _coef2 * dvy_dy)
            _syy_surf = syy_pre_fs + dt * (_coef2 * dvx_dx + _coef * dvy_dy)
            if topo_rows is not None:
                sxx = overwrite_at_topo(sxx, _sxx_surf, topo_rows, axis=-3)
                syy = overwrite_at_topo(syy, _syy_surf, topo_rows, axis=-3)
            else:
                sxx = overwrite_top_row(sxx, _sxx_surf, top_halo, axis=-3)
                syy = overwrite_top_row(syy, _syy_surf, top_halo, axis=-3)
        if topo_rows is not None:
            # z-low free surface: zero ONLY the normal stress szz (its node is
            # on the surface).  sxz/syz live +h/2 inside the medium, so zeroing
            # them is a spurious over-constraint that adds ~6% Rayleigh grid
            # dispersion (Kristek 2002 Table 1; cf. Elastic._fs_zero_traction).
            szz = zero_at_topo(szz, topo_rows, axis=-3)
        else:
            szz = zero_top_row(szz, top_halo, axis=-3)
    
    out = (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
           m_vxx, m_vxy, m_vxz,
           m_vyx, m_vyy, m_vyz,
           m_vzx, m_vzy, m_vzz,
           m_sxxx, m_szzz,
           m_sxyx, m_sxyy,
           m_sxzx, m_sxzz,
           m_syyy,
           m_syzy, m_syzz)
    if return_gradients:
        # The CPML-corrected velocity gradients the stress update just consumed.
        # DASMu3D derives its strain-rate state from exactly these; exporting
        # them is what lets its step be this step plus six strain lines instead
        # of a 130-line copy. Nothing is recomputed and nothing reordered, so
        # the flag cannot perturb the default path.
        return out + ((dvx_dx, dvy_dy, dvz_dz, dvx_dy, dvy_dx,
                       dvx_dz, dvz_dx, dvy_dz, dvz_dy),)
    return out


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
         return_gradients=False,
         ):
    """One leapfrog step: :func:`velocity_substep`, then :func:`stress_substep`
    (``Elastic.interior_substeps`` reuses the two halves)."""
    fields = velocity_substep(vx, vy, vz, sxx, syy, szz, sxy, sxz, syz,
                              m_vxx, m_vxy, m_vxz,
                              m_vyx, m_vyy, m_vyz,
                              m_vzx, m_vzy, m_vzz,
                              m_sxxx, m_szzz,
                              m_sxyx, m_sxyy,
                              m_sxzx, m_sxzz,
                              m_syyy,
                              m_syzy, m_syzz,
                              rho=rho, dt=dt, pd=pd, pml=pml,
                              free_surface=free_surface, topo_rows=topo_rows)
    return stress_substep(*fields, lame_lambda=lame_lambda, lame_mu=lame_mu, dt=dt, pd=pd, pml=pml,
                          free_surface=free_surface, topo_rows=topo_rows,
                          return_gradients=return_gradients)


@register_equation('Elastic3D')
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
    C_NAME = "elastic3d"

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
        FieldSpec("m_vxx", description="CPML memory variable for dvx/dx.", internal=True, boundary_related=True, cpml_axis="x"),
        FieldSpec("m_vxy", description="CPML memory variable for dvx/dy.", internal=True, boundary_related=True, cpml_axis="y"),
        FieldSpec("m_vxz", description="CPML memory variable for dvx/dz.", internal=True, boundary_related=True, cpml_axis="z"),
        FieldSpec("m_vyx", description="CPML memory variable for dvy/dx.", internal=True, boundary_related=True, cpml_axis="x"),
        FieldSpec("m_vyy", description="CPML memory variable for dvy/dy.", internal=True, boundary_related=True, cpml_axis="y"),
        FieldSpec("m_vyz", description="CPML memory variable for dvy/dz.", internal=True, boundary_related=True, cpml_axis="z"),
        FieldSpec("m_vzx", description="CPML memory variable for dvz/dx.", internal=True, boundary_related=True, cpml_axis="x"),
        FieldSpec("m_vzy", description="CPML memory variable for dvz/dy.", internal=True, boundary_related=True, cpml_axis="y"),
        FieldSpec("m_vzz", description="CPML memory variable for dvz/dz.", internal=True, boundary_related=True, cpml_axis="z"),
        FieldSpec("m_sxxx", description="CPML memory variable for dsxx/dx.", internal=True, boundary_related=True, cpml_axis="x"),
        FieldSpec("m_szzz", description="CPML memory variable for dszz/dz.", internal=True, boundary_related=True, cpml_axis="z"),
        FieldSpec("m_sxyx", description="CPML memory variable for dsxy/dx.", internal=True, boundary_related=True, cpml_axis="x"),
        FieldSpec("m_sxyy", description="CPML memory variable for dsxy/dy.", internal=True, boundary_related=True, cpml_axis="y"),
        FieldSpec("m_sxzx", description="CPML memory variable for dsxz/dx.", internal=True, boundary_related=True, cpml_axis="x"),
        FieldSpec("m_sxzz", description="CPML memory variable for dsxz/dz.", internal=True, boundary_related=True, cpml_axis="z"),
        FieldSpec("m_syyy", description="CPML memory variable for dsyy/dy.", internal=True, boundary_related=True, cpml_axis="y"),
        FieldSpec("m_syzy", description="CPML memory variable for dsyz/dy.", internal=True, boundary_related=True, cpml_axis="y"),
        FieldSpec("m_syzz", description="CPML memory variable for dsyz/dz.", internal=True, boundary_related=True, cpml_axis="z"),
    )

    default_pml_type = "cpmls"  # staggered-grid CPML: step() unpacks 12 profiles
    # The eager boundary-saving sub-steps carry the free-surface BC (the image
    # derivatives and the surface rows of stress_substep).
    supports_bs_free_surface = True
    supports_image_topography = True   # eager func applies the staircase
    # supports_image_topography_c stays False: the CUDA kernels never read
    # the topo rows (topo3d_gradient_mode_suite's docstring says the same).

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
    
    @staticmethod
    def _moduli(models):
        """``(vp, vs, rho, lame_lambda, lame_mu)`` from the raw or prepared models."""
        if len(models) == 5:
            return tuple(models)
        if len(models) == 3:
            vp, vs, rho = models
            return vp, vs, rho, rho * (vp**2 - 2 * vs**2), rho * vs**2
        raise ValueError(f"Elastic3D.func expected 3 or 5 models, got {len(models)}")

    def func(self, wavefields, models, dt, h, b, **kwargs):
        vp, vs, rho, lame_lambda, lame_mu = self._moduli(models)

        # Image-method / flat step.  ``topo_rows`` is 2-D
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

    def interior_substeps(self):
        """The leapfrog as two reversible sub-steps, REUSING the
        ``velocity_substep`` / ``stress_substep`` that :func:`step` (and thus
        ``func``) is built from -- no rewritten copy of the physics. Called with
        ``pml`` zeroed (boundary saving reconstructs only the lossless interior)
        but the free-surface BC kept, so the eager boundary-saving 'substep'
        driver inverts the step exactly by composing the list in reverse order
        at ``-dt``."""
        pd = self.pd
        fs = bool(getattr(self, "free_surface", False))
        topo = getattr(self, "_topo_rows_runtime", None)
        kw = dict(pd=pd, pml=(0.0,) * 12, free_surface=fs, topo_rows=topo)

        def sub_v(wf, models, dt, h):
            return list(velocity_substep(*wf, rho=self._moduli(models)[2], dt=dt, **kw))

        def sub_s(wf, models, dt, h):
            _, _, _, lam, mu = self._moduli(models)
            return list(stress_substep(*wf, lame_lambda=lam, lame_mu=mu, dt=dt, **kw))

        return [sub_v, sub_s]   # forward order; reverse = reversed(...) at -dt

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            record_shape=record_multi(),
            save_all_shape=history_fields(3),   # vx, vy, vz
            base_nvar=9,
            pml_nvar=27,
            last_two_nvar=1,
            last_two_storage_nvar=9,
            # The compiled backward's scratch (common/elastic.h
            # ElasticAdjointWorkspaceTensor), one padded grid per shot each:
            # [0-17]  = the nine q* then the nine p*, the stress- /
            #           velocity-adjoint prepare->apply scratch (every mode);
            # [18-20] = next_segment_v (ckpt) / current_v (recursive): the v(it)
            #           carriers the imaging reads, vx / vy / vz;
            # [21-23] = prev_segment_next_v (ckpt) / next_v (recursive): the
            #           v(it+1) carriers, vx / vy / vz
            # -- [18-23] in the checkpoint modes only (sg_driver.cuh
            # SgCarrierSlots after elastic3d WS_CARRIERS = 18); the full mode
            # keeps one read-only zero grid at [18] instead (v(nt)).
            backward_workspace_nvar=18,
            backward_workspace_shapes=lambda B, nt, shape, mode: [[B, 1, *shape]] * (
                24 if mode in ("ckpt", "recursive") else 19 if mode == "full" else 18),
            # The chunked backward's replayed velocity histories (elastic3d
            # driver_traits seg_buffers): vx, vy, vz of one chunk, (interval + 1,
            # B, 1, grid) each -- row 0 = v(start), row k = v(start + k).  The
            # recursive mode replays per step into the carriers above and keeps none.
            checkpoint_replay_shapes=lambda B, nt, grid, seg, mode: (
                [(seg + 1, B, 1, *grid)] * 3 if mode == "ckpt" else []),
            derived_model_nvar=2,   # mu, lambda (common/derived_models.h LameSlot)
            # CPML memory variables in C++ bind order: nine (x, y, z) triples
            # (m_vx*, m_vy*, m_vz*, m_sxx*, m_syy*, m_szz*, m_sxy*, m_sxz*,
            # m_syz*); the differencing axis is the name's last letter.  All
            # elastic aux access is own-cell, so the adjoint uses slabs too.
            pml_slot_axes=("x", "y", "z") * 9,
            checkpoint_slot_axes=(None,) * 9 + ("x", "y", "z") * 9,
            adjoint_pml_slab=True,
            slots=slot_table.ELASTIC3D,
            stepped=True,
            dd_backward_phases=True,
            grads_out_has_wavelet=False,
            illum_nvar=0,
        )
