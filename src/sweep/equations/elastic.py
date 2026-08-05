from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from ._elastic_step_core import (
    elastic_step_core,
    elastic_velocity_substep,
    elastic_stress_substep,
)
from ._free_surface import (
    zero_at_topo,
    zero_top_row,
    zero_surface_row,
    sides_from_fs_faces_2d,
)
from ._topography import (
    classify_topography,
    enforce_apm_traction_bc,
    precompute_apm_moduli,
    zero_at_air,
)


class Elastic(FirstOrderEquation):
    """First-order 2-D elastic wave equation on a staggered grid (Virieux 1986).

    Velocity-stress formulation. The five physical fields
    ``(vx, vz, sxx, szz, sxz)`` are evolved together with ten CPML memory
    variables (one per first-derivative direction). Sources are typically an
    explosion (`['sxx', 'szz']` — the default in the propagator) or a
    directional body force; receivers usually read particle velocities
    (`['vx']` or `['vz']`) or stresses.

    Reference: J. Virieux, 1986, *P-SV wave propagation in heterogeneous
    media: velocity-stress finite-difference method*, Geophysics 51(4),
    [10.1190/1.1442147](https://doi.org/10.1190/1.1442147).

    
    """
    MODEL_SPECS = (
        ModelSpec("vp", aliases=("p_velocity",), description="Elastic P-wave velocity model.", unit="m/s"),
        ModelSpec("vs", aliases=("s_velocity",), description="Elastic S-wave velocity model.", unit="m/s"),
        ModelSpec("rho", aliases=("density",), description="Density model.", unit="kg/m^3"),
    )
    FIELD_SPECS = (
        FieldSpec("vx", aliases=("velocity_x",), description="Particle velocity in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("vz", aliases=("velocity_z",), description="Particle velocity in the z direction.", supports_source=True, supports_receiver=True),
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

    default_pml_type = "cpmls"  # staggered-grid CPML: step() unpacks 8 profiles
    supports_apm = True          # APM (Cao & Chen 2018) dispatch in ``func``
    # Eager ``func`` honours a per-edge free surface (``self.fs_faces``): the
    # staggered image method on any subset of the 4 faces, with the z<->x
    # Robertsson analogue and traction zeroing per face.  See ``_fs_zero_traction``.
    supports_per_edge_free_surface = True
    # The 2-D Elastic CUDA kernels honour a per-edge free surface on ALL four
    # faces (top/bottom/left/right, incl. z∩x corners) via the generalized image
    # mirror + hand-written adjoint (z and x) in ``elastic_free_surface.cuh``.
    supports_per_edge_free_surface_c = True
    # interior_substeps carry the free-surface BC, so eager boundary saving may
    # run with free_surface=True (the propagator's guard skips this equation).
    supports_bs_free_surface = True

    # The 2-D elastic CUDA kernels (forward / adjoint / gradient) stride EVERY
    # model buffer per batch index ``b`` (``rho + b*nz*nx``, ``lambda + b*nz*nx``,
    # ``mu + b*nz*nx``) and the C wire derives ``lambda``/``mu`` from vp/vs/rho
    # with element-wise ops that preserve a leading batch axis. A per-shot
    # batched model ``(B, nz, nx)`` for each of vp/vs/rho is therefore supported
    # directly by ``impl='c'``: shot ``b`` propagates in its own (vp,vs,rho)[b]
    # and each parameter's gradient is kept per-shot. Same for the eager path.
    supports_batched_models = True

    def __init__(self, spatial_order=4, device='cpu', backend='torch'):
        """Build the elastic equation operator.

        Args:
            spatial_order: FD accuracy order of the staggered first-derivative operator — e.g.
                ``spatial_order=4`` is fourth-order accurate.
                Internally the half-stencil width is
                ``M = spatial_order // 2`` (used for loop bounds and PML padding). Must
                be an even integer (``2, 4, 6, 8, 10, …``). Higher orders
                reduce grid dispersion — most visibly on the slower S-wave,
                which is the first thing to alias at marginal ppw — at the
                cost of more compute per step and a wider PML halo.
                **Performance note (`impl='c'` on CUDA):** the compiled
                kernels are template-specialised only for
                ``spatial_order ∈ {2, 4, 6, 8}``. Above 8 the dispatcher
                falls through to the generic ``order = -1`` runtime path
                (see ``src/sweep/csrc/cuda/equations/elastic2d/forward.cu``)
                which is noticeably slower; the PyTorch eager path is
                unaffected. Defaults to 4.
            device: Device for the operator's static gradient kernels.
                Use ``'cuda'`` / a ``torch.device`` for GPU runs so the
                propagator can follow without a host↔device copy.
                Defaults to ``'cpu'``.
            backend: Array / programming backend, ``'torch'`` or ``'jax'``.
                When you later want ``impl='c'``, leave this on ``'torch'``
                — the compiled CUDA kernels go through the Torch binding.
                Defaults to ``'torch'``.
        """
        super().__init__(spatial_order, device, backend)
        # APM (Cao & Chen 2018) cache.  The propagator's ``_process_topography``
        # attaches ``_apm_air_mask_runtime`` when the user passes a 2-D
        # ``topography=`` array; ``func`` then computes the per-category
        # effective moduli once per (air_mask, models) pair and reuses them
        # across every time step.  ``cache_key`` keys on the tuple of
        # ``id()`` of (air_mask, vp, vs, rho).
        self._apm_cache_key = None
        self._apm_cache = None

    @property
    def default_source_fields(self):
        return ["sxx", "szz"]

    @property
    def default_receiver_fields(self):
        return ["vx", "vz"]

    @staticmethod
    def _lame(models):
        """Normalise the model list to the canonical 6-tuple
        ``(vp, vs, rho, λ, μ, λ+2μ)`` — the single source of truth for the
        vp/vs→Lamé mapping, shared by ``func``, ``prepare_models`` and
        ``interior_substeps``.

        Accepts three levels of preparation:
          * 3  ``(vp, vs, rho)``                     — raw physical models;
          * 5  ``(vp, vs, rho, λ, μ)``               — λ+2μ derived;
          * 6  ``(vp, vs, rho, λ, μ, λ+2μ)``         — already prepared (the
            runtime path; returned as-is, so calling this per step is a cheap
            no-op and does NOT recompute the moduli).
        """
        n = len(models)
        if n == 6:
            return tuple(models)
        if n == 5:
            vp, vs, rho, lam, mu = models
            return vp, vs, rho, lam, mu, lam + 2 * mu
        if n == 3:
            vp, vs, rho = models
            lam = rho * (vp ** 2 - 2 * vs ** 2)
            mu = rho * vs ** 2
            return vp, vs, rho, lam, mu, lam + 2 * mu
        raise ValueError(f"Elastic models must be a 3-, 5- or 6-tuple, got {n}")

    def prepare_models(self, models):
        return list(self._lame(models))

    def func(self, wavefields, models, dt, h, b, **kwargs):
        vp, vs, rho, lame_lambda, lame_mu, lame_lambda_2mu = self._lame(models)

        # ---- APM dispatch ---------------------------------------------------
        # If the propagator's ``topography=`` was a 2-D air_mask it stored a
        # padded runtime version on the equation.  Re-route through the
        # parameter-modified step (Cao & Chen 2018).
        air_mask_rt = getattr(self, "_apm_air_mask_runtime", None)
        if air_mask_rt is not None:
            return self._func_apm(
                wavefields, lame_lambda, lame_mu, rho, air_mask_rt,
                dt, h, b, **kwargs,
            )

        # ---- Standard (image-method / flat) dispatch -----------------------
        # The wave equation (Virieux 1986 first-order velocity-stress) is
        # solved by ``elastic_step_core`` — shared verbatim with the APM
        # branch above.  For the bulk path we alias ρ_x = ρ_z = ρ and
        # μ_xz = μ (no node-centred averaging, no per-cell modifications).
        free_surface = getattr(self, "free_surface", False)
        topo_rows = getattr(self, "_topo_rows_runtime", None)
        out = elastic_step_core(
            *wavefields,
            lame_lambda=lame_lambda,
            lame_mu=lame_mu,
            mu_xz=lame_mu,
            rho_x=rho,
            rho_z=rho,
            dt=dt, h=h, b=b, pd=self.pd, pml=self.b,
            free_surface=free_surface,
            topo_rows=topo_rows,
            fs_faces=getattr(self, "fs_faces", None),
            lame_lambda_2mu=lame_lambda_2mu,
        )
        (vx, vz, sxx, szz, sxz,
         m_vxx, m_vxz, m_vzx, m_vzz,
         m_txxx, m_txxz, m_tzzx, m_tzzz,
         m_txzx, m_txzz) = out

        # Image-method post-step traction zeroing (per active free-surface face).
        if free_surface:
            szz, sxz, sxx = self._fs_zero_traction(szz, sxz, sxx, topo_rows)

        return (
            vx, vz, sxx, szz, sxz,
            m_vxx, m_vxz, m_vzx, m_vzz,
            m_txxx, m_txxz, m_tzzx, m_tzzz,
            m_txzx, m_txzz,
        )

    def _fs_zero_traction(self, szz, sxz, sxx, topo_rows):
        """Zero the traction components at each active free-surface face: the
        shear ``sxz`` at every face; the normal ``szz`` at z-faces (top/bottom);
        the normal ``sxx`` at x-faces (left/right).  For the top-only default the
        z-low branch is bit-identical to ``zero_top_row(szz/sxz, ...)``."""
        top_halo = self.pd.coes.shape[0]
        if topo_rows is not None:
            szz = zero_at_topo(szz, topo_rows, axis=-2)
            sxz = zero_at_topo(sxz, topo_rows, axis=-2)
            return szz, sxz, sxx
        fs_faces = getattr(self, "fs_faces", None)
        if fs_faces is None:
            szz = zero_top_row(szz, top_halo, axis=-2)
            sxz = zero_top_row(sxz, top_halo, axis=-2)
            return szz, sxz, sxx
        z_sides, x_sides = sides_from_fs_faces_2d(fs_faces)
        for side in z_sides:
            szz = zero_surface_row(szz, top_halo, axis=-2, side=side)
            sxz = zero_surface_row(sxz, top_halo, axis=-2, side=side)
        for side in x_sides:
            sxx = zero_surface_row(sxx, top_halo, axis=-1, side=side)
            sxz = zero_surface_row(sxz, top_halo, axis=-1, side=side)
        return szz, sxz, sxx

    def interior_substeps(self):
        """The Virieux leapfrog as reversible sub-steps, REUSING the shared
        ``elastic_velocity_substep`` / ``elastic_stress_substep`` that
        :func:`elastic_step_core` (and thus ``func``) is built from — no
        rewritten copy of the physics.  Called with ``pml`` zeroed (boundary
        saving reconstructs only the lossless interior) but the free-surface BC
        KEPT, so the eager boundary-saving 'substep' driver inverts the step
        exactly by composing the list in reverse order at ``-dt``."""
        pd = self.pd
        fs = bool(getattr(self, "free_surface", False))
        topo = getattr(self, "_topo_rows_runtime", None)
        top_halo = pd.coes.shape[0]
        pml_zero = (0.0,) * 8

        fs_faces = getattr(self, "fs_faces", None)

        def _kw(models, dt, h):
            _, _, rho, lam, mu, lam2mu = self._lame(models)
            return dict(
                lame_lambda=lam, lame_mu=mu, mu_xz=mu, rho_x=rho, rho_z=rho,
                dt=dt, h=h, b=None, pd=pd, pml=pml_zero,
                free_surface=fs, topo_rows=topo, fs_faces=fs_faces,
                lame_lambda_2mu=lam2mu,
            )

        def sub_v(wf, models, dt, h):
            return list(elastic_velocity_substep(*wf, **_kw(models, dt, h)))

        def sub_s(wf, models, dt, h):
            out = list(elastic_stress_substep(*wf, **_kw(models, dt, h)))
            if fs:  # post-step free-surface traction zeroing (mirrors ``func``)
                # out order (vx, vz, sxx, szz, sxz, ...): [2]=sxx [3]=szz [4]=sxz
                out[3], out[4], out[2] = self._fs_zero_traction(out[3], out[4], out[2], topo)
            return out

        return [sub_v, sub_s]   # forward order; reverse = reversed(...) at -dt

    def _func_apm(self, wavefields, lame_lambda, lame_mu, rho, air_mask_rt,
                  dt, h, b, **kwargs):
        """One leapfrog step with APM-modified moduli (Cao & Chen 2018).

        Caches the per-category effective moduli by the id() of the input
        tensors — the same (air_mask, λ, μ, ρ) inputs hit the cache on
        every subsequent time step within a propagator run.
        """
        cache_key = (id(air_mask_rt), id(lame_lambda), id(lame_mu), id(rho))
        if cache_key == self._apm_cache_key and self._apm_cache is not None:
            cat, lam_eff, mu_eff, mu_xz, rho_x, rho_z = self._apm_cache
        else:
            cat = classify_topography(air_mask_rt)
            lam_eff, mu_eff, mu_xz, rho_x, rho_z = precompute_apm_moduli(
                lame_lambda, lame_mu, rho, cat
            )
            self._apm_cache_key = cache_key
            self._apm_cache = (cat, lam_eff, mu_eff, mu_xz, rho_x, rho_z)

        out = elastic_step_core(
            *wavefields,
            lame_lambda=lam_eff,
            lame_mu=mu_eff,
            mu_xz=mu_xz,
            rho_x=rho_x,
            rho_z=rho_z,
            dt=dt, h=h, b=b, pd=self.pd, pml=self.b,
            free_surface=False,    # APM IS the free surface
            topo_rows=None,
        )
        (vx, vz, sxx, szz, sxz,
         m_vxx, m_vxz, m_vzx, m_vzz,
         m_txxx, m_txxz, m_tzzx, m_tzzz,
         m_txzx, m_txzz) = out

        # APM traction-free BC: pointwise stress zero per cell type.
        sxx, szz, sxz = enforce_apm_traction_bc(sxx, szz, sxz, cat)
        # Vacuum approximation: kill any wavefield in pure-air cells.
        vx = zero_at_air(vx, air_mask_rt)
        vz = zero_at_air(vz, air_mask_rt)
        sxx = zero_at_air(sxx, air_mask_rt)
        szz = zero_at_air(szz, air_mask_rt)
        sxz = zero_at_air(sxz, air_mask_rt)

        return (
            vx, vz, sxx, szz, sxz,
            m_vxx, m_vxz, m_vzx, m_vzz,
            m_txxx, m_txxz, m_tzzx, m_tzzz,
            m_txzx, m_txzz,
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

    def _C_apm(self):
        """CUDA APM (Cao & Chen 2018) entry points.  Forward is fully
        implemented; backward is a stub — gradients should be computed
        via the eager autograd path (see :func:`_func_apm`)."""
        import sweep._C as _C
        return (
            _C.elastic2d_apm_forward,
            _C.elastic2d_apm_backward,
            _C.elastic2d_apm_backward_bs,
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
