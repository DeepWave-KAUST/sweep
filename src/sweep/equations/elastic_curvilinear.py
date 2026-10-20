"""Elastic 2-D velocity-stress modelling on a boundary-fitted curvilinear
grid, following

    Bartana, A., Xu, J. & Rickett, J. (SLB) -- *Solving the elastodynamic
    equations in curvilinear coordinates using asymmetric operators*.

Three ingredients, all different from the earlier staggered chain-rule
implementation of this class:

1. **Conservative form on a collocated grid.**  All five fields live at
   the same grid points.  The momentum equation is discretised in the
   divergence form of the paper's eq. (3),

       rho dv_i/dt = (1/J) d/dzeta_m ( J T_mj sigma_ij )
       dsigma_ij/dt = c_ijkl T_ml dv_k/dzeta_m

   where ``T = F^-1`` is the inverse of the forward mapping and
   ``J = det F``.  For the boundary-fitted stretch used by
   :class:`sweep.utils.curvilinear.CurvilinearGrid`
   (``x = xi``, ``z = h(xi) + D(xi) eta``) this is ``T_31 = alpha``,
   ``T_33 = beta = 1/D`` and ``J = D``.  Writing the momentum equation as
   a divergence -- rather than expanding the chain rule -- is what makes
   the spatial operator self-adjoint, and hence the scheme stable.

2. **Asymmetric operators** (:mod:`sweep.operators.asymmetric`) instead of
   staggering.  A collocated grid differentiated centrally decouples odd
   and even points; the asymmetric stencil has a non-vanishing symbol at
   Nyquist and removes that null mode.  The gradient operator acts on the
   velocities, its negative transpose on the stress fluxes -- the pairing
   the paper refers to when it calls the operator self-adjoint.

3. **Free surface by tensor rotation, no ghost points.**  The top row of
   the computational grid *is* the topography.  There the stress rate is
   not taken from the wave equation but built in the frame aligned with
   the surface normal: the normal strain rate is eliminated with the
   paper's eq. (10), ``dv3'/dx3' = -(c13/c33) dv1'/dx1'``, giving
   ``sigma'_33 = sigma'_13 = 0`` and

       dsigma'_11/dt = (c11 - c13^2/c33) dv1'/dx1'

   which is then rotated back.  For the isotropic case this is
   ``M_surf = 4 mu (lambda + mu) / (lambda + 2 mu)``.  The tangential
   strain rate uses only derivatives *along* the surface, so no value is
   ever needed above it.

Why the rotated *rate* rather than a rotated *value*.  The previous
implementation enforced the same rotated traction condition by
overwriting ``sigma_xz`` / ``sigma_zz`` on the surface row after a normal
update.  The wave equation does not drive the stresses toward the
constraint by itself, so the per-step difference between the two acted as
an energy source and made the instability worse, not better (this is
observed on the old implementation).  Updating the surface
row with a *rate* that is rank-1 along the surface tangent keeps the
traction identically zero for all time once it starts at zero: the
constraint is preserved, not re-imposed.
"""

from __future__ import annotations

import numpy as np

from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from ._registry import register_equation


def _tangent_image(vx, vz, geom):
    """Fill the rows above the surface with the velocity reflected about
    the surface tangent: ``v_ghost = (2 t t^T - I) v``.

    For a flat surface this is ``vx`` even and ``vz`` odd -- the standard
    elastic image -- and it stays correct on a dipping one because the
    reflection follows the tangent.
    """
    import torch

    surf = geom["surface_row"]
    hp, il2 = geom["h_prime"], geom["inv_len2"]
    a = (1.0 - hp * hp) * il2          # 2 t1^2 - 1
    b = 2.0 * hp * il2                 # 2 t1 t3
    src_x = vx[..., surf + 1:surf + 1 + surf, :].flip(-2)
    src_z = vz[..., surf + 1:surf + 1 + surf, :].flip(-2)
    ghost_x = a * src_x + b * src_z
    ghost_z = b * src_x - a * src_z
    return (torch.cat([ghost_x, vx[..., surf:, :]], dim=-2),
            torch.cat([ghost_z, vz[..., surf:, :]], dim=-2))


def step_curv_asym(
    vx, vz, sxx, szz, sxz,
    m_vxx, m_vxz, m_vzx, m_vzz,
    m_txxx, m_txxz, m_tzzx, m_tzzz,
    m_txzx, hg_x, hg_z,
    vp, vs, rho,
    lame_lambda, lame_mu,
    dt, h, b, pd,
    pml=None,
    free_surface=False,
    lame_lambda_2mu=None,
    op=None,
    geom=None,
    closure="shifted",
    damping=None,
    damp_velocity=False,
    hourglass=None,
):
    """One collocated velocity-stress step in the computational (xi, eta) grid.

    ``op`` is a
    :class:`sweep.operators.asymmetric.CollocatedAsymmetricOperator` and
    ``geom`` the derived metric bundle built by
    :meth:`ElasticCurvilinear._geometry`.
    """
    az, bz, azh, bzh, ax, bx, axh, bxh = pml
    lame_lambda_2mu = (lame_lambda + 2 * lame_mu
                       if lame_lambda_2mu is None else lame_lambda_2mu)

    d_xi, d_eta = geom["d_xi"], geom["d_eta"]
    jac, jalpha = geom["jac"], geom["jalpha"]
    alpha, beta = geom["alpha"], geom["beta"]
    surf = geom["surface_row"] if free_surface else None
    below = geom["below"]

    # ---------------------------------------------------------------
    # Momentum:  rho dv_i/dt = (1/J) d_zeta_m ( J T_mj sigma_ij )
    #
    # xi-flux   F1_i = J T_1j sigma_ij = J sigma_i1   (T_11 = 1, T_13 = 0)
    # eta-flux  F3_i = J T_3j sigma_ij = J alpha sigma_i1 + sigma_i3
    #                                    (J beta == 1 by construction)
    # F3 is the surface traction times J|grad zeta_3|, so the free-surface
    # condition makes it vanish on the surface row; above it the medium is
    # void and every field is held at zero.
    # ---------------------------------------------------------------
    f1x = jac * sxx
    f1z = jac * sxz
    f3x = jalpha * sxx + sxz
    f3z = jalpha * sxz + szz

    flux_mode = {"shifted": "shifted", "zero": "zero",
                 "mirror": "mirror_odd", "mirror_full": "mirror_odd"}[closure]
    d_f1x = op.d_x_adj(f1x) / d_xi
    d_f3x = op.d_z_adj(f3x, surf, flux_mode) / d_eta
    d_f1z = op.d_x_adj(f1z) / d_xi
    d_f3z = op.d_z_adj(f3z, surf, flux_mode) / d_eta

    m_txxx = ax * m_txxx + bx * d_f1x
    d_f1x = d_f1x + m_txxx
    m_txxz = az * m_txxz + bz * d_f3x
    d_f3x = d_f3x + m_txxz
    m_tzzx = ax * m_tzzx + bx * d_f1z
    d_f1z = d_f1z + m_tzzx
    m_tzzz = az * m_tzzz + bz * d_f3z
    d_f3z = d_f3z + m_tzzz

    acc = dt / (rho * jac)
    if free_surface and geom.get("mass_weight") is not None:
        acc = acc * geom["mass_weight"]
    vx = vx + acc * (d_f1x + d_f3x)
    vz = vz + acc * (d_f1z + d_f3z)
    if free_surface and hourglass is not None:
        # Hourglass stiffness (see ``hourglass_frequency``): the pair
        #     v <- v - dt a S h ,   h <- h + dt a S v ,   S symmetric
        # is symplectic, so this adds no dissipation at all -- it only
        # raises omega for the vertically grid-scale modes that S sees.
        rate, apply_s = hourglass
        vx = vx - rate * apply_s(hg_x)
        vz = vz - rate * apply_s(hg_z)
    if free_surface:
        vx = vx * below
        vz = vz * below
        if damping is not None and damp_velocity:
            vx = damping(vx)
            vz = damping(vz)

    # ---------------------------------------------------------------
    # Constitutive law:  dsigma_ij/dt = c_ijkl T_ml dv_k/dzeta_m
    # ---------------------------------------------------------------
    vx_xi = op.d_x(vx) / d_xi
    vz_xi = op.d_x(vz) / d_xi
    if closure == "mirror_full":
        # Image the velocity as well: reflect the vector about the surface
        # tangent, which sends the tangential component to an even image
        # and the normal component to an odd one -- the classical elastic
        # image method, written so it also holds on a dipping surface.
        # With the ghost rows filled the full-width interior stencil is
        # usable at every row, so no row runs at reduced order.
        vx_g, vz_g = _tangent_image(vx, vz, geom)
        vx_eta = op.d_z(vx_g) / d_eta
        vz_eta = op.d_z(vz_g) / d_eta
    else:
        # No exact continuation for the velocity, so the one-sided
        # (``shifted``) stencils are used even when the flux uses its
        # exact odd image.
        grad_mode = "zero" if closure == "zero" else "shifted"
        vx_eta = op.d_z(vx, surf, grad_mode) / d_eta
        vz_eta = op.d_z(vz, surf, grad_mode) / d_eta

    m_vxx = ax * m_vxx + bx * vx_xi
    vx_xi = vx_xi + m_vxx
    m_vxz = az * m_vxz + bz * vx_eta
    vx_eta = vx_eta + m_vxz
    m_vzx = ax * m_vzx + bx * vz_xi
    vz_xi = vz_xi + m_vzx
    m_vzz = az * m_vzz + bz * vz_eta
    vz_eta = vz_eta + m_vzz

    dvx_dx = vx_xi + alpha * vx_eta
    dvx_dz = beta * vx_eta
    dvz_dx = vz_xi + alpha * vz_eta
    dvz_dz = beta * vz_eta

    rate_xx = lame_lambda_2mu * dvx_dx + lame_lambda * dvz_dz
    rate_zz = lame_lambda * dvx_dx + lame_lambda_2mu * dvz_dz
    rate_xz = lame_mu * (dvx_dz + dvz_dx)

    if free_surface:
        # Rotated traction-free stress rate on the surface row.  With
        # t the unit surface tangent, the only surviving component is
        # sigma'_11, so the rate is rank one along t (x) t and the
        # traction stays zero for all time.
        hp = geom["h_prime"]
        inv_len = geom["inv_len"]        # 1 / sqrt(1 + h'^2)
        inv_len2 = geom["inv_len2"]      # 1 / (1 + h'^2)
        v_t = (vx + hp * vz) * inv_len
        dvt_xi = op.d_x(v_t) / d_xi
        m_txzx = ax * m_txzx + bx * dvt_xi
        dvt_xi = dvt_xi + m_txzx
        # d(v.t)/ds = eps'_11 + v . dt/ds  ->  subtract the curvature term
        eps_tt = inv_len * dvt_xi - (vx * geom["tau_x"] + vz * geom["tau_z"])
        m_surf = 4.0 * lame_mu * (lame_lambda + lame_mu) / lame_lambda_2mu
        a_surf = m_surf * eps_tt
        mask = geom["surf_mask"]
        rate_xx = rate_xx + mask * (a_surf * inv_len2 - rate_xx)
        rate_xz = rate_xz + mask * (a_surf * hp * inv_len2 - rate_xz)
        rate_zz = rate_zz + mask * (a_surf * hp * hp * inv_len2 - rate_zz)

    sxx = sxx + dt * rate_xx
    szz = szz + dt * rate_zz
    sxz = sxz + dt * rate_xz
    if free_surface:
        sxx = sxx * below
        szz = szz * below
        sxz = sxz * below

    if free_surface and damping is not None:
        # Hourglass control for the free-surface closure.  The closure
        # cannot be both a consistent derivative and the exact negative
        # transpose of the gradient (an identity-norm summation-by-parts
        # closure does not exist for an asymmetric interior operator), and
        # the residual supports a near-static stress mode in the first
        # rows that oscillates at high wavenumber and grows slowly.  A
        # high-order Kreiss-Oliger filter, tapered into the surface band
        # and applied only to the stresses, removes it.  Its response is
        # ``1 - kappa sin^4(theta/2)`` per direction, so at 20 points per
        # wavelength it costs ~1e-6 per step and at Nyquist ~kappa.
        sxx = damping(sxx)
        szz = damping(szz)
        sxz = damping(sxz)

    if free_surface and hourglass is not None:
        rate, apply_s = hourglass
        hg_x = hg_x + rate * apply_s(vx)
        hg_z = hg_z + rate * apply_s(vz)

    return (
        vx, vz, sxx, szz, sxz,
        m_vxx, m_vxz, m_vzx, m_vzz,
        m_txxx, m_txxz, m_tzzx, m_tzzz,
        m_txzx, hg_x, hg_z,
    )


@register_equation()
class ElasticCurvilinear(FirstOrderEquation):
    """Curvilinear-grid elastic 2-D for irregular topography.

    Collocated fields, asymmetric operators and a rotated traction-free
    surface -- see the module docstring.  Same field / source / receiver
    layout as :class:`Elastic`; the propagator attaches the metrics.

    Parameters
    ----------
    spatial_order : int
        ``2L``: the interior stencil has ``2L + 1`` coefficients and the
        propagator allocates an ``L``-cell halo.  The paper uses ``L = 8``
        (``spatial_order=16``), which is the default here.
    kmax_frac, nyquist_real, surface_reach : see
        :class:`sweep.operators.asymmetric.CollocatedAsymmetricOperator`.
    source_spread : int
        Number of binomial passes applied to the source / receiver
        footprint (``2`` -> a 5x5 stencil).  See
        :attr:`source_receiver_stencil`; ``0`` gives a bare point source.
    surface_damping : float
        Strength of the Kreiss-Oliger filter that suppresses the
        near-static high-wavenumber stress mode the free-surface closure
        supports (see :meth:`_surface_damper`).  ``0`` disables it, which
        is useful for showing that the mode is really there.  ``4e-3`` is
        the smallest value that turns the per-step amplification factor
        from ``1.00059`` into ``0.99998`` at a third of the CFL limit, but
        the *record* keeps improving well past that: on a flat half-space
        the trailing tail behind the Rayleigh arrival is 10.4% of peak at
        ``4e-3``, 4.2% at ``1e-2`` and 2.9% at the default ``3e-2``, and
        the agreement with the staggered image-method solver improves with
        it (cos 0.9962 -> 0.9989), so the filter is removing the spurious
        mode rather than the signal.  Being a per-step filter it needs
        raising if the run sits much closer to the CFL limit.
    damp_velocity : bool
        Apply the surface filter to the velocities as well as the stresses.
        The spurious surface mode is stress-dominated but it is *sustained*
        by the velocity-stress loop, so filtering only one side leaves the
        loop running: with the stresses alone the trailing tail on a flat
        half-space is 2.9% of peak, with both it is 1.5% -- the level of
        the staggered image-method solver on the same grid -- and the
        waveform agreement improves too (cos 0.9989 -> 0.9992).
    surface_damping_rows : int or None
        Depth of the filter taper in cells; ``None`` uses ``4L``.  The
        taper wants to be wider than the closure band it is cleaning up:
        at ``2L`` the flat-surface tail is 4.2% of peak, at ``4L`` it is
        2.9%, because a narrow taper is itself a sharp enough profile to
        scatter the surface wave.
    adjoint_boundary_weight : float
        Alternative to the filter -- see
        :class:`sweep.operators.asymmetric.CollocatedAsymmetricOperator`.
        Blending toward the exact adjoint also stabilises the closure but
        costs accuracy in the divergence right where the surface wave
        lives, so the filter is the default route.
    surface_closure : {'shifted', 'mirror', 'zero'}
        How the eta-stencils are closed at the free surface.  The paper
        says only that there are no ghost points and that stencil points
        above the surface contribute zero, which are two different
        prescriptions; all three readings are implemented.

        ``'shifted'`` takes "no ghost points" literally: the stencils are
        truncated to the rows at or below the surface.  ``'zero'`` takes
        "contribute zero" literally; it is first-order *inconsistent* --
        the eta-derivative of the traction flux on the surface row comes
        out ~60% low -- and is kept only as a control.

        ``'mirror'`` (the default) reflects the traction flux
        anti-symmetrically about the surface row.  That is exact through
        the quadratic term because the flux -- the surface traction --
        vanishes there, so it is a symmetry of a quantity known to be
        zero rather than an extrapolation, and on a boundary-fitted grid
        the reflection is about a flat row and costs nothing.  It departs
        from the paper's "there are no ghost points" wording, and it is
        the default because it measures better on every count: against
        the staggered image-method solver on a flat half-space it reaches
        cos 0.981 versus 0.938 for ``'shifted'``, its trailing tail is
        2.13% of peak versus 2.48%, and on the steep hill its
        near-surface band is 7.4x the interior rms versus 8.9x.  Keep
        ``source_spread`` at 2 or below with it: a 7x7 footprint and the
        mirror together were measurably worse than either alone.
    """

    MODEL_SPECS = (
        ModelSpec("vp", aliases=("p_velocity",),
                  description="Elastic P-wave velocity model.", unit="m/s"),
        ModelSpec("vs", aliases=("s_velocity",),
                  description="Elastic S-wave velocity model.", unit="m/s"),
        ModelSpec("rho", aliases=("density",),
                  description="Density model.", unit="kg/m^3"),
    )
    FIELD_SPECS = (
        FieldSpec("vx", aliases=("velocity_x",), description="Particle velocity in the x direction.", supports_receiver=True),
        FieldSpec("vz", aliases=("velocity_z",), description="Particle velocity in the z direction.", supports_receiver=True),
        FieldSpec("sxx", aliases=("stress_xx",), description="Normal stress in the x direction.", supports_source=True, supports_receiver=True),
        FieldSpec("szz", aliases=("stress_zz",), description="Normal stress in the z direction.", supports_source=True, supports_receiver=True),
        FieldSpec("sxz", aliases=("stress_xz", "shear_xz"), description="Shear stress component.", supports_source=True),
        FieldSpec("m_vxx", description="CPML memory for dvx/dxi.", internal=True, boundary_related=True),
        FieldSpec("m_vxz", description="CPML memory for dvx/deta.", internal=True, boundary_related=True),
        FieldSpec("m_vzx", description="CPML memory for dvz/dxi.", internal=True, boundary_related=True),
        FieldSpec("m_vzz", description="CPML memory for dvz/deta.", internal=True, boundary_related=True),
        FieldSpec("m_txxx", description="CPML memory for the xi-flux of sigma_x.", internal=True, boundary_related=True),
        FieldSpec("m_txxz", description="CPML memory for the eta-flux of sigma_x.", internal=True, boundary_related=True),
        FieldSpec("m_tzzx", description="CPML memory for the xi-flux of sigma_z.", internal=True, boundary_related=True),
        FieldSpec("m_tzzz", description="CPML memory for the eta-flux of sigma_z.", internal=True, boundary_related=True),
        FieldSpec("m_txzx", description="CPML memory for the free-surface tangential velocity derivative.", internal=True, boundary_related=True),
        FieldSpec("hg_x", description="Hourglass-stiffness auxiliary for vx.", internal=True, boundary_related=True),
        FieldSpec("hg_z", description="Hourglass-stiffness auxiliary for vz.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmls"

    def __init__(self, spatial_order=16, device="cpu", backend="torch",
                 kmax_frac=0.56, nyquist_real=-2.0, surface_reach=1,
                 surface_closure="mirror", surface_damping=3e-2,
                 surface_damping_rows=None, adjoint_boundary_weight=0.0,
                 project_adjoint=False, boundary_nyquist_real=-0.5,
                 source_spread=2, surface_mass_weight=1.0,
                 damp_velocity=True, hourglass_frequency=0.0,
                 hourglass_rows=None):
        if spatial_order % 2:
            raise ValueError(
                f"spatial_order must be even (it is 2L for an L-point "
                f"half-stencil); got {spatial_order}")
        super().__init__(spatial_order, device, backend)
        from sweep.operators.asymmetric import CollocatedAsymmetricOperator

        self.op = CollocatedAsymmetricOperator(
            half_width=spatial_order // 2,
            kmax_frac=kmax_frac,
            nyquist_real=nyquist_real,
            surface_reach=surface_reach,
            boundary_nyquist_real=boundary_nyquist_real,
            adjoint_boundary_weight=adjoint_boundary_weight,
            project_adjoint=project_adjoint,
        )
        self.source_spread = int(source_spread)
        self.damp_velocity = bool(damp_velocity)
        # Finite-volume weight of the surface node (its control volume is
        # half an interior one).  1.0 = the plain pointwise update.
        self.surface_mass_weight = float(surface_mass_weight)
        self.surface_damping = float(surface_damping)
        # Hourglass stiffness: an energy-conserving alternative to the
        # Kreiss-Oliger filter.  ``hourglass_frequency`` is the angular
        # frequency (in Hz) given to the vertically grid-scale modes; the
        # response goes as ``sin^8(theta/2)``, twice as selective as the
        # filter's ``sin^4``, so the resolved band is untouched.
        self.hourglass_frequency = float(hourglass_frequency)
        self.hourglass_rows = (
            spatial_order // 2 if hourglass_rows is None
            else int(hourglass_rows))
        self.surface_damping_rows = (
            4 * (spatial_order // 2) if surface_damping_rows is None
            else int(surface_damping_rows))
        self._curv_alpha = None
        self._curv_beta = None
        self._curv_d_eta = None
        self._curv_h_prime = None
        if surface_closure not in ("shifted", "mirror", "mirror_full", "zero"):
            raise ValueError(
                f"surface_closure must be 'shifted', 'mirror', 'mirror_full' "
                f"or 'zero'; got {surface_closure!r}")
        self.surface_closure = surface_closure
        self._geom_cache = None
        self.is_curvilinear = True

    # -- propagator hooks --------------------------------------------------

    def set_curvilinear_metrics(self, alpha, metric_pηη, metric_pη, d_eta):
        """Attach metric tensors.

        Only ``alpha`` (= ``T_31``) and ``d_eta`` come through this hook;
        ``beta`` (= ``T_33``) and ``h_prime`` are assigned directly by the
        propagator.  The combined ``metric_p*`` fields are acoustic-only
        and accepted for interface compatibility.
        """
        self._curv_alpha = alpha
        self._curv_d_eta = float(d_eta)
        self._geom_cache = None

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

    # -- derived geometry --------------------------------------------------

    @property
    def source_receiver_stencil(self):
        """Binomial footprint folded into source injection / receiver sampling.

        A spatial delta excites every wavenumber the grid carries, and near
        the free surface those are exactly the wavenumbers the boundary
        closure handles worst -- with the source three cells down, the
        grid-scale content of the top rows is ~170x what it is with the
        same source below the closure band.  Band-limiting the injection
        removes that at no physical cost (the wavelet is band-limited
        anyway); it is the same reason a spectral-element source, spread
        over an element's nodes, leaves a cleaner near surface.  ``None``
        (set ``source_spread=0``) restores the single-cell injection.
        """
        n = self.source_spread
        if not n:
            return None
        k = np.ones(1, dtype=np.float64)
        for _ in range(n):
            k = np.convolve(k, np.array([1.0, 2.0, 1.0]) / 4.0)
        return np.outer(k, k).astype(np.float32)

    @property
    def surface_row(self):
        """Index of the free-surface row in the padded runtime array."""
        return self.so // 2

    def _geometry(self, like, d_xi):
        """Build (and cache) the derived metric bundle.

        ``like`` is any runtime tensor, used for device / dtype.
        """
        cache = self._geom_cache
        key = (str(like.device), like.dtype, float(d_xi))
        if cache is not None and cache["key"] == key:
            return cache

        import torch

        if self._curv_alpha is None or self._curv_beta is None:
            raise RuntimeError(
                "ElasticCurvilinear stepped before set_curvilinear_metrics() "
                "and _curv_beta were attached by the propagator.")

        beta = self._curv_beta.to(device=like.device, dtype=like.dtype)
        jac = 1.0 / beta                       # J = det F = D(xi)
        surf = self.surface_row
        nz = beta.shape[-2]

        # --- metrics rebuilt to satisfy the geometric conservation law ---
        #
        # The conservative momentum equation only annihilates a uniform
        # stress state if  d_zeta_m (J T_mj) = 0  holds *discretely*.  With
        # the metric slope taken from an independent formula (a plain
        # central difference of the surface) it does not: the residual
        # reaches 22% of |dJ/dxi| on the 10% hill and 71% on the 30% one,
        # and it acts as a slope-dependent body force.
        #
        # For this mapping J = D(xi) and J T_31 = -(h' + D' eta), and the
        # eta-operator is exact on functions linear in eta, so writing
        #     J T_31 = (1 - eta) * d_xi J      with d_xi from the scheme
        # makes the identity hold to round-off by construction.  h' = -d_xi J
        # then follows, and is what the free-surface rotation should use too:
        # it is the slope the scheme actually sees.
        # eta runs 0..1 over the physical rows; clamp outside so the PML
        # and halo rows inherit the nearest physical metric, exactly as the
        # replicate padding of the original metric fields does.  (Letting
        # the linear form extrapolate past the bottom row flips the sign of
        # J T_31 inside the bottom PML and the run diverges.)
        eta = torch.clamp(
            (torch.arange(nz, device=like.device, dtype=like.dtype) - surf)
            * float(self._curv_d_eta), 0.0, 1.0).reshape(1, 1, nz, 1)
        # Replicate-pad before differentiating: the convolution zero-pads
        # outside the array, which would turn the constant J of a flat
        # surface into a +-165 m/m slope in the outermost L columns and
        # blow the run up.
        import torch.nn.functional as F

        pad = self.op.L
        j_pad = F.pad(jac[None, None], (pad, pad, 0, 0), mode="replicate")
        dj_dxi = self.op.d_x_adj(j_pad)[..., pad:-pad] / float(d_xi)
        jalpha = ((1.0 - eta) * dj_dxi)[0, 0]
        alpha = jalpha / jac

        rows = torch.arange(nz, device=like.device, dtype=like.dtype)
        below = (rows >= surf).to(like.dtype).reshape(1, 1, nz, 1)
        surf_mask = (rows == surf).to(like.dtype).reshape(1, 1, nz, 1)

        # Surface tangent t = (1, h') / sqrt(1 + h'^2) and its arclength
        # derivative dt/ds (the curvature correction in the tangential
        # strain rate).
        # Surface slope consistent with the metric above: h' = -d_xi J.
        hp_np = -(dj_dxi[0, 0, surf].detach().cpu().numpy()
                  ).astype(np.float64).reshape(-1)
        inv_len_np = 1.0 / np.sqrt(1.0 + hp_np**2)
        t_x = inv_len_np
        t_z = hp_np * inv_len_np
        # d/ds = (1/sqrt(1+h'^2)) d/dxi
        tau_x = inv_len_np * np.gradient(t_x, float(d_xi))
        tau_z = inv_len_np * np.gradient(t_z, float(d_xi))

        def _row(a):
            return torch.as_tensor(np.ascontiguousarray(a, dtype=np.float64),
                                   device=like.device,
                                   dtype=like.dtype).reshape(1, 1, 1, -1)

        cache = {
            "key": key,
            "alpha": alpha,
            "beta": beta,
            "jac": jac,
            "jalpha": jalpha,
            "d_xi": float(d_xi),
            "d_eta": float(self._curv_d_eta),
            "surface_row": surf,
            "below": below,
            "surf_mask": surf_mask,
            "h_prime": _row(hp_np),
            "inv_len": _row(inv_len_np),
            "inv_len2": _row(inv_len_np**2),
            "tau_x": _row(tau_x),
            "tau_z": _row(tau_z),
            "mass_weight": None if self.surface_mass_weight == 1.0 else
                torch.where(rows == surf,
                            torch.as_tensor(float(self.surface_mass_weight),
                                            device=like.device, dtype=like.dtype),
                            torch.ones((), device=like.device, dtype=like.dtype)
                            ).reshape(1, 1, nz, 1),
        }
        self._geom_cache = cache
        return cache

    def _surface_damper(self, geom):
        """Kreiss-Oliger filter tapered into the free-surface band.

        Fourth order in both directions: the per-step response is
        ``1 - kappa sin^4(theta/2)``, i.e. ``kappa`` at Nyquist and
        ``kappa * 4e-4`` at 20 points per wavelength.  The vertical filter
        starts two rows below the surface so its five-point stencil never
        needs a value from the void.
        """
        import torch
        import torch.nn.functional as F

        cache = geom.get("damper")
        if cache is None:
            surf, nz = geom["surface_row"], geom["below"].shape[-2]
            rows = np.arange(nz) - surf
            n = max(self.surface_damping_rows, 3)
            taper = np.clip(1.0 - rows / float(n), 0.0, 1.0)
            taper[rows < 0] = 0.0
            like = geom["below"]
            k = np.array([1.0, -4.0, 6.0, -4.0, 1.0]) / 16.0
            # Fade the filter out across the horizontal PML.  Filtering the
            # velocities inside it desynchronises them from the CPML memory
            # variables, and the free-surface/PML corner then grows: the
            # peak of the late field parks itself in the outermost column
            # and climbs by ~2/s.  ``bx`` is non-zero exactly over the PML,
            # so 1 - |bx|/max|bx| is a ready-made smooth taper.
            bx = self.b[5]
            bx = bx.detach() if hasattr(bx, "detach") else torch.as_tensor(bx)
            bx = bx.to(device=like.device, dtype=like.dtype).reshape(1, 1, 1, -1)
            xmask = 1.0 - bx.abs() / bx.abs().max().clamp_min(1e-30)
            wz = torch.as_tensor(
                np.where(rows >= 2, taper, 0.0) * self.surface_damping,
                device=like.device, dtype=like.dtype).reshape(1, 1, nz, 1)
            wx = torch.as_tensor(
                taper * self.surface_damping,
                device=like.device, dtype=like.dtype).reshape(1, 1, nz, 1)
            cache = {
                "wz": wz * xmask,
                "wx": wx * xmask,
                "kz": torch.as_tensor(k, device=like.device,
                                      dtype=like.dtype).reshape(1, 1, 5, 1),
                "kx": torch.as_tensor(k, device=like.device,
                                      dtype=like.dtype).reshape(1, 1, 1, 5),
            }
            geom["damper"] = cache

        def apply(u):
            u = u - cache["wz"] * F.conv2d(u, cache["kz"], padding=(2, 0))
            u = u - cache["wx"] * F.conv2d(u, cache["kx"], padding=(0, 2))
            return u

        return apply

    def _hourglass(self, geom, dt):
        """``(dt * a, S)`` for the symplectic stiffness pair.

        ``S = W K W`` with ``K`` the fourth-order Kreiss-Oliger operator in
        eta and ``W`` the surface-band taper (faded out across the
        horizontal PML, exactly as the filter is).  ``S`` is symmetric, so
        ``v -= dt a S h`` followed by ``h += dt a S v`` is a symplectic
        Euler pair: it conserves a discrete energy and cannot damp
        anything.  Its only effect is ``omega**2 += a**2 (eig S)**2``,
        which lifts the vertically grid-scale free-surface mode out of the
        seismic band and leaves the Rayleigh wave where it is.
        """
        import torch
        import torch.nn.functional as F

        cache = geom.get("hourglass")
        if cache is None:
            surf, nz = geom["surface_row"], geom["below"].shape[-2]
            rows = np.arange(nz) - surf
            # Ramp in over a few rows and then stay at one all the way
            # down.  The weight enters S = W K W, so any kink in W is a
            # fourth difference of order one that the stiffness would then
            # apply to *smooth* fields; a raised cosine over m rows keeps
            # it at (pi/m)^4/16 instead, i.e. nothing.  Rows above the
            # surface, and the two the five-point stencil cannot reach,
            # stay at zero.
            m = max(self.hourglass_rows, 4)
            x = np.clip((rows - 2.0) / float(m), 0.0, 1.0)
            taper = 0.5 * (1.0 - np.cos(np.pi * x))
            taper[rows < 2] = 0.0
            like = geom["below"]
            bx = self.b[5]
            bx = bx.detach() if hasattr(bx, "detach") else torch.as_tensor(bx)
            bx = bx.to(device=like.device, dtype=like.dtype).reshape(1, 1, 1, -1)
            xmask = 1.0 - bx.abs() / bx.abs().max().clamp_min(1e-30)
            w = torch.as_tensor(taper, device=like.device,
                                dtype=like.dtype).reshape(1, 1, nz, 1) * xmask
            k = np.array([1.0, -4.0, 6.0, -4.0, 1.0]) / 16.0
            cache = {
                "w": w,
                "kz": torch.as_tensor(k, device=like.device,
                                      dtype=like.dtype).reshape(1, 1, 5, 1),
            }
            geom["hourglass"] = cache

        def apply_s(u):
            u = cache["w"] * u
            u = F.conv2d(u, cache["kz"], padding=(2, 0))
            return cache["w"] * u

        return float(dt) * 2.0 * np.pi * self.hourglass_frequency, apply_s

    def cfl_limit(self, max_velocity):
        """Largest stable ``dt`` for this operator and the attached grid.

        The vertical physical spacing on a boundary-fitted grid is
        ``D(xi) * d_eta`` and shrinks wherever the surface is elevated, so
        the limit is taken over the whole grid.
        """
        if self._curv_beta is None:
            raise RuntimeError("metrics not attached yet")
        import torch

        dz_min = float((1.0 / self._curv_beta).min()) * float(self._curv_d_eta)
        d_xi = float(self.pd._spacing["x"])
        return self.op.cfl_dt(max_velocity, (dz_min, d_xi))

    # -- step --------------------------------------------------------------

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
            raise ValueError(
                f"ElasticCurvilinear.func expected 3, 5, or 6 models, "
                f"got {len(models)}")

        d_xi = h if isinstance(h, (int, float)) else h[-1]
        geom = self._geometry(wavefields[0], float(d_xi))

        return step_curv_asym(
            *wavefields,
            vp, vs, rho, lame_lambda, lame_mu,
            dt, h, b,
            pd=self.pd,
            pml=self.b,
            free_surface=getattr(self, "free_surface", False),
            lame_lambda_2mu=lame_lambda_2mu,
            op=self.op,
            geom=geom,
            closure=self.surface_closure,
            damping=(self._surface_damper(geom)
                     if self.surface_damping else None),
            damp_velocity=self.damp_velocity,
            hourglass=(self._hourglass(geom, dt)
                       if self.hourglass_frequency else None),
            **kwargs,
        )

    def _C(self):
        raise NotImplementedError(
            "ElasticCurvilinear does not have a CUDA kernel; use impl='eager'.")

    @property
    def cuda_layout(self):
        return CUDALayoutSpec(
            base_nvar=5,
            pml_nvar=10,
            last_two_nvar=1,
            last_two_storage_nvar=5,
            backward_workspace_nvar=8,
        )
