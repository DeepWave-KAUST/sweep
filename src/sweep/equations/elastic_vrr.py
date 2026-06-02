"""Elastic vector-reflectivity 2D wave equation (1st-order momentum-stress).

Soares & Sacchi 2025 (SEG, doi 10.1190/image2025-4316471.1) replace the
particle velocity ``v_i`` with the particle momentum ``p_i = rho * v_i``
as the state variable. The momentum equation then collapses to the
standard Virieux form ``d_t p_i = d_j sigma_ij`` with NO rho dependence.
All complexity moves into the constitutive equation, parameterised by
``(V_P, V_S, R_P, R_S)`` where ``R_alpha,i = (1/2) Z_alpha^{-1} d_i Z_alpha``
is the vector reflectivity of impedance ``Z_alpha = rho * V_alpha``.

State (per Virieux 1986 staggered grid):

* ``px`` -- particle momentum x (lives where vx lives in Virieux SSG)
* ``pz`` -- particle momentum z (lives where vz lives in Virieux SSG)
* ``sxx, szz, sxz`` -- physical Cartesian stress (same locations as Elastic)
* 10 CPML memory variables (same layout as :class:`Elastic`)

Models: ``vp, vs, Rp_x, Rp_z, Rs_x, Rs_z`` -- NO density.

Per-step update (Soares & Sacchi 2025 Eq. 23-26, 2D expansion):

::

  gamma_alpha_ij = V_alpha * (d_i V_alpha) * p_j           [velocity-gradient]
                 - 2 * V_alpha**2 * R_alpha_i * p_j        [reflectivity]
                 + V_alpha**2 * d_i p_j                    [divergence-of-p]

  d_t sxx = (gamma_P_xx + gamma_P_zz) - 2 * gamma_S_zz       <- NB: -2 gamma_S_zz
  d_t szz = (gamma_P_xx + gamma_P_zz) - 2 * gamma_S_xx       <- swap zz <-> xx
  d_t sxz = gamma_S_xz + gamma_S_zx                          <- S-only, symmetric
  d_t p_i = d_j sigma_ij                                     <- standard Virieux

The ``-2 gamma_S_zz`` (not ``-2 gamma_S_xx``) in the sigma_xx update is the
sign-bug trap; it comes from ``lambda = rho(V_P^2 - 2 V_S^2)`` contributing
a ``-2 mu eps_zz`` deviatoric term to sigma_xx through delta_ij.

Reference: Soares, A. S. Q. and Sacchi, M. D. (2025). "Vector-reflectivity
Elastic Wave Equation with First-order formulation", SEG Technical Program
Expanded Abstracts, doi 10.1190/image2025-4316471.1.
"""

from __future__ import annotations

import numpy as np

from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from ._free_surface import zero_top_row, top_free_surface_derivative
from sweep.scalars import fd_coefficients


# ---------------------------------------------------------------------------
# Centered first derivatives for the smooth model fields (velocity gradients
# vp_x/vp_z/vs_x/vs_z and the log-impedance gradient in the R helper).
#
# Coefficients come from the canonical ``sweep.scalars.fd_coefficients`` --
# the same source the Laplacian and the regular-grid gradient kernels use --
# so there are no hand-rolled stencil constants and the accuracy order is
# whatever the caller passes. ``fd_coefficients(1, order)`` returns the
# positive-offset half of the centered first-derivative stencil
# (e.g. order 4 -> ``[8/12, -1/12]`` at offsets +1, +2); the negative side is
# the antisymmetric mirror. ``order`` must match the CUDA ``evr_model_grad_*``
# template order so impl='eager' and impl='c' agree.
# ---------------------------------------------------------------------------

_GRAD_COEF_CACHE: dict[int, np.ndarray] = {}


def _grad_coefs(order):
    if order not in _GRAD_COEF_CACHE:
        _GRAD_COEF_CACHE[order] = np.asarray(fd_coefficients(1, order), dtype=np.float64)
    return _GRAD_COEF_CACHE[order]


def _take(u, axis, start, stop):
    idx = [slice(None)] * u.ndim
    idx[axis] = slice(start, stop)
    return u[tuple(idx)]


def _pad_axis_zeros(c, axis, n, ref):
    """Zero-pad ``c`` by ``n`` cells on each side of ``axis`` to match ``ref``."""
    ax = axis if axis >= 0 else ref.ndim + axis
    try:
        import torch
        if isinstance(c, torch.Tensor):
            import torch.nn.functional as F
            pad = []
            for d in range(ref.ndim - 1, -1, -1):
                pad += [n, n] if d == ax else [0, 0]
            return F.pad(c, pad)
    except ImportError:
        pass
    widths = [(0, 0)] * ref.ndim
    widths[ax] = (n, n)
    return np.pad(c, widths)


def _centered_first_derivative(u, h, axis, order):
    """Order-``order`` centered first derivative of ``u`` along ``axis``.

    ``n = order // 2`` boundary cells on each side of ``axis`` are left zero
    (matching the staggered-stencil halo handling). Works for numpy arrays
    and torch tensors of any rank.
    """
    coefs = _grad_coefs(order)
    n = len(coefs)
    L = u.shape[axis]
    acc = None
    for k in range(1, n + 1):
        plus = _take(u, axis, n + k, L - n + k)
        minus = _take(u, axis, n - k, L - n - k)
        term = float(coefs[k - 1]) * (plus - minus)
        acc = term if acc is None else acc + term
    acc = acc / h
    return _pad_axis_zeros(acc, axis, n, u)


def compute_vector_reflectivity(vp, vs, rho, h, *, order=4, eps=1e-30):
    """Helper -- compute ``(Rp_x, Rp_z, Rs_x, Rs_z)`` from
    ``(vp, vs, rho)`` via the impedance gradient definition

    ``R_alpha,i = (1/2) Z_alpha^{-1} d_i Z_alpha = (1/2) d_i ln(Z_alpha)``

    where ``Z_P = rho * vp`` and ``Z_S = rho * vs``. Uses an order-``order``
    centered FD (default 4) sourced from :func:`sweep.scalars.fd_coefficients`;
    the ``order // 2`` boundary cells on each side get zero. ``h`` is the
    grid spacing (scalar; for non-isotropic dh pass a 2-tuple ``(dz, dx)``).
    """
    if np.isscalar(h):
        hz = hx = float(h)
    else:
        hz, hx = float(h[0]), float(h[1])

    zp = rho * vp
    zs = rho * vs
    # log( ) for stable formulation; eps to guard if anyone passes a
    # zero-impedance layer.
    try:
        import torch
        if isinstance(vp, torch.Tensor):
            lnzp = torch.log(zp + eps)
            lnzs = torch.log(zs + eps)
        else:
            lnzp = np.log(zp + eps)
            lnzs = np.log(zs + eps)
    except ImportError:
        lnzp = np.log(zp + eps)
        lnzs = np.log(zs + eps)

    Rp_x = 0.5 * _centered_first_derivative(lnzp, hx, axis=-1, order=order)
    Rp_z = 0.5 * _centered_first_derivative(lnzp, hz, axis=-2, order=order)
    Rs_x = 0.5 * _centered_first_derivative(lnzs, hx, axis=-1, order=order)
    Rs_z = 0.5 * _centered_first_derivative(lnzs, hz, axis=-2, order=order)
    return Rp_x, Rp_z, Rs_x, Rs_z


def elastic_vr_step_core(
    px, pz, sxx, szz, sxz,
    m_pxx, m_pxz, m_pzx, m_pzz,
    m_sxxx, m_sxxz, m_szzx, m_szzz, m_sxzx, m_sxzz,
    *,
    vp, vs,
    Rp_x, Rp_z, Rs_x, Rs_z,
    vp_x, vp_z, vs_x, vs_z,
    dt, h, b, pd, pml,
    free_surface=False,
):
    """One leapfrog step of the Soares & Sacchi 2025 elastic vector
    reflectivity equation on a Virieux SSG.

    No post-step BC enforcement (callers handle image-method top-row
    zeroing). Same staggered-CPML layout as :func:`elastic_step_core`
    -- 8 profile arrays ``(az, bz, azh, bzh, ax, bx, axh, bxh)`` --
    so the propagator's CPML setup is reused verbatim.

    For terms with mismatched stagger (``V (dV) p_j`` and ``R p_j`` --
    cell-centred model factors multiplied by ``p_j`` at its half-grid
    stagger), we follow the same convention as standard Elastic
    (``mu_xz * (vx_z + vz_x)`` in :func:`elastic_step_core`): naive
    multiplication, accepting the O(h) staggered-interpolation noise
    that S&S 2025 Fig 2 explicitly tolerates.
    """
    az, bz, azh, bzh, ax, bx, axh, bxh = pml
    top_halo = pd.coes.shape[0]

    vp2 = vp * vp
    vs2 = vs * vs

    # IMPORTANT -- update order is momentum-FIRST, then stress, matching both
    # standard ``Elastic`` (``_elastic_step_core`` updates velocity then
    # stress) and the CUDA ``forward.cu`` (LAUNCH_EVR_MOMENTUM then
    # LAUNCH_EVR_STRESS). The gamma terms must consume the JUST-UPDATED
    # momentum p^{it}, not the incoming p^{it-1}. The two sub-step orderings
    # give the same recorded momentum (so the forward records agree either
    # way), but they are DIFFERENT discrete operators whose model gradients
    # differ at O(dt) -- so eager and the CUDA adjoint only agree when both
    # use this same order.

    # ---- 1. Stress spatial derivatives (from incoming stress) ----
    dsxx_dx = pd.x_forward(sxx)    # sxx (i, j)          -> (i+1/2, j)   feeds px
    dsxz_dx = pd.x_backward(sxz)   # sxz (i+1/2, j+1/2)  -> (i, j+1/2)   feeds pz
    # z-derivatives of stress: image-method mirror at the top free surface
    # (odd parity, mirroring _elastic_step_core txz_z/tzz_z), else plain FD.
    if free_surface:
        dszz_dz = top_free_surface_derivative(szz, pd.z_forward, top_halo, odd=True, axis=-2)
        dsxz_dz = top_free_surface_derivative(sxz, pd.z_backward, top_halo, odd=True, axis=-2)
    else:
        dszz_dz = pd.z_forward(szz)    # szz (i, j)          -> (i, j+1/2)   feeds pz
        dsxz_dz = pd.z_backward(sxz)   # sxz (i+1/2, j+1/2)  -> (i+1/2, j)   feeds px

    # ---- 2. CPML accumulation on stress derivatives ----
    m_sxxx = axh * m_sxxx + bxh * dsxx_dx
    dsxx_dx = dsxx_dx + m_sxxx
    m_szzz = azh * m_szzz + bzh * dszz_dz
    dszz_dz = dszz_dz + m_szzz
    m_sxzz = az * m_sxzz + bz * dsxz_dz
    dsxz_dz = dsxz_dz + m_sxzz
    m_sxzx = ax * m_sxzx + bx * dsxz_dx
    dsxz_dx = dsxz_dx + m_sxzx

    # ---- 3. Momentum update (S&S Eq. 23) -- NO division by rho ----
    # p_i = rho v_i absorbs the density, so d_t p_i = d_j sigma_ij is exact.
    px = px + dt * (dsxx_dx + dsxz_dz)
    pz = pz + dt * (dsxz_dx + dszz_dz)

    # ---- 4. Momentum spatial derivatives (from the UPDATED momentum) ----
    # Same stagger pattern as standard Elastic. Each output lives at the
    # location of the corresponding sigma component it feeds:
    dpx_dx = pd.x_backward(px)     # px (i+1/2, j)   -> (i, j)         feeds sigma_xx
    dpz_dx = pd.x_forward(pz)      # pz (i, j+1/2)   -> (i+1/2, j+1/2) feeds sigma_xz
    # z-derivatives of momentum: image-method mirror at the top free surface
    # (pz odd like vz, px even like vx; mirrors _elastic_step_core), else plain.
    if free_surface:
        dpz_dz = top_free_surface_derivative(pz, pd.z_backward, top_halo, odd=True, axis=-2)
        dpx_dz = top_free_surface_derivative(px, pd.z_forward, top_halo, odd=False, axis=-2)
    else:
        dpz_dz = pd.z_backward(pz)     # pz (i, j+1/2)   -> (i, j)         feeds sigma_zz
        dpx_dz = pd.z_forward(px)      # px (i+1/2, j)   -> (i+1/2, j+1/2) feeds sigma_xz

    # ---- 5. CPML accumulation on momentum derivatives ----
    # Pattern from _elastic_step_core: half-grid axis uses (axh, bxh)
    # or (azh, bzh); node-centred axis uses (ax, bx) or (az, bz).
    m_pxx = ax * m_pxx + bx * dpx_dx
    dpx_dx = dpx_dx + m_pxx
    m_pzz = az * m_pzz + bz * dpz_dz
    dpz_dz = dpz_dz + m_pzz
    m_pxz = azh * m_pxz + bzh * dpx_dz
    dpx_dz = dpx_dz + m_pxz
    m_pzx = axh * m_pzx + bxh * dpz_dx
    dpz_dx = dpz_dx + m_pzx

    # ---- 6. 8 gamma components (S&S Eq. 25-26), from updated momentum ----
    # gamma_alpha_ij = V_alpha (d_i V_alpha) p_j - 2 V_alpha^2 R_alpha_i p_j
    #                + V_alpha^2 (d_i p_j)
    # Index pairing (i = derivative axis, j = momentum component):
    #   xx -> dpx_dx        zz -> dpz_dz        xz -> dpz_dx        zx -> dpx_dz
    gamma_P_xx = vp * vp_x * px - 2.0 * vp2 * Rp_x * px + vp2 * dpx_dx
    gamma_P_zz = vp * vp_z * pz - 2.0 * vp2 * Rp_z * pz + vp2 * dpz_dz
    gamma_S_xx = vs * vs_x * px - 2.0 * vs2 * Rs_x * px + vs2 * dpx_dx
    gamma_S_zz = vs * vs_z * pz - 2.0 * vs2 * Rs_z * pz + vs2 * dpz_dz
    gamma_S_xz = vs * vs_x * pz - 2.0 * vs2 * Rs_x * pz + vs2 * dpz_dx
    gamma_S_zx = vs * vs_z * px - 2.0 * vs2 * Rs_z * px + vs2 * dpx_dz

    # ---- 7. Stress update (S&S Eq. 24, 2D expansion) ----
    # Trap: sigma_xx gets -2 gamma_S_zz, NOT -2 gamma_S_xx. Comes from
    # delta_ij in lambda * eps_kk delta_ij; sigma_xx picks up
    # lambda * eps_kk = lambda (eps_xx + eps_zz), and -2 mu eps_zz is
    # the deviatoric S contribution. See module docstring.
    gamma_P_kk = gamma_P_xx + gamma_P_zz
    sxx = sxx + dt * (gamma_P_kk - 2.0 * gamma_S_zz)
    szz = szz + dt * (gamma_P_kk - 2.0 * gamma_S_xx)
    sxz = sxz + dt * (gamma_S_xz + gamma_S_zx)

    if free_surface:
        sxz = zero_top_row(sxz, top_halo, axis=-2)
        szz = zero_top_row(szz, top_halo, axis=-2)

    return (
        px, pz, sxx, szz, sxz,
        m_pxx, m_pxz, m_pzx, m_pzz,
        m_sxxx, m_sxxz, m_szzx, m_szzz, m_sxzx, m_sxzz,
    )


class ElasticVRR(FirstOrderEquation):
    """First-order 2-D elastic vector-reflectivity wave equation.

    Soares & Sacchi 2025 momentum-stress formulation. State variables
    are ``(px, pz, sxx, szz, sxz)`` with ``p_i = rho * v_i`` particle
    momentum; physical Cartesian stress is unchanged. Models are
    ``(vp, vs, Rp_x, Rp_z, Rs_x, Rs_z)`` -- NO density.

    Reference: Soares, A. S. Q. and Sacchi, M. D. (2025).
    *Vector-reflectivity Elastic Wave Equation with First-order
    formulation*, SEG Technical Program Expanded Abstracts,
    doi 10.1190/image2025-4316471.1.

    !!! info "Models (constructor input order)"

        - ``vp`` (m/s): Elastic P-wave velocity model.
        - ``vs`` (m/s): Elastic S-wave velocity model.
        - ``Rp_x``: P-impedance vector reflectivity, x component.
        - ``Rp_z``: P-impedance vector reflectivity, z component.
        - ``Rs_x``: S-impedance vector reflectivity, x component.
        - ``Rs_z``: S-impedance vector reflectivity, z component.

    !!! info "Wavefields"

        - ``px`` (aliases: ``momentum_x``): Particle momentum x-component
          (= rho * vx); default receiver.
        - ``pz`` (aliases: ``momentum_z``): Particle momentum z-component
          (= rho * vz); default receiver.
        - ``sxx`` (aliases: ``stress_xx``): Normal stress in x; default source.
        - ``szz`` (aliases: ``stress_zz``): Normal stress in z; default source.
        - ``sxz`` (aliases: ``stress_xz``, ``shear_xz``): Shear stress.
        - ``m_pxx, m_pxz, m_pzx, m_pzz``: CPML memory variables for
          momentum-gradient terms (internal).
        - ``m_sxxx, m_sxxz, m_szzx, m_szzz, m_sxzx, m_sxzz``: CPML memory
          variables for stress-gradient terms (internal). Two slots
          (``m_sxxz``, ``m_szzx``) are kept for layout symmetry with
          :class:`Elastic` but never written.

    !!! info "Defaults"

        - ``source_type``: ``['sxx', 'szz']``
        - ``receiver_type``: ``['px', 'pz']``
        - ``pml_type``: ``'cpmls'``

    Notes
    -----
    Velocity-vs-momentum at sources / receivers. With density removed from
    the model, the natural physical receiver is particle MOMENTUM
    ``p_i = rho v_i`` rather than velocity. Multiply the receiver record by
    the auxiliary ``1/rho`` at the receiver location to recover velocity if
    that's the desired output. Same applies to a body-force source: inject
    into ``px/pz`` directly (the standard explosive ``[sxx, szz]`` source
    needs no rho rescaling because stress is unchanged).
    """

    MODEL_SPECS = (
        ModelSpec("vp", aliases=("p_velocity",),
                  description="Elastic P-wave velocity model.", unit="m/s"),
        ModelSpec("vs", aliases=("s_velocity",),
                  description="Elastic S-wave velocity model.", unit="m/s"),
        ModelSpec("Rp_x",
                  description="P-impedance vector reflectivity, x component."),
        ModelSpec("Rp_z",
                  description="P-impedance vector reflectivity, z component."),
        ModelSpec("Rs_x",
                  description="S-impedance vector reflectivity, x component."),
        ModelSpec("Rs_z",
                  description="S-impedance vector reflectivity, z component."),
    )
    FIELD_SPECS = (
        FieldSpec("px", aliases=("momentum_x",),
                  description="Particle momentum x-component (= rho*vx).",
                  supports_source=True, supports_receiver=True),
        FieldSpec("pz", aliases=("momentum_z",),
                  description="Particle momentum z-component (= rho*vz).",
                  supports_source=True, supports_receiver=True),
        FieldSpec("sxx", aliases=("stress_xx",),
                  description="Normal stress in x.",
                  supports_source=True, supports_receiver=True),
        FieldSpec("szz", aliases=("stress_zz",),
                  description="Normal stress in z.",
                  supports_source=True, supports_receiver=True),
        FieldSpec("sxz", aliases=("stress_xz", "shear_xz"),
                  description="Shear stress component.",
                  supports_source=True),
        FieldSpec("m_pxx", description="CPML memory variable for dpx/dx.",
                  internal=True, boundary_related=True),
        FieldSpec("m_pxz", description="CPML memory variable for dpx/dz.",
                  internal=True, boundary_related=True),
        FieldSpec("m_pzx", description="CPML memory variable for dpz/dx.",
                  internal=True, boundary_related=True),
        FieldSpec("m_pzz", description="CPML memory variable for dpz/dz.",
                  internal=True, boundary_related=True),
        FieldSpec("m_sxxx", description="CPML memory variable for dsxx/dx.",
                  internal=True, boundary_related=True),
        FieldSpec("m_sxxz", description="Reserved auxiliary field (layout parity with Elastic).",
                  internal=True, boundary_related=True),
        FieldSpec("m_szzx", description="Reserved auxiliary field (layout parity with Elastic).",
                  internal=True, boundary_related=True),
        FieldSpec("m_szzz", description="CPML memory variable for dszz/dz.",
                  internal=True, boundary_related=True),
        FieldSpec("m_sxzx", description="CPML memory variable for dsxz/dx.",
                  internal=True, boundary_related=True),
        FieldSpec("m_sxzz", description="CPML memory variable for dsxz/dz.",
                  internal=True, boundary_related=True),
    )

    default_pml_type = "cpmls"

    def __init__(self, spatial_order=4, device='cpu', backend='torch'):
        """Build the elastic vector-reflectivity operator.

        Args:
            spatial_order: FD accuracy order of the staggered first-derivative
                operator. Must be even (2, 4, 6, 8, ...). Defaults to 4.
            device: Device for the operator. Defaults to 'cpu'.
            backend: 'torch' or 'jax'. Defaults to 'torch'.
        """
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
        return ["px", "pz"]

    def prepare_models(self, models):
        """Precompute the spatial gradients of ``vp`` and ``vs`` once per
        run.  These appear in every per-step gamma computation; doing them
        once at setup saves 4 FD passes per timestep at the cost of 4
        modelfield-sized tensors.
        """
        vp, vs, Rp_x, Rp_z, Rs_x, Rs_z = models
        # The grid spacing here is unknown at prepare_models time (the
        # propagator passes h at step time). We compute the gradients
        # using a unit-h placeholder and rescale at step time via the
        # propagator-passed h. Simpler: defer to first call of func()
        # by caching a key on (id(vp), id(vs)).
        return [vp, vs, Rp_x, Rp_z, Rs_x, Rs_z]

    def _velocity_gradient_order(self):
        """Centered-FD order for the cached velocity gradients. Tracks the
        equation's spatial order for the CUDA template specialisations
        ({2,4,6,8}); for the rare runtime path (spatial_order > 8) the CUDA
        kernel falls back to 4th order, so match it here for eager/c parity.
        """
        return self.so if self.so in (2, 4, 6, 8) else 4

    def _precompute_velocity_gradients(self, vp, vs, h):
        """Per-run cache of ``(vp_x, vp_z, vs_x, vs_z)`` computed via the
        canonical centered FD (order = :meth:`_velocity_gradient_order`).
        Re-keys on ``id(vp), id(vs), h``.
        """
        if np.isscalar(h):
            hz = hx = float(h)
        else:
            hz, hx = float(h[0]), float(h[1])
        order = self._velocity_gradient_order()
        key = (id(vp), id(vs), hz, hx, order)
        cached = getattr(self, "_vgrad_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        vp_x = _centered_first_derivative(vp, hx, axis=-1, order=order)
        vp_z = _centered_first_derivative(vp, hz, axis=-2, order=order)
        vs_x = _centered_first_derivative(vs, hx, axis=-1, order=order)
        vs_z = _centered_first_derivative(vs, hz, axis=-2, order=order)
        out = (vp_x, vp_z, vs_x, vs_z)
        self._vgrad_cache = (key, out)
        return out

    def func(self, wavefields, models, dt, h, b, **kwargs):
        vp, vs, Rp_x, Rp_z, Rs_x, Rs_z = models
        vp_x, vp_z, vs_x, vs_z = self._precompute_velocity_gradients(vp, vs, h)
        free_surface = getattr(self, "free_surface", False)
        out = elastic_vr_step_core(
            *wavefields,
            vp=vp, vs=vs,
            Rp_x=Rp_x, Rp_z=Rp_z, Rs_x=Rs_x, Rs_z=Rs_z,
            vp_x=vp_x, vp_z=vp_z, vs_x=vs_x, vs_z=vs_z,
            dt=dt, h=h, b=b, pd=self.pd, pml=self.b,
            free_surface=free_surface,
        )
        return out

    def _C(self):
        """Compiled CUDA entry points (added in Phase 2)."""
        import sweep._C as _C
        return (
            _C.elastic_vr2d_forward,
            _C.elastic_vr2d_backward,
            _C.elastic_vr2d_backward_bs,
            _C.elastic_vr2d_backward_ckpt,
            _C.elastic_vr2d_backward_recursive_ckpt,
        )

    @property
    def cuda_layout(self):
        # EVR backward needs 14 workspace tensors per cell:
        #   4 q* (qxx, qzz, qxz, qzx)         stress-adjoint workspace
        #   4 p* (pxx, pzz, pxz, pzx)         momentum-adjoint workspace
        #   2 point_p* (point_px, point_pz)   gamma multiplicative term
        #                                     adjoint contribution
        #   4 L_dV* (L_dVp_dx, L_dVp_dz,
        #            L_dVs_dx, L_dVs_dz)      chain-rule source tensors for
        #                                     grad_vp / grad_vs through cached
        #                                     velocity gradients (4th-order
        #                                     central FD, transpose = -A)
        return CUDALayoutSpec(
            base_nvar=5,
            pml_nvar=10,
            last_two_nvar=1,
            last_two_storage_nvar=5,
            backward_workspace_nvar=14,
        )
