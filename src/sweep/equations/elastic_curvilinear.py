"""Elastic 2-D first-order (velocity-stress) wave equation on a
boundary-fitted curvilinear grid (irregular free-surface topography).

Companion to :class:`Elastic` for the case where the surface is
non-flat. The chain rule is applied to every ∂/∂X and ∂/∂Z derivative
in the standard Virieux (1986) staggered system:

    ∂f/∂X = f_ξ + α(ξ, η) · f_η
    ∂f/∂Z = β(ξ) · f_η

The free-surface BC σ_zz = σ_xz = 0 is enforced on the **flat** η = 0
top row of the computational domain — exactly the same machinery used
by the flat ``Elastic.step``. This eliminates the staircase corner
energy injection that drives the Stage-2 staircase-image-method
exponential instability.

MVP simplification: metric tensors α, β are evaluated at cell centres
and used by all staggered derivatives regardless of their offset. The
accuracy hit is O(dh / 2) at the surface (acceptable for moderate
topography); a future refinement would interpolate metrics to half-grid
positions.
"""

from __future__ import annotations

from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec
from .fields import FieldSpec, ModelSpec
from ._free_surface import top_free_surface_derivative, zero_top_row


def step_curv(
    vx, vz, sxx, szz, sxz,
    m_vxx, m_vxz, m_vzx, m_vzz,
    m_txxx, m_txxz, m_tzzx, m_tzzz,
    m_txzx, m_txzz,
    vp, vs, rho,
    lame_lambda, lame_mu,
    dt, h, b, pd,
    pml=None,
    free_surface=False,
    lame_lambda_2mu=None,
    alpha=None,
    beta=None,
    d_eta=None,
):
    """Staggered-grid velocity-stress elastic step in the (ξ, η) grid.

    ``alpha``, ``beta`` are the inverse-metric tensors (η-derivative
    coefficients for ∂/∂X and ∂/∂Z respectively); see
    :class:`sweep.utils.curvilinear.CurvilinearGrid`. ``d_eta`` is the
    computational η spacing (dimensionless, 1/(nz-1)) and is
    **explicitly passed to every** ``pd.z_*`` call so the FD operator
    divides by ``d_eta`` instead of ``self.pd._spacing["z"]`` (which the
    propagator initialised to the user's physical ``dh``).
    """
    az, bz, azh, bzh, ax, bx, axh, bxh = pml
    top_halo = pd.coes.shape[0]
    lame_lambda_2mu = lame_lambda + 2 * lame_mu if lame_lambda_2mu is None else lame_lambda_2mu
    if d_eta is None:
        raise ValueError("step_curv requires d_eta (η-spacing) — pass it from the equation.")

    # Convenience wrappers that route z-derivatives through the
    # dimensionless ``d_eta`` spacing instead of the user's physical
    # dh that ``pd`` was initialised with.
    def _z_fwd(u):
        return pd.z_forward(u, h=d_eta)
    def _z_bwd(u):
        return pd.z_backward(u, h=d_eta)

    # --- ξ-direction and η-direction derivatives of stress ---
    # ξ-derivatives: same as flat case (computational ξ ≡ physical X).
    txx_xi = pd.x_forward(sxx)
    txz_xi_back = pd.x_backward(sxz)
    # η-derivatives of stress (for chain-rule term). On the flat η=0 row
    # the image-method mirror enforces σ_zz / σ_xz = 0 — same flags as
    # the flat elastic.
    if free_surface:
        txz_eta = top_free_surface_derivative(sxz, _z_bwd, top_halo, odd=True, axis=-2)
        tzz_eta = top_free_surface_derivative(szz, _z_fwd, top_halo, odd=True, axis=-2)
    else:
        txz_eta = _z_bwd(sxz)
        tzz_eta = _z_fwd(szz)

    # σxx_η is a cell-centred derivative (σxx lives on cell centres) —
    # average the staggered forward / backward z derivatives at the
    # cell centre, both invoked with the d_eta spacing override.
    txx_eta = 0.5 * (_z_fwd(sxx) + _z_bwd(sxx))

    # Stagger-consistent chain rule. Each ``∂/∂X = (∂/∂ξ) + α·(∂/∂η)``
    # term must be evaluated at the same staggered location as the
    # bare ξ-derivative; ``α`` (and ``β``) are stored at cell centres
    # so we cheaply shift them to whichever right-/forward-edge the
    # current operator lives on by averaging adjacent x cells. Without
    # this, the cell-centre chain-rule term mixes with the edge ξ
    # derivative and excites a low-amplitude grid-scale instability
    # that explodes within ~0.5 s for non-trivial topography.
    alpha_xfwd = 0.5 * (alpha + _x_shift_left(alpha))     # forward-x stagger
    alpha_xbwd = 0.5 * (alpha + _x_shift_right(alpha))    # backward-x stagger

    # Physical-coord derivatives via chain rule (stagger-matched)
    txx_X = txx_xi + alpha_xfwd * txx_eta
    txz_X = txz_xi_back + alpha_xbwd * txz_eta
    tzz_Z = beta * tzz_eta
    txz_Z = beta * txz_eta

    # --- Update velocities ---
    m_tzzz = azh * m_tzzz + bzh * tzz_Z
    tzz_Z_pml = tzz_Z + m_tzzz
    m_txzx = ax * m_txzx + bx * txz_X
    txz_X_pml = txz_X + m_txzx
    vz = vz + dt / rho * (tzz_Z_pml + txz_X_pml)

    m_txzz = az * m_txzz + bz * txz_Z
    txz_Z_pml = txz_Z + m_txzz
    m_txxx = axh * m_txxx + bxh * txx_X
    txx_X_pml = txx_X + m_txxx
    vx = vx + dt / rho * (txx_X_pml + txz_Z_pml)

    # --- ξ-derivatives of velocity ---
    vx_xi = pd.x_backward(vx)
    vz_xi = pd.x_forward(vz)
    # η-derivatives of velocity (for chain-rule term)
    if free_surface:
        vz_eta = top_free_surface_derivative(vz, _z_bwd, top_halo, odd=True, axis=-2)
        vx_eta = top_free_surface_derivative(vx, _z_fwd, top_halo, odd=False, axis=-2)
    else:
        vz_eta = _z_bwd(vz)
        vx_eta = _z_fwd(vx)

    # Stagger-shift α to match the ξ-derivative stagger.
    # vx_xi from pd.x_backward → at x-cell-left-edge → use bwd-shift.
    # vz_xi from pd.x_forward  → at x-cell-right-edge → use fwd-shift.
    vx_X = vx_xi + alpha_xbwd * vx_eta
    vz_X = vz_xi + alpha_xfwd * vz_eta
    vx_Z = beta * vx_eta
    vz_Z = beta * vz_eta

    # --- Update stresses ---
    m_vzz = az * m_vzz + bz * vz_Z
    vz_Z_pml = vz_Z + m_vzz
    m_vxx = ax * m_vxx + bx * vx_X
    vx_X_pml = vx_X + m_vxx
    szz = szz + dt * (lame_lambda_2mu * vz_Z_pml + lame_lambda * vx_X_pml)
    sxx = sxx + dt * (lame_lambda_2mu * vx_X_pml + lame_lambda * vz_Z_pml)

    m_vxz = azh * m_vxz + bzh * vx_Z
    vx_Z_pml = vx_Z + m_vxz
    m_vzx = axh * m_vzx + bxh * vz_X
    vz_X_pml = vz_X + m_vzx
    sxz = sxz + dt * lame_mu * (vx_Z_pml + vz_X_pml)

    if free_surface:
        szz = zero_top_row(szz, top_halo, axis=-2)
        sxz = zero_top_row(sxz, top_halo, axis=-2)

    return (
        vx, vz, sxx, szz, sxz,
        m_vxx, m_vxz, m_vzx, m_vzz,
        m_txxx, m_txxz, m_tzzx, m_tzzz,
        m_txzx, m_txzz,
    )


def _z_central(pd, u):
    """Central η-derivative for the chain-rule term on cell-centred
    stresses (σxx, σzz). Approximated as the average of the staggered
    forward and backward z derivatives."""
    return 0.5 * (pd.z_forward(u) + pd.z_backward(u))


def _x_shift_left(t):
    """Average each cell with its left neighbour along the last axis
    (gives a forward-x staggered version of the cell-centre field)."""
    import torch

    pad = torch.cat([t[..., :1], t[..., :-1]], dim=-1)
    return 0.5 * (t + pad)


def _x_shift_right(t):
    """Average each cell with its right neighbour along the last axis
    (gives a backward-x staggered version of the cell-centre field)."""
    import torch

    pad = torch.cat([t[..., 1:], t[..., -1:]], dim=-1)
    return 0.5 * (t + pad)


class ElasticCurvilinear(FirstOrderEquation):
    """Curvilinear-grid Elastic 2-D for irregular topography.

    Same field / source / receiver layout as :class:`Elastic`. Metrics
    are attached by the Propagator's curvilinear path.
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
        FieldSpec("m_vxx", description="CPML memory for dvx/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vxz", description="CPML memory for dvx/dz.", internal=True, boundary_related=True),
        FieldSpec("m_vzx", description="CPML memory for dvz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_vzz", description="CPML memory for dvz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_txxx", description="CPML memory for dsxx/dx.", internal=True, boundary_related=True),
        FieldSpec("m_txxz", description="Reserved.", internal=True, boundary_related=True),
        FieldSpec("m_tzzx", description="Reserved.", internal=True, boundary_related=True),
        FieldSpec("m_tzzz", description="CPML memory for dszz/dz.", internal=True, boundary_related=True),
        FieldSpec("m_txzx", description="CPML memory for dsxz/dx.", internal=True, boundary_related=True),
        FieldSpec("m_txzz", description="CPML memory for dsxz/dz.", internal=True, boundary_related=True),
    )

    default_pml_type = "cpmls"

    def __init__(self, spatial_order=4, device="cpu", backend="torch"):
        super().__init__(spatial_order, device, backend)
        self._curv_alpha = None
        self._curv_beta = None
        self._curv_d_eta = None
        self.is_curvilinear = True

    def set_curvilinear_metrics(self, alpha, metric_pηη, metric_pη, d_eta):
        """Attach metric tensors. The elastic equation only needs ``alpha``
        and ``beta``; the combined ``metric_p*`` fields (acoustic-only)
        are accepted and ignored for interface compatibility."""
        self._curv_alpha = alpha
        self._curv_d_eta = float(d_eta)
        # ``_curv_beta`` is set separately by the propagator via direct
        # attribute assignment, since the acoustic ``set_curvilinear_metrics``
        # signature doesn't include it.

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
        if self._curv_alpha is None or self._curv_beta is None:
            raise RuntimeError(
                "ElasticCurvilinear stepped before set_curvilinear_metrics() "
                "and _curv_beta were attached by the propagator."
            )
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
                f"ElasticCurvilinear.func expected 3, 5, or 6 models, got {len(models)}"
            )

        return step_curv(
            *wavefields,
            vp, vs, rho, lame_lambda, lame_mu,
            dt, h, b,
            pd=self.pd,
            pml=self.b,
            free_surface=getattr(self, "free_surface", False),
            lame_lambda_2mu=lame_lambda_2mu,
            alpha=self._curv_alpha,
            beta=self._curv_beta,
            d_eta=self._curv_d_eta,
            **kwargs,
        )

    def _C(self):
        raise NotImplementedError(
            "ElasticCurvilinear does not have a CUDA kernel; use impl='eager'."
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
