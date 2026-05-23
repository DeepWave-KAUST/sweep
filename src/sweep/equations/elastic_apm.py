"""2-D elastic wave equation with irregular free-surface topography
via the Adaptive Parameter-Modified (APM) method.

Implements Cao & Chen (2018) Geophysics 83(6), T313-T322 — the 2-D
elastic limit of the parameter-modified family also developed by
Dong et al. (2023) for 3-D viscoelastic. The free surface is realised
on a staircase quantisation of the topography by **modifying the
constitutive moduli and densities** at every solid cell that touches
the air region (categories H / VL / VR / OC / IC; see
``_topography.py``). The FD stencil stays exactly the Virieux (1986)
staggered grid used by :class:`Elastic` — no mirroring helpers, no
per-row branching inside the time step. APM is well-known to be
stable for arbitrary simulation time on staircase topography (Cao &
Chen 2018 Fig 4), unlike the image-method-based ``Elastic`` +
``topography`` path which has a slow corner-induced instability for
high Poisson ratio.

The class is a strict superset of :class:`Elastic` in terms of fields
(15 identical specs) and source / receiver types — users can swap
``Elastic`` ↔ ``ElasticAPM`` without changing the rest of their
pipeline. The single API addition is the ``air_mask`` model
parameter (a 2-D boolean grid; passed alongside ``vp, vs, rho``).

References
----------
- Cao, J. & Chen, J.-B. (2018), Geophysics 83(6), T313-T322.
- Dong, S.-L. et al. (2023), Geophysics 88(4), T211-T226 (Table 1).
- Moczo, P. et al. (2002), BSSA 92, 3042-3066 (equivalent-medium averaging).
- Virieux, J. (1986), Geophysics 51(4), 1933-1942 (staggered grid).
"""

from __future__ import annotations

from .base import FirstOrderEquation
from .cuda_layout import CUDALayoutSpec
from .elastic import Elastic
from .fields import FieldSpec, ModelSpec
from ._topography import (
    classify_topography,
    enforce_apm_traction_bc,
    precompute_apm_moduli,
    zero_at_air,
)


def step_apm(
    vx, vz, sxx, szz, sxz,
    m_vxx, m_vxz, m_vzx, m_vzz,
    m_txxx, m_txxz, m_tzzx, m_tzzz,
    m_txzx, m_txzz,
    vp, vs, rho, air_mask,
    lame_lambda_eff, lame_mu_eff, mu_xz_node,
    rho_x_eff, rho_z_eff,
    category,
    dt, h, b, pd,
    pml=None,
):
    """One leapfrog step for 2D elastic APM on a staircase free surface.

    The APM modifications live in ``lame_lambda_eff``, ``lame_mu_eff``,
    ``mu_xz_node``, ``rho_x_eff``, ``rho_z_eff`` (precomputed once in
    ``prepare_models``). The FD stencil is identical to
    :func:`Elastic.step` — no mirror, no per-row branch.

    After the standard update, two pointwise corrections are applied:

      1. ``enforce_apm_traction_bc(sxx, szz, sxz, category)`` —
         pointwise zero the traction-free stress components per cell
         type (H → σ_zz=0, VL/VR → σ_xx=0, OC → all three).
      2. ``zero_at_air(field, air_mask)`` — wipe AIR-cell wavefields
         (vacuum approximation; the energy in AIR cells stays bounded
         to numerical noise).
    """
    az, bz, azh, bzh, ax, bx, axh, bxh = pml
    lame_lambda_2mu_eff = lame_lambda_eff + 2 * lame_mu_eff

    # --- Stress gradients (forward / backward stagger as in Elastic) ---
    txx_x = pd.x_forward(sxx)
    txz_z = pd.z_backward(sxz)
    tzz_z = pd.z_forward(szz)
    txz_x = pd.x_backward(sxz)

    # --- CPML accumulation + velocity update ---
    m_tzzz = azh * m_tzzz + bzh * tzz_z
    tzz_z = tzz_z + m_tzzz
    m_txzx = ax * m_txzx + bx * txz_x
    txz_x = txz_x + m_txzx
    vz = vz + dt / rho_z_eff * (tzz_z + txz_x)

    m_txzz = az * m_txzz + bz * txz_z
    txz_z = txz_z + m_txzz
    m_txxx = axh * m_txxx + bxh * txx_x
    txx_x = txx_x + m_txxx
    vx = vx + dt / rho_x_eff * (txx_x + txz_z)

    # --- Velocity gradients ---
    vx_x = pd.x_backward(vx)
    vz_z = pd.z_backward(vz)
    vx_z = pd.z_forward(vx)
    vz_x = pd.x_forward(vz)

    # --- CPML accumulation + stress update ---
    m_vzz = az * m_vzz + bz * vz_z
    vz_z = vz_z + m_vzz
    m_vxx = ax * m_vxx + bx * vx_x
    vx_x = vx_x + m_vxx

    szz = szz + dt * (lame_lambda_2mu_eff * vz_z + lame_lambda_eff * vx_x)
    sxx = sxx + dt * (lame_lambda_2mu_eff * vx_x + lame_lambda_eff * vz_z)

    m_vxz = azh * m_vxz + bzh * vx_z
    vx_z = vx_z + m_vxz
    m_vzx = axh * m_vzx + bxh * vz_x
    vz_x = vz_x + m_vzx
    sxz = sxz + dt * mu_xz_node * (vx_z + vz_x)

    # --- APM traction-free BC: pointwise stress zero per cell type ---
    sxx, szz, sxz = enforce_apm_traction_bc(sxx, szz, sxz, category)

    # --- Vacuum approximation: kill any wavefield in pure air cells ---
    vx = zero_at_air(vx, air_mask)
    vz = zero_at_air(vz, air_mask)
    sxx = zero_at_air(sxx, air_mask)
    szz = zero_at_air(szz, air_mask)
    sxz = zero_at_air(sxz, air_mask)

    return (
        vx, vz, sxx, szz, sxz,
        m_vxx, m_vxz, m_vzx, m_vzz,
        m_txxx, m_txxz, m_tzzx, m_tzzz,
        m_txzx, m_txzz,
    )


class ElasticAPM(FirstOrderEquation):
    """Elastic 2-D with topographic free surface via APM (Cao & Chen 2018).

    Model order is ``(vp, vs, rho, air_mask)``. ``air_mask`` is a 2-D
    array (``bool`` or numeric — treated as ``True`` above 0.5) marking
    cells in air (above the topographic surface).

    Field layout (15 fields, identical to :class:`Elastic`) — sources
    default to ``[sxx, szz]``, receivers to ``[vx, vz]``.

    Notes
    -----
    * Use ``pml_type='cpmls'`` (the staggered-CPML, matches
      :class:`Elastic`).
    * Use ``free_surface=False`` at the propagator — APM IS the free
      surface, encoded via the modified moduli. Setting
      ``free_surface=True`` is harmless but redundant; the existing
      ``zero_top_row``-style helpers won't be invoked because this
      class never imports them.
    * Sources / receivers placed at AIR cells are silently zeroed by
      the vacuum-cell mechanism.  Validate placement at the user
      level; the equation will not warn.
    """

    MODEL_SPECS = (
        ModelSpec("vp", aliases=("p_velocity",),
                  description="P-wave velocity.", unit="m/s"),
        ModelSpec("vs", aliases=("s_velocity",),
                  description="S-wave velocity.", unit="m/s"),
        ModelSpec("rho", aliases=("density",),
                  description="Density.", unit="kg/m^3"),
        ModelSpec(
            "air_mask",
            description=("Boolean 2-D mask, True where the cell is in air "
                         "(above the topographic surface)."),
        ),
    )
    # Field layout copied verbatim from Elastic so source/receiver wiring
    # carries across.
    FIELD_SPECS = Elastic.FIELD_SPECS

    default_pml_type = "cpmls"

    def __init__(self, spatial_order=4, device="cpu", backend="torch"):
        super().__init__(spatial_order, device, backend)
        # Cache the most recently classified topography so the ``func``
        # call doesn't reclassify every time step (the per-step work is
        # already O(nz·nx) — reclassifying that would 2× the cost).
        # The cache key is the id() of the air_mask tensor passed by
        # ``prepare_models``.
        self._apm_cache_key = None
        self._apm_cache = None

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
        """Classify topography and precompute APM-modified moduli once."""
        vp, vs, rho, air_mask = models
        # Bulk Lamé parameters
        lame_lambda = rho * (vp ** 2 - 2 * vs ** 2)
        lame_mu = rho * vs ** 2

        # Cache hit?
        cache_key = id(air_mask)
        if cache_key == self._apm_cache_key and self._apm_cache is not None:
            cat, lam_eff, mu_eff, mu_xz, rho_x, rho_z = self._apm_cache
        else:
            cat = classify_topography(air_mask)
            lam_eff, mu_eff, mu_xz, rho_x, rho_z = precompute_apm_moduli(
                lame_lambda, lame_mu, rho, cat
            )
            self._apm_cache_key = cache_key
            self._apm_cache = (cat, lam_eff, mu_eff, mu_xz, rho_x, rho_z)

        return [
            vp, vs, rho, air_mask,
            lam_eff, mu_eff, mu_xz, rho_x, rho_z,
            cat,
        ]

    def func(self, wavefields, models, dt, h, b, **kwargs):
        # Accept the long (10-model) prepared list or the short
        # (4-model raw) form — the propagator may pass either.
        if len(models) == 10:
            (vp, vs, rho, air_mask,
             lam_eff, mu_eff, mu_xz, rho_x, rho_z, cat) = models
        elif len(models) == 4:
            prepared = self.prepare_models(list(models))
            (vp, vs, rho, air_mask,
             lam_eff, mu_eff, mu_xz, rho_x, rho_z, cat) = prepared
        else:
            raise ValueError(
                f"ElasticAPM.func expected 4 or 10 models, got {len(models)}"
            )
        return step_apm(
            *wavefields,
            vp, vs, rho, air_mask,
            lam_eff, mu_eff, mu_xz, rho_x, rho_z, cat,
            dt, h, b,
            pd=self.pd,
            pml=self.b,
        )

    def _C(self):
        raise NotImplementedError(
            "ElasticAPM has no CUDA kernel; use impl='eager'."
        )

    @classmethod
    def supports_torch_binding(cls):
        return False

    @property
    def cuda_layout(self):
        return Elastic.cuda_layout.fget(self)
