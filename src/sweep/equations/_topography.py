"""Topography classification and APM (Adaptive Parameter-Modified)
modulus precomputation for elastic free-surface modelling with
irregular topography on the standard staggered grid.

Implements the 2-D elastic limit of the parameter-modified family:

  - Cao, J. & Chen, J.-B. (2018), "A parameter-modified method for
    implementing surface topography in elastic-wave finite-difference
    modeling", Geophysics 83(6), T313–T322. **primary reference**
  - Dong, S.-L. et al. (2023), "Finite-difference modeling with
    topography using 3D viscoelastic parameter-modified free-surface
    condition", Geophysics 88(4), T211–T226. **Table 1 cell taxonomy**

The free-surface effect is encoded entirely by modifying constitutive
moduli and densities at cells that touch the air region, plus a
selective stress-zeroing step. The FD stencil itself is unchanged from
the bulk Virieux (1986) update — no mirroring helpers are needed and
the existing ``pd.x_forward / x_backward / z_forward / z_backward``
operators are reused as-is.

Per-cell category codes
-----------------------

::

    INTERIOR (0) — bulk solid, no air neighbour
    AIR      (1) — pure air cell (above topography)
    H        (2) — solid cell with air above   (horizontal free face)
    VL       (3) — solid cell with air to the left (vertical, air on left)
    VR       (4) — solid cell with air to the right
    OC       (5) — outer corner: air on two adjacent 4-conn sides
    IC       (6) — inner corner: no 4-conn air, but a diagonal neighbour is air

The classification uses 4-connected neighbour tests plus a diagonal
fallback for inner corners; ties resolve by priority OC > H > VL > VR > IC.

Modulus modifications
---------------------

Define the APM-modified Lamé combinations::

    α = 2 μ (λ + μ) / (λ + 2 μ)             effective P-modulus
    β = λ μ / (λ + 2 μ)                     effective coupling (Cao 2018 Eq 17)

For each cell category we replace ``(λ, μ)`` and ``ρ`` in the wave
equation with:

==========  ===============  ===============  ===============  ===========  ===========
Category    ``λ_eff``         ``μ_eff`` for    ``μ_xz_node``    ``ρ_x``       ``ρ_z``
                              normal stresses
==========  ===============  ===============  ===============  ===========  ===========
INTERIOR    ``λ``             ``μ``             ``μ``             ``ρ``         ``ρ``
H           ``0``             ``α / 2``         ``μ / 2``         ``0.5 ρ``    ``ρ``
VL          ``0``             ``α / 2``         ``μ / 2``         ``ρ``         ``0.5 ρ``
VR          ``0``             ``α / 2``         ``μ / 2``         ``0.5 ρ``    ``0.5 ρ``
OC          ``0``             ``0``             ``0``             ``0.25 ρ``  ``0.25 ρ``
IC          ``λ``             ``μ``             ``μ``             ``0.75 ρ``  ``0.75 ρ``
AIR         ``0``             ``0``             ``0``             ``ρ``         ``ρ``
==========  ===============  ===============  ===============  ===========  ===========

At AIR cells we keep ``ρ`` at its original value (NOT zero) to avoid
``dt / ρ`` divide-by-zero — the wave equation in air still runs, but
with all moduli zeroed the stress can never accumulate so the kinetic
energy stays bounded. As a belt-and-braces measure ``zero_at_air``
also wipes wavefield cells whose ``category == AIR`` at the end of
each step.

Selective stress zeroing
------------------------

After the standard stress update, traction-free components on the
surface row of each category are zeroed::

    H   →  σ_zz = 0
    VL  →  σ_xx = 0
    VR  →  σ_xx = 0
    OC  →  σ_xx = σ_zz = σ_xz = 0

These are pointwise overwrites (no FD stencil involvement) and exactly
satisfy the traction-free BC at the per-cell free face. For h' = 0
(flat surface, all topographic cells are H) this reduces to the
existing ``zero_top_row`` behaviour bit-for-bit.
"""

from __future__ import annotations

import numpy as np


# Category codes — kept as module constants for use in tests and step.
INTERIOR = 0
AIR = 1
H = 2
VL = 3
VR = 4
OC = 5
IC = 6


def classify_topography(air_mask):
    """Return the per-cell category int code array (``int32``) given
    a 2-D boolean ``air_mask`` of shape ``(nz, nx)``.

    Parameters
    ----------
    air_mask : numpy.ndarray or torch.Tensor
        Shape ``(nz, nx)``. ``True`` / non-zero where the cell is in air
        (above the topographic surface). Float inputs (e.g. after
        propagator padding) are interpreted with the threshold ``> 0.5``.

    Returns
    -------
    category : numpy.ndarray of int32, shape ``(nz, nx)``
        Codes per the module-level constants ``INTERIOR / AIR / H / VL /
        VR / OC / IC``. The function is pure numpy under the hood; if
        ``air_mask`` is a torch tensor, conversion is automatic.
    """
    is_torch = hasattr(air_mask, "device") and hasattr(air_mask, "is_cuda")
    if is_torch:
        am_np = (air_mask.detach().cpu().numpy() > 0.5)
    else:
        am_np = np.asarray(air_mask)
        if am_np.dtype != np.bool_:
            am_np = am_np > 0.5
    if am_np.ndim != 2:
        raise ValueError(f"air_mask must be 2-D; got shape {am_np.shape}")

    nz, nx = am_np.shape
    cat = np.full((nz, nx), INTERIOR, dtype=np.int32)
    cat[am_np] = AIR

    # 4-connected neighbour booleans. Cells outside the grid are
    # treated as solid (False) — boundary cells don't get topographic
    # categories.
    above = np.zeros_like(am_np); above[1:, :] = am_np[:-1, :]
    below = np.zeros_like(am_np); below[:-1, :] = am_np[1:, :]
    left  = np.zeros_like(am_np); left[:, 1:]  = am_np[:, :-1]
    right = np.zeros_like(am_np); right[:, :-1] = am_np[:, 1:]

    # Diagonal neighbours (for IC detection)
    dnw = np.zeros_like(am_np); dnw[1:, 1:]   = am_np[:-1, :-1]
    dne = np.zeros_like(am_np); dne[1:, :-1]  = am_np[:-1, 1:]
    dsw = np.zeros_like(am_np); dsw[:-1, 1:]  = am_np[1:, :-1]
    dse = np.zeros_like(am_np); dse[:-1, :-1] = am_np[1:, 1:]

    solid = ~am_np

    is_OC = solid & ((above & left) | (above & right)
                     | (below & left) | (below & right))
    is_H  = solid & ~is_OC & above
    is_VL = solid & ~is_OC & ~above & left
    is_VR = solid & ~is_OC & ~above & ~left & right
    has_diag = dnw | dne | dsw | dse
    has_4conn = above | below | left | right
    is_IC = solid & ~has_4conn & has_diag

    cat[is_H]  = H
    cat[is_VL] = VL
    cat[is_VR] = VR
    cat[is_OC] = OC
    cat[is_IC] = IC

    return cat


def precompute_apm_moduli(lame_lambda, lame_mu, rho, category):
    """Return the five APM-modified arrays used by ``ElasticAPM.step``.

    Parameters
    ----------
    lame_lambda, lame_mu, rho : torch.Tensor or numpy.ndarray
        Shape ``(nz, nx)``, float. Bulk Lamé parameters (NOT yet
        modified).
    category : numpy.ndarray of int32, shape ``(nz, nx)``
        Output of :func:`classify_topography`. Torch tensors are
        accepted and cast.

    Returns
    -------
    lame_lambda_eff : same type as ``lame_lambda``
        λ multiplier on normal-stress updates after APM modification.
    lame_mu_eff : same type
        μ multiplier on normal-stress updates after APM modification.
        Note: this is NOT the μ used in the σ_xz update — see
        ``mu_xz_node`` below.
    mu_xz_node : same type
        μ multiplier on the σ_xz update. Reduced to ``μ / 2`` at
        surface-touching nodes, ``0`` at outer-corner / air nodes.
    rho_x_eff, rho_z_eff : same type
        Densities used by the ``v_x`` and ``v_z`` updates respectively
        (different because they live at different half-grid offsets).

    Notes
    -----
    * AIR cells get all moduli zeroed and ``ρ`` kept at the bulk value
      (to avoid ``dt/ρ`` divide-by-zero). The wavefield in air cells is
      additionally zeroed in :func:`zero_at_air` each step.
    * The mapping at H/VL/VR cells ``(λ_eff, μ_eff) = (0, α/2)``
      reproduces the APM identities
      ``(λ_eff + 2 μ_eff) = α`` (so ``∂σ_xx/∂t = α · ∂v_x/∂x`` at H)
      and ``λ_eff · ∂v_z/∂z = 0`` (so the cross-term drops, exactly
      the Cao 2018 modified equation).
    """
    is_torch = hasattr(lame_lambda, "device") and hasattr(lame_lambda, "is_cuda")
    if is_torch:
        import torch

        cat = (
            category
            if isinstance(category, torch.Tensor)
            else torch.from_numpy(category).to(lame_lambda.device)
        )
        where = torch.where
        zeros_like = torch.zeros_like
    else:
        cat = np.asarray(category, dtype=np.int32)
        where = np.where
        zeros_like = np.zeros_like

    # α and β per Cao 2018 (defined per cell so the ratios use the
    # local ``λ, μ``). Guard the denominator against zero μ to avoid
    # NaNs in pure-fluid cells.
    safe_denom = lame_lambda + 2 * lame_mu
    if is_torch:
        safe_denom = where(safe_denom > 0, safe_denom, zeros_like(safe_denom) + 1.0)
    else:
        safe_denom = np.where(safe_denom > 0, safe_denom, 1.0)
    alpha = 2.0 * lame_mu * (lame_lambda + lame_mu) / safe_denom

    # Start from bulk (INTERIOR) values.
    lam_eff = lame_lambda
    mu_eff  = lame_mu
    mu_xz   = lame_mu
    rho_x   = rho
    rho_z   = rho

    is_H  = cat == H
    is_VL = cat == VL
    is_VR = cat == VR
    is_OC = cat == OC
    is_IC = cat == IC
    is_AIR = cat == AIR

    # H / VL / VR: λ_eff = 0, μ_eff = α/2, μ_xz = μ/2
    surf = is_H | is_VL | is_VR
    lam_eff = where(surf, zeros_like(lam_eff), lam_eff)
    mu_eff  = where(surf, alpha * 0.5, mu_eff)
    mu_xz   = where(surf, lame_mu * 0.5, mu_xz)

    # OC: all stiffness zero
    lam_eff = where(is_OC, zeros_like(lam_eff), lam_eff)
    mu_eff  = where(is_OC, zeros_like(mu_eff), mu_eff)
    mu_xz   = where(is_OC, zeros_like(mu_xz), mu_xz)

    # AIR: all stiffness zero (ρ kept at bulk value)
    lam_eff = where(is_AIR, zeros_like(lam_eff), lam_eff)
    mu_eff  = where(is_AIR, zeros_like(mu_eff), mu_eff)
    mu_xz   = where(is_AIR, zeros_like(mu_xz), mu_xz)

    # IC: stiffness unchanged (still bulk), density at 0.75 ρ.

    # Density modifications per Dong 2023 Table 1 (2D limit)
    rho_x = where(is_H, 0.5 * rho, rho_x)
    rho_z = where(is_VL, 0.5 * rho, rho_z)
    rho_x = where(is_VR, 0.5 * rho, rho_x)
    rho_z = where(is_VR, 0.5 * rho, rho_z)
    rho_x = where(is_OC, 0.25 * rho, rho_x)
    rho_z = where(is_OC, 0.25 * rho, rho_z)
    rho_x = where(is_IC, 0.75 * rho, rho_x)
    rho_z = where(is_IC, 0.75 * rho, rho_z)

    return lam_eff, mu_eff, mu_xz, rho_x, rho_z


def zero_at_air(field, air_mask_float, *, eps=0.5):
    """Multiplicatively zero ``field`` wherever ``air_mask`` is True.

    ``air_mask_float`` is the post-padding float version (0 or 1, but
    after replicate-padding may have intermediate values at PML/halo
    edges if the user supplied a float mask). Uses an explicit
    threshold rather than bool casting to stay differentiable under
    autograd / torch.compile.
    """
    is_torch = hasattr(field, "device") and hasattr(field, "is_cuda")
    if is_torch:
        keep = (air_mask_float < eps).to(field.dtype)
    else:
        keep = (np.asarray(air_mask_float) < eps).astype(field.dtype)
    return field * keep


def enforce_apm_traction_bc(sxx, szz, sxz, category):
    """Pointwise zero the traction-free stress components per cell type.

      H  → σ_zz = 0
      VL → σ_xx = 0
      VR → σ_xx = 0
      OC → σ_xx = σ_zz = σ_xz = 0
      AIR → σ_xx = σ_zz = σ_xz = 0

    Cell-centred fields (σ_xx, σ_zz) and the σ_xz node (which lives at
    the (i+½, j+½) corner — same tensor shape, different staggered
    interpretation) are all treated by index lookup on ``category``.
    """
    is_torch = hasattr(sxx, "device") and hasattr(sxx, "is_cuda")
    if is_torch:
        import torch

        cat = (
            category
            if isinstance(category, torch.Tensor)
            else torch.from_numpy(category).to(sxx.device)
        )
        # Broadcast over leading dims.
        while cat.ndim < sxx.ndim:
            cat = cat.unsqueeze(0)
        zero_sxx = (cat == VL) | (cat == VR) | (cat == OC) | (cat == AIR)
        zero_szz = (cat == H) | (cat == OC) | (cat == AIR)
        zero_sxz = (cat == OC) | (cat == AIR)
        sxx_out = torch.where(zero_sxx, torch.zeros_like(sxx), sxx)
        szz_out = torch.where(zero_szz, torch.zeros_like(szz), szz)
        sxz_out = torch.where(zero_sxz, torch.zeros_like(sxz), sxz)
        return sxx_out, szz_out, sxz_out

    cat = np.asarray(category, dtype=np.int32)
    while cat.ndim < sxx.ndim:
        cat = cat[None, ...]
    sxx_out = np.where((cat == VL) | (cat == VR) | (cat == OC) | (cat == AIR), 0.0, sxx)
    szz_out = np.where((cat == H) | (cat == OC) | (cat == AIR), 0.0, szz)
    sxz_out = np.where((cat == OC) | (cat == AIR), 0.0, sxz)
    return sxx_out, szz_out, sxz_out
