"""Topography classification and APM (Adaptive Parameter-Modified)
modulus precomputation for elastic free-surface modelling with
irregular topography on the standard staggered grid.

Implements the elastic parameter-modified free-surface method of:

  - Cao, J. & Chen, J.-B. (2018), "A parameter-modified method for
    implementing surface topography in elastic-wave finite-difference
    modeling", Geophysics 83(6), T313–T322. **primary reference (2D + 3D)**

Cao & Chen 2018 already gives the full 2-D and 3-D parameter
modifications and cell taxonomy used here. Dong et al. (2023) extends
the same scheme to viscoelasticity (Q / memory variables); we do NOT
implement the viscoelastic part — the kernels here are the pure
elastic case, which is exactly Cao & Chen 2018.

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
# 0..6 are the 2-D codes (and a subset is reused by the 3-D classifier);
# 7..8 add the y-direction face categories that only exist in 3-D.
INTERIOR = 0
AIR = 1
H = 2     # air above (-z)
VL = 3    # air on the left (-x)
VR = 4    # air on the right (+x)
OC = 5    # outer corner / edge: >=2 face-air neighbours
IC = 6    # inner corner: 0 face-air but >=1 diagonal-air
# 3-D-only categories:
VF = 7    # air in front (-y)
VB = 8    # air in back  (+y)


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

    # Density modifications per Cao & Chen 2018 (2D)
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


# ===========================================================================
# 3-D extension — Cao & Chen 2018 (3-D parameter-modified scheme)
# ===========================================================================
#
# In 3-D each cell can touch the free surface through any of FIVE of the
# six face directions:
#
#   H   air above   (-z)
#   VL  air left    (-x)
#   VR  air right   (+x)
#   VF  air front   (-y)
#   VB  air back    (+y)
#
# (``below`` = +z is intentionally excluded — a hole *underneath* the
# cell is not a topographic free surface; it's a buried cavity, handled
# separately by the AIR-cell short-circuit.)
#
# Cells with two or more face-air neighbours are categorised as OC (outer
# corner / edge), following Cao & Chen 2018 + Hayashi 2001: the simpler "ρ/4"
# rule used for OC in 2-D is reused here for both edges and 3-face
# corners.  IC (inner corner) is reserved for cells with zero face-air
# neighbours but at least one of the 12 edge-diagonals or 8
# corner-diagonals is air (concave notch).


def classify_topography_3d(air_mask):
    """3-D cell classifier — Cao & Chen 2018 (3-D APM, topographical free surface).

    Parameters
    ----------
    air_mask : ndarray or torch.Tensor, shape ``(nz, ny, nx)``
        ``True``/non-zero where the cell is in air.

    Returns
    -------
    category : numpy.ndarray of int32, shape ``(nz, ny, nx)``
        One of the codes ``INTERIOR / AIR / H / VL / VR / VF / VB / OC / IC``.
    """
    is_torch = hasattr(air_mask, "device") and hasattr(air_mask, "is_cuda")
    if is_torch:
        am_np = (air_mask.detach().cpu().numpy() > 0.5)
    else:
        am_np = np.asarray(air_mask)
        if am_np.dtype != np.bool_:
            am_np = am_np > 0.5
    if am_np.ndim != 3:
        raise ValueError(f"3-D air_mask must be 3-D; got shape {am_np.shape}")

    nz, ny, nx = am_np.shape
    cat = np.full(am_np.shape, INTERIOR, dtype=np.int32)
    cat[am_np] = AIR

    solid = ~am_np

    # 6-connected neighbour booleans.  Out-of-grid neighbours are False
    # (treated as solid) — boundary cells stay INTERIOR.
    above = np.zeros_like(am_np);  above[1:, :, :]    = am_np[:-1, :, :]    # -z
    below = np.zeros_like(am_np);  below[:-1, :, :]   = am_np[1:, :, :]     # +z (only for AIR fully-surrounded check)
    front = np.zeros_like(am_np);  front[:, 1:, :]    = am_np[:, :-1, :]    # -y
    back  = np.zeros_like(am_np);  back[:, :-1, :]    = am_np[:, 1:, :]     # +y
    left  = np.zeros_like(am_np);  left[:, :, 1:]     = am_np[:, :, :-1]    # -x
    right = np.zeros_like(am_np);  right[:, :, :-1]   = am_np[:, :, 1:]     # +x

    # Surface faces — only the 5 "above-the-cell" directions count.
    n_air = (above.astype(np.int8) + left.astype(np.int8) + right.astype(np.int8)
             + front.astype(np.int8) + back.astype(np.int8))

    # Decision tree (>=2 → OC, ==1 → category by direction, ==0 → INTERIOR or IC).
    is_OC = solid & (n_air >= 2)
    is_H  = solid & ~is_OC & (n_air == 1) & above
    is_VL = solid & ~is_OC & ~is_H & (n_air == 1) & left
    is_VR = solid & ~is_OC & ~is_H & ~is_VL & (n_air == 1) & right
    is_VF = solid & ~is_OC & ~is_H & ~is_VL & ~is_VR & (n_air == 1) & front
    is_VB = solid & ~is_OC & ~is_H & ~is_VL & ~is_VR & ~is_VF & (n_air == 1) & back

    # Inner corner (concave notch): 0 face-air but at least one diagonal-air.
    # Roll the mask in all 18 diagonal directions and OR them.
    diag = np.zeros_like(am_np)
    for dz, dy, dx in (
        # 12 edge diagonals (two axes differ)
        (-1, -1,  0), (-1, +1,  0), (+1, -1,  0), (+1, +1,  0),
        (-1,  0, -1), (-1,  0, +1), (+1,  0, -1), (+1,  0, +1),
        ( 0, -1, -1), ( 0, -1, +1), ( 0, +1, -1), ( 0, +1, +1),
        # 8 corner diagonals (all three axes differ)
        (-1, -1, -1), (-1, -1, +1), (-1, +1, -1), (-1, +1, +1),
        (+1, -1, -1), (+1, -1, +1), (+1, +1, -1), (+1, +1, +1),
    ):
        sliced_dst = (
            slice(max(0,  dz), nz + min(0, dz)),
            slice(max(0,  dy), ny + min(0, dy)),
            slice(max(0,  dx), nx + min(0, dx)),
        )
        sliced_src = (
            slice(max(0, -dz), nz + min(0, -dz)),
            slice(max(0, -dy), ny + min(0, -dy)),
            slice(max(0, -dx), nx + min(0, -dx)),
        )
        diag[sliced_dst] |= am_np[sliced_src]

    is_IC = solid & (n_air == 0) & diag

    cat[is_H]  = H
    cat[is_VL] = VL
    cat[is_VR] = VR
    cat[is_VF] = VF
    cat[is_VB] = VB
    cat[is_OC] = OC
    cat[is_IC] = IC

    return cat


def precompute_apm_moduli_3d(lame_lambda, lame_mu, rho, category):
    """Per-cell modified moduli and densities for 3-D elastic APM
    (Cao & Chen 2018, 3-D parameter-modified moduli).

    Parameters
    ----------
    lame_lambda, lame_mu, rho : ndarray or torch.Tensor, shape ``(nz, ny, nx)``
        Bulk Lamé parameters at every cell.
    category : int32 array, shape ``(nz, ny, nx)``
        Output of :func:`classify_topography_3d`.

    Returns
    -------
    A 15-tuple of arrays, all same shape and type as ``lame_lambda``:

    Normal-stress effective moduli (9)::

        alpha_xx, alpha_yy, alpha_zz                 (each multiplies ε_NN
                                                       in d/dt σ_NN)
        lambda_xx_yy, lambda_xx_zz                   (cross-coupling λ in σ_xx)
        lambda_yy_xx, lambda_yy_zz                   (cross-coupling λ in σ_yy)
        lambda_zz_xx, lambda_zz_yy                   (cross-coupling λ in σ_zz)

    Shear node moduli (3)::

        mu_xy_node, mu_xz_node, mu_yz_node

    Inverse densities, masked at air (3)::

        inv_rho_x, inv_rho_y, inv_rho_z
            ``= 1 / ρ_*_eff`` for solid half-grid nodes; ``0`` where the
            half-grid velocity node sits in air (so the velocity update
            multiplies by zero and no division by zero occurs).
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

    eta = lame_lambda + 2.0 * lame_mu                # η = λ + 2μ
    safe_denom = eta
    if is_torch:
        safe_denom = where(safe_denom > 0, safe_denom, zeros_like(safe_denom) + 1.0)
    else:
        safe_denom = np.where(safe_denom > 0, safe_denom, 1.0)
    alpha = 2.0 * lame_mu * (lame_lambda + lame_mu) / safe_denom   # 2μ(λ+μ)/η
    beta  = lame_lambda * lame_mu / safe_denom                     # λμ/η

    is_INT = cat == INTERIOR
    is_H   = cat == H
    is_VL  = cat == VL
    is_VR  = cat == VR
    is_VF  = cat == VF
    is_VB  = cat == VB
    is_OC  = cat == OC
    is_IC  = cat == IC
    is_AIR = cat == AIR

    # Convenience disjunctions.
    is_VLR = is_VL | is_VR              # air on -x or +x (σ_xx = 0)
    is_VFB = is_VF | is_VB              # air on -y or +y (σ_yy = 0)

    bulk_or_IC = is_INT | is_IC         # bulk stiffness applies

    # ---- Normal-stress effective moduli ----
    # Start from bulk (η on the diagonal, λ off-diagonal).
    alpha_xx = where(bulk_or_IC, eta, zeros_like(eta))
    alpha_yy = where(bulk_or_IC, eta, zeros_like(eta))
    alpha_zz = where(bulk_or_IC, eta, zeros_like(eta))
    lam_xx_yy = where(bulk_or_IC, lame_lambda, zeros_like(lame_lambda))
    lam_xx_zz = where(bulk_or_IC, lame_lambda, zeros_like(lame_lambda))
    lam_yy_xx = where(bulk_or_IC, lame_lambda, zeros_like(lame_lambda))
    lam_yy_zz = where(bulk_or_IC, lame_lambda, zeros_like(lame_lambda))
    lam_zz_xx = where(bulk_or_IC, lame_lambda, zeros_like(lame_lambda))
    lam_zz_yy = where(bulk_or_IC, lame_lambda, zeros_like(lame_lambda))

    # H: σ_xx = α·ε_xx + β·ε_yy, σ_yy = β·ε_xx + α·ε_yy, σ_zz = 0
    alpha_xx  = where(is_H, alpha, alpha_xx)
    alpha_yy  = where(is_H, alpha, alpha_yy)
    alpha_zz  = where(is_H, zeros_like(eta), alpha_zz)
    lam_xx_yy = where(is_H, beta, lam_xx_yy)
    lam_xx_zz = where(is_H, zeros_like(lame_lambda), lam_xx_zz)
    lam_yy_xx = where(is_H, beta, lam_yy_xx)
    lam_yy_zz = where(is_H, zeros_like(lame_lambda), lam_yy_zz)
    lam_zz_xx = where(is_H, zeros_like(lame_lambda), lam_zz_xx)
    lam_zz_yy = where(is_H, zeros_like(lame_lambda), lam_zz_yy)

    # VL / VR: σ_xx = 0, σ_yy = α·ε_yy + β·ε_zz, σ_zz = β·ε_yy + α·ε_zz
    alpha_xx  = where(is_VLR, zeros_like(eta), alpha_xx)
    alpha_yy  = where(is_VLR, alpha, alpha_yy)
    alpha_zz  = where(is_VLR, alpha, alpha_zz)
    lam_xx_yy = where(is_VLR, zeros_like(lame_lambda), lam_xx_yy)
    lam_xx_zz = where(is_VLR, zeros_like(lame_lambda), lam_xx_zz)
    lam_yy_xx = where(is_VLR, zeros_like(lame_lambda), lam_yy_xx)
    lam_yy_zz = where(is_VLR, beta, lam_yy_zz)
    lam_zz_xx = where(is_VLR, zeros_like(lame_lambda), lam_zz_xx)
    lam_zz_yy = where(is_VLR, beta, lam_zz_yy)

    # VF / VB: σ_yy = 0, σ_xx = α·ε_xx + β·ε_zz, σ_zz = β·ε_xx + α·ε_zz
    alpha_xx  = where(is_VFB, alpha, alpha_xx)
    alpha_yy  = where(is_VFB, zeros_like(eta), alpha_yy)
    alpha_zz  = where(is_VFB, alpha, alpha_zz)
    lam_xx_yy = where(is_VFB, zeros_like(lame_lambda), lam_xx_yy)
    lam_xx_zz = where(is_VFB, beta, lam_xx_zz)
    lam_yy_xx = where(is_VFB, zeros_like(lame_lambda), lam_yy_xx)
    lam_yy_zz = where(is_VFB, zeros_like(lame_lambda), lam_yy_zz)
    lam_zz_xx = where(is_VFB, beta, lam_zz_xx)
    lam_zz_yy = where(is_VFB, zeros_like(lame_lambda), lam_zz_yy)

    # OC and AIR: all stiffness zero (already initialised to zero for non-bulk;
    # OC/AIR are not in ``bulk_or_IC`` so they stay zero.  Just ensure
    # any spurious writes above are overridden.)
    for surface in (is_OC, is_AIR):
        alpha_xx  = where(surface, zeros_like(eta), alpha_xx)
        alpha_yy  = where(surface, zeros_like(eta), alpha_yy)
        alpha_zz  = where(surface, zeros_like(eta), alpha_zz)
        lam_xx_yy = where(surface, zeros_like(lame_lambda), lam_xx_yy)
        lam_xx_zz = where(surface, zeros_like(lame_lambda), lam_xx_zz)
        lam_yy_xx = where(surface, zeros_like(lame_lambda), lam_yy_xx)
        lam_yy_zz = where(surface, zeros_like(lame_lambda), lam_yy_zz)
        lam_zz_xx = where(surface, zeros_like(lame_lambda), lam_zz_xx)
        lam_zz_yy = where(surface, zeros_like(lame_lambda), lam_zz_yy)

    # ---- Shear node moduli (Cao & Chen 2018, with the 2ε→μ·(∂v+∂v) absorbed) ----
    # Bulk: μ everywhere.
    mu_xy = where(bulk_or_IC, lame_mu, zeros_like(lame_mu))
    mu_xz = where(bulk_or_IC, lame_mu, zeros_like(lame_mu))
    mu_yz = where(bulk_or_IC, lame_mu, zeros_like(lame_mu))
    half_mu = 0.5 * lame_mu

    # H: μ_xy = μ/2, μ_xz = μ, μ_yz = μ
    mu_xy = where(is_H, half_mu, mu_xy)
    mu_xz = where(is_H, lame_mu, mu_xz)
    mu_yz = where(is_H, lame_mu, mu_yz)

    # VL: μ_xy = μ, μ_xz = μ, μ_yz = μ/2
    mu_xy = where(is_VL, lame_mu, mu_xy)
    mu_xz = where(is_VL, lame_mu, mu_xz)
    mu_yz = where(is_VL, half_mu, mu_yz)

    # VR: μ_xy = 0, μ_xz = 0, μ_yz = μ/2
    mu_xy = where(is_VR, zeros_like(lame_mu), mu_xy)
    mu_xz = where(is_VR, zeros_like(lame_mu), mu_xz)
    mu_yz = where(is_VR, half_mu, mu_yz)

    # VF: μ_xy = μ, μ_xz = μ/2, μ_yz = μ
    mu_xy = where(is_VF, lame_mu, mu_xy)
    mu_xz = where(is_VF, half_mu, mu_xz)
    mu_yz = where(is_VF, lame_mu, mu_yz)

    # VB: μ_xy = 0, μ_xz = μ/2, μ_yz = 0
    mu_xy = where(is_VB, zeros_like(lame_mu), mu_xy)
    mu_xz = where(is_VB, half_mu, mu_xz)
    mu_yz = where(is_VB, zeros_like(lame_mu), mu_yz)

    # OC and AIR: all shear zero.
    for surface in (is_OC, is_AIR):
        mu_xy = where(surface, zeros_like(lame_mu), mu_xy)
        mu_xz = where(surface, zeros_like(lame_mu), mu_xz)
        mu_yz = where(surface, zeros_like(lame_mu), mu_yz)

    # ---- Effective densities + air masking ----
    rho_x_eff = rho
    rho_y_eff = rho
    rho_z_eff = rho

    half_rho = 0.5 * rho
    quarter_rho = 0.25 * rho
    threeq_rho = 0.75 * rho

    # H: ρ_x = ρ/2, ρ_y = ρ/2, ρ_z = ρ
    rho_x_eff = where(is_H, half_rho, rho_x_eff)
    rho_y_eff = where(is_H, half_rho, rho_y_eff)
    # rho_z_eff stays ρ

    # VL: ρ_x = ρ, ρ_y = ρ/2, ρ_z = ρ/2
    rho_y_eff = where(is_VL, half_rho, rho_y_eff)
    rho_z_eff = where(is_VL, half_rho, rho_z_eff)

    # VR: ρ_x mask (air), ρ_y = ρ/2, ρ_z = ρ/2  (mask handled below)
    rho_y_eff = where(is_VR, half_rho, rho_y_eff)
    rho_z_eff = where(is_VR, half_rho, rho_z_eff)

    # VF: ρ_x = ρ/2, ρ_y = ρ, ρ_z = ρ/2
    rho_x_eff = where(is_VF, half_rho, rho_x_eff)
    rho_z_eff = where(is_VF, half_rho, rho_z_eff)

    # VB: ρ_x = ρ/2, ρ_y mask, ρ_z = ρ/2
    rho_x_eff = where(is_VB, half_rho, rho_x_eff)
    rho_z_eff = where(is_VB, half_rho, rho_z_eff)

    # OC: ρ/4 on all 3 nodes.
    rho_x_eff = where(is_OC, quarter_rho, rho_x_eff)
    rho_y_eff = where(is_OC, quarter_rho, rho_y_eff)
    rho_z_eff = where(is_OC, quarter_rho, rho_z_eff)

    # IC: 3ρ/4 on all 3 nodes.
    rho_x_eff = where(is_IC, threeq_rho, rho_x_eff)
    rho_y_eff = where(is_IC, threeq_rho, rho_y_eff)
    rho_z_eff = where(is_IC, threeq_rho, rho_z_eff)

    # Inverse density with masking.  Velocity nodes whose half-grid
    # position is "in air" (sentinel ρ=0 cases) are zeroed.  Concretely:
    # VR  → v_x node is in air        →  inv_rho_x = 0
    # VB  → v_y node is in air        →  inv_rho_y = 0
    # AIR → all three nodes in air    →  inv_rho_* = 0
    safe_rho_x = where(rho_x_eff > 0, rho_x_eff, zeros_like(rho_x_eff) + 1.0)
    safe_rho_y = where(rho_y_eff > 0, rho_y_eff, zeros_like(rho_y_eff) + 1.0)
    safe_rho_z = where(rho_z_eff > 0, rho_z_eff, zeros_like(rho_z_eff) + 1.0)
    inv_rho_x = 1.0 / safe_rho_x
    inv_rho_y = 1.0 / safe_rho_y
    inv_rho_z = 1.0 / safe_rho_z

    air_x = is_VR | is_AIR
    air_y = is_VB | is_AIR
    air_z = is_AIR
    inv_rho_x = where(air_x, zeros_like(inv_rho_x), inv_rho_x)
    inv_rho_y = where(air_y, zeros_like(inv_rho_y), inv_rho_y)
    inv_rho_z = where(air_z, zeros_like(inv_rho_z), inv_rho_z)

    return (
        alpha_xx, alpha_yy, alpha_zz,
        lam_xx_yy, lam_xx_zz,
        lam_yy_xx, lam_yy_zz,
        lam_zz_xx, lam_zz_yy,
        mu_xy, mu_xz, mu_yz,
        inv_rho_x, inv_rho_y, inv_rho_z,
    )


def enforce_apm_traction_bc_3d(
    sxx, syy, szz, sxy, sxz, syz, category
):
    """Pointwise zero traction-free σ components per 3-D cell category.

    H   →  σ_zz = σ_xz = σ_yz = 0  (free face normal to +z)
    VL/VR → σ_xx = σ_xy = σ_xz = 0  (free face normal to ±x)
    VF/VB → σ_yy = σ_xy = σ_yz = 0  (free face normal to ±y)
    OC  →  all 6 stresses zero
    AIR →  all 6 stresses zero
    """
    is_torch = hasattr(sxx, "device") and hasattr(sxx, "is_cuda")
    if is_torch:
        import torch

        cat = (
            category
            if isinstance(category, torch.Tensor)
            else torch.from_numpy(category).to(sxx.device)
        )
        while cat.ndim < sxx.ndim:
            cat = cat.unsqueeze(0)
        zero = lambda u, m: torch.where(m, torch.zeros_like(u), u)
    else:
        cat = np.asarray(category, dtype=np.int32)
        while cat.ndim < sxx.ndim:
            cat = cat[None, ...]
        zero = lambda u, m: np.where(m, 0.0, u)

    is_H   = cat == H
    is_VL  = cat == VL
    is_VR  = cat == VR
    is_VF  = cat == VF
    is_VB  = cat == VB
    is_OC  = cat == OC
    is_AIR = cat == AIR
    is_VLR = is_VL | is_VR
    is_VFB = is_VF | is_VB

    zero_sxx = is_VLR | is_OC | is_AIR
    zero_syy = is_VFB | is_OC | is_AIR
    zero_szz = is_H   | is_OC | is_AIR
    zero_sxy = is_VLR | is_VFB | is_OC | is_AIR
    zero_sxz = is_H   | is_VLR | is_OC | is_AIR
    zero_syz = is_H   | is_VFB | is_OC | is_AIR

    return (
        zero(sxx, zero_sxx),
        zero(syy, zero_syy),
        zero(szz, zero_szz),
        zero(sxy, zero_sxy),
        zero(sxz, zero_sxz),
        zero(syz, zero_syz),
    )
