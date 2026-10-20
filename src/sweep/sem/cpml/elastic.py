"""SPECFEM2D-style CPML for the 2nd-order *displacement* elastic SEM (K_MAX = 1).

Faithful port of SPECFEM2D's elastic C-PML (``pml_compute.f90``,
``pml_compute_memory_variables.f90``, ``pml_compute_accel_contribution.f90``).
The 2nd-order displacement form needs **three** ingredients, all of which this
profile precomputes per element-GLL node:

1. **Strain stretching** (``lik_parameter_computation``).  The displacement
   gradients are convolution-stretched *asymmetrically* before the stress:
   the ``∂/∂x`` derivatives (``dux_dx, duz_dx``) use one coefficient triple
   ``(Sx0, Sx1, Sx2)`` (SPECFEM ``A5,A6,A7`` from the ``index_ik = 31`` call),
   the ``∂/∂z`` derivatives (``dux_dz, duz_dz``) use ``(Sz0, Sz1, Sz2)``
   (``A8,A9,A10`` from the ``index_ik = 13`` call).  Each derivative carries two
   recursive-convolution memory branches.

2. **Asymmetric stress** (in the equation, not here).  With κ = 1::

       σ_xx = (λ+2μ)·dux_dx̃ + λ·duz_dz          (dux_dx̃ stretched, duz_dz raw)
       σ_zz = (λ+2μ)·duz_dz̃ + λ·dux_dx
       σ_zx = μ·(duz_dx + dux_dz̃)                (goes into accel_x)
       σ_xz = μ·(dux_dz + duz_dx̃)                (goes into accel_z)

   so σ_xz ≠ σ_zx in the layer — the divergence is assembled non-symmetrically.

3. **Acceleration / mass contribution** (``l_parameter_computation`` +
   ``pml_compute_accel_contribution_elastic``).  A damping term is subtracted
   from the acceleration::

       accel_PML = w_α w_β ρ |J| · (La1·v + La2·u + La3·rmem1 + La4·rmem2)

   with its own two memory branches ``rmem1, rmem2`` convolving the
   displacement.  This LHS term (missing from the earlier opt-in attempt) is
   what makes the 2nd-order displacement CPML *stable* — without it the layer
   blows up around nt≈2000 even on a flat mesh.

Because ``K_MAX = 1`` (no real coordinate stretching, ``κ = 1``) the PML mass
factor ``A0`` is identically 1, so the global mass matrix is **unchanged** and
no separate PML mass needs to be assembled.

Region constants (implicit in ``d_x>0`` / ``d_z>0``) match SPECFEM2D
``setup/constants.h.in``: ``CPML_X_ONLY = 1, CPML_Z_ONLY = 2, CPML_XZ = 3``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..mesh.quad9 import Quad9Mesh

_TINY = 1.0e-20


def _compute_coef_convolution(bb: torch.Tensor, dt: float):
    """Discrete recursive-convolution coefficients (SPECFEM ``compute_coef_convolution``).

    ``coef0 = exp(-bb·dt)``; ``coef1 = (1 − exp(-½ bb dt)) / bb``;
    ``coef2 = coef1 · exp(-½ bb dt)``.  Limit ``bb → 0``: coef0 → 1, coef1,
    coef2 → ½ dt.
    """
    half = 0.5 * bb * dt
    e_half = torch.exp(-half)
    coef0 = e_half * e_half
    bb_safe = torch.where(bb.abs() > _TINY, bb, torch.ones_like(bb))
    coef1 = torch.where(bb.abs() > _TINY, (1.0 - e_half) / bb_safe,
                        torch.full_like(bb, 0.5 * dt))
    coef2 = coef1 * e_half
    return coef0, coef1, coef2


def _build_damping_axis(coord, lo, hi, *, side_low, side_high,
                        pml_L, d_max, alpha_max, poly, dtype):
    """SPECFEM damping/frequency-shift profile along one axis."""
    d = torch.zeros_like(coord)
    alpha = torch.zeros_like(coord)
    if side_low:
        in_lo = coord < (lo + pml_L)
        s = ((lo + pml_L - coord) / pml_L).clamp(min=0.0, max=1.0)
        d = d + in_lo.to(dtype) * d_max * s.pow(poly)
        alpha = alpha + in_lo.to(dtype) * alpha_max * (1.0 - s)
    if side_high:
        in_hi = coord > (hi - pml_L)
        s = ((coord - (hi - pml_L)) / pml_L).clamp(min=0.0, max=1.0)
        d = d + in_hi.to(dtype) * d_max * s.pow(poly)
        alpha = alpha + in_hi.to(dtype) * alpha_max * (1.0 - s)
    return d, alpha


def _safe_div(num, den):
    return num / torch.where(den.abs() > _TINY, den, torch.ones_like(den))


def outer_pml_edge_nodes(mesh, sides) -> torch.Tensor:
    """Global node indices on the OUTER boundary of the PML (the domain edge on
    each absorbing side): the rigid-wall termination.

    A C-PML must be closed by a Dirichlet (u=0) wall at its outer edge. There
    the damping profile's frequency shift α→0, so the recursive-convolution
    memory coefficient ``coef0 = exp(-α·Δt) → 1`` — an *undamped integrator* —
    while its paired stretch coefficient is ``~d_max`` (largest at the outer
    edge). Left free (Neumann), that integrator accumulates the outer-edge
    field without bound and the layer blows up (worst at a free-surface∩PML
    corner). Pinning u=0 there — SPECFEM2D's standard PML termination — keeps
    the field, hence the integrator input, at zero. Excludes the ``top`` free
    surface even if the profile does not damp it."""
    xg, zg = mesh.x_global, mesh.z_global
    tol = max(1e-9, 1e-6 * max(float(mesh.x_max), float(mesh.z_max)))
    mask = torch.zeros(mesh.n_global, dtype=torch.bool, device=xg.device)
    if "left" in sides:
        mask |= xg.abs() < tol
    if "right" in sides:
        mask |= (xg - mesh.x_max).abs() < tol
    if "top" in sides:
        mask |= zg.abs() < tol
    if "bottom" in sides:
        mask |= (zg - mesh.z_max).abs() < tol
    return torch.nonzero(mask, as_tuple=False).reshape(-1)


def _lik_parameters(kx, bx, ax, kz, bz, az, mX, mZ, mXZ, index_ik, dt):
    """Vectorised port of SPECFEM ``lik_parameter_computation`` (strain stretch).

    Arguments are the *first* coordinate triple ``(kx,bx,ax)`` and the *second*
    triple ``(kz,bz,az)`` exactly as passed at the Fortran call site; ``index_ik``
    is 13 or 31 and permutes which physical region maps to the X-only / Z-only
    branch.  ``mX, mZ, mXZ`` are boolean masks of the *actual* element region
    (X-only / Z-only / corner).  Returns ``(A0, A1, A2, c1, c2)`` where ``c1``
    and ``c2`` are ``(coef0, coef1, coef2)`` tuples for branch 1 (bb = ax) and
    branch 2 (bb = bz)."""
    if index_ik == 13:
        m_xbranch, m_zbranch = mX, mZ          # X_ONLY_TEMP = X_ONLY
    elif index_ik == 31:
        m_xbranch, m_zbranch = mZ, mX          # X_ONLY_TEMP = Z_ONLY
    else:
        raise ValueError("index_ik must be 13 or 31")

    A0 = torch.ones_like(ax)
    A1 = torch.zeros_like(ax)
    A2 = torch.zeros_like(ax)

    # corner (XZ)
    gamma_x = _safe_div(ax * bz + ax ** 2 + 2.0 * bx * az - 2.0 * ax * (bx + az),
                        bz - ax)
    gamma_z = _safe_div(ax * bz + bz ** 2 + 2.0 * bx * az - 2.0 * bz * (bx + az),
                        ax - bz)
    A0 = torch.where(mXZ, kx / kz, A0)
    A1 = torch.where(mXZ, 0.5 * (kx / kz) * (gamma_x - ax), A1)
    A2 = torch.where(mXZ, 0.5 * (kx / kz) * (gamma_z - bz), A2)

    # X-branch content (bar_A_0 = kappa_x, A1 = beta_x - alpha_x, A2 = 0)
    A0 = torch.where(m_xbranch, kx, A0)
    A1 = torch.where(m_xbranch, bx - ax, A1)
    A2 = torch.where(m_xbranch, torch.zeros_like(ax), A2)

    # Z-branch content (bar_A_0 = 1/kappa_z, A1 = 0, A2 = -(beta_z - alpha_z))
    A0 = torch.where(m_zbranch, 1.0 / kz, A0)
    A1 = torch.where(m_zbranch, torch.zeros_like(ax), A1)
    A2 = torch.where(m_zbranch, -(bz - az), A2)

    c1 = _compute_coef_convolution(ax, dt)     # bb_1 = alpha_x_arg
    c2 = _compute_coef_convolution(bz, dt)     # bb_2 = beta_z_arg
    return A0, A1, A2, c1, c2


def _l_parameters(kx, bx, ax, kz, bz, az, mX, mZ, mXZ, dt):
    """Vectorised port of SPECFEM ``l_parameter_computation`` (accel / mass term).

    Returns ``(A0, A1, A2, A3, A4, c1, c2)`` with ``c1`` (bb = alpha_x) and
    ``c2`` (bb = alpha_z) the ``(coef0, coef1, coef2)`` branches."""
    A0 = torch.ones_like(ax)
    A1 = torch.zeros_like(ax)
    A2 = torch.zeros_like(ax)
    A3 = torch.zeros_like(ax)
    A4 = torch.zeros_like(ax)

    # corner (XZ)
    gamma_x = _safe_div(ax * az + ax ** 2 + 2.0 * bx * bz - 2.0 * ax * (bx + bz),
                        az - ax)
    gamma_z = _safe_div(ax * az + az ** 2 + 2.0 * bx * bz - 2.0 * az * (bx + bz),
                        ax - az)
    barA0 = kx * kz
    A0 = torch.where(mXZ, barA0, A0)
    A1 = torch.where(mXZ, 0.5 * barA0 * (gamma_x - ax + gamma_z - az), A1)
    A2 = torch.where(mXZ, 0.5 * barA0 * (ax ** 2 - gamma_x * ax
                                         + az ** 2 - gamma_z * az), A2)
    A3 = torch.where(mXZ, 0.5 * barA0 * ax ** 2 * (gamma_x - ax), A3)
    A4 = torch.where(mXZ, 0.5 * barA0 * az ** 2 * (gamma_z - az), A4)

    # X-only
    A0 = torch.where(mX, kx, A0)
    A1 = torch.where(mX, kx * (bx - ax), A1)
    A2 = torch.where(mX, -kx * ax * (bx - ax), A2)
    A3 = torch.where(mX, kx * ax ** 2 * (bx - ax), A3)
    A4 = torch.where(mX, torch.zeros_like(ax), A4)

    # Z-only
    A0 = torch.where(mZ, kz, A0)
    A1 = torch.where(mZ, kz * (bz - az), A1)
    A2 = torch.where(mZ, -kz * az * (bz - az), A2)
    A3 = torch.where(mZ, torch.zeros_like(ax), A3)
    A4 = torch.where(mZ, kz * az ** 2 * (bz - az), A4)

    c1 = _compute_coef_convolution(ax, dt)     # bb_1 = alpha_x
    c2 = _compute_coef_convolution(az, dt)     # bb_2 = alpha_z
    return A0, A1, A2, A3, A4, c1, c2


@dataclass
class ElasticCPMLProfile:
    """Per-element-GLL SPECFEM-style coefficients for the 2nd-order elastic CPML.

    Every field has shape ``(E, ngll, ngll)``.  Outside the PML the strain
    coefficients reduce to a no-op (``S?0 = 1``, ``S?1 = S?2 = 0``) and the
    accel coefficients vanish, so the recursions are safe to run everywhere.
    The recursive-convolution coefficients are masked to zero outside the PML so
    the (unused) memory branches never accumulate.
    """
    sides: tuple
    pml_thickness: float
    node_in_pml: torch.Tensor          # (E, ngll, ngll) bool — d_x>0 or d_z>0

    # 1) strain stretch of ∂/∂x derivatives (dux_dx, duz_dx)  [SPECFEM A5,A6,A7]
    Sx0: torch.Tensor
    Sx1: torch.Tensor
    Sx2: torch.Tensor
    zx1: tuple                         # (coef0, coef1, coef2) branch 1
    zx2: tuple

    # 1) strain stretch of ∂/∂z derivatives (dux_dz, duz_dz)  [SPECFEM A8,A9,A10]
    Sz0: torch.Tensor
    Sz1: torch.Tensor
    Sz2: torch.Tensor
    xz1: tuple
    xz2: tuple

    # 3) acceleration / mass contribution                     [SPECFEM A1..A4]
    La1: torch.Tensor
    La2: torch.Tensor
    La3: torch.Tensor
    La4: torch.Tensor
    l1: tuple                         # displ-memory branch 1 (bb = alpha_x)
    l2: tuple                         # displ-memory branch 2 (bb = alpha_z)

    # global node indices of the rigid-wall PML termination (see
    # outer_pml_edge_nodes); the equation pins u=0 here every step.
    outer_edge_idx: torch.Tensor

    # Multiaxial-PML cross-damping fraction (0 = classical uniaxial C-PML).
    mpml_ratio: float = 0.0


def build_elastic_cpml(
    mesh: Quad9Mesh,
    *,
    dt: float,
    sides: tuple = ("left", "right", "bottom"),
    pml_thickness_elements: int = 8,
    c_max: float,
    f_dominant: float,
    reflection_target: float = 1e-3,
    poly: float = 2.0,
    mpml_ratio: float = 0.0,
) -> ElasticCPMLProfile:
    """Compute the full SPECFEM2D elastic-CPML coefficient set at every GLL node.

    SPECFEM recipe (K_MAX = 1, no rotation):
        d_max  = -ln(R)·(p+1)·c / (2 L);   α_max = π f₀
        d(s)   = d_max·(s/L)^p   inward;   α(s) = α_max·(1 − s/L)
        β = α + d   (since κ = 1)
    then ``lik`` (strain stretch, two calls) and ``l`` (accel term).

    ``mpml_ratio`` (``p`` in Meza-Fajardo & Papageorgiou 2008): the **Multiaxial
    PML** cross-damping fraction.  Classical (uniaxial) C-PML damps only normal
    to each boundary (``d_x`` in the left/right bands, ``d_z`` in the bottom band),
    which is weakly reflective/unstable for grazing and surface (Rayleigh) waves in
    elastic media.  ``mpml_ratio > 0`` adds a small fraction of the normal damping
    in the tangential direction — ``d_x += p·d_z⁰``, ``d_z += p·d_x⁰`` — so every
    band becomes multiaxial (a "corner" in the region sense).  This stabilises the
    surface-wave absorption at the cost of the layer no longer being perfectly
    matched (a small reflection).  Typical ``p ∈ [0.1, 0.3]``; ``0`` = classical
    C-PML (default, exact SPECFEM2D behaviour).
    """
    dtype, device = mesh.x_e.dtype, mesh.x_e.device

    Lx = pml_thickness_elements * mesh.dx_e
    Lz = pml_thickness_elements * mesh.dz_e
    d_max_x = -np.log(reflection_target) * (poly + 1.0) * c_max / (2.0 * Lx)
    d_max_z = -np.log(reflection_target) * (poly + 1.0) * c_max / (2.0 * Lz)
    alpha_max = float(np.pi * f_dominant)

    d_x, alpha_x = _build_damping_axis(
        mesh.x_e, 0.0, mesh.x_max,
        side_low=("left" in sides), side_high=("right" in sides),
        pml_L=Lx, d_max=float(d_max_x), alpha_max=alpha_max, poly=poly, dtype=dtype)
    d_z, alpha_z = _build_damping_axis(
        mesh.z_e, 0.0, mesh.z_max,
        side_low=("top" in sides), side_high=("bottom" in sides),
        pml_L=Lz, d_max=float(d_max_z), alpha_max=alpha_max, poly=poly, dtype=dtype)

    # Multiaxial (M-PML) cross-damping: add p·(other-axis normal damping) so a
    # left/right band (classically X-only) also damps z and vice versa. This turns
    # every band into a multiaxial "corner" (has_dx & has_dz), which the region
    # logic + corner formulas below handle; the α frequency-shift stays per-axis
    # (0 on the non-normal axis), and the corner min_sep separation guards the
    # γ-divisions where the augmented β crosses the other axis's α.
    if mpml_ratio > 0.0:
        d_x0, d_z0 = d_x.clone(), d_z.clone()
        d_x = d_x0 + mpml_ratio * d_z0
        d_z = d_z0 + mpml_ratio * d_x0

    kappa_x = torch.ones_like(d_x)             # K_MAX = 1
    kappa_z = torch.ones_like(d_z)

    has_dx = d_x > 0
    has_dz = d_z > 0
    mX = has_dx & ~has_dz
    mZ = has_dz & ~has_dx
    mXZ = has_dx & has_dz
    in_pml = has_dx | has_dz

    # ------------------------------------------------------------------ #
    # XZ-corner parameter separation (SPECFEM2D pml_init.F90 lines ~1290-1347).
    # The corner γ-coefficients divide by (α_z−α_x), (β_z−α_x), (β_x−α_z); push
    # those apart by a small `min_sep` so they never blow up.  Only the corner
    # region is touched (X-only / Z-only use the singularity-free branches).
    # ------------------------------------------------------------------ #
    gll_gap = float((mesh.xi[1:] - mesh.xi[:-1]).abs().min()) * 0.5   # ref [-1,1]
    distance_min = gll_gap * min(float(mesh.dx_e), float(mesh.dz_e))
    min_sep = alpha_max / 8.0 * distance_min / max(float(Lx), float(Lz))
    sep2 = 2.0 * min_sep

    ax, az = alpha_x, alpha_z
    close = (ax - az).abs() < min_sep
    ax = torch.where(mXZ & close & (ax > az), az + sep2, ax)
    az = torch.where(mXZ & close & (ax <= az), ax + sep2, az)
    bx = ax + d_x                                  # κ = 1
    bz = az + d_z
    bx = torch.where(mXZ & ((bx - az).abs() < min_sep), az + sep2, bx)
    bz = torch.where(mXZ & ((bz - ax).abs() < min_sep), ax + sep2, bz)
    d_x = torch.where(mXZ, bx - ax, d_x)
    d_z = torch.where(mXZ, bz - az, d_z)
    alpha_x, alpha_z, beta_x, beta_z = ax, az, bx, bz

    # --- strain stretch -------------------------------------------------- #
    # x-derivatives: SPECFEM zx call  lik(kappa_z,beta_z,alpha_z, kappa_x,beta_x,alpha_x, 31)
    Sx0, Sx1, Sx2, zx1, zx2 = _lik_parameters(
        kappa_z, beta_z, alpha_z, kappa_x, beta_x, alpha_x, mX, mZ, mXZ, 31, dt)
    # z-derivatives: SPECFEM xz call  lik(kappa_x,beta_x,alpha_x, kappa_z,beta_z,alpha_z, 13)
    Sz0, Sz1, Sz2, xz1, xz2 = _lik_parameters(
        kappa_x, beta_x, alpha_x, kappa_z, beta_z, alpha_z, mX, mZ, mXZ, 13, dt)

    # --- acceleration / mass term --------------------------------------- #
    La0, La1, La2, La3, La4, l1, l2 = _l_parameters(
        kappa_x, beta_x, alpha_x, kappa_z, beta_z, alpha_z, mX, mZ, mXZ, dt)
    # K_MAX = 1  =>  La0 == 1 everywhere; assert so the mass matrix can stay raw.
    assert torch.allclose(La0, torch.ones_like(La0)), "K_MAX!=1 mass factor unsupported"

    # Mask the recursive-convolution coefficients to zero outside the PML so the
    # (unused) memory branches never accumulate over the large-strain interior.
    m = in_pml.to(dtype)

    def _mask(c):
        c0, c1, c2 = c
        return (c0 * m, c1 * m, c2 * m)

    return ElasticCPMLProfile(
        sides=tuple(sides), pml_thickness=float(max(Lx, Lz)),
        node_in_pml=in_pml,
        Sx0=Sx0, Sx1=Sx1, Sx2=Sx2, zx1=_mask(zx1), zx2=_mask(zx2),
        Sz0=Sz0, Sz1=Sz1, Sz2=Sz2, xz1=_mask(xz1), xz2=_mask(xz2),
        La1=La1, La2=La2, La3=La3, La4=La4, l1=_mask(l1), l2=_mask(l2),
        outer_edge_idx=outer_pml_edge_nodes(mesh, sides),
        mpml_ratio=float(mpml_ratio),
    )


def mask_elastic_profile(profile: ElasticCPMLProfile,
                         is_elastic: "torch.Tensor") -> ElasticCPMLProfile:
    """Domain-restrict a profile: freeze every memory recursion on NON-elastic
    elements (fluid-solid coupling).

    Multiplying all six convolution-coefficient triples by the elastic element
    mask turns their recursions into ``r' = 0·r + 0·d + 0·d_old ≡ 0`` there, so
    a cross-interface fluid element (whose edge nodes carry a live solid
    displacement) can never accumulate dead elastic memory. The stress-stretch
    ``S*`` fields are left untouched — their output is killed downstream by the
    equation's element mask — and ``La*`` are masked as hygiene (their output
    already dies with the ρ-mass weight). ``node_in_pml`` is intersected with
    the mask so boundary saving attributes the fluid part of the bands to the
    acoustic profile only.
    """
    m3 = is_elastic.reshape(-1, 1, 1)
    mf = m3.to(profile.La1.dtype)
    mask_t = lambda tri: tuple(c * mf for c in tri)
    return ElasticCPMLProfile(
        sides=profile.sides, pml_thickness=profile.pml_thickness,
        node_in_pml=profile.node_in_pml & m3,
        Sx0=profile.Sx0, Sx1=profile.Sx1, Sx2=profile.Sx2,
        zx1=mask_t(profile.zx1), zx2=mask_t(profile.zx2),
        Sz0=profile.Sz0, Sz1=profile.Sz1, Sz2=profile.Sz2,
        xz1=mask_t(profile.xz1), xz2=mask_t(profile.xz2),
        La1=profile.La1 * mf, La2=profile.La2 * mf,
        La3=profile.La3 * mf, La4=profile.La4 * mf,
        l1=mask_t(profile.l1), l2=mask_t(profile.l2),
        outer_edge_idx=profile.outer_edge_idx,
        mpml_ratio=profile.mpml_ratio,
    )
