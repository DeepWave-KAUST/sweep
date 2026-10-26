"""Per-axis C-PML damping profiles shared by the builders.

Two families, deliberately NOT merged into one function:

* :func:`damping_profile` returns raw ``(sigma, alpha)`` -- the acoustic
  double-convolution builders (2-D and 3-D) turn those into their own
  recursion coefficients. The two copies were byte-identical.
* :func:`vs_axis_profile` returns the FD-style ``(a, b)`` recursion pair used
  by the velocity-stress builders (2-D and 3-D). The two copies differed only
  in line-wrapping; the profiles they build are bit-compared against
  pre-refactor snapshots in test_sem_cpml_axis_shared.py.

The SPECFEM2D displacement port (:mod:`.elastic`) keeps its own
``_build_damping_axis``: it feeds a three-coefficient strain-stretching scheme
whose needs genuinely differ, and forcing it through this module would trade
one copy for a parameter soup.
"""
from __future__ import annotations

import torch


def damping_profile(
    coord: torch.Tensor,
    lo: float,
    hi: float,
    side_low: bool,
    side_high: bool,
    pml_thickness: float,
    sigma_max: float,
    alpha_max: float,
    poly: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (σ, α) sampled at ``coord`` with low/high optional PML strips."""
    sigma = torch.zeros_like(coord)
    alpha = torch.zeros_like(coord)
    if side_low:
        in_low = coord < (lo + pml_thickness)
        s_low = ((lo + pml_thickness - coord) / pml_thickness).clamp(min=0.0)
        sigma = sigma + in_low.to(sigma.dtype) * sigma_max * s_low.pow(poly)
        alpha = alpha + in_low.to(sigma.dtype) * alpha_max * (1.0 - s_low)
    if side_high:
        in_high = coord > (hi - pml_thickness)
        s_high = ((coord - (hi - pml_thickness)) / pml_thickness).clamp(min=0.0)
        sigma = sigma + in_high.to(sigma.dtype) * sigma_max * s_high.pow(poly)
        alpha = alpha + in_high.to(sigma.dtype) * alpha_max * (1.0 - s_high)
    return sigma, alpha


def vs_axis_profile(coord, lo, hi, *, side_low, side_high,
                    pml_L, sigma_0, alpha_max, poly, dt, dtype):
    """Build (a, b) at every node for one axis (broadcasts coord shape)."""
    eps = 1e-12

    pml_frac0 = torch.zeros_like(coord)
    pml_frac1 = torch.zeros_like(coord)
    if side_low:
        pml_frac0 = ((lo + pml_L - coord) / pml_L).clamp(min=0.0, max=1.0)
    if side_high:
        pml_frac1 = ((coord - (hi - pml_L)) / pml_L).clamp(min=0.0, max=1.0)
    pml_frac = torch.maximum(pml_frac0, pml_frac1)

    sigma = sigma_0 * pml_frac.pow(poly)
    alpha = alpha_max * (1.0 - pml_frac)
    sa = sigma + alpha
    sa_safe = torch.where(sa.abs() < eps, torch.full_like(sa, eps), sa)

    a = torch.exp(-sa_safe * dt)
    b = (sigma / sa_safe) * (a - 1.0)

    # Outside PML: zero both (no recursion, no stretching).
    inside = pml_frac > 0
    a = torch.where(inside, a, torch.zeros_like(a))
    b = torch.where(inside, b, torch.zeros_like(b))
    return a.to(dtype=dtype), b.to(dtype=dtype)
