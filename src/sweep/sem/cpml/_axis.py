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
one copy for a parameter soup. All three decide "inside the layer" with
:func:`inner_edge_tol`.
"""
from __future__ import annotations

import numpy as np
import torch


def inner_edge_tol(lo: float, hi: float, dtype: torch.dtype) -> float:
    """Depth into a layer below which a node counts as ON its inner edge.

    A node on the inner edge sits at ``lo + L`` (``hi - L``) only up to
    round-off: its coordinate comes out of the element map and ``L`` from
    ``n * x_max / nex``, each an ulp or so off, in either direction. A strict
    ``coord < lo + L`` then puts it in the layer (``sigma ~ 1e-30 sigma_max``,
    ``alpha = alpha_max``) on whichever side the round-off falls, and every
    element touching it joins the C-PML band: one extra physical element
    column, on that side only. The offsets measured on flat, topography and
    3-D meshes stay within 3 ulps of the axis extent; 16 leaves a margin and
    is still far below the depth of the first GLL node inside the layer.
    """
    return 16.0 * torch.finfo(dtype).eps * max(abs(float(lo)), abs(float(hi)))


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
    tol = inner_edge_tol(lo, hi, coord.dtype)
    if side_low:
        depth = lo + pml_thickness - coord
        in_low = depth > tol
        s_low = (depth / pml_thickness).clamp(min=0.0)
        sigma = sigma + in_low.to(sigma.dtype) * sigma_max * s_low.pow(poly)
        alpha = alpha + in_low.to(sigma.dtype) * alpha_max * (1.0 - s_low)
    if side_high:
        depth = coord - (hi - pml_thickness)
        in_high = depth > tol
        s_high = (depth / pml_thickness).clamp(min=0.0)
        sigma = sigma + in_high.to(sigma.dtype) * sigma_max * s_high.pow(poly)
        alpha = alpha + in_high.to(sigma.dtype) * alpha_max * (1.0 - s_high)
    return sigma, alpha


def vs_axis_profile(coord, lo, hi, *, side_low, side_high,
                    pml_L, sigma_0, alpha_max, poly, dt, dtype):
    """Build (a, b) at every node for one axis (broadcasts coord shape)."""
    eps = 1e-12
    tol = inner_edge_tol(lo, hi, coord.dtype)

    pml_frac0 = torch.zeros_like(coord)
    pml_frac1 = torch.zeros_like(coord)
    inside = torch.zeros_like(coord, dtype=torch.bool)
    if side_low:
        depth = lo + pml_L - coord
        pml_frac0 = (depth / pml_L).clamp(min=0.0, max=1.0)
        inside |= depth > tol
    if side_high:
        depth = coord - (hi - pml_L)
        pml_frac1 = (depth / pml_L).clamp(min=0.0, max=1.0)
        inside |= depth > tol
    pml_frac = torch.maximum(pml_frac0, pml_frac1)

    sigma = sigma_0 * pml_frac.pow(poly)
    alpha = alpha_max * (1.0 - pml_frac)
    sa = sigma + alpha
    sa_safe = torch.where(sa.abs() < eps, torch.full_like(sa, eps), sa)

    a = torch.exp(-sa_safe * dt)
    b = (sigma / sa_safe) * (a - 1.0)

    # Outside PML: zero both (no recursion, no stretching).
    a = torch.where(inside, a, torch.zeros_like(a))
    b = torch.where(inside, b, torch.zeros_like(b))
    return a.to(dtype=dtype), b.to(dtype=dtype)


def gradient_along(f, x, axis):
    """``df/dx`` along ``axis`` of the NumPy arrays ``f`` and ``x`` (same shape), each grid
    line with its own coordinates: ``np.gradient(f_line, x_line)`` line by line, with its
    arithmetic (the non-uniform branch: second order inside, first order at the ends). A
    mesh graded column by column -- topography, internal interfaces -- has different
    coordinates on every line, which ``np.gradient`` with one coordinate axis cannot take."""
    f, x = np.moveaxis(f, axis, 0), np.moveaxis(x, axis, 0)
    dx = np.diff(x, axis=0)
    dx1, dx2 = dx[:-1], dx[1:]
    a = -(dx2) / (dx1 * (dx1 + dx2))
    b = (dx2 - dx1) / (dx1 * dx2)
    c = dx1 / (dx2 * (dx1 + dx2))
    out = np.empty_like(f)
    out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
    out[0] = (f[1] - f[0]) / dx[0]
    out[-1] = (f[-1] - f[-2]) / dx[-1]
    return np.moveaxis(out, 0, axis)
