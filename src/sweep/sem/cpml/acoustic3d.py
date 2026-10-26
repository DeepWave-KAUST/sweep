"""CPML profiles for SEM acoustic 3-D — double-convolution form, the 3-D
analogue of :mod:`sweep.sem.cpml.acoustic` (Komatitsch & Martin 2007).

Per-direction (x, y, z) two auxiliary fields ψ and ζ are carried at every
global mesh node. The recursive update uses the sweep-FD convention

    a(s) = exp(-(σ + α) Δt)            # decay factor
    b(s) = (σ / (σ + α)) · (a - 1)     # convolution kernel weight (≤ 0)

Outside the PML strip σ = α = 0 → a = 1, b = 0 so ψ, ζ stay frozen at zero.
Inside the strip the standard Collino-Tsogka polynomial profile is used:

    σ(s) = σ_max · (s/L)^p ,    α(s) = α_max · (1 - s/L)
    σ_max = -ln(R) · (p+1) · c_max / (2 L)
    α_max = π · f_dominant

The spatial derivatives ``∂b/∂x``, ``∂b/∂y``, ``∂b/∂z`` (needed in the
double-convolution wave step) are precomputed with ``np.gradient`` on the
structured global mesh (row-major (z, y, x)), which handles the non-uniform
GLL spacing exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ._axis import damping_profile as _damping_profile

from ..mesh.hex8 import Hex8Mesh


@dataclass
class CPML3DProfile:
    """Per-global-node CPML recursion coefficients (3-D double-convolution)."""

    sides: tuple[str, ...]
    pml_thickness: float
    a_x: torch.Tensor
    b_x: torch.Tensor
    dbx_dx: torch.Tensor
    a_y: torch.Tensor
    b_y: torch.Tensor
    dby_dy: torch.Tensor
    a_z: torch.Tensor
    b_z: torch.Tensor
    dbz_dz: torch.Tensor


def _ab_coeffs(sigma, alpha, dt):
    """sweep-FD-convention (a, b) from (σ, α)."""
    a = torch.exp(-dt * (sigma + alpha))
    denom = sigma + alpha
    safe = torch.where(denom > 0, denom, torch.ones_like(denom))
    b = (sigma / safe) * (a - 1.0)
    b = torch.where(sigma > 0, b, torch.zeros_like(b))
    return a, b


def build_cpml3d(
    mesh: Hex8Mesh,
    *,
    dt: float,
    sides: tuple[str, ...] = ("left", "right", "front", "back", "bottom"),
    pml_thickness_elements: int = 8,
    c_max: float,
    f_dominant: float,
    reflection_target: float = 1e-3,
    poly: float = 2.0,
) -> CPML3DProfile:
    """Build a Komatitsch-Martin 2007 double-convolution 3-D CPML profile.

    Parameters
    ----------
    sides
        Subset of ``{"left", "right", "front", "back", "top", "bottom"}``.
        Top is usually omitted (free surface).
    pml_thickness_elements
        Number of mesh elements in each PML strip.
    c_max, f_dominant
        Peak velocity and dominant source frequency (drive σ_max / α_max).
    reflection_target
        Target normal-incidence reflection coefficient (Collino-Tsogka 2001).
    poly
        Polynomial taper power (2 = conventional quadratic).
    """
    valid = {"left", "right", "front", "back", "top", "bottom"}
    if any(s not in valid for s in sides):
        raise ValueError(f"unknown side in {sides!r}")

    dtype, device = mesh.x_global.dtype, mesh.x_global.device
    Lx = pml_thickness_elements * mesh.dx_e
    Ly = pml_thickness_elements * mesh.dy_e
    Lz = pml_thickness_elements * mesh.dz_e
    smax = lambda L: -np.log(reflection_target) * (poly + 1) * c_max / (2.0 * L)
    alpha_max = float(np.pi * f_dominant)

    sigma_x, alpha_x = _damping_profile(
        mesh.x_global, 0.0, mesh.x_max,
        side_low="left" in sides, side_high="right" in sides,
        pml_thickness=Lx, sigma_max=float(smax(Lx)),
        alpha_max=alpha_max, poly=poly)
    sigma_y, alpha_y = _damping_profile(
        mesh.y_global, 0.0, mesh.y_max,
        side_low="front" in sides, side_high="back" in sides,
        pml_thickness=Ly, sigma_max=float(smax(Ly)),
        alpha_max=alpha_max, poly=poly)
    sigma_z, alpha_z = _damping_profile(
        mesh.z_global, 0.0, mesh.z_max,
        side_low="top" in sides, side_high="bottom" in sides,
        pml_thickness=Lz, sigma_max=float(smax(Lz)),
        alpha_max=alpha_max, poly=poly)

    a_x, b_x = _ab_coeffs(sigma_x, alpha_x, dt)
    a_y, b_y = _ab_coeffs(sigma_y, alpha_y, dt)
    a_z, b_z = _ab_coeffs(sigma_z, alpha_z, dt)

    # ∂b/∂* via numerical gradient on the structured global mesh (row-major
    # (z, y, x)). np.gradient handles non-uniform GLL spacing when given coords.
    nx_g = mesh.nex * mesh.n_order + 1
    ny_g = mesh.ney * mesh.n_order + 1
    nz_g = mesh.nez * mesh.n_order + 1
    shape3 = (nz_g, ny_g, nx_g)
    x_axis = mesh.x_global.reshape(shape3)[0, 0, :].detach().cpu().numpy()
    y_axis = mesh.y_global.reshape(shape3)[0, :, 0].detach().cpu().numpy()
    z_axis = mesh.z_global.reshape(shape3)[:, 0, 0].detach().cpu().numpy()
    dbx_dx = np.gradient(b_x.reshape(shape3).detach().cpu().numpy(), x_axis, axis=2)
    dby_dy = np.gradient(b_y.reshape(shape3).detach().cpu().numpy(), y_axis, axis=1)
    dbz_dz = np.gradient(b_z.reshape(shape3).detach().cpu().numpy(), z_axis, axis=0)
    to_t = lambda arr: torch.as_tensor(arr.reshape(-1), dtype=dtype, device=device)

    return CPML3DProfile(
        sides=tuple(sides),
        pml_thickness=max(Lx, Ly, Lz),
        a_x=a_x.to(dtype), b_x=b_x.to(dtype), dbx_dx=to_t(dbx_dx),
        a_y=a_y.to(dtype), b_y=b_y.to(dtype), dby_dy=to_t(dby_dy),
        a_z=a_z.to(dtype), b_z=b_z.to(dtype), dbz_dz=to_t(dbz_dz),
    )
