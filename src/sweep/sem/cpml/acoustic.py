"""CPML profiles for SEM acoustic 2-D — double-convolution form matching
``sweep.equations.acoustic.step_cpml`` (Komatitsch & Martin 2007).

Per-direction, two auxiliary fields ψ and ζ are carried at every global
mesh node. The recursive update uses the sweep-FD convention

    a(x) = exp(-(σ + α) Δt)            # decay factor
    b(x) = (σ / (σ + α)) · (a - 1)     # convolution kernel weight (≤ 0)

Outside the PML strip σ = α = 0 → a = 1, b = 0 so ψ, ζ stay frozen at
zero (no extra cost). Inside the strip the standard Collino-Tsogka
polynomial profile is used:

    σ(s) = σ_max · (s/L)^p ,    α(s) = α_max · (1 - s/L)
    σ_max = -ln(R) · (p+1) · c_max / (2 L)
    α_max = π · f_dominant

with ``s ∈ [0, L]`` the depth inside the PML and ``p`` the taper order
(default 2).

The spatial derivative ``∂b/∂x`` (needed in the double-convolution wave
step) is precomputed by finite differences on the structured global mesh,
``np.gradient``'s arithmetic along every grid line with its own coordinates
(``_axis.gradient_along``), which takes the non-uniform GLL spacing and the
graded columns of a topography or interfaces mesh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from ._axis import damping_profile as _damping_profile
from ._axis import gradient_along as _gradient_along

from ..mesh.quad4 import Quad4Mesh


@dataclass
class CPMLProfile:
    """Per-global-node CPML recursion coefficients (double-convolution form)."""

    sides: tuple[str, ...]
    pml_thickness: float
    a_x: torch.Tensor              # (n_global,)
    b_x: torch.Tensor              # (n_global,)
    dbx_dx: torch.Tensor           # (n_global,)
    a_z: torch.Tensor              # (n_global,)
    b_z: torch.Tensor              # (n_global,)
    dbz_dz: torch.Tensor           # (n_global,)
    # the damping and frequency shift a/b are built from, (n_global,) each;
    # the fluid-solid coupling stretches its interface load with them
    sigma_x: Optional[torch.Tensor] = None
    alpha_x: Optional[torch.Tensor] = None
    sigma_z: Optional[torch.Tensor] = None
    alpha_z: Optional[torch.Tensor] = None


def build_cpml(
    mesh: Quad4Mesh,
    *,
    dt: float,
    sides: tuple[str, ...] = ("left", "right", "bottom"),
    pml_thickness_elements: int = 8,
    c_max: float,
    f_dominant: float,
    reflection_target: float = 1e-3,
    poly: float = 2.0,
) -> CPMLProfile:
    """Build a Komatitsch-Martin 2007 double-convolution CPML profile.

    Parameters
    ----------
    sides
        Subset of ``{"left", "right", "top", "bottom"}``. Top is usually
        omitted (free surface).
    pml_thickness_elements
        Number of mesh elements in each PML strip; physical thickness =
        ``pml_thickness_elements · dh_e``.
    c_max, f_dominant
        Peak velocity and dominant source frequency, used to derive
        ``σ_max`` and ``α_max``.
    reflection_target
        Target normal-incidence reflection coefficient (drives σ_max via
        Collino-Tsogka 2001 formula).
    poly
        Polynomial taper power; 2 is the conventional quadratic profile.
    """
    if any(s not in {"left", "right", "top", "bottom"} for s in sides):
        raise ValueError(f"unknown side in {sides!r}")

    dtype, device = mesh.x_global.dtype, mesh.x_global.device
    Lx = pml_thickness_elements * mesh.dx_e
    Lz = pml_thickness_elements * mesh.dz_e
    sigma_max_x = -np.log(reflection_target) * (poly + 1) * c_max / (2.0 * Lx)
    sigma_max_z = -np.log(reflection_target) * (poly + 1) * c_max / (2.0 * Lz)
    alpha_max = float(np.pi * f_dominant)

    sigma_x, alpha_x = _damping_profile(
        mesh.x_global, 0.0, mesh.x_max,
        side_low="left" in sides, side_high="right" in sides,
        pml_thickness=Lx, sigma_max=float(sigma_max_x),
        alpha_max=alpha_max, poly=poly,
    )
    sigma_z, alpha_z = _damping_profile(
        mesh.z_global, 0.0, mesh.z_max,
        side_low="top" in sides, side_high="bottom" in sides,
        pml_thickness=Lz, sigma_max=float(sigma_max_z),
        alpha_max=alpha_max, poly=poly,
    )

    # sweep-FD-convention coefficients.
    a_x = torch.exp(-dt * (sigma_x + alpha_x))
    denom_x = sigma_x + alpha_x
    safe_x = torch.where(denom_x > 0, denom_x, torch.ones_like(denom_x))
    b_x = (sigma_x / safe_x) * (a_x - 1.0)
    b_x = torch.where(sigma_x > 0, b_x, torch.zeros_like(b_x))

    a_z = torch.exp(-dt * (sigma_z + alpha_z))
    denom_z = sigma_z + alpha_z
    safe_z = torch.where(denom_z > 0, denom_z, torch.ones_like(denom_z))
    b_z = (sigma_z / safe_z) * (a_z - 1.0)
    b_z = torch.where(sigma_z > 0, b_z, torch.zeros_like(b_z))

    # ∂b/∂x and ∂b/∂z by finite differences on the structured global mesh
    # (global node (i_z, i_x), row-major), along every row and column with its
    # own coordinates: a topography or interfaces mesh grades each column
    # differently, and column 0's z axis made ∂b/∂z off by the ratio of the
    # column heights.
    nx_g = mesh.nex * mesh.n_order + 1
    nz_g = mesh.nez * mesh.n_order + 1
    x_2d = mesh.x_global.reshape(nz_g, nx_g).detach().cpu().numpy()
    z_2d = mesh.z_global.reshape(nz_g, nx_g).detach().cpu().numpy()
    b_x_2d = b_x.reshape(nz_g, nx_g).detach().cpu().numpy()
    b_z_2d = b_z.reshape(nz_g, nx_g).detach().cpu().numpy()
    dbx_dx_2d = _gradient_along(b_x_2d, x_2d, axis=1)
    dbz_dz_2d = _gradient_along(b_z_2d, z_2d, axis=0)
    dbx_dx = torch.as_tensor(dbx_dx_2d.reshape(-1), dtype=dtype, device=device)
    dbz_dz = torch.as_tensor(dbz_dz_2d.reshape(-1), dtype=dtype, device=device)

    return CPMLProfile(
        sides=tuple(sides),
        pml_thickness=max(Lx, Lz),
        a_x=a_x.to(dtype),  b_x=b_x.to(dtype),  dbx_dx=dbx_dx,
        a_z=a_z.to(dtype),  b_z=b_z.to(dtype),  dbz_dz=dbz_dz,
        sigma_x=sigma_x.to(dtype), alpha_x=alpha_x.to(dtype),
        sigma_z=sigma_z.to(dtype), alpha_z=alpha_z.to(dtype),
    )
