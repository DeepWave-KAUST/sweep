"""FD-style single-recursion CPML for the 3-D 1st-order elastic velocity-stress
SEM — the 3-D analogue of :mod:`sweep.sem.cpml.elastic_vs`.

Per element-GLL node we store the discrete Komatitsch-Martin recursion
coefficients

    a = exp(-(σ + α) · dt)
    b = σ / (σ + α) · (a − 1)

for each of the three stretching axes (x, y, z), with σ = σ₀·(depth/L)^p and
α = α_max·(1 − depth/L). Outside the PML σ = 0 so the recursion
``m' = a·m + b·∂`` is a no-op. One memory variable per derivative-direction-and-
field, applied at the derivative level. Consumed by :class:`ElasticVSSEM3D`.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch

from ._axis import vs_axis_profile as _axis_profile

from ..mesh.hex8 import Hex8Mesh


@dataclass
class SimpleElasticCPML3DProfile:
    """FD-style 1-stage 3-D CPML profile, per element-GLL node ``(E,g,g,g)``."""
    sides: tuple
    pml_thickness: float
    a_x: torch.Tensor
    b_x: torch.Tensor
    a_y: torch.Tensor
    b_y: torch.Tensor
    a_z: torch.Tensor
    b_z: torch.Tensor


def build_simple_elastic_cpml_3d(
    mesh: Hex8Mesh,
    *,
    dt: float,
    sides: tuple = ("left", "right", "front", "back", "bottom"),
    pml_thickness_elements: int = 8,
    c_max: float,
    f_dominant: float,
    reflection_target: float = 1e-3,
    poly: float = 2.0,
) -> SimpleElasticCPML3DProfile:
    """Build the FD-style 3-D CPML profile for ``mesh`` (one (a,b) pair per axis).

      σ_0   = -(n + 1) · c_max · ln(R) / (2 L)
      α_max = π · f_0
      σ(s) = σ_0 · (depth/L)^n ;  α(s) = α_max · (1 − depth/L)
    """
    valid = {"left", "right", "front", "back", "top", "bottom"}
    if any(s not in valid for s in sides):
        raise ValueError(f"unknown side in {sides!r}")
    dtype = mesh.x_e.dtype
    Lx = pml_thickness_elements * mesh.dx_e
    Ly = pml_thickness_elements * mesh.dy_e
    Lz = pml_thickness_elements * mesh.dz_e
    log_R = math.log(reflection_target)
    s0 = lambda L: -(poly + 1) * c_max * log_R / (2.0 * L)
    alpha_max = math.pi * f_dominant

    a_x, b_x = _axis_profile(
        mesh.x_e, 0.0, mesh.x_max, side_low=("left" in sides),
        side_high=("right" in sides), pml_L=Lx, sigma_0=s0(Lx),
        alpha_max=alpha_max, poly=poly, dt=dt, dtype=dtype)
    a_y, b_y = _axis_profile(
        mesh.y_e, 0.0, mesh.y_max, side_low=("front" in sides),
        side_high=("back" in sides), pml_L=Ly, sigma_0=s0(Ly),
        alpha_max=alpha_max, poly=poly, dt=dt, dtype=dtype)
    a_z, b_z = _axis_profile(
        mesh.z_e, 0.0, mesh.z_max, side_low=("top" in sides),
        side_high=("bottom" in sides), pml_L=Lz, sigma_0=s0(Lz),
        alpha_max=alpha_max, poly=poly, dt=dt, dtype=dtype)

    return SimpleElasticCPML3DProfile(
        sides=tuple(sides), pml_thickness=max(Lx, Ly, Lz),
        a_x=a_x, b_x=b_x, a_y=a_y, b_y=b_y, a_z=a_z, b_z=b_z,
    )
