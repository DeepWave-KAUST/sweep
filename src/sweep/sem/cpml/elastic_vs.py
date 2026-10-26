"""FD-style single-recursion CPML for 1st-order elastic velocity-stress SEM.

Mirrors :func:`sweep.equations.pml.setup_pml` (the same Komatitsch-Martin
CPML that the SWEEP FD elastic kernels use). Per element-GLL node we
store the discrete recursion coefficients

    a = exp(-(σ + α) · dt)
    b = σ / (σ + α) · (a − 1)

separately for the x- and z-stretching, with σ = σ₀ · (depth/L)^p and
α = α_max · (1 − depth/L). Outside the PML σ = 0 and the (a, b)
recursion becomes a no-op.

One memory variable per derivative-direction-and-field, applied at the
derivative level (velocity gradients in the stress update, stress
gradients in the velocity update). Consumed by
:class:`ElasticVSSEM2DDeformed`.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch

from ._axis import vs_axis_profile as _axis_profile

from ..mesh.quad9 import Quad9Mesh


@dataclass
class SimpleElasticCPMLProfile:
    """FD-style 1-stage CPML profile, per element-GLL node.

    Storage: each tensor has shape ``(E, ngll, ngll)``. Outside the PML
    both ``a_x`` and ``b_x`` (or ``a_z``, ``b_z``) are zero, so the
    recursion ``m' = a · m + b · ∂`` becomes ``m' = 0`` and ``∂̃ = ∂``.
    """
    sides: tuple
    pml_thickness: float

    a_x: torch.Tensor                 # (E, ngll, ngll)
    b_x: torch.Tensor
    a_z: torch.Tensor
    b_z: torch.Tensor


def build_simple_elastic_cpml(
    mesh: Quad9Mesh,
    *,
    dt: float,
    sides: tuple = ("left", "right", "bottom"),
    pml_thickness_elements: int = 8,
    c_max: float,
    f_dominant: float,
    reflection_target: float = 1e-3,
    poly: float = 2.0,
) -> SimpleElasticCPMLProfile:
    """Build the FD-style CPML profile for ``mesh``.

    Parameters mirror :func:`sweep.equations.pml.setup_pml`:

      σ_0   = -(n + 1) · c_max · ln(R) / (2 L)
      α_max = π · f_0
      σ(x) = σ_0 · (depth/L)^n
      α(x) = α_max · (1 − depth/L)

    Then per axis:

      a = exp(-(σ + α) · dt)
      b = σ / (σ + α) · (a − 1)
    """
    dtype, device = mesh.x_e.dtype, mesh.x_e.device

    Lx = pml_thickness_elements * mesh.dx_e
    Lz = pml_thickness_elements * mesh.dz_e
    log_R = math.log(reflection_target)
    sigma_0_x = -(poly + 1) * c_max * log_R / (2.0 * Lx)
    sigma_0_z = -(poly + 1) * c_max * log_R / (2.0 * Lz)
    alpha_max = math.pi * f_dominant

    a_x, b_x = _axis_profile(
        mesh.x_e, 0.0, mesh.x_max,
        side_low=("left" in sides), side_high=("right" in sides),
        pml_L=Lx, sigma_0=sigma_0_x, alpha_max=alpha_max,
        poly=poly, dt=dt, dtype=dtype,
    )
    a_z, b_z = _axis_profile(
        mesh.z_e, 0.0, mesh.z_max,
        side_low=("top" in sides), side_high=("bottom" in sides),
        pml_L=Lz, sigma_0=sigma_0_z, alpha_max=alpha_max,
        poly=poly, dt=dt, dtype=dtype,
    )

    return SimpleElasticCPMLProfile(
        sides=tuple(sides),
        pml_thickness=max(Lx, Lz),
        a_x=a_x, b_x=b_x, a_z=a_z, b_z=b_z,
    )
