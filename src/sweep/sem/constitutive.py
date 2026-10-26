"""Layout-agnostic constitutive laws shared across SEM equations.

These are pure point-wise tensor functions — they do not know about the SEM
mesh, GLL nodes, or the FD grid, so the same code can (in principle) back both
the SEM velocity-stress equations and the FD-eager constitutive step. They map
material parameters to the 2-D plane (x, z) Voigt stiffness coefficients used by
the velocity-stress update ``dσ = dt · C : ∇v``.
"""

from __future__ import annotations

import torch


def thomsen_to_vti_cij(vp, vs, epsilon, delta, rho):
    """VTI (vertical transverse isotropy) Voigt stiffnesses from Thomsen (1986).

    Returns ``(c11, c13, c33, c55)`` for the in-plane P-SV system, where the
    velocity-stress constitutive update is

        σ_xx += dt (c11 ∂x v_x + c13 ∂z v_z)
        σ_zz += dt (c13 ∂x v_x + c33 ∂z v_z)
        σ_xz += dt  c55 (∂z v_x + ∂x v_z)

    with ``vp`` / ``vs`` the *vertical* P / S velocities. Thomsen ε, δ are the
    standard anisotropy parameters:

        c33 = ρ vp²            (vertical P modulus)
        c55 = ρ vs²            (vertical S modulus = c44 in 2D)
        c11 = c33 (1 + 2ε)     (horizontal P modulus)
        c13 = √[2δ c33 (c33−c55) + (c33−c55)²] − c55

    ``ε = δ = 0`` reduces exactly to isotropy: ``c11 = c33 = λ+2μ``,
    ``c13 = λ``, ``c55 = μ`` — so a VTI equation degenerates bit-for-bit to the
    isotropic one (verified in the test-suite).
    """
    c33 = rho * vp * vp
    c55 = rho * vs * vs
    c11 = c33 * (1.0 + 2.0 * epsilon)
    radicand = 2.0 * delta * c33 * (c33 - c55) + (c33 - c55) ** 2
    c13 = torch.sqrt(torch.clamp(radicand, min=0.0)) - c55
    return c11, c13, c33, c55
