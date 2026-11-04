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


def thomsen_to_tti_cij(vp, vs, epsilon, delta, rho, theta):
    """TTI (tilted transverse isotropy) Voigt stiffnesses: the VTI medium of
    :func:`thomsen_to_vti_cij` with its symmetry axis tilted by ``theta``
    (radians) in the (x, z) plane.

    Returns ``(c11, c13, c15, c33, c35, c55)`` for the velocity-stress update

        σ_xx += dt (c11 ∂x v_x + c13 ∂z v_z + c15 (∂z v_x + ∂x v_z))
        σ_zz += dt (c13 ∂x v_x + c33 ∂z v_z + c35 (∂z v_x + ∂x v_z))
        σ_xz += dt (c15 ∂x v_x + c35 ∂z v_z + c55 (∂z v_x + ∂x v_z))

    by the closed-form 2-D Bond rotation of the FD ``ElasticTTI2nd`` (Oh et al.
    2020, eq. 10), in its axes (z down) and with its sign of ``theta``.
    ``theta = 0`` gives the VTI stiffnesses exactly, with ``c15 = c35 = 0``.
    """
    c11, c13, c33, c55 = thomsen_to_vti_cij(vp, vs, epsilon, delta, rho)
    ct, st = torch.cos(theta), torch.sin(theta)
    ct2, st2 = ct * ct, st * st
    s2c2 = st2 * ct2
    mix = 2.0 * c13 + 4.0 * c55
    a = c13 - c11 + 2.0 * c55
    b = c13 - c33 + 2.0 * c55
    return (c11 * ct2 * ct2 + c33 * st2 * st2 + mix * s2c2,
            (c11 + c33 - 4.0 * c55) * s2c2 + c13 * (st2 * st2 + ct2 * ct2),
            a * ct2 * ct * st - b * ct * st2 * st,
            c11 * st2 * st2 + c33 * ct2 * ct2 + mix * s2c2,
            a * st2 * st * ct - b * st * ct2 * ct,
            (c11 + c33 - 2.0 * c13) * s2c2 + c55 * (ct2 - st2) ** 2)
