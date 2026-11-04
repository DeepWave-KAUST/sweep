"""Layout-agnostic constitutive laws shared across SEM equations.

These are pure point-wise tensor functions — they do not know about the SEM
mesh, GLL nodes, or the FD grid, so the same code can (in principle) back both
the SEM velocity-stress equations and the FD-eager constitutive step. They map
material parameters to the Voigt stiffness coefficients used by the velocity-stress
update ``dσ = dt · C : ∇v``: those of the 2-D plane (x, z) and the 21 of 3-D.
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


def thomsen_to_cij3d(vp, vs, epsilon, delta, gamma, rho, theta=None, phi=None):
    """The 21 Voigt stiffnesses of a transversely isotropic medium in 3-D, in the order of the
    FD ``ElasticTTISG3D``'s ``STIFFNESS_KEYS_3D`` (``c11, c12, ..., c66``; rows and columns
    ``xx, yy, zz, yz, xz, xy``; z down).

    The VTI medium has the vertical velocities ``vp``, ``vs`` and Thomsen's ``epsilon``,
    ``delta``, ``gamma``::

        c33 = rho vp^2,  c44 = c55 = rho vs^2,  c11 = c22 = c33 (1 + 2 epsilon),
        c66 = c44 (1 + 2 gamma),  c12 = c11 - 2 c66,
        c13 = c23 = sqrt(2 delta c33 (c33 - c44) + (c33 - c44)^2) - c44

    (the 2-D :func:`thomsen_to_vti_cij` with ``c44``). ``theta`` (radians) tilts its symmetry
    axis from the vertical toward the azimuth ``phi`` (radians, from x toward y): the Bond
    rotation of ``ElasticTTISG3D`` (its ``R(theta, phi)``), which at ``phi = 0`` is the 2-D
    :func:`thomsen_to_tti_cij`. ``theta=None`` returns the VTI entries (the twelve others
    zero). Where ``|theta| < 1e-7`` the values are the VTI ones exactly, while the gradient stays
    the rotation's: an inversion that starts from ``theta = 0`` still sees ``d/d theta``,
    which the FD class's switch to the VTI entries there cuts.
    """
    from ..equations.elastic_tti import ElasticTTI
    from ..equations.elastic_tti_sg3d import _KEY_TO_VOIGT, STIFFNESS_KEYS_3D, ElasticTTISG3D

    c33 = rho * vp * vp
    c44 = rho * vs * vs
    c11 = c33 * (1.0 + 2.0 * epsilon)
    c66 = c44 * (1.0 + 2.0 * gamma)
    radicand = 2.0 * delta * c33 * (c33 - c44) + (c33 - c44) ** 2
    c13 = torch.sqrt(torch.clamp(radicand, min=0.0)) - c44
    zero = torch.zeros_like(c11)
    vti = {"C11": c11, "C12": c11 - 2.0 * c66, "C13": c13, "C22": c11, "C23": c13,
           "C33": c33, "C44": c44, "C55": c44, "C66": c66}
    if theta is None:
        return tuple(vti.get(k, zero) for k in STIFFNESS_KEYS_3D)
    if phi is None:
        phi = torch.zeros_like(theta)
    rot = ElasticTTISG3D._bond_matrix_torch(ElasticTTI._bond_rotation_matrix_torch(theta, phi))
    C = rot @ ElasticTTI._vti_voigt_torch(c11, c13, c33, c44, c66) @ rot.transpose(-1, -2)
    flat = theta.detach().abs() < 1e-7
    out = []
    for k in STIFFNESS_KEYS_3D:
        i, j = _KEY_TO_VOIGT[k]
        v = C[..., i, j]
        out.append(v + torch.where(flat, (vti.get(k, zero) - v).detach(), zero))
    return tuple(out)
