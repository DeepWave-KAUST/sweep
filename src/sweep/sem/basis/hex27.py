"""Q2 (triquadratic) shape functions for 27-node hexahedral elements — the 3-D
analogue of :mod:`sweep.sem.basis.quad9` (SPECFEM3D ``NGNOD = 27``).

A Hex27 element is parametrised on the reference cube [−1, 1]³ by the tensor
product of the three 1-D three-point Lagrange polynomials (reused from
``quad9``). For control node ``(k, i, j) ∈ {0,1,2}³`` the shape function is
``Φ_{kij}(ξ, η, γ) = ψ_j(ξ) · ψ_i(η) · ψ_k(γ)``. Index convention: ``k`` ↔ γ (z),
``i`` ↔ η (y), ``j`` ↔ ξ (x). Element control coordinates live in a
``(E, 3, 3, 3, 3)`` tensor — ``[:, k, i, j, 0/1/2]`` is x/y/z.
"""

from __future__ import annotations

import torch

from .quad9 import q2_basis_at, q2_basis_deriv_at


def evaluate_hex27_map(
    control_xyz: torch.Tensor,        # (E, 3, 3, 3, 3)  [k(γ), i(η), j(ξ), xyz]
    xi: torch.Tensor,                 # (P_ξ,)
    eta: torch.Tensor,                # (P_η,)
    gamma: torch.Tensor,              # (P_γ,)
) -> torch.Tensor:                    # (E, P_γ, P_η, P_ξ, 3)
    """Evaluate the Q2 mapping ``(ξ, η, γ) → (x, y, z)`` at a grid of points."""
    psi_xi    = q2_basis_at(xi)       # (P_ξ, 3)
    psi_eta   = q2_basis_at(eta)
    psi_gamma = q2_basis_at(gamma)
    return torch.einsum("gk,hi,xj,ekijd->eghxd",
                        psi_gamma, psi_eta, psi_xi, control_xyz)


def evaluate_hex27_jacobian(
    control_xyz: torch.Tensor,        # (E, 3, 3, 3, 3)
    xi: torch.Tensor,
    eta: torch.Tensor,
    gamma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(J, inv_J, det_J)`` at every grid point.

    Shapes::

        J     : (E, P_γ, P_η, P_ξ, 3, 3)   J[..., d, m] = ∂x_d/∂ξ_m
        inv_J : (E, P_γ, P_η, P_ξ, 3, 3)   inv_J[..., m, d] = ∂ξ_m/∂x_d
        det_J : (E, P_γ, P_η, P_ξ)

    ``d ∈ {x, y, z}`` (physical), ``m ∈ {ξ, η, γ}`` (reference). The chain rule
    is ``∂/∂x_d = Σ_m inv_J[m, d] ∂/∂ξ_m``.
    """
    psi_xi,    dpsi_xi    = q2_basis_at(xi),    q2_basis_deriv_at(xi)
    psi_eta,   dpsi_eta   = q2_basis_at(eta),   q2_basis_deriv_at(eta)
    psi_gamma, dpsi_gamma = q2_basis_at(gamma), q2_basis_deriv_at(gamma)

    # column m = ξ/η/γ : derivative along that ref axis, plain basis on the others
    dxi   = torch.einsum("gk,hi,xj,ekijd->eghxd", psi_gamma, psi_eta, dpsi_xi, control_xyz)
    deta  = torch.einsum("gk,hi,xj,ekijd->eghxd", psi_gamma, dpsi_eta, psi_xi, control_xyz)
    dgam  = torch.einsum("gk,hi,xj,ekijd->eghxd", dpsi_gamma, psi_eta, psi_xi, control_xyz)
    J = torch.stack([dxi, deta, dgam], dim=-1)    # (E, ., ., ., 3_d, 3_m)

    a, b, c = J[..., 0, 0], J[..., 0, 1], J[..., 0, 2]   # x: ξ,η,γ
    d, e, f = J[..., 1, 0], J[..., 1, 1], J[..., 1, 2]   # y
    g, h, i = J[..., 2, 0], J[..., 2, 1], J[..., 2, 2]   # z
    det_J = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    safe = torch.where(det_J.abs() > 0, det_J, torch.ones_like(det_J))

    # inv_J[m, d] = adj(J)[m, d] / det  (adjugate = transpose of cofactors)
    m00 = (e * i - f * h); m01 = (c * h - b * i); m02 = (b * f - c * e)
    m10 = (f * g - d * i); m11 = (a * i - c * g); m12 = (c * d - a * f)
    m20 = (d * h - e * g); m21 = (b * g - a * h); m22 = (a * e - b * d)
    inv_J = torch.stack([
        torch.stack([m00, m01, m02], dim=-1),
        torch.stack([m10, m11, m12], dim=-1),
        torch.stack([m20, m21, m22], dim=-1),
    ], dim=-2) / safe[..., None, None]
    return J, inv_J, det_J
