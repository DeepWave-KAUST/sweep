"""Q2 (biquadratic) shape functions for 9-node quadrilateral elements.

A Q2 element is parametrised on the reference square [−1, 1]² by the
tensor product of three-point 1-D Lagrange polynomials evaluated at the
nodes ``ξ ∈ {−1, 0, +1}``:

    ψ_0(ξ) = ξ(ξ − 1) / 2
    ψ_1(ξ) = 1 − ξ²
    ψ_2(ξ) = ξ(ξ + 1) / 2

    ψ_0'(ξ) = ξ − 1/2
    ψ_1'(ξ) = −2 ξ
    ψ_2'(ξ) = ξ + 1/2

For node ``(i, j)`` ∈ {0, 1, 2}² the 2-D shape function is
``Φ_{ij}(ξ, η) = ψ_j(ξ) · ψ_i(η)``. Index convention: ``i`` ↔ η (z),
``j`` ↔ ξ (x). Element control coordinates live in a ``(E, 3, 3, 2)``
tensor — ``[:, i, j, 0]`` is x, ``[:, i, j, 1]`` is z.
"""

from __future__ import annotations

import torch


def q2_basis_at(xi: torch.Tensor) -> torch.Tensor:
    """ψ_0(ξ), ψ_1(ξ), ψ_2(ξ) at every point in ``xi``.

    ``xi`` shape ``(P,)`` → result shape ``(P, 3)``.
    """
    return torch.stack([
        0.5 * xi * (xi - 1.0),
        1.0 - xi * xi,
        0.5 * xi * (xi + 1.0),
    ], dim=-1)


def q2_basis_deriv_at(xi: torch.Tensor) -> torch.Tensor:
    """ψ_0'(ξ), ψ_1'(ξ), ψ_2'(ξ) at every point in ``xi``.

    ``xi`` shape ``(P,)`` → result shape ``(P, 3)``.
    """
    return torch.stack([
        xi - 0.5,
        -2.0 * xi,
        xi + 0.5,
    ], dim=-1)


def evaluate_q2_map(
    control_xy: torch.Tensor,         # (E, 3, 3, 2)
    xi: torch.Tensor,                 # (P_ξ,)
    eta: torch.Tensor,                # (P_η,)
) -> torch.Tensor:                    # (E, P_η, P_ξ, 2)
    """Evaluate the Q2 mapping ``(ξ, η) → (x, z)`` at a grid of points."""
    psi_xi  = q2_basis_at(xi)         # (P_ξ, 3)
    psi_eta = q2_basis_at(eta)        # (P_η, 3)
    # Φ_{α β, i, j} = ψ_i(η_α) · ψ_j(ξ_β); contract with control_xy[..., i, j, :]
    return torch.einsum("ai,bj,eijk->eabk", psi_eta, psi_xi, control_xy)


def evaluate_q2_jacobian(
    control_xy: torch.Tensor,         # (E, 3, 3, 2)
    xi: torch.Tensor,                 # (P_ξ,)
    eta: torch.Tensor,                # (P_η,)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(J, inv_J, det_J)`` evaluated at every grid point.

    Shapes::

        J     : (E, P_η, P_ξ, 2, 2)   row 0 = (∂x/∂ξ, ∂x/∂η), row 1 = (∂z/∂ξ, ∂z/∂η)
        inv_J : (E, P_η, P_ξ, 2, 2)
        det_J : (E, P_η, P_ξ)
    """
    psi_xi   = q2_basis_at(xi)
    dpsi_xi  = q2_basis_deriv_at(xi)
    psi_eta  = q2_basis_at(eta)
    dpsi_eta = q2_basis_deriv_at(eta)

    # ∂(x,z)/∂ξ : derivative of ξ-basis times eta-basis · control
    dxdxi = torch.einsum("ai,bj,eijk->eabk", psi_eta, dpsi_xi, control_xy)
    # ∂(x,z)/∂η
    dxdeta = torch.einsum("ai,bj,eijk->eabk", dpsi_eta, psi_xi, control_xy)
    # J[..., 0, 0] = ∂x/∂ξ ; J[..., 0, 1] = ∂x/∂η ; row 1 = same for z
    J = torch.stack([dxdxi, dxdeta], dim=-1)    # (E, P_η, P_ξ, 2_xy, 2_ξη)
    # Re-arrange so J[..., row=xy, col=ξη]
    # currently J[..., k, m] where k ∈ {x=0,z=1} via control_xy[..., k], m ∈ {ξ=0, η=1}
    # That's exactly the convention we want.

    a = J[..., 0, 0]; b = J[..., 0, 1]
    c = J[..., 1, 0]; d = J[..., 1, 1]
    det_J = a * d - b * c
    # inv = (1/det) · [[d, -b], [-c, a]]
    safe_det = torch.where(det_J.abs() > 0, det_J, torch.ones_like(det_J))
    inv_a = d / safe_det
    inv_b = -b / safe_det
    inv_c = -c / safe_det
    inv_d = a / safe_det
    inv_J = torch.stack([
        torch.stack([inv_a, inv_b], dim=-1),
        torch.stack([inv_c, inv_d], dim=-1),
    ], dim=-2)
    return J, inv_J, det_J
