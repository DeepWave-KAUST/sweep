"""GLL nodes/weights and Lagrange-basis derivative matrix on [-1, 1].

Reference: Canuto, Hussaini, Quarteroni, Zang -- *Spectral Methods*.
The N+1 Gauss-Lobatto-Legendre nodes are the roots of
(1 - x^2) P'_N(x), where P_N is the N-th Legendre polynomial.
Quadrature is exact for polynomials up to degree 2N-1.
"""

from __future__ import annotations

import numpy as np
import torch


def _legendre_p_and_deriv(n: int, x: np.ndarray):
    """Evaluate P_n(x) and P'_n(x) by Bonnet recurrence."""
    p_prev = np.ones_like(x)
    p_curr = x.copy()
    if n == 0:
        return p_prev, np.zeros_like(x)
    if n == 1:
        return p_curr, np.ones_like(x)
    for k in range(1, n):
        p_next = ((2 * k + 1) * x * p_curr - k * p_prev) / (k + 1)
        p_prev, p_curr = p_curr, p_next
    # P'_n(x) = n (x P_n(x) - P_{n-1}(x)) / (x^2 - 1)
    # Endpoints (x = ±1) divide by zero; callers overwrite those values.
    with np.errstate(divide="ignore", invalid="ignore"):
        dp = n * (x * p_curr - p_prev) / (x * x - 1.0)
    return p_curr, dp


def _gll_nodes_weights_numpy(n_order: int) -> tuple[np.ndarray, np.ndarray]:
    """N+1 GLL nodes and quadrature weights on [-1, 1], NumPy float64."""
    n = n_order
    if n < 1:
        raise ValueError("polynomial order must be >= 1")
    npts = n + 1
    # Initial guess: Chebyshev-Lobatto, then Newton iteration on (1-x^2) P'_N
    x = -np.cos(np.pi * np.arange(npts) / n)
    # Endpoints are exact
    for _ in range(50):
        p_n, dp_n = _legendre_p_and_deriv(n, x)
        # f = (1 - x^2) P'_n = - n(n+1) ... actually we directly use the
        # well-known relation: GLL roots satisfy P'_N(x) = 0 in the interior,
        # plus x = +/- 1. So Newton on dp_n for interior points.
        interior = (np.abs(x) < 1.0 - 1e-12)
        if not interior.any():
            break
        # second derivative: P''_n(x) = (2 x P'_n - n(n+1) P_n) / (1 - x^2)
        denom = 1.0 - x * x
        ddp_n = (2 * x * dp_n - n * (n + 1) * p_n) / np.where(denom == 0, 1.0, denom)
        delta = np.where(interior, dp_n / ddp_n, 0.0)
        x_new = x - delta
        if np.max(np.abs(x_new - x)) < 1e-15:
            x = x_new
            break
        x = x_new
    x[0] = -1.0
    x[-1] = 1.0
    p_n, _ = _legendre_p_and_deriv(n, x)
    w = 2.0 / (n * (n + 1) * p_n * p_n)
    return x, w


def gll_nodes_weights(
    n_order: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """N+1 GLL nodes and weights on [-1, 1] as torch tensors.

    Always computed in NumPy float64 then cast — sub-ULP precision matters
    for stiffness assembly. ``n_order`` is the polynomial degree N
    (SPECFEM default N=4 → 5 nodes).
    """
    xi, wi = _gll_nodes_weights_numpy(n_order)
    return (
        torch.as_tensor(xi, dtype=dtype, device=device),
        torch.as_tensor(wi, dtype=dtype, device=device),
    )


def lagrange_derivative_matrix(
    nodes: torch.Tensor,
) -> torch.Tensor:
    """Derivative matrix D[i, j] = ℓ'_j(x_i) for Lagrange basis at ``nodes``.

    Standard closed form: for distinct nodes {x_k},
        ℓ'_j(x_i) = (1 / (x_i - x_j)) · Π_{k≠i,j} (x_i - x_k) / (x_j - x_k)   (i ≠ j)
        ℓ'_j(x_j) = Σ_{k ≠ j} 1 / (x_j - x_k)
    For GLL nodes there's an even simpler closed form via P_N values
    (see Canuto Appendix B), but we use the general one for clarity.
    """
    x = nodes.detach().cpu().numpy().astype(np.float64)
    n = x.size
    D = np.zeros((n, n), dtype=np.float64)
    # Barycentric weights c_j = 1 / Π_{k≠j} (x_j - x_k)
    c = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                c[j] *= (x[j] - x[k])
        c[j] = 1.0 / c[j]
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (c[j] / c[i]) / (x[i] - x[j])
        D[i, i] = -np.sum(D[i, :])
    return torch.as_tensor(D, dtype=nodes.dtype, device=nodes.device)
