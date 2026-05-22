"""Boundary-fitted curvilinear-grid metrics for irregular free-surface
topography.

The physical (X, Z) domain has an irregular top boundary
``Z = z_top(X)``. We map it onto a rectangular computational (ξ, η)
domain via the simple linear stretch:

    X = ξ
    Z = z_top(ξ) + (z_bottom − z_top(ξ)) · η,   η ∈ [0, 1]

The wave equation in physical coordinates becomes a regular-grid PDE in
(ξ, η) at the cost of extra "metric" coefficients on every spatial
derivative. This module precomputes those coefficients once at
propagator init.

For the linear stretch the inverse metrics are particularly simple:

    ∂ξ/∂X = 1                         ∂ξ/∂Z = 0
    ∂η/∂X = α(ξ, η) = −h'(ξ)·(1 − η) / D(ξ)
    ∂η/∂Z = β(ξ)    = 1 / D(ξ)

where ``h(ξ) = z_top``, ``D(ξ) = z_bottom − z_top``, ``h'`` is the
slope of the surface.

The acoustic-2nd-order Laplacian becomes::

    Δ_phys p = p_ξξ + 2α p_ξη + (α² + β²) p_ηη + (α_ξ + α α_η) p_η

so the precomputed fields are:

    alpha          α(ξ, η)
    beta           β(ξ)           (η-independent, stored broadcast)
    alpha_xi       ∂α/∂ξ
    alpha_eta      ∂α/∂η
    metric_pηη     α² + β²
    metric_pη      α_ξ + α·α_η

Elastic 1st-order uses the same α and β directly inside chain-rule
substitutions for ∂/∂X and ∂/∂Z; the combined ``metric_p*`` fields are
not used there.

References
----------
Tessmer, E. & Kosloff, D. (1994), Geophysics 59, 464–473.
Hestholm, S. & Ruud, B. (1998), Geophysics 63, 613–622.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _gaussian_smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth ``x`` with a discrete Gaussian kernel (reflect at edges).

    Used to take the bite out of integer-row topography before computing
    the surface slope ``h'(ξ)`` — a raw central difference of stair-step
    elevations produces a saw-tooth derivative that destabilises the
    metric ``α``.
    """
    if sigma <= 0:
        return x.copy()
    radius = max(1, int(np.ceil(3.0 * sigma)))
    t = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()
    # Reflect-pad to avoid edge artefacts
    padded = np.concatenate([x[:radius][::-1], x, x[-radius:][::-1]])
    out = np.convolve(padded, kernel, mode="valid")
    return out.astype(x.dtype, copy=False)


class CurvilinearGrid:
    """Precomputed inverse-Jacobian metric fields for the linear-stretch
    boundary-fitted grid.

    Parameters
    ----------
    topography : array_like or None
        Integer row index of the surface per column, length ``nx_phys``.
        ``None`` → flat surface at row 0 (identity transform).
    nz_phys, nx_phys : int
        Physical-grid size.
    dh : float
        Physical grid spacing in X (uniform).
    device : str | torch.device
        Device for the precomputed tensors.

    Attributes
    ----------
    alpha, beta, alpha_xi, alpha_eta, metric_pηη, metric_pη : torch.Tensor
        Shape ``(nz_phys, nx_phys)``, ``float32``.
    d_eta : float
        η-grid spacing = ``1 / (nz_phys - 1)``.
    topo_rows : torch.LongTensor
        The integer surface-row array used to build the grid (for
        later cropping / display).
    """

    def __init__(
        self,
        topography: Optional[np.ndarray],
        nz_phys: int,
        nx_phys: int,
        dh: float,
        device: str = "cpu",
    ) -> None:
        import torch

        if topography is None:
            topo = np.zeros(nx_phys, dtype=np.int64)
        else:
            topo = np.asarray(topography, dtype=np.int64).reshape(-1)
            if topo.shape[0] != nx_phys:
                raise ValueError(
                    f"topography length {topo.shape[0]} != nx_phys ({nx_phys})"
                )

        # Surface elevation in physical units (m)
        h = topo.astype(np.float32) * float(dh)               # (nx,)
        z_bottom = float(nz_phys - 1) * float(dh)
        D = z_bottom - h                                       # (nx,)
        if (D <= 0).any():
            raise ValueError(
                f"topography must leave at least one row of physical domain "
                f"below the surface; got iz_surf range "
                f"[{int(topo.min())}, {int(topo.max())}] with nz_phys={nz_phys}"
            )

        # First derivative of the surface in X. Integer-row topography
        # produces a saw-tooth ``h(ξ)`` whose raw central difference is
        # noisy; smooth ``h`` with a short Gaussian before differencing
        # so the metric ``α`` is well-behaved. A 4-cell stddev keeps the
        # large-scale shape (topo features span tens of cells) while
        # killing the per-cell quantisation jitter.
        h_smooth = _gaussian_smooth_1d(h, sigma=4.0)
        h_prime = np.gradient(h_smooth, dh)                     # (nx,)

        # η grid on the computational domain
        eta_1d = np.linspace(0.0, 1.0, nz_phys, dtype=np.float32)
        d_eta = 1.0 / max(nz_phys - 1, 1)
        eta = eta_1d[:, None]                                   # (nz, 1)

        # Inverse metrics
        alpha = -h_prime[None, :] * (1.0 - eta) / D[None, :]    # (nz, nx)
        beta_1d = 1.0 / D                                       # (nx,)
        beta = np.broadcast_to(beta_1d[None, :], (nz_phys, nx_phys)).copy()

        # Derivatives of α
        alpha_xi = np.gradient(alpha, dh, axis=-1)              # ∂α/∂ξ
        alpha_eta = np.gradient(alpha, d_eta, axis=-2)          # ∂α/∂η

        # Combined coefficients used by the acoustic Laplacian
        metric_pηη = alpha * alpha + beta * beta                # α² + β²
        metric_pη = alpha_xi + alpha * alpha_eta                # α_ξ + α·α_η

        def _t(a):
            return torch.from_numpy(a.astype(np.float32)).to(device)

        self.alpha = _t(alpha)
        self.beta = _t(beta)
        self.alpha_xi = _t(alpha_xi)
        self.alpha_eta = _t(alpha_eta)
        self.metric_pηη = _t(metric_pηη)
        self.metric_pη = _t(metric_pη)
        self.topo_rows = torch.from_numpy(topo).to(device).long()
        self.d_eta = float(d_eta)
        self.dh = float(dh)
        self.nz_phys = int(nz_phys)
        self.nx_phys = int(nx_phys)
        self.device = device

    # ------------------------------------------------------------------
    # Pad-to-runtime helpers
    # ------------------------------------------------------------------

    def padded_metrics(self, padding):
        """Return a dict of metric tensors padded to runtime shape.

        ``padding`` is the same flat F.pad-style tuple the propagator
        uses for vp models: ``(pad_x_left, pad_x_right, pad_z_top, pad_z_bottom)``
        for 2D. Metrics are extended by edge replication so the side+
        bottom PML zones inherit the nearest physical metric value.
        """
        import torch
        import torch.nn.functional as F

        def _pad(t):
            t4 = t.unsqueeze(0).unsqueeze(0)
            t4 = F.pad(t4, padding, mode="replicate")
            return t4.squeeze(0).squeeze(0)

        return {
            "alpha": _pad(self.alpha),
            "beta": _pad(self.beta),
            "alpha_xi": _pad(self.alpha_xi),
            "alpha_eta": _pad(self.alpha_eta),
            "metric_pηη": _pad(self.metric_pηη),
            "metric_pη": _pad(self.metric_pη),
        }

    def physical_to_computational(self, x_phys, z_phys):
        """Map a physical (x_phys, z_phys) point — both in grid-index units
        — to its computational (ξ, η) cell-index location.

        Returns ``(i_xi, i_eta)`` as Python floats (the caller can round
        for use as source/receiver placement). The ξ index is identical
        to ``x_phys``; the η index is ``(z_phys − topo[ix]) /
        (nz_phys − 1 − topo[ix]) · (nz_phys − 1)``.
        """
        import torch

        ix = int(round(float(x_phys)))
        ix = max(0, min(ix, self.nx_phys - 1))
        s = int(self.topo_rows[ix].item())
        denom = self.nz_phys - 1 - s
        if denom <= 0:
            raise ValueError(f"column {ix} has no physical depth below the surface")
        eta_norm = (float(z_phys) - s) / denom
        eta_norm = max(0.0, min(1.0, eta_norm))
        i_eta = eta_norm * (self.nz_phys - 1)
        return float(ix), float(i_eta)
