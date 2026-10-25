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
        Surface position per column, in grid rows, length ``nx_phys``.
        **May be fractional**: a boundary-fitted grid has no reason to snap
        the surface to grid rows, and snapping it is actively harmful --
        an integer surface is a staircase, and a staircase forces
        grid-scale energy into the free surface at every step.  Integer
        input is still accepted (and then smoothed, see below) for
        backwards compatibility with the image-method topography API.
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
            topo = np.zeros(nx_phys, dtype=np.float64)
        else:
            topo = np.asarray(topography, dtype=np.float64).reshape(-1)
            if topo.shape[0] != nx_phys:
                raise ValueError(
                    f"topography length {topo.shape[0]} != nx_phys ({nx_phys})"
                )

        # Surface elevation in physical units (m).  An integer-row surface
        # is a staircase; smooth it here so that the mapping *and* its
        # metric describe the same surface.  (The old code smoothed only
        # the slope, which left the grid stepped while the metric claimed
        # it was smooth -- that mismatch is a per-step forcing of
        # grid-scale energy right at the free surface.)  A fractional
        # surface is taken as given: the caller already supplied a smooth
        # one.
        is_stepped = bool(np.allclose(topo, np.round(topo)))
        h = topo * float(dh)                                   # (nx,)
        if is_stepped and nx_phys > 1:
            h = _gaussian_smooth_1d(h.astype(np.float32), sigma=4.0
                                    ).astype(np.float64)
        z_bottom = float(nz_phys - 1) * float(dh)
        D = z_bottom - h                                       # (nx,)
        if (D <= 0).any():
            raise ValueError(
                f"topography must leave at least one row of physical domain "
                f"below the surface; got iz_surf range "
                f"[{int(topo.min())}, {int(topo.max())}] with nz_phys={nz_phys}"
            )

        # First derivative of the surface in X, taken from the very same
        # ``h`` the mapping uses, so grid and metric always agree.
        h_prime = np.gradient(h, dh)                            # (nx,)

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
        # 1D surface-slope h'(ξ) for the elastic rotated free-surface BC
        # ``σ_xz = h' σ_xx, σ_zz = h'² σ_xx`` on a curved surface.
        self.h_prime = _t(h_prime)
        # Surface actually used by the mapping (rows / metres); ``topo_rows``
        # keeps the legacy integer form for display and cropping.
        self.h_surface = _t(h)
        self.topo_rows = torch.from_numpy(
            np.round(h / float(dh)).astype(np.int64)).to(device).long()
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

        # 1D ``h_prime`` padded along x only (matches the surface row's
        # x-extent on the runtime grid). ``F.pad(..., mode='replicate')``
        # with a 2-tuple expects ≤ 3D input.
        h_prime_3d = self.h_prime[None, None, :]          # (1, 1, nx)
        h_prime_padded = F.pad(h_prime_3d, (padding[0], padding[1]), mode="replicate")
        h_prime_runtime = h_prime_padded.squeeze(0).squeeze(0)  # (nx_runtime,)

        return {
            "alpha": _pad(self.alpha),
            "beta": _pad(self.beta),
            "alpha_xi": _pad(self.alpha_xi),
            "alpha_eta": _pad(self.alpha_eta),
            "metric_pηη": _pad(self.metric_pηη),
            "metric_pη": _pad(self.metric_pη),
            "h_prime": h_prime_runtime,
        }

    def row_spacing(self, x_phys=None):
        """Physical vertical spacing (metres) of the boundary-fitted grid.

        **This is not** ``dh``.  The mapping spreads ``eta`` over the whole
        ``nz_phys - 1`` computational rows while the physical column below
        the surface is only ``nz_phys - 1 - topo[ix]`` rows deep, so one
        computational row spans ``D(x) / (nz_phys - 1)`` metres, and with
        topography that varies **column by column**.  Assuming ``dh`` here
        is the single easiest way to place a source at the wrong depth:
        with ``nz_phys=192``, ``topo=24``, ``dh=5`` the spacing is 4.372 m,
        12.6 % short, and the crest of a 50 m hill gives 4.634 m.
        """
        import numpy as np

        # the surface the MAPPING uses, in ROWS.  ``h_surface`` is stored in
        # metres and ``topo_rows`` is its rounded integer shadow -- neither
        # is directly usable here.
        topo = np.asarray(self.h_surface.detach().cpu(),
                          dtype=np.float64) / float(self.dh)
        d = ((self.nz_phys - 1) - topo) * float(self.dh) / (self.nz_phys - 1)
        if x_phys is None:
            return d
        ix = np.clip(np.round(np.asarray(x_phys)).astype(int), 0, self.nx_phys - 1)
        return d[ix]

    def physical_to_computational(self, x_phys, z_phys):
        """Map physical ``(x_phys, z_phys)`` grid-index coordinates to the
        computational ``(xi, eta)`` cell-index location.

        Accepts scalars or arrays and broadcasts them together.  The xi
        index is identical to ``x_phys``; the eta index is
        ``(z_phys - topo[ix]) / (nz_phys - 1 - topo[ix]) * (nz_phys - 1)``
        and comes back **fractional**; :meth:`row_spacing` gives what a
        computational row is in metres.

        The propagator applies it to sources and receivers itself (they are
        physical indices; ``PropBase._place_on_curvilinear``), so this is for
        looking at where a position lands: ``k`` computational rows below
        the surface are ``k * D(x)/(nz_phys - 1)`` metres, not ``k * dh``.
        """
        import numpy as np

        ix = np.clip(np.round(np.asarray(x_phys, dtype=np.float64)).astype(int),
                     0, self.nx_phys - 1)
        # the surface the MAPPING uses, in ROWS.  ``h_surface`` is stored in
        # metres and ``topo_rows`` is its rounded integer shadow -- neither
        # is directly usable here.
        topo = np.asarray(self.h_surface.detach().cpu(),
                          dtype=np.float64) / float(self.dh)
        s = topo[ix]
        denom = (self.nz_phys - 1) - s
        if np.any(denom <= 0):
            raise ValueError("some column has no physical depth below the surface")
        eta_norm = np.clip((np.asarray(z_phys, dtype=np.float64) - s)/denom, 0.0, 1.0)
        i_eta = eta_norm * (self.nz_phys - 1)
        if np.isscalar(x_phys) and np.isscalar(z_phys):
            return float(ix), float(i_eta)
        return ix.astype(np.float64), i_eta

    def computational_to_physical(self, x_phys, i_eta):
        """Inverse of :meth:`physical_to_computational` — the physical row
        a computational eta index actually lands on."""
        import numpy as np

        ix = np.clip(np.round(np.asarray(x_phys, dtype=np.float64)).astype(int),
                     0, self.nx_phys - 1)
        # the surface the MAPPING uses, in ROWS.  ``h_surface`` is stored in
        # metres and ``topo_rows`` is its rounded integer shadow -- neither
        # is directly usable here.
        topo = np.asarray(self.h_surface.detach().cpu(),
                          dtype=np.float64) / float(self.dh)
        s = topo[ix]
        return s + np.asarray(i_eta, dtype=np.float64)/(self.nz_phys - 1) * \
            ((self.nz_phys - 1) - s)
