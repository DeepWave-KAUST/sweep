"""Suggest SEM mesh resolution + a stable time step from physical parameters.

Picking ``nex``, ``nez`` and ``dt`` by hand is error-prone — they follow
directly from the domain, the velocities and the source frequency:

* resolution: ~``ppw`` GLL points per shortest wavelength
  (``λ_min = v_min / f_max``) → element size ``h_e ≈ N·λ_min/ppw``;
* time step: CFL on the smallest GLL spacing,
  ``dt ≲ courant · Δx_min^GLL / v_max``.

:func:`suggest_discretization` does both and returns the numbers to feed
:func:`build_topography_quad9` / :func:`build_quad4_mesh` and the propagator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .basis.gll import gll_nodes_weights


@dataclass
class SEMDiscretization:
    """Suggested SEM discretization parameters + the diagnostics behind them."""

    n_order: int
    nex: int
    nez: int
    dt: float
    nt: int | None
    # diagnostics
    lambda_min: float                 # shortest wavelength (m)
    h_elem_x: float                   # actual element size (m)
    h_elem_z: float
    points_per_wavelength: float      # achieved (on the coarser element)
    courant: float

    def __repr__(self) -> str:
        nt = f", nt={self.nt}" if self.nt is not None else ""
        return (f"SEMDiscretization(n_order={self.n_order}, nex={self.nex}, "
                f"nez={self.nez}, dt={self.dt:.2e}{nt}  |  λ_min={self.lambda_min:.1f} m, "
                f"h_e≈{self.h_elem_x:.0f}×{self.h_elem_z:.0f} m, "
                f"{self.points_per_wavelength:.1f} pts/λ)")


def _floor_2sig(x: float) -> float:
    """Floor ``x`` to 2 significant figures (keeps dt a clean, safe value)."""
    if x <= 0.0:
        return x
    d = 1 - math.floor(math.log10(x))
    f = 10.0 ** d
    return math.floor(x * f) / f


def suggest_discretization(
    x_max: float,
    z_max: float,
    *,
    v_min: float,
    v_max: float,
    f0: float,
    n_order: int = 4,
    ppw: float = 5.0,
    f_max_factor: float = 2.5,
    courant: float = 0.3,
    record_length: float | None = None,
) -> SEMDiscretization:
    """Suggest ``(nex, nez, dt[, nt])`` for a 2-D SEM run.

    Parameters
    ----------
    x_max, z_max
        Domain extent (m).
    v_min
        Slowest wave speed (m/s) — sets the spatial resolution. Use ``vs`` for
        elastic, ``vp`` for constant-density acoustic.
    v_max
        Fastest wave speed (m/s) — sets the CFL time step. Use ``vp``.
    f0
        Ricker central frequency (Hz).
    n_order
        GLL polynomial degree ``N`` (``ngll = N + 1``). Default 4.
    ppw
        Target GLL points per shortest wavelength. Default 5 (standard for
        ``N = 4``); raise to 6–8 for extra margin / prettier wavefields.
    f_max_factor
        ``f_max / f0`` for the source bandwidth. Default 2.5 (Ricker).
    courant
        CFL safety factor, below the ``~0.4–0.5`` N=4 stability limit.
        Default 0.3 (leaves margin for mild mesh deformation / topography).
    record_length
        If given (s), also return ``nt = ceil(record_length / dt)``.

    Returns
    -------
    SEMDiscretization
        ``.nex .nez .dt`` (and ``.nt``) ready to use, plus diagnostics.

    Examples
    --------
    >>> d = suggest_discretization(1280, 900, v_min=1200, v_max=2400, f0=12,
    ...                            record_length=0.6)
    >>> mesh = build_topography_quad9(nex=d.nex, nez=d.nez, x_max=1280,
    ...                               z_bottom=900, z_top_fn=z_top, n_order=d.n_order)
    >>> # propagate with dt=d.dt, nt=d.nt
    """
    if min(x_max, z_max, v_min, v_max, f0) <= 0:
        raise ValueError("x_max, z_max, v_min, v_max, f0 must all be positive")

    f_max = f_max_factor * f0
    lam_min = v_min / f_max
    # points per wavelength = N · λ / h_e  →  h_e = N·λ/ppw for the target ppw
    h_target = n_order * lam_min / ppw
    nex = max(1, math.ceil(x_max / h_target))
    nez = max(1, math.ceil(z_max / h_target))
    h_x, h_z = x_max / nex, z_max / nez
    ppw_actual = n_order * lam_min / max(h_x, h_z)      # coarser element bounds ppw

    # CFL: smallest GLL spacing on the finer element.
    xi, _ = gll_nodes_weights(n_order)
    min_ref_gap = float((xi[1:] - xi[:-1]).min())       # min gap on [-1, 1]
    dx_min = min_ref_gap * min(h_x, h_z) / 2.0          # → physical (half-element)
    dt = _floor_2sig(courant * dx_min / v_max)

    nt = math.ceil(record_length / dt) if record_length else None
    return SEMDiscretization(
        n_order=n_order, nex=nex, nez=nez, dt=dt, nt=nt,
        lambda_min=lam_min, h_elem_x=h_x, h_elem_z=h_z,
        points_per_wavelength=ppw_actual, courant=courant,
    )
