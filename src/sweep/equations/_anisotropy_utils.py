"""Utility helpers for anisotropic wave-equation implementations.

Currently contains:
  smooth_delta_to_epsilon_disk — Duveneck (2008) source-region shear-artifact
  suppression via a raised-cosine taper of δ → ε around each source location.
"""

import numpy as np


def smooth_delta_to_epsilon_disk(epsilon, delta, source_grid_pos, r_taper_grid=5):
    """Return a modified delta array equal to epsilon inside a smooth disk.

    Inside a raised-cosine taper of radius *r_taper_grid* grid cells centred
    at *source_grid_pos*, delta is blended toward epsilon so that the
    spurious source-generated S-wave artefact described by Duveneck et al.
    (2008) is suppressed.  Outside the disk delta is unchanged.

    Parameters
    ----------
    epsilon : np.ndarray, shape (nz, nx) or (ny, nz, nx)
        Thomsen epsilon model (spatially varying).
    delta : np.ndarray, same shape as epsilon
        Thomsen delta model to be modified.
    source_grid_pos : sequence of int
        Grid coordinates of the source, ordered (x, z) for 2-D or
        (x, y, z) for 3-D — matching the SWEEP axis convention where the
        first positional index is x and the last is z.
    r_taper_grid : int or float, optional
        Radius of the suppression disk in grid cells.  Default 5.

    Returns
    -------
    np.ndarray
        A copy of *delta* with the values inside the disk smoothly replaced
        by *epsilon* via a raised-cosine profile.

    Notes
    -----
    The approach is described in Duveneck et al. (2008), §"Acoustic VTI wave
    field modeling", second paragraph.  The returned array should be passed to
    the propagator *before* it is constructed — the equation class itself is
    not modified.

    Reference
    ---------
    Duveneck, E., Milcik, P., Bakker, P. M., Perkins, C. (2008),
    "Acoustic VTI wave equations and their application for anisotropic
    reverse-time migration", SEG Las Vegas 2008 Annual Meeting, pp. 2186–2190.
    DOI: 10.1190/1.3059320
    """
    epsilon = np.asarray(epsilon, dtype=np.float32)
    delta = np.asarray(delta, dtype=np.float32).copy()

    ndim = epsilon.ndim
    r = float(r_taper_grid)

    if ndim == 2:
        nz, nx = epsilon.shape
        # source_grid_pos is (ix, iz) in SWEEP convention
        ix_src, iz_src = int(source_grid_pos[0]), int(source_grid_pos[1])
        iz_grid, ix_grid = np.ogrid[0:nz, 0:nx]
        dist = np.sqrt((ix_grid - ix_src) ** 2 + (iz_grid - iz_src) ** 2)
    elif ndim == 3:
        ny, nz, nx = epsilon.shape
        ix_src, iy_src, iz_src = (
            int(source_grid_pos[0]),
            int(source_grid_pos[1]),
            int(source_grid_pos[2]),
        )
        iz_grid, iy_grid, ix_grid = np.ogrid[0:ny, 0:nz, 0:nx]
        dist = np.sqrt(
            (ix_grid - ix_src) ** 2
            + (iy_grid - iy_src) ** 2
            + (iz_grid - iz_src) ** 2
        )
    else:
        raise ValueError(f"epsilon must be 2-D or 3-D, got {ndim}-D.")

    inside = dist <= r
    transition = (dist > r) & (dist <= 2 * r)

    # raised-cosine weight: 1 at dist=r, 0 at dist=2r
    w = np.zeros_like(dist)
    w[inside] = 1.0
    w[transition] = 0.5 * (1.0 + np.cos(np.pi * (dist[transition] - r) / r))

    delta = delta * (1.0 - w) + epsilon * w
    return delta
