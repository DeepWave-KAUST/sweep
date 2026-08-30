"""Analytic qP/qS group-velocity surfaces of a TTI medium, for verification.

Deliberately independent of ``sweep``: the medium is built from Thomsen
parameters with the textbook VTI formulas and the Christoffel problem is solved
in the CRYSTAL frame, so nothing here goes through the solver's Bond matrix or
its rotated stiffness. Only the rotation R(theta, phi) crosses between frames,
and it is written from the documented meaning of tilt and azimuth: the third
column of R is the symmetry axis in the lab frame.

Why it is a test at all: a homogeneous medium radiates a wavefront that IS the
group-velocity surface, so at time ``t`` the qP front sits at ``|Vg(ray)| * t``
along every ray. Comparing the two checks the anisotropic kinematics end to end.
Note that eager and the compiled backend share ``prepare_models``, so no
c-vs-eager test can see an error in the rotation -- this can.
"""
import numpy as np

# Voigt index of each (i, j) pair: xx yy zz yz xz xy
_VOIGT = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])


def vti_stiffness(vp0, vs0, rho, epsilon, delta, gamma):
    """Thomsen parameters -> the five independent VTI constants."""
    C33 = rho * vp0 ** 2
    C44 = rho * vs0 ** 2
    C11 = C33 * (1.0 + 2.0 * epsilon)
    C66 = C44 * (1.0 + 2.0 * gamma)
    # exact Thomsen delta: (C13 + C44)^2 = 2 C33 (C33 - C44) delta + (C33 - C44)^2
    C13 = np.sqrt(2.0 * C33 * (C33 - C44) * delta + (C33 - C44) ** 2) - C44
    return C11, C13, C33, C44, C66


def vti_tensor(C11, C13, C33, C44, C66):
    """The five constants -> the rank-4 tensor c_ijkl in the crystal frame."""
    C12 = C11 - 2.0 * C66
    V = np.array([[C11, C12, C13, 0.0, 0.0, 0.0],
                  [C12, C11, C13, 0.0, 0.0, 0.0],
                  [C13, C13, C33, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, C44, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, C44, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, C66]])
    return V[_VOIGT[:, :, None, None], _VOIGT[None, None, :, :]]


def rotation(theta, phi):
    """Lab <- crystal. The third column is the symmetry axis: tilt ``theta``
    from z, azimuth ``phi`` from x. Matches the solver's documented convention."""
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    return np.array([[cp * ct, -sp, cp * st],
                     [sp * ct, cp, sp * st],
                     [-st, 0.0, ct]])


def qp_group_surface(vp0, vs0, rho, epsilon, delta, gamma, theta, phi,
                     n_samp=700, tol=3e-3):
    """qP group-velocity vectors whose ray lies in the y = 0 plane, sorted by
    ray angle. Returns ``(psi_deg, speed)``; multiply by a time to get a front.

    Traced parametrically from the phase-direction sphere, so folds of the
    surface come out on their own -- no root finding, no assumption that the
    surface is single valued in ray angle.
    """
    c = vti_tensor(*vti_stiffness(vp0, vs0, rho, epsilon, delta, gamma))
    R = rotation(theta, phi)

    u = np.linspace(-1.0, 1.0, n_samp)
    az = np.linspace(0.0, 2.0 * np.pi, n_samp, endpoint=False)
    U, A = np.meshgrid(u, az, indexing="ij")
    s = np.sqrt(np.clip(1.0 - U ** 2, 0.0, None))
    n_lab = np.stack([s * np.cos(A), s * np.sin(A), U], axis=-1).reshape(-1, 3)

    n = n_lab @ R                                        # crystal frame
    G = np.einsum("ijkl,pj,pl->pik", c, n, n)            # Christoffel
    lam, vec = np.linalg.eigh(G)
    v, g = np.sqrt(lam[:, 2] / rho), vec[:, :, 2]        # qP = largest
    vg = np.einsum("ijkl,pj,pk,pl->pi", c, g, g, n) / (rho * v[:, None])
    vg = vg @ R.T                                        # back to the lab frame

    speed = np.linalg.norm(vg, axis=-1)
    keep = np.abs(vg[:, 1]) / speed < tol                # the y = 0 branch
    vg, speed = vg[keep], speed[keep]
    psi = np.degrees(np.arctan2(vg[:, 0], vg[:, 2])) % 360.0
    order = np.argsort(psi)
    return psi[order], speed[order]


def qp_radius(psi_deg, travel, **medium):
    """qP front radius along the given in-plane ray angles at ``travel`` seconds."""
    psi, speed = qp_group_surface(**medium)
    psi = np.concatenate([psi - 360.0, psi, psi + 360.0])   # periodic
    speed = np.concatenate([speed, speed, speed])
    return np.interp(np.asarray(psi_deg) % 360.0, psi, speed) * travel


# --------------------------------------------------------------------------- #
#  picking the front out of a snapshot (numpy only -- tests must not need scipy)
# --------------------------------------------------------------------------- #
def _envelope(x):
    """|analytic signal| via the FFT, i.e. a Hilbert envelope without scipy."""
    n = len(x)
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(np.fft.fft(x) * h))


def _bilinear(img, zz, xx):
    z0 = np.clip(np.floor(zz).astype(int), 0, img.shape[0] - 2)
    x0 = np.clip(np.floor(xx).astype(int), 0, img.shape[1] - 2)
    dz, dx = zz - z0, xx - x0
    return ((1 - dz) * (1 - dx) * img[z0, x0] + (1 - dz) * dx * img[z0, x0 + 1]
            + dz * (1 - dx) * img[z0 + 1, x0] + dz * dx * img[z0 + 1, x0 + 1])


def pick_front(sl, dh, psi_deg, r_max_cells, r_min_cells=3.0, step=0.25):
    """Radius of the dominant arrival along each in-plane ray of a snapshot.

    ``sl`` is the (nz, nx) slice through the source, which sits at its centre.
    The GLOBAL envelope maximum is taken, not the outermost peak: in a
    homogeneous medium the wanted arrival dominates every ray, whereas an
    outermost-peak rule latches onto whatever leaks in near the PML.
    ``r_max_cells`` must keep the search well inside the absorbing band.
    """
    c = sl.shape[0] // 2
    r = np.arange(r_min_cells, r_max_cells, step) * dh
    out = np.empty(len(psi_deg))
    for k, psi in enumerate(np.deg2rad(psi_deg)):
        prof = _bilinear(sl, c + (r / dh) * np.cos(psi), c + (r / dh) * np.sin(psi))
        env = _envelope(prof)
        i = int(np.argmax(env))
        shift = 0.0
        if 0 < i < len(env) - 1:                      # parabolic refinement
            d = env[i - 1] - 2 * env[i] + env[i + 1]
            if d != 0:
                shift = 0.5 * (env[i - 1] - env[i + 1]) / d
        out[k] = r[i] + shift * (r[1] - r[0])
    return out


def shape_error(measured, predicted):
    """Split the comparison into the part that tests the anisotropy and the part
    that does not.

    A common radial offset is grid dispersion plus the finite-bandwidth bias of
    an envelope pick -- it is there for an isotropic medium too. Only the
    variation of ``measured / predicted`` WITH RAY ANGLE tests the shape of the
    group-velocity surface. Returns ``(bias, rms, max_dev)`` as fractions.
    """
    ratio = np.asarray(measured) / np.asarray(predicted)
    spread = ratio - ratio.mean()
    return ratio.mean() - 1.0, float(spread.std()), float(np.abs(spread).max())
