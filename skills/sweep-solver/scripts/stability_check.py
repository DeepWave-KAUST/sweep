#!/usr/bin/env python3
"""Time-step stability and grid-dispersion check for a sweep finite-difference run.

Standalone (numpy only). Prints, for the given grid / time step / frequency:

* the stability ratio of the explicit leapfrog scheme (> 1 means the run blows
  up, exactly, for a homogeneous medium at vmax);
* the largest stable dt;
* points per minimum wavelength (ppw) at fmax and the resulting spatial
  phase-velocity error of the stencil.

Two scheme families, matching sweep's equations:

  --scheme second     second-order-in-time pressure equations on a regular grid
                      (Acoustic, Acoustic3D, AcousticVRZ*, AcousticBorn*,
                      ViscoAcoustic, AcousticVTI/TTI with eps = delta = 0).
                      Bound: dt*vmax*sqrt(sum 1/dh_i^2) <= sqrt(2)/sqrt(sum_m c_m (1-(-1)^m)),
                      c_m = central 2nd-derivative Taylor coefficients. This is the
                      formula sweep itself uses for its qP stability warning
                      (sweep/equations/qP_vti.py: qp_stability_ratio) with eps = delta = 0.
  --scheme staggered  first-order velocity-stress on a staggered grid
                      (Elastic, Elastic3D, DAS*, ElasticTTISG*, AcousticVTI1st*,
                      Acoustic1st, ViscoElastic, Biot).
                      Bound: dt*vmax*sqrt(sum 1/dh_i^2) * sum_m |a_m| <= 1,
                      a_m = staggered 1st-derivative Taylor coefficients (the ones
                      sweep uses: sweep.scalars.staggered_grid_coes). Standard von
                      Neumann bound; sweep does not check it for you.

Anisotropic equations need a smaller dt than these isotropic bounds (use the
fastest phase velocity, e.g. vp*sqrt(1+2*eps), as --vmax).

The ppw / phase-error numbers are spatial-only. sweep enforces no ppw threshold
for finite differences; the "rule of thumb" printed below is just that.

Examples:
  python stability_check.py --vmin 1500 --vmax 4500 --dh 10 --dt 1e-3 --fmax 25 --order 8
  python stability_check.py --scheme staggered --vmin 800 --vmax 3000 --dh 5 \\
      --dt 5e-4 --fmax 30 --order 4 --ndim 3        # elastic: vmin = min(vs)
"""
from __future__ import annotations

import argparse
import math

import numpy as np


def second_derivative_coefficients(order: int) -> np.ndarray:
    """c_1..c_M of the central 2nd derivative (same as sweep.scalars.regular_grid_coes)."""
    M = order // 2
    c = np.zeros(M)
    for m in range(1, M + 1):
        prod = 1.0
        for n in range(1, M + 1):
            if n != m:
                prod *= abs(n * n / (n * n - m * m))
        c[m - 1] = (-1) ** (m + 1) / m ** 2 * prod
    return c


def staggered_coefficients(order: int) -> np.ndarray:
    """a_1..a_M of the staggered 1st derivative (same as sweep.scalars.staggered_grid_coes)."""
    M = order // 2
    a = np.zeros(M)
    for m in range(1, M + 1):
        prod = 1.0
        for n in range(1, M + 1):
            if n != m:
                num = (2 * n - 1) ** 2
                prod *= abs(num / (num - (2 * m - 1) ** 2))
        a[m - 1] = (-1) ** (m + 1) / (2 * m - 1) * prod
    return a


def stability_ratio(scheme, vmax, dt, spacing, order):
    s = math.sqrt(sum(1.0 / h ** 2 for h in spacing))
    if scheme == "second":
        c = second_derivative_coefficients(order)
        nyq = sum(cm * (1 - (-1) ** m) for m, cm in enumerate(c, 1))
        return dt * vmax * s * math.sqrt(nyq) / math.sqrt(2.0)
    a = staggered_coefficients(order)
    return dt * vmax * s * float(np.abs(a).sum())


def phase_error(scheme, ppw, order):
    """Relative spatial phase-velocity error (1-D) of the stencil at `ppw`."""
    kh = 2 * math.pi / ppw
    if scheme == "second":
        c = second_derivative_coefficients(order)
        k2 = sum(cm * 2 * (1 - math.cos(m * kh)) for m, cm in enumerate(c, 1))
        kn = math.sqrt(k2)
    else:
        a = staggered_coefficients(order)
        kn = 2 * sum(am * math.sin((2 * m - 1) * kh / 2) for m, am in enumerate(a, 1))
    return kn / kh - 1.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vmin", type=float, required=True,
                    help="slowest wave speed that must be resolved (elastic: min vs; m/s)")
    ap.add_argument("--vmax", type=float, required=True, help="fastest wave speed (m/s)")
    ap.add_argument("--dh", type=float, nargs="+", required=True,
                    help="grid spacing: one value, or one per axis in model order (dz dx | dz dy dx)")
    ap.add_argument("--dt", type=float, required=True, help="time step (s)")
    ap.add_argument("--fmax", type=float, required=True,
                    help="highest frequency with significant energy (Ricker: ~2.5-3x the peak f)")
    ap.add_argument("--order", type=int, default=4, help="spatial_order (even), default 4")
    ap.add_argument("--ndim", type=int, default=None, help="2 or 3 (default: len(--dh) or 2)")
    ap.add_argument("--scheme", choices=("second", "staggered"), default="second")
    args = ap.parse_args(argv)

    if args.order % 2 or args.order < 2:
        ap.error("--order must be an even integer >= 2")
    ndim = args.ndim or (len(args.dh) if len(args.dh) > 1 else 2)
    spacing = args.dh if len(args.dh) == ndim else [args.dh[0]] * ndim
    if len(spacing) != ndim:
        ap.error(f"--dh needs 1 or {ndim} values")

    r = stability_ratio(args.scheme, args.vmax, args.dt, spacing, args.order)
    dt_max = args.dt / r
    ppw = args.vmin / (args.fmax * max(spacing))
    err = phase_error(args.scheme, ppw, args.order)

    print(f"scheme={args.scheme} order={args.order} ndim={ndim} spacing={tuple(spacing)} m")
    print(f"stability ratio  = {r:.3f}   ({'UNSTABLE: reduce dt' if r > 1 else 'stable'}; must be <= 1)")
    print(f"largest stable dt = {dt_max:.4g} s   (use ~0.8x of it for margin / heterogeneity)")
    print(f"ppw at fmax       = {ppw:.2f}   (vmin / (fmax * max dh))")
    print(f"spatial phase-velocity error at that ppw = {100 * err:+.2f} %")
    if abs(err) > 0.01:
        print("rule of thumb (not enforced by sweep): |error| > 1 % -> visible grid dispersion; "
              "refine dh, raise --order, or lower fmax.")
    if args.scheme == "staggered" and ppw < 5:
        print("rule of thumb (not enforced by sweep): elastic runs want >= 5 points per minimum "
              "S wavelength; the S wave aliases first.")
    return 0 if r <= 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
