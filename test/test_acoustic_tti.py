"""Regression tests for AcousticTTI (qP pseudo-acoustic TTI, Liang 2022).

The headline test pins the ``theta`` unit convention: ``theta`` is the tilt
angle in RADIANS (matching ``ModelSpec(unit="rad")`` and every other TTI
equation, e.g. ``elastic_tti``).  A historical bug applied ``deg2rad(theta)``
inside the step kernel, double-converting a radian input — e.g. 45° = 0.785 rad
became deg2rad(0.785) ≈ 0.0137 rad ≈ 0.78°, collapsing the tilt to almost
nothing.  We propagate a point source in a homogeneous anisotropic medium and
measure the principal-axis tilt of the wavefront; it must track the input
radian angle, not the ~1° the bug produced.
"""

import numpy as np
import pytest
import torch

from sweep.equations import AcousticTTI
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

NZ = NX = 140
DH = 10.0
DT = 1.0e-3
NT = 180
ABCN = 20
VP_BG = 2500.0
DOM_FREQ = 12.0
EPS = 0.25   # ε ≠ δ ⇒ genuinely anisotropic, so the wavefront has an axis to tilt
DELTA = 0.10


def _wavelet(nt=NT):
    t = np.arange(nt, dtype=np.float32) * DT - 1.0 / DOM_FREQ
    return torch.tensor((1e3 * ricker(t, f=DOM_FREQ)).astype(np.float32))[None, :]


def _src(nz, nx):
    # Source at the model centre — the canonical view for wavefront shape.
    return torch.tensor(np.array([[nx // 2, nz // 2]], dtype=np.int64)[None])


def _wavefront_tilt_deg(theta_rad, nz=NZ, nx=NX):
    """Propagate AcousticTTI with a constant ``theta`` (radians) and return the
    principal-axis tilt of the final |h1| snapshot in degrees."""
    shape = (nz, nx)
    models = [
        torch.full(shape, VP_BG, dtype=torch.float32),
        torch.full(shape, EPS, dtype=torch.float32),
        torch.full(shape, DELTA, dtype=torch.float32),
        torch.full(shape, float(theta_rad), dtype=torch.float32),
    ]
    eq = AcousticTTI(spatial_order=4, device="cpu", backend="torch")
    src = _src(nz, nx)
    prop = PropTorch(
        eq, shape,
        source_type=["h1"], receiver_type=["h1"],
        abcn=ABCN, dh=DH, dt=DT, nt=NT, device="cpu", impl="eager",
        use_ckpt=False,
    )
    with torch.no_grad():
        _, wf = prop(_wavelet(), src, src, models=models,
                     return_wavefield=True, snapshot_times=[NT - 1])
    idx = eq.wavefields.index("h1")
    h1_raw = wf[0, idx, 0, 0]
    field = prop.crop(h1_raw.unsqueeze(0).unsqueeze(0))[0, 0].numpy()

    # Principal-axis angle from the second moments of |field|.
    w = np.abs(field)
    assert w.sum() > 0, "TTI run produced an all-zero wavefield."
    n0, n1 = w.shape
    zz, xx = np.mgrid[0:n0, 0:n1]
    z0 = (w * zz).sum() / w.sum()
    x0 = (w * xx).sum() / w.sum()
    dz = zz - z0
    dx = xx - x0
    cxx = (w * dx * dx).sum() / w.sum()
    czz = (w * dz * dz).sum() / w.sum()
    cxz = (w * dx * dz).sum() / w.sum()
    return 0.5 * np.degrees(np.arctan2(2 * cxz, cxx - czz))


def test_theta_is_radians_not_degrees():
    """θ in radians must produce a matching wavefront tilt.

    Guards against the ``deg2rad(theta)`` double-conversion regression: with the
    bug, θ = π/4 rad would tilt the wavefront by only ~0.78°.
    """
    tilt_0 = _wavefront_tilt_deg(0.0)
    tilt_45 = _wavefront_tilt_deg(np.pi / 4.0)

    # Vertical symmetry axis (θ=0) ⇒ no tilt.
    assert abs(tilt_0) < 5.0, f"θ=0 should give ~0° tilt, got {tilt_0:.2f}°."

    # θ = π/4 rad = 45°.  The bug would yield ~0.78°; require a real tilt.
    assert abs(tilt_45) > 30.0, (
        f"θ=π/4 rad gave only {tilt_45:.2f}° tilt — theta is being treated as "
        f"degrees (deg2rad regression)."
    )
    assert abs(abs(tilt_45) - 45.0) < 8.0, (
        f"θ=π/4 rad should tilt the wavefront ~45°, got {tilt_45:.2f}°."
    )


def test_theta_tilt_tracks_angle():
    """The measured tilt should increase monotonically with θ (in radians)."""
    angles_deg = [0.0, 20.0, 40.0]
    tilts = [abs(_wavefront_tilt_deg(np.deg2rad(a))) for a in angles_deg]
    # Each larger input angle yields a larger wavefront tilt.
    assert tilts[0] < tilts[1] < tilts[2], (
        f"Wavefront tilt does not track θ: inputs {angles_deg}° → tilts "
        f"{[round(t, 2) for t in tilts]}°."
    )
    # And each tracks its input angle reasonably closely.
    for a, t in zip(angles_deg, tilts):
        assert abs(t - a) < 8.0, f"θ={a}° → tilt {t:.2f}° (off by >8°)."
