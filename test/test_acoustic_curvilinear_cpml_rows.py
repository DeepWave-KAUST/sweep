"""AcousticCurvilinear: the CPML terms must use the physical row spacing.

One computational row of the boundary-fitted grid is ``D(x) d_eta`` metres,
not ``dh`` (see ``CurvilinearGrid.row_spacing``).  The z-direction CPML terms
took their derivatives with the propagator's ``dh`` and its ``dbz/dz``, so
whenever the surface sits below row 0 the stretched-coordinate correction was
scaled by ``(row_spacing / dh)**2`` and the bottom layer reflected: with the
surface at row 40 of 121 the reflection was 1.2% of the record, against
~1e-4 without topography.  The interior (zero profiles) is unaffected.

The reflection is read against a model twice as deep with the same row
spacing: there the bottom is too far to return within the record.
"""
import numpy as np
import pytest
import torch

from sweep.equations import AcousticCurvilinear
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

DH, DT, NT, ABCN, NX = 10.0, 1e-3, 700, 30, 120


def _record(nz, surface_row, depth_m):
    eq = AcousticCurvilinear(spatial_order=4, device="cpu", backend="torch")
    prop = PropTorch(eq, shape=(nz, NX), free_surface=True,
                     topography=np.full(NX, float(surface_row)),
                     abcn=ABCN, dh=DH, dt=DT, use_ckpt=False, impl="eager",
                     eager_options={"use_compile": False})
    t = np.arange(NT, dtype=np.float32) * DT - 0.12
    w = torch.tensor(ricker(t, f=10.0).astype(np.float32))
    # source 200 m, receivers 300 m above the bottom of the shallow model
    src = torch.tensor([[NX // 2, surface_row + int((depth_m - 200) / DH)]])
    rx = np.arange(10, NX - 10, 5)
    rec = torch.tensor(np.stack([rx, np.full_like(rx, surface_row + int((depth_m - 300) / DH))], -1)[None])
    with torch.no_grad():
        return prop(w, src, rec, models=[torch.full((nz, NX), 2000.0)]).numpy()


@pytest.mark.parametrize("surface_row", [0, 40])
def test_bottom_cpml_reflection(surface_row):
    depth = (120 - surface_row) * DH
    near = _record(121, surface_row, depth)
    # nz - 1 and the surface row both doubled: same row spacing, bottom 2x deeper
    far = _record(241, 2 * surface_row, depth)
    refl = np.linalg.norm(near - far) / np.linalg.norm(far)
    assert refl < 5e-4, refl        # was 1.2e-2 at surface_row=40
