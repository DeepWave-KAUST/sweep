"""ViscoAcoustic on a grid with dz != dx.

The spectral attenuation and dispersion terms need |k| on the runtime grid.
Both the eager grid (``init_abc``) and the CUDA one (``c_eq_aux``) were built
from the FIRST spacing alone, so with ``dh=(dz, dx)`` the x wavenumbers were
scaled by dz: the filters saw a wrong |k| everywhere and the Q response
depended on the grid orientation.  The FD Laplacian already used both.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import ricker

_SRC = Path(__file__).resolve().parents[1] / "src"
if (_SRC / "sweep").exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

torch = pytest.importorskip("torch")

NA, NB = 40, 48            # cells along the fine and the coarse axis
DFINE, DCOARSE = 5.0, 10.0
DT, NT, ABCN, F0 = 5e-4, 300, 15, 20.0


def _record(transpose):
    """Same physical problem on an (NA x NB) grid with dz=5, dx=10, or on its
    transpose (NB x NA, dz=10, dx=5) with source and receivers swapped too."""
    from sweep.equations import ViscoAcoustic
    from sweep.propagator.options import EagerOptions
    from sweep.propagator.torch import PropTorch

    shape, dh = ((NB, NA), (DCOARSE, DFINE)) if transpose else ((NA, NB), (DFINE, DCOARSE))
    eq = ViscoAcoustic(spatial_order=4, backend="torch", device="cpu")
    prop = PropTorch(eq, shape=shape, dh=dh, dt=DT, dev=torch.device("cpu"),
                     source_type=["h1"], receiver_type=["h1"], abcn=ABCN,
                     impl="eager", use_ckpt=False,
                     eager_options=EagerOptions(use_compile=False))
    # (a, b) = (index along the fine axis, index along the coarse axis)
    src_ab = (NA // 2, NB // 2)
    rec_ab = [(a, 6) for a in range(4, NA - 4, 3)] + [(6, b) for b in range(4, NB - 4, 3)]
    # PropTorch takes (x, z) = (column, row)
    if transpose:       # rows = coarse axis b, columns = fine axis a
        src = np.array([[src_ab[0], src_ab[1]]], dtype=np.int64)
        rec = np.array([[(a, b) for a, b in rec_ab]], dtype=np.int64)
    else:               # rows = fine axis a, columns = coarse axis b
        src = np.array([[src_ab[1], src_ab[0]]], dtype=np.int64)
        rec = np.array([[(b, a) for a, b in rec_ab]], dtype=np.int64)
    models = [torch.full(shape, 2000.0), torch.full(shape, 20.0),
              torch.full(shape, 2 * np.pi * F0)]
    with torch.no_grad():
        out = prop.forward(ricker(NT, DT, F0, 1.0 / F0), src, rec, models=models)
    d = out[0] if isinstance(out, (tuple, list)) else out
    return eq, d.detach().cpu().numpy()


def test_record_independent_of_grid_orientation():
    _, a = _record(False)
    _, b = _record(True)
    rel = np.linalg.norm(a - b) / np.linalg.norm(b)
    assert rel < 1e-4, rel


def test_eager_wavenumbers_use_both_spacings():
    eq, _ = _record(False)
    k = eq.k.detach().cpu().numpy()
    nz, nx = k.shape[-2:]
    kz = 2 * np.pi * np.fft.fftfreq(nz, d=DFINE)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=DCOARSE)
    np.testing.assert_allclose(k, np.hypot(kz[:, None], kx[None, :]), rtol=1e-6)


def test_c_wavenumbers_use_both_spacings():
    """``c_eq_aux`` builds the CUDA filters on CPU; no compiled core needed."""
    from types import SimpleNamespace

    from sweep.equations import ViscoAcoustic

    eq = ViscoAcoustic(spatial_order=4, backend="torch", device="cpu",
                       phase_shift=True, amplitude_damping=False)
    eq._gbar = torch.zeros(1)            # exponent 0: D_k2 = |k|^2, D_frac = |k|^2
    nz, nx = 64, 50
    prop = SimpleNamespace(shape_cuda=(1, nz, nx), _grid_spacing=(DFINE, DCOARSE),
                           dev=torch.device("cpu"))
    d_k2 = eq.c_eq_aux(prop)[0].numpy()
    kz = 2 * np.pi * np.fft.fftfreq(nz, d=DFINE)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=DCOARSE)
    np.testing.assert_allclose(d_k2, kz[:, None] ** 2 + kx[None, :] ** 2, rtol=1e-5)
