"""dL/dvp on the GRID model against central finite differences, eager fp64,
for the acoustic SEM classes with and without a C-PML.

The node fields ``vp_squared_node`` / ``kappa_node`` used to be built with
``node[ibool] = per_element``, whose backward gives the node's gradient to
every element copy; the copies all interpolate the same grid point, so the
grid gradient came out multiplied by the share count (measured rel 2.7 on
both 2-D classes with a C-PML). Equation-vs-equation and c-vs-eager tests
cannot see it -- every path scaled the same way -- so this compares with FD.

Per the FD checks elsewhere: tolerance with no absolute floor, two step sizes
(a real derivative does not depend on h), and a direction that leaves the
outer ring alone, so the edge-padded C-PML band and its ``c_max`` (the model
maximum is the bottom row) do not move.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import PropSEM
from sweep.sem.equations.acoustic2d import AcousticSEM2D
from sweep.sem.equations.acoustic2d_deformed import AcousticSEM2DDeformed
from sweep.sem.equations.acoustic3d import AcousticSEM3D
from sweep.sem.equations.acoustic3d_deformed import AcousticSEM3DDeformed
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D

DT64 = torch.float64
DH, DT, F0, NO = 10.0, 1e-3, 10.0, 4
RIM = 2                                     # untouched outer cells

CASES_2D = (AcousticSEM2D, AcousticSEM2DDeformed, AcousticPotentialSEM2D)
CASES_3D = (AcousticSEM3D, AcousticSEM3DDeformed)


def _wav(nt):
    t = np.arange(nt) * DT
    a = (np.pi * F0 * (t - 0.12)) ** 2
    return torch.tensor((1 - 2 * a) * np.exp(-a), dtype=DT64)


def _check(misfit, vp0, label):
    torch.manual_seed(0)
    d = torch.zeros_like(vp0)
    inner = tuple(slice(RIM, n - RIM) for n in vp0.shape)
    d[inner] = torch.randn_like(d[inner])
    v = vp0.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(misfit(v), v)
    an = float((g * d).sum())
    with torch.no_grad():
        for h in (1e-1, 1e-2):
            fd = (float(misfit(vp0 + h * d)) - float(misfit(vp0 - h * d))) / (2 * h)
            rel = abs(an - fd) / abs(fd)
            assert rel < 1e-5, f"{label} h={h:g}: analytic {an:+.6e} fd {fd:+.6e} rel {rel:.1e}"


@pytest.mark.parametrize("abcn", [0, 2 * NO], ids=["bare", "cpml"])
@pytest.mark.parametrize("equation", CASES_2D, ids=lambda c: c.__name__)
def test_grid_gradient_2d_matches_fd(equation, abcn):
    nz, nx, nt = 25, 33, 250
    p = PropSEM(equation, shape=(nz, nx), dh=DH, dt=DT, nt=nt, n_order=NO,
                abcn=abcn, free_surface=True, f_dominant=F0, dev="cpu",
                dtype=DT64, impl="eager")
    src = np.array([[16, 6]])
    rix = np.arange(4, nx - 3, 4)
    rec = np.stack([rix, np.full_like(rix, 3)], -1)[None]
    wav = _wav(nt)
    vp0 = (2000.0 + 15.0 * torch.arange(nz, dtype=DT64)[:, None]
           + 40.0 * torch.sin(torch.arange(nx, dtype=DT64)[None] / 4.0))
    _check(lambda v: p(wav, src, rec, models=[v]).pow(2).sum(), vp0,
           f"{equation.__name__} abcn={abcn}")


@pytest.mark.parametrize("abcn", [0, NO], ids=["bare", "cpml"])
@pytest.mark.parametrize("equation", CASES_3D, ids=lambda c: c.__name__)
def test_grid_gradient_3d_matches_fd(equation, abcn):
    nz = ny = nx = 13
    nt = 150
    p = PropSEM(equation, shape=(nz, ny, nx), dh=DH, dt=DT, nt=nt, n_order=NO,
                abcn=abcn, free_surface=True, f_dominant=F0, dev="cpu",
                dtype=DT64, impl="eager")
    src = np.array([[6, 6, 5]])
    rix = np.arange(3, nx - 2, 3)
    rec = np.stack([rix, np.full_like(rix, 6), np.full_like(rix, 2)], -1)[None]
    wav = _wav(nt)
    z = torch.arange(nz, dtype=DT64)[:, None, None]
    x = torch.arange(nx, dtype=DT64)[None, None, :]
    vp0 = (2000.0 + 15.0 * z + 40.0 * torch.sin(x / 3.0)).expand(nz, ny, nx).contiguous()
    _check(lambda v: p(wav, src, rec, models=[v]).pow(2).sum(), vp0,
           f"{equation.__name__} abcn={abcn}")
