"""PropSEM impl='c' (CUDA) vs impl='eager' for the 2-D AcousticPotentialSEM2D,
mimicking notebook-17's Marmousi FWI usage: grid-index sources/receivers, a grid
vp model + gradient, CPML (abcn, added outside the model), multi-shot,
free_surface=False.
Both impls use the SAME equation, so forward + dL/dvp match to fp32 (cos>=1-1e-4).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving
from sweep.sem import PropSEM
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NZ, NX, DH, DT, NT, FREQ, ABCN = 65, 97, 20.0, 2e-3, 400, 8.0, 10


def _vp():
    z = np.arange(NZ)[:, None]; x = np.arange(NX)[None, :]
    return (1800 + 6.0 * z + 200 * np.sin(x / 12.0)).astype(np.float32)


def _wav(dev):
    t = np.arange(NT) * DT - 0.12
    a = (np.pi * FREQ * t) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device=dev)


def _cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1); b = np.asarray(b, np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _run(impl):
    dev = "cuda"
    vp = torch.tensor(_vp(), device=dev, requires_grad=True)
    prop = PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                   n_order=4, free_surface=False,
                   abcn=ABCN * 4, f_dominant=FREQ, dev=dev, impl=impl)
    srcs = np.array([[15, 2], [45, 2], [75, 2]], dtype=np.int64)
    rix = np.arange(4, NX - 4, 3)
    rec = np.stack([rix, np.full_like(rix, 2)], -1)[None].astype(np.int64)
    seis = prop(_wav(dev), srcs, rec, models=[vp])
    seis = seis / (seis.abs().max().detach() + 1e-12)
    (seis ** 2).sum().backward()
    return seis.detach().cpu().numpy(), vp.grad.detach().cpu().numpy()


def test_propsem_c_matches_eager():
    se, ge = _run("eager")
    sc, gc = _run("c")
    assert _cos(sc, se) >= 1 - 1e-4, f"forward cos={_cos(sc, se):.6f}"
    assert _cos(gc, ge) >= 1 - 1e-4, f"grad cos={_cos(gc, ge):.6f}"


def test_propsem_c_topography_matches_eager():
    """PropSEM topography + impl='c' (Quad9 curved free surface), with and
    without boundary saving. The CUDA kernels are metric-general (they consume
    the deformed-mesh ixx/ixz/iex/iez), so topo needs no kernel changes."""
    topo = lambda xx: 120.0 * torch.exp(-((xx - NX * DH / 2) / 300.0) ** 2)
    # the surface row of every model column
    rows = (topo(torch.arange(NX, dtype=torch.float64) * DH) / DH).numpy()
    src = np.array([[32, 6]], dtype=np.int64)
    rix = np.arange(4, NX - 4, 3)
    rec = np.stack([rix, np.full_like(rix, 6)], -1)[None].astype(np.int64)

    def run(impl, mem=None):
        p = PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                    n_order=4, free_surface=True,
                    abcn=ABCN * 4, f_dominant=FREQ, dev="cuda", impl=impl,
                    topography=rows, memory=mem)
        vp = torch.tensor(_vp(), device="cuda", requires_grad=True)
        s = p(_wav("cuda"), src, rec, models=[vp])
        (s ** 2).sum().backward()
        return s.detach().cpu().numpy(), vp.grad.detach().cpu().numpy()

    se, ge = run("eager")
    for tag, (sc, gc) in {"c": run("c"),
                          "c+bs": run("c", BoundarySaving())}.items():
        assert _cos(sc, se) >= 1 - 1e-4, f"{tag} fwd cos={_cos(sc, se):.6f}"
        assert _cos(gc, ge) >= 1 - 1e-4, f"{tag} grad cos={_cos(gc, ge):.6f}"
