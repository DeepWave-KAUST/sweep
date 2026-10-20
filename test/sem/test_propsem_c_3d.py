"""PropSEM 3-D impl='c' (CUDA) vs impl='eager' for AcousticSEM3D: grid-index
sources/receivers, grid vp model + gradient, CPML (abcn, added outside the
model).

The source is amplitude-calibrated first (notebook-17 style): a unit wavelet
gives O(1e-10) data in 3-D and the gradients then sit at the fp32 noise floor,
where cos is meaningless (P2 lesson).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.sem import PropSEM
from sweep.sem.equations.acoustic3d import AcousticSEM3D
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NZ, NY, NX = 25, 21, 29
DH, DT, NT, F0, ABCN = 20.0, 1.5e-3, 250, 8.0, 3


def _vp():
    z = np.arange(NZ)[:, None, None]
    x = np.arange(NX)[None, None, :]
    return (1800 + 15.0 * z + 40 * np.sin(x / 5.0)
            + 0 * np.arange(NY)[None, :, None]).astype(np.float32)


def _wav(dev):
    t = np.arange(NT, dtype=np.float32) * DT
    a = (np.pi * F0 * (t - 0.15)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device=dev)


def _cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _mk(impl):
    return PropSEM(AcousticSEM3D, shape=(NZ, NY, NX), dh=DH, dt=DT, nt=NT,
                   n_order=4, free_surface=False,
                   abcn=ABCN * 4, f_dominant=F0, dev="cuda",
                   dtype=torch.float32, impl=impl)


def test_propsem_c_3d_matches_eager():
    src = np.array([[14, 10, 3]], dtype=np.int64)
    rix = np.arange(3, NX - 3, 2)
    rec = np.stack([rix, np.full_like(rix, 10), np.full_like(rix, 4)], -1)[None]
    wav0 = _wav("cuda")
    with torch.no_grad():
        o0 = _mk("eager")(wav0, src, rec,
                          models=[torch.tensor(_vp(), device="cuda")])
    wav = wav0 * (1.0 / float(o0.pow(2).mean().sqrt()))

    def run(impl):
        vp = torch.tensor(_vp(), device="cuda", requires_grad=True)
        s = _mk(impl)(wav, src, rec, models=[vp])
        (s ** 2).sum().backward()
        return s.detach().cpu().numpy(), vp.grad.cpu().numpy()

    se, ge = run("eager")
    sc, gc = run("c")
    assert _cos(sc, se) >= 1 - 1e-4, f"forward cos={_cos(sc, se):.6f}"
    assert _cos(gc, ge) >= 1 - 1e-4, f"grad cos={_cos(gc, ge):.6f}"
