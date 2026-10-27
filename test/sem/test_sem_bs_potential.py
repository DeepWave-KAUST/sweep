"""Boundary saving for AcousticPotentialSEM2D on BOTH impls, via
PropSEM(memory=BoundarySaving()).

eager: swap2nd reverse + exact-band ring (chi + the 4 CPML memories saved on
the full band, so the reconstruction is exact and the BS gradient equals the
full-tape gradient — including with a 4-sided band, no free surface).
cuda:  reverse chi march in the reversible interior + exact band restore from
the per-step band histories (grad identical to save-all).

All comparisons: cos >= 1-1e-4 on forward and dL/dvp.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving, Full
from sweep.sem import PropSEM
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.equations.acoustic3d import AcousticSEM3D
from sweep.sem.prop_c import sem_c_available

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and sem_c_available()),
    reason="needs CUDA + compiled sweep._C SEM binding")

NZ, NX, DH, DT, NT, F0, ABCN = 49, 65, 20.0, 2e-3, 300, 8.0, 6


def _vp():
    z = np.arange(NZ)[:, None]
    x = np.arange(NX)[None, :]
    return (1800 + 8.0 * z + 150 * np.sin(x / 9.0)).astype(np.float32)


def _wav(dev):
    t = np.arange(NT, dtype=np.float32) * DT
    a = (np.pi * F0 * (t - 0.15)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device=dev)


def _cos(a, b):
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def _run(impl, fs, mem):
    p = PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                n_order=4, free_surface=fs, abcn=ABCN * 4,
                f_dominant=F0, dev="cuda", dtype=torch.float32, impl=impl,
                memory=mem)
    src = np.array([[32, 2]], dtype=np.int64)
    rix = np.arange(3, NX - 3, 2)
    rec = np.stack([rix, np.full_like(rix, 2)], -1)[None]
    vp = torch.tensor(_vp(), device="cuda", requires_grad=True)
    s = p(_wav("cuda"), src, rec, models=[vp])
    (s ** 2).sum().backward()
    return s.detach().cpu().numpy(), vp.grad.cpu().numpy()


@pytest.mark.parametrize("impl", ["eager", "c"])
@pytest.mark.parametrize("fs", [False, True], ids=["nofs", "fs"])
def test_boundary_saving_matches_full(impl, fs):
    sf, gf = _run(impl, fs, Full())
    sb, gb = _run(impl, fs, BoundarySaving())
    assert _cos(sb, sf) >= 1 - 1e-4, f"fwd cos={_cos(sb, sf):.6f}"
    assert _cos(gb, gf) >= 1 - 1e-4, f"grad cos={_cos(gb, gf):.6f}"


@pytest.mark.parametrize("fs", [False, True], ids=["nofs", "fs"])
def test_cuda_bs_matches_eager_full(fs):
    se, ge = _run("eager", fs, Full())
    sc, gc = _run("c", fs, BoundarySaving())
    assert _cos(sc, se) >= 1 - 1e-4, f"fwd cos={_cos(sc, se):.6f}"
    assert _cos(gc, ge) >= 1 - 1e-4, f"grad cos={_cos(gc, ge):.6f}"


# --------------------------- 3-D (AcousticSEM3D) --------------------------- #
NZ3, NY3, NX3 = 25, 21, 29


def _vp3():
    z = np.arange(NZ3)[:, None, None]
    x = np.arange(NX3)[None, None, :]
    return (1800 + 15.0 * z + 40 * np.sin(x / 5.0)
            + 0 * np.arange(NY3)[None, :, None]).astype(np.float32)


def _run3(impl, fs, mem, wav):
    p = PropSEM(AcousticSEM3D, shape=(NZ3, NY3, NX3), dh=20.0, dt=1.5e-3, nt=250,
                n_order=4, free_surface=fs, abcn=3 * 4, f_dominant=8.0,
                dev="cuda", dtype=torch.float32, impl=impl, memory=mem)
    src = np.array([[14, 10, 3]], dtype=np.int64)
    rix = np.arange(3, NX3 - 3, 2)
    rec = np.stack([rix, np.full_like(rix, 10), np.full_like(rix, 4)], -1)[None]
    vp = torch.tensor(_vp3(), device="cuda", requires_grad=True)
    s = p(wav, src, rec, models=[vp])
    (s ** 2).sum().backward()
    return s.detach().cpu().numpy(), vp.grad.cpu().numpy()


def _wav3():
    t = np.arange(250, dtype=np.float32) * 1.5e-3
    a = (np.pi * 8.0 * (t - 0.15)) ** 2
    w0 = torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32),
                      device="cuda")
    p = PropSEM(AcousticSEM3D, shape=(NZ3, NY3, NX3), dh=20.0, dt=1.5e-3, nt=250,
                n_order=4, free_surface=False, abcn=3 * 4, f_dominant=8.0,
                dev="cuda", dtype=torch.float32, impl="eager")
    src = np.array([[14, 10, 3]], dtype=np.int64)
    rix = np.arange(3, NX3 - 3, 2)
    rec = np.stack([rix, np.full_like(rix, 10), np.full_like(rix, 4)], -1)[None]
    with torch.no_grad():
        o0 = p(w0, src, rec, models=[torch.tensor(_vp3(), device="cuda")])
    return w0 * (1.0 / float(o0.pow(2).mean().sqrt()))   # data O(1): grads off
                                                         # the fp32 noise floor


@pytest.mark.parametrize("impl", ["eager", "c"])
@pytest.mark.parametrize("fs", [False, True], ids=["nofs", "fs"])
def test_boundary_saving_3d_matches_full(impl, fs):
    wav = _wav3()
    sf, gf = _run3(impl, fs, Full(), wav)
    sb, gb = _run3(impl, fs, BoundarySaving(), wav)
    assert _cos(sb, sf) >= 1 - 1e-4, f"fwd cos={_cos(sb, sf):.6f}"
    assert _cos(gb, gf) >= 1 - 1e-4, f"grad cos={_cos(gb, gf):.6f}"


@pytest.mark.parametrize("fs", [False, True], ids=["nofs", "fs"])
def test_cuda_bs_3d_matches_eager_full(fs):
    """Direct cross-impl check: impl='c' + boundary saving vs eager full tape."""
    wav = _wav3()
    se, ge = _run3("eager", fs, Full(), wav)
    sc, gc = _run3("c", fs, BoundarySaving(), wav)
    assert _cos(sc, se) >= 1 - 1e-4, f"fwd cos={_cos(sc, se):.6f}"
    assert _cos(gc, ge) >= 1 - 1e-4, f"grad cos={_cos(gc, ge):.6f}"


def _run3_bare(impl, mem, wav):
    p = PropSEM(AcousticSEM3D, shape=(NZ3, NY3, NX3), dh=20.0, dt=1.5e-3, nt=250,
                n_order=4, free_surface=True, abcn=0, f_dominant=8.0,
                dev="cuda", dtype=torch.float32, impl=impl, memory=mem)
    src = np.array([[14, 10, 3]], dtype=np.int64)
    rix = np.arange(3, NX3 - 3, 2)
    rec = np.stack([rix, np.full_like(rix, 10), np.full_like(rix, 4)], -1)[None]
    vp = torch.tensor(_vp3(), device="cuda", requires_grad=True)
    s = p(wav, src, rec, models=[vp])
    (s ** 2).sum().backward()
    return s.detach().cpu().numpy(), vp.grad.cpu().numpy()


def test_cuda_3d_bare_is_refused():
    """Bare 3-D (abcn=0): the sem3d_acoustic kernel solves div(c² grad u), not
    the eager c² Laplace u, so impl='c' refuses it and 'auto' stays eager."""
    with pytest.raises(NotImplementedError, match="without a C-PML"):
        _run3_bare("c", BoundarySaving(), _wav3())
    p = PropSEM(AcousticSEM3D, shape=(NZ3, NY3, NX3), dh=20.0, dt=1.5e-3, nt=4,
                n_order=4, free_surface=True, abcn=0, f_dominant=8.0,
                dev="cuda", dtype=torch.float32)
    assert p.impl == "eager"
