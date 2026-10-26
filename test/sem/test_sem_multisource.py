"""FD-parity multi-source (source-encoding) tests for PropSEM.

Mirrors the FD ``PropTorch`` source modes (``base._normalize_io``): mode B feeds
``sources=(1, nsrc, nd)`` — one super-shot of ``nsrc`` superposed point sources —
with the wavelet either shared ``(nt,)`` or per-source ``(nsrc, nt)`` (distinct
polarity / amplitude / time function per source).

The decisive correctness check is linear superposition: the constant-density
acoustic wave equation is linear in the source, so a per-source mode-B forward
must equal the sum of the individual single-source forwards to machine precision.
"""
import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving
from sweep.sem import PropSEM, AcousticSEM2D
from sweep.sem.equations.acoustic3d import AcousticSEM3D

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float64

NZ, NX = 49, 57                       # (nz-1, nx-1) = (48, 56), both div by n_order=4
DH, DT, NT = 10.0, 1.5e-3, 120


def _ricker(f0, delay, scale=1.0):
    t = np.arange(NT) * DT
    a = (np.pi * f0 * (t - delay)) ** 2
    return torch.tensor((1.0 - 2.0 * a) * np.exp(-a) * scale, device=DEV, dtype=DTYPE)


def _vp_grid():
    z = np.arange(NZ)[:, None] * DH
    return torch.tensor(1800.0 + 0.5 * z + np.zeros((NZ, NX)),
                        device=DEV, dtype=DTYPE)


def _prop(**kw):
    # the defaults these tests were written against: a free top, eager
    kw.setdefault("free_surface", True)
    kw.setdefault("impl", "eager")
    # Two absorbing elements a side, not the default ten: superposition holds
    # whatever the boundary does, and ten a side made the mesh 748 elements
    # for a model of 168 (two: 252). The boundary-saving tests set their own.
    kw.setdefault("abcn", 2 * 4)
    return PropSEM(AcousticSEM2D, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                   dev=DEV, dtype=DTYPE, **kw)


def _cos(x, y):
    x, y = x.reshape(-1), y.reshape(-1)
    return (x @ y / (x.norm() * y.norm())).item()           # bare fp64 cosine


# two sources at the surface, well separated in x; one receiver line along x
SRC_B = np.array([[[18, 2], [38, 2]]], dtype=np.int64)       # (1, nsrc=2, 2)
SRC_0 = np.array([[18, 2]], dtype=np.int64)                  # (1, 2) single shot
SRC_1 = np.array([[38, 2]], dtype=np.int64)
_rix = np.arange(6, NX - 6, 6)
REC = np.stack([_rix, np.full_like(_rix, 2)], -1)[None].astype(np.int64)  # (1,nrec,2)


# --------------------------------------------------------------------------- #
# 1. Linear superposition — the core correctness proof.
# --------------------------------------------------------------------------- #
def test_mode_b_forward_superposition():
    """Per-source mode-B forward == Σ_k single-source forward (acoustic is linear
    in the source). Two *different* wavelets, so it is a genuine per-source test."""
    vp = _vp_grid()
    w0, w1 = _ricker(10.0, 0.06), _ricker(14.0, 0.05)
    prop = _prop()

    seis_B = prop(torch.stack([w0, w1]), SRC_B, REC, models=[vp])    # (1, nrec, nt)
    seis_0 = prop(w0, SRC_0, REC, models=[vp])
    seis_1 = prop(w1, SRC_1, REC, models=[vp])

    assert seis_B.shape == seis_0.shape == (1, NT, len(_rix), 1)
    ref = seis_0 + seis_1
    assert (seis_B - ref).norm() / ref.norm() < 1e-9


def test_mode_b_per_source_equals_shared():
    """A per-source wavelet whose rows are identical reproduces the shared-wavelet
    path bit-for-bit (cross-checks the scatter path vs the summed-pattern path)."""
    vp = _vp_grid()
    w = _ricker(10.0, 0.06)
    prop = _prop()

    seis_shared = prop(w, SRC_B, REC, models=[vp])                  # (nt,)  → summed
    seis_persrc = prop(w.unsqueeze(0).repeat(2, 1), SRC_B, REC, models=[vp])  # (2,nt)
    assert (seis_persrc - seis_shared).norm() / seis_shared.norm() < 1e-9


def test_mode_b_polarity_encoding():
    """Per-source polarity: [w, -w] differs from [w, w], and equals s0 - s1."""
    vp = _vp_grid()
    w = _ricker(10.0, 0.06)
    prop = _prop()

    seis_pp = prop(torch.stack([w, w]), SRC_B, REC, models=[vp])
    seis_pm = prop(torch.stack([w, -w]), SRC_B, REC, models=[vp])
    assert (seis_pp - seis_pm).norm() / seis_pp.norm() > 1e-2       # genuinely differ

    s0 = prop(w, SRC_0, REC, models=[vp])
    s1 = prop(w, SRC_1, REC, models=[vp])
    assert (seis_pm - (s0 - s1)).norm() / (s0 - s1).norm() < 1e-9


def test_mode_b_wavelet_shape_validation():
    """Mode-B wavelet must be (nt,) or (nsrc, nt) — a wrong source count raises."""
    vp = _vp_grid()
    prop = _prop()
    with pytest.raises(ValueError):
        prop(torch.zeros(3, NT, device=DEV, dtype=DTYPE), SRC_B, REC, models=[vp])


# --------------------------------------------------------------------------- #
# 2. Gradients — per-source source encoding is differentiable, and BS == full.
# --------------------------------------------------------------------------- #
# The PML sits *outside* the (NZ, NX) model, so the whole model stays a
# reconstructable interior for boundary saving; six elements of it (not the
# default ten) keep the mesh small.
def _prop_bs(memory=None):
    return _prop(free_surface=True, abcn=6 * 4, memory=memory)


def test_mode_b_model_gradient_bs_matches_full():
    """Source-encoded vp gradient (the true cross-talk gradient): boundary-saving
    matches the full-tape autograd reference."""
    w0, w1 = _ricker(10.0, 0.06, scale=1e6), _ricker(14.0, 0.05, scale=1e6)
    wav = torch.stack([w0, w1])

    def grad(memory):
        vp = _vp_grid().requires_grad_(True)
        seis = _prop_bs(memory)(wav, SRC_B, REC, models=[vp])
        seis.pow(2).sum().backward()
        return seis.detach(), vp.grad.detach()

    seis_full, g_full = grad(None)
    seis_bs, g_bs = grad(BoundarySaving())
    assert (seis_bs - seis_full).norm() / seis_full.norm() < 1e-5
    assert _cos(g_bs, g_full) > 0.999                    # interior reconstruction exact
    assert (g_bs - g_full).norm() / g_full.norm() < 1e-2


@pytest.mark.parametrize("memory", [None, BoundarySaving()],
                         ids=["None", "boundary"])
def test_mode_b_wavelet_gradient(memory):
    """The per-source wavelet itself is a differentiable leaf (need_wav path),
    in both the full-tape and the boundary-saved rollouts."""
    vp = _vp_grid()
    wav = torch.stack([_ricker(10.0, 0.06, scale=1e6),
                       _ricker(14.0, 0.05, scale=1e6)]).requires_grad_(True)
    seis = _prop_bs(memory)(wav, SRC_B, REC, models=[vp])
    seis.pow(2).sum().backward()
    assert wav.grad is not None and wav.grad.shape == (2, NT)
    assert torch.isfinite(wav.grad).all() and wav.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# 3. 3-D: the same source-encoding path on a Hex8 mesh (CUDA-gated).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(DEV != "cuda", reason="3-D SEM is heavy; gate on CUDA")
def test_mode_b_3d_superposition():
    nz, ny, nx, dh, dt, nt = 25, 25, 25, 20.0, 1.5e-3, 60
    vp = torch.full((nz, ny, nx), 2000.0, device=DEV, dtype=DTYPE)
    t = np.arange(nt) * dt
    mk = lambda f0, d: torch.tensor(
        (1 - 2 * (np.pi * f0 * (t - d)) ** 2) * np.exp(-(np.pi * f0 * (t - d)) ** 2),
        device=DEV, dtype=DTYPE)
    w0, w1 = mk(12.0, 0.05), mk(16.0, 0.04)

    prop = PropSEM(AcousticSEM3D, shape=(nz, ny, nx), dh=dh, dt=dt, nt=nt,
                   free_surface=True, abcn=4 * 4, dev=DEV, dtype=DTYPE,
                   impl="eager")
    s_B = np.array([[[8, 12, 2], [16, 12, 2]]], dtype=np.int64)        # (1,2,3)
    rix = np.arange(4, nx - 4, 4)
    rec = np.stack([rix, np.full_like(rix, ny // 2),
                    np.full_like(rix, 2)], -1)[None].astype(np.int64)

    seis_B = prop(torch.stack([w0, w1]), s_B, rec, models=[vp])
    seis_0 = prop(w0, s_B[:, 0], rec, models=[vp])
    seis_1 = prop(w1, s_B[:, 1], rec, models=[vp])
    ref = seis_0 + seis_1
    assert (seis_B - ref).norm() / ref.norm() < 1e-9


@pytest.mark.skipif(DEV != "cuda", reason="3-D elastic is heavy; gate on CUDA")
def test_mode_b_3d_elastic_superposition():
    """Same source-encoding path on the velocity-stress elastic equation, whose
    force source spreads over *three* source fields (vx, vy, vz) — exercises the
    multi-source-field scatter that the acoustic case (1 field) doesn't."""
    from sweep.sem.equations.elastic_vs3d import ElasticVSSEM3D
    nz, ny, nx, dh, dt, nt = 21, 17, 21, 25.0, 4e-4, 80
    mk = lambda v: torch.full((nz, ny, nx), v, device=DEV, dtype=DTYPE)
    models = [mk(3000.0), mk(1732.0), mk(2200.0)]                    # vp, vs, rho
    t = np.arange(nt) * dt
    mw = lambda f0, d: torch.tensor(
        (1 - 2 * (np.pi * f0 * (t - d)) ** 2) * np.exp(-(np.pi * f0 * (t - d)) ** 2),
        device=DEV, dtype=DTYPE)
    w0, w1 = mw(12.0, 0.06), mw(15.0, 0.05)

    prop = PropSEM(ElasticVSSEM3D, shape=(nz, ny, nx), dh=dh, dt=dt, nt=nt,
                   free_surface=False, abcn=2 * 4,
                   f_dominant=12.0, dev=DEV, dtype=DTYPE, impl="eager")
    s_B = np.array([[[7, 8, 5], [14, 8, 5]]], dtype=np.int64)         # (1, 2, 3)
    rx = np.arange(3, nx - 3, 4)
    rec = np.stack([rx, np.full_like(rx, 8), np.full_like(rx, 9)], -1)[None]

    seis_B = prop(torch.stack([w0, w1]), s_B, rec, models=models)     # (1,3,nrec,nt)
    seis_0 = prop(w0, s_B[:, 0], rec, models=models)
    seis_1 = prop(w1, s_B[:, 1], rec, models=models)
    assert seis_B.shape == (1, nt, len(rx), 3)
    ref = seis_0 + seis_1
    assert (seis_B - ref).norm() / ref.norm() < 1e-9
