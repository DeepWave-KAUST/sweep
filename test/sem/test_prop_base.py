"""PropSEM (PropBase-compatible SEM adapter) regression tests."""
import numpy as np
import torch

from sweep.propagator import Ckpt, Full
from sweep.sem import (
    PropSEM,
    AcousticSEM2D,
    SEMPropagator,
)

NZ, NX = 49, 57            # (nz-1, nx-1) = (48, 56), both divisible by n_order=4
DH, DT, NT = 10.0, 1.5e-3, 120
DTYPE = torch.float64


def ricker(t, f0, delay):
    a = (np.pi * f0 * (t - delay)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def _vp_grid():
    z = np.arange(NZ)[:, None] * DH
    return torch.tensor(1800.0 + 0.5 * z + np.zeros((NZ, NX)), dtype=DTYPE)


def _wav():
    t = np.arange(NT) * DT
    return torch.tensor(ricker(t, 10.0, 0.06), dtype=DTYPE)


def _prop(**kw):
    # the defaults these tests were written against: a free top, eager
    kw.setdefault("free_surface", True)
    kw.setdefault("impl", "eager")
    return PropSEM(AcousticSEM2D, shape=(NZ, NX), dh=DH, dt=DT, nt=NT,
                   dtype=DTYPE, **kw)


def test_shape_and_finite():
    prop = _prop(free_surface=True)
    rix = np.arange(4, NX - 4, 6)
    src = np.array([[NX // 2, 2]], dtype=np.int64)                  # (1, 2) [ix, iz]
    rec = np.stack([rix, np.full_like(rix, 2)], -1)[None]          # (1, nrec, 2)
    seis = prop(_wav(), src, rec, models=[_vp_grid()])
    assert seis.shape == (1, NT, len(rix), 1)      # FD layout (B, nt, nrec, ncomp)
    assert torch.isfinite(seis).all()


def test_matches_native_sempropagator():
    """Adapter output is bit-for-bit a hand-built SEMPropagator call."""
    vp, wav = _vp_grid(), _wav()
    six, siz = NX // 2, 2
    rix, riz = np.arange(4, NX - 4, 6), 2
    src = np.array([[six, siz]], dtype=np.int64)
    rec = np.stack([rix, np.full_like(rix, riz)], -1)[None].astype(np.int64)

    prop = _prop(free_surface=True)
    seis_adapter = prop(wav, src, rec, models=[vp]).detach()

    # Same eq (same mesh + interp), coords converted by hand (index + pad)*dh:
    # the model sits inside the absorbing layer, _pad0 cells in.
    eq = prop._build_eq(vp)
    pz, px = prop._pad0
    sp = SEMPropagator.build(
        eq, dt=DT, nt=NT,
        source_xz=[((six + px) * DH, (siz + pz) * DH)],
        receiver_xz=[((float(ix) + px) * DH, (riz + pz) * DH) for ix in rix],
    )
    seis_native = sp.forward(wav)[..., 0].T          # (nrec, nt)
    # the adapter now returns the FD layout; compare in the native one
    assert torch.allclose(seis_adapter[0, :, :, 0].T, seis_native,
                          atol=1e-12, rtol=0)


def test_autograd_grid_gradient():
    """loss.backward() returns d loss / d vp on the (nz, nx) grid."""
    vp = _vp_grid().requires_grad_(True)
    prop = _prop()
    src = np.array([[NX // 2, 2]], dtype=np.int64)
    rec = np.array([[10, 2], [20, 2], [30, 2]], dtype=np.int64)[None]
    seis = prop(_wav(), src, rec, models=[vp])
    (0.5 * seis.pow(2).sum()).backward()
    assert vp.grad is not None and vp.grad.shape == (NZ, NX)
    assert torch.isfinite(vp.grad).all() and vp.grad.abs().sum() > 0


def test_multishot_shape():
    prop = _prop()
    src = np.array([[20, 2], [36, 2]], dtype=np.int64)             # (2, 2) → 2 shots
    rec = np.array([[10, 2], [44, 2]], dtype=np.int64)[None]       # (1, nrec, 2) broadcast
    seis = prop(_wav(), src, rec, models=[_vp_grid()])
    assert seis.shape == (2, NT, 2, 1)


def test_batched_matches_loop():
    """nshots>1 (vmap-batched) == per-shot single-shot loop, bit-for-bit."""
    vp, wav = _vp_grid(), _wav()
    prop = _prop(free_surface=True)
    src = np.array([[15, 2], [28, 2], [42, 2]], dtype=np.int64)        # 3 shots
    rix, riz = np.arange(4, NX - 4, 8), 2
    rec = np.stack([rix, np.full_like(rix, riz)], -1)[None].astype(np.int64)

    seis_batched = prop(wav, src, rec, models=[vp]).detach()           # (3, nrec, nt)

    eq = prop._build_eq(vp)
    pz, px = prop._pad0                                 # model origin in the mesh
    ref = []
    for b in range(src.shape[0]):
        sp = SEMPropagator.build(
            eq, dt=DT, nt=NT,
            source_xz=[((float(src[b, 0]) + px) * DH,
                        (float(src[b, 1]) + pz) * DH)],
            receiver_xz=[((float(ix) + px) * DH, (riz + pz) * DH) for ix in rix],
        )
        ref.append(sp.forward(wav)[..., 0].T)
    ref = torch.stack(ref)                                # (B, nrec, nt)
    assert torch.allclose(seis_batched[..., 0].permute(0, 2, 1), ref,
                          atol=1e-10, rtol=0)


def test_batched_autograd_grid_gradient():
    """Batched multi-shot path is differentiable w.r.t. the grid model."""
    vp = _vp_grid().requires_grad_(True)
    prop = _prop()
    src = np.array([[15, 2], [40, 2]], dtype=np.int64)                 # 2 shots
    rec = np.array([[10, 2], [30, 2], [50, 2]], dtype=np.int64)[None]
    seis = prop(_wav(), src, rec, models=[vp])
    assert seis.shape == (2, NT, 3, 1)
    (0.5 * seis.pow(2).sum()).backward()
    assert vp.grad is not None and vp.grad.shape == (NZ, NX)
    assert torch.isfinite(vp.grad).all() and vp.grad.abs().sum() > 0


def test_checkpoint_matches_full_gradient():
    """Checkpointed batched gradient == full-graph batched gradient."""
    src = np.array([[15, 2], [28, 2], [42, 2]], dtype=np.int64)
    rec = np.array([[10, 2], [30, 2], [50, 2]], dtype=np.int64)[None]
    wav = _wav()

    def grad(chunk):
        vp = _vp_grid().requires_grad_(True)
        prop = _prop(memory=Ckpt(chunks=chunk) if chunk else Full())
        seis = prop(wav, src, rec, models=[vp])
        (0.5 * seis.pow(2).sum()).backward()
        return vp.grad.detach().clone()

    g_full = grad(None)
    g_ckpt = grad(40)                                   # 3 checkpointed chunks
    assert torch.allclose(g_full, g_ckpt, atol=1e-10, rtol=1e-6)


def test_gll_native_model_passthrough():
    """A GLL-native (E, ngll, ngll) model is accepted as-is."""
    prop = _prop()
    vp_e = prop._interp_to_gll(_vp_grid())          # GLL-native tensor
    src = np.array([[NX // 2, 2]], dtype=np.int64)
    rec = np.array([[10, 2], [30, 2]], dtype=np.int64)[None]
    seis = prop(_wav(), src, rec, models=[vp_e])
    assert seis.shape == (1, NT, 2, 1) and torch.isfinite(seis).all()
