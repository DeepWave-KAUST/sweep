"""Batched (vmapped) boundary saving must equal looping the single-shot path.

The two rollouts are separate implementations that have already drifted once
(frame checkpoints exist only in the single-shot one), and the only previous
equivalence test was CUDA-only and covered one acoustic equation. This one
runs on CPU in float64, where the tolerance can be tight: vmap changes kernel
shapes so bit-exactness is not expected, but 1e-12 relative is.

The elastic displacement equation is asserted to be REFUSED by the batched
path -- it needs the element-local C-PML memory ring the vmapped rollout does
not implement. Silently accepting it and zeroing the ring would corrupt the
interior gradient (measured 22% of |g| lives in the band at nt=1200).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sweep.propagator import BoundarySaving
from sweep.sem import PropSEM
from sweep.sem.boundary_saving import SEMReconState
from sweep.sem.equations.acoustic_potential2d import AcousticPotentialSEM2D
from sweep.sem.equations.elastic2d_deformed import ElasticSEM2DDeformed

NO, DH, F0 = 4, 20.0, 10.0
NZ = NX = 33
DT = 0.30 * (0.172 * NO * DH) / (3000.0 * 1.2)
NT = 160


def _shot(dtype):
    t = np.arange(NT) * DT
    a = (np.pi * F0 * (t - 1.5 / F0)) ** 2
    w = torch.tensor(((1 - 2 * a) * np.exp(-a)), dtype=dtype)
    src = np.array([[10, 3], [22, 3]])
    rix = np.arange(6, NX - 6, 2)
    rec = np.stack([rix, np.full_like(rix, 3)], -1)[None]
    return w, src, rec


def _prop(equation, dtype, memory=None):
    return PropSEM(equation, shape=(NZ, NX), dh=DH, dt=DT, nt=NT, n_order=NO,
                   free_surface=("top",), abcn=3 * NO,
                   f_dominant=F0, dev="cpu", dtype=dtype, impl="eager",
                   memory=memory)


def test_batched_bs_equals_looped_single_shot_bs():
    D = torch.float64
    w, src, rec = _shot(D)
    vp0 = torch.full((NZ, NX), 2500.0, dtype=D)
    vp0 += 40.0 * torch.linspace(0, 1, NZ)[:, None]      # a gradient, not air

    # batched: both shots in one vmapped rollout
    vp_b = vp0.clone().requires_grad_(True)
    sb = _prop(AcousticPotentialSEM2D, D, memory=BoundarySaving())(
        w, src, rec, models=[vp_b])
    (sb ** 2).sum().backward()

    # looped: one single-shot rollout per shot, gradients accumulate
    vp_l = vp0.clone().requires_grad_(True)
    outs = []
    for k in range(src.shape[0]):
        s1 = _prop(AcousticPotentialSEM2D, D, memory=BoundarySaving())(
            w, src[k:k + 1], rec, models=[vp_l])
        outs.append(s1)
        (s1 ** 2).sum().backward()
    sl = torch.cat(outs, dim=0)

    rel_s = float((sb - sl).abs().max() / sl.abs().max())
    rel_g = float((vp_b.grad - vp_l.grad).abs().max() / vp_l.grad.abs().max())
    assert rel_s < 1e-12, f"seismograms differ: {rel_s:.3e}"
    assert rel_g < 1e-11, f"gradients differ: {rel_g:.3e}"


def test_batched_int8_ring_gives_each_shot_its_own_scale():
    """The batched int8 ring takes its amax over the NODE axis alone, so every
    shot owns its quantisation step.

    A batch-shared amax is set by the loudest shot: with amplitudes spanning
    1e3 the quiet shot's band rounds to all zeros and its reverse march
    reconstructs nothing. The C backend pins the same invariant end to end
    (test_sem_shot_batch::test_batched_bs_int8_scales_are_per_shot).

    This one goes straight at ``SEMReconState`` instead, and that is
    deliberate: two end-to-end versions of it were written first and BOTH were
    vacuous. The first used per-shot models, which make ``_forward_bs_batched``
    fall back to looping the single-shot path, so it compared the loop against
    itself (0.00e+00 for every store dtype). The second used a shared model and
    the wavelet gradient -- which does reach the batched rollout, but the
    wavelet gradient is fed by the source injection in the INTERIOR, so a band
    that reconstructs to zero barely moves it, and the test passed with the
    bug deliberately reinstated. The ring is where the invariant lives, so the
    ring is what to assert on.
    """
    nt, n_global, B = 3, 40, 2
    idx = torch.arange(4, 12)
    zero_idx = torch.arange(0, 4)
    recon = SEMReconState(nt, [0], idx, zero_idx, torch.device("cpu"),
                          torch.float32, store_dtype="int8", batch=B)
    band = torch.zeros(B, n_global)
    torch.manual_seed(0)
    band[0, idx] = 1e-3 * (1.0 + torch.rand(idx.numel()))   # quiet shot
    band[1, idx] = 1.0 * (1.0 + torch.rand(idx.numel()))    # loud shot
    recon.save(0, [band])
    out = recon.restore([torch.zeros(B, n_global)], 0, [(0, 0, 0)])[0]

    for k, tag in ((0, "quiet"), (1, "loud")):
        ref, got = band[k, idx], out[k, idx]
        rel = float((got - ref).abs().max() / ref.abs().max())
        assert rel < 1e-2, (
            f"{tag} shot: int8 ring reconstructs to {rel:.2e} relative -- with "
            f"one amax over the whole batch the quiet shot rounds to zero")
    assert float(out[0, idx].abs().max()) > 0.0, "the quiet shot came back empty"


@pytest.mark.parametrize("store", [None, "fp32", "bf16", "fp16", "int8"])
def test_the_eager_ring_takes_the_same_spellings_the_c_ring_does(store):
    """``bs_store_dtype`` used to reach ``torch.zeros`` unresolved on this
    path, so "bf16" -- which the C ring accepts (prop_c._band_store) -- came
    back as a TypeError from inside the allocator. The two support different
    SETS (the C kernels have no fp16), but they must fail the same way."""
    recon = SEMReconState(2, [0], torch.arange(4), torch.arange(0),
                          torch.device("cpu"), torch.float32,
                          store_dtype=store)
    assert recon.ring[0].dtype is not None
    with pytest.raises(ValueError, match="bs_store_dtype"):
        SEMReconState(2, [0], torch.arange(4), torch.arange(0),
                      torch.device("cpu"), torch.float32,
                      store_dtype="bfloat16")          # the C path's ValueError


def test_the_batched_rollout_is_the_one_a_shared_model_takes(monkeypatch):
    """``_forward_bs_batched`` takes the vmapped rollout only for a SHARED
    model; per-shot models loop the single-shot path instead. That is what made
    the first version of the test above vacuous, so it is pinned here."""
    import sweep.sem.boundary_saving as BS
    seen = []
    real = BS.SEMReconState

    class Spy(real):
        def __init__(self, *a, **k):
            seen.append(k.get("batch"))
            super().__init__(*a, **k)

    monkeypatch.setattr(BS, "SEMReconState", Spy)
    D = torch.float32
    w, src, rec = _shot(D)
    vp = torch.full((NZ, NX), 2500.0, dtype=D)
    _prop(AcousticPotentialSEM2D, D,
          memory=BoundarySaving(storage_dtype="int8"))(w, src, rec, models=[vp])
    assert seen == [src.shape[0]], (
        f"expected one batched recon of {src.shape[0]} shots, saw {seen}")


def test_batched_bs_refuses_the_element_ring_equation():
    D = torch.float32
    w, src, rec = _shot(D)
    models = [torch.full((NZ, NX), v) for v in (2400.0, 1250.0, 2000.0)]
    models = [m.requires_grad_(True) for m in models]
    with pytest.raises(NotImplementedError, match="element-local"):
        _prop(ElasticSEM2DDeformed, D, memory=BoundarySaving())(
            w, src, rec, models=models)
