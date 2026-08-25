"""3-D two-tile DD gradient vs single domain — bit-exact (M4/M5 closure).

Same protocol as test_dd_backward_two_tile.py, on acoustic3d (x-cut):
DD forward (BS gpu ring, per-step u_now halo) -> per-step descending
backward segments with cut_face_mask + lambda/recon halo exchanges ->
owned-slice gradient assembly, compared bitwise against the single-domain
monolithic backward. 3-D layouts: forward 12 slots, adjoint 15 slots.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sweep.equations import Acoustic3D  # noqa: E402
from sweep.parallel import MeshTopology  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import (  # noqa: E402
    SteppedBackwardRunner,
    SteppedBindingRunner,
    acoustic_adj_pairs,
    acoustic_psi_pairs,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")

NZ, NY, NX = 24, 20, 32
NT = 60
DT = 0.0015
SO = 4
M = SO // 2
ABCN = 10
PAD = ABCN + M
NXP = NX // 2

SRC = (NX // 2, NY // 2, NZ // 4)      # on the cut -> owned by tile1
REC_GX = list(range(2, NX - 2, 5))
REC_Y, REC_Z = NY // 2, 2
X_LO_BIT, X_HI_BIT = 1, 2


def ricker(nt, dt, fm=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def vp_global():
    vp = 1800.0 + 600.0 * np.linspace(0, 1, NZ, dtype=np.float32)[:, None, None]
    vp = np.broadcast_to(vp, (NZ, NY, NX)).copy()
    vp[8:14, 5:15, 10:22] += 180.0  # anomaly straddling the cut at x=16
    return vp


def make_prop(shape, topo=None):
    eq = Acoustic3D(spatial_order=SO, device=DEV, backend="torch")
    kw = dict(
        backend="torch", impl="c", shape=shape, dev=DEV, dh=10.0, dt=DT,
        source_type=["h1"], receiver_type=["h1"], abcn=ABCN,
        free_surface=False, pml_type="cpmlr", nt=NT, B=1, use_ckpt=False,
        boundary_saving_config={"enabled": True, "storage": "gpu",
                                "transfer_interval": 1,
                                "pinned_memory": False},
    )
    if topo is not None:
        kw["model_parallel"] = topo
    return PropTorch(eq, **kw)


def capture_both(prop):
    cap = {}
    impl = prop._backend_impl
    fwd_orig = impl.forward_func

    def fwd_wrapper(params):
        out = fwd_orig(params)
        cap["fp"], cap["fwd_raw_out"], cap["fwd_func"] = params, out, fwd_orig
        return out

    impl.forward_func = fwd_wrapper
    bwd_orig = impl.backward_bs_func

    def bwd_wrapper(params):
        out = bwd_orig(params)
        cap["bp"], cap["bwd_func"] = params, bwd_orig
        return out

    impl.backward_bs_func = bwd_wrapper
    return cap


def run_public_once(prop, wavelet, sources, receivers, vp_np):
    models = [torch.tensor(vp_np, device=DEV, requires_grad=True)]
    syn = prop(wavelet, sources, receivers, models=models)
    rec = syn[0] if isinstance(syn, (tuple, list)) else syn
    rec.sum().backward()


class TileState3D:
    def __init__(self, cap):
        self.fp = cap["fp"]
        self.fwd_func = cap["fwd_func"]
        self.bp = cap["bp"]
        self.bwd_func = cap["bwd_func"]
        self.fwd_record_raw = cap["fwd_raw_out"][2]

        L = list(self.fp.wavefields)
        if not L:
            L = [torch.zeros_like(self.fp.models[0]) for _ in range(12)]
        assert len(L) == 12
        self.L_fwd = L
        self.record = torch.zeros_like(self.fwd_record_raw)
        self.fp.record_out = self.record

        self.L_adj = list(self.bp.adjoint_wavefields)
        assert len(self.L_adj) == 15
        self.recon = [torch.zeros_like(self.bp.models[0]) for _ in range(3)]
        self.gbufs = [torch.zeros_like(self.bp.forward_source)] + [
            torch.zeros_like(m) for m in self.bp.models
        ]
        self.ibufs = [torch.zeros_like(self.bp.models[0]) for _ in range(2)]
        self.bp.grads_out = self.gbufs
        self.bp.illum_out = self.ibufs

    def fwd_runner(self):
        for t in self.L_fwd:
            t.zero_()
        return SteppedBindingRunner(
            self.fwd_func, self.fp, self.L_fwd, acoustic_psi_pairs(3)
        )

    def zero_backward_state(self):
        for t in self.L_adj + self.recon + self.gbufs + self.ibufs:
            t.zero_()

    def bwd_runner(self):
        return SteppedBackwardRunner(
            self.bwd_func, self.bp, self.L_adj, self.recon,
            acoustic_adj_pairs(3),
        )

    def backward_snapshot(self):
        return {
            "grads": [t.clone() for t in self.gbufs],
            "illum": [t.clone() for t in self.ibufs],
        }


def build_reference():
    sources = np.array([[list(SRC)]], dtype=np.int32)
    receivers = np.array(
        [[[gx, REC_Y, REC_Z] for gx in REC_GX]], dtype=np.int32
    )
    prop = make_prop((NZ, NY, NX))
    cap = capture_both(prop)
    run_public_once(prop, ricker(NT, DT), sources, receivers, vp_global())
    return TileState3D(cap)


def build_tiles(residual_raw):
    vp = vp_global()
    wavelet = ricker(NT, DT)
    tiles = []
    for xi in range(2):
        topo = MeshTopology(py=1, px=2, shot_groups=1, world_size=2, rank=xi)
        x0 = xi * NXP
        vp_tile = vp[..., x0:x0 + NXP].copy()

        owns_src = x0 <= SRC[0] < x0 + NXP
        if owns_src:
            t_src = np.array([[[SRC[0] - x0, SRC[1], SRC[2]]]], dtype=np.int32)
            t_wav = wavelet
        else:
            t_src = np.array([[[1, 1, 1]]], dtype=np.int32)
            t_wav = np.zeros_like(wavelet)

        own = [gx for gx in REC_GX if x0 <= gx < x0 + NXP]
        idxs = [REC_GX.index(gx) for gx in own]
        t_rec = np.array(
            [[[gx - x0, REC_Y, REC_Z] for gx in own]], dtype=np.int32
        )

        prop = make_prop((NZ, NY, NXP), topo=topo)
        cap = capture_both(prop)
        run_public_once(prop, t_wav, t_src, t_rec, vp_tile)
        t = TileState3D(cap)
        t.owns_src = owns_src
        t.rec_idxs = idxs
        t.cut_face_mask = X_HI_BIT if xi == 0 else X_LO_BIT
        t.x_off = x0
        t.lo = prop.padding[0] + M       # cut-aware x interior offset
        t.hi = t.lo + NXP

        adj = torch.zeros_like(t.bp.adjoint_source)
        for j, gi in enumerate(idxs):
            adj[:, j] = residual_raw[:, gi]
        t.bp.adjoint_source = adj
        tiles.append(t)
    return tiles


@cuda_only
def test_dd_backward_two_tile_3d_bitexact():
    ref = build_reference()
    residual_raw = ref.fwd_record_raw.clone()
    ref.bp.adjoint_source = residual_raw

    # ---------------- DD forward ----------------
    tiles = build_tiles(residual_raw)
    g_model = ref.fp.models[0]
    with torch.no_grad():
        for t in tiles:
            t.fp.cut_face_mask = t.cut_face_mask
            start = PAD + t.x_off - t.lo
            w = t.fp.models[0].shape[-1]
            sl = g_model[..., :, :, start:start + w]
            t.fp.models[0].copy_(sl)
            t.bp.models[0].copy_(sl)
        r0, r1 = tiles[0].fwd_runner(), tiles[1].fwd_runner()
        lo0, hi0, lo1, hi1 = tiles[0].lo, tiles[0].hi, tiles[1].lo, tiles[1].hi
        for it in range(NT):
            r0.run_to(it + 1)
            r1.run_to(it + 1)
            u0, u1 = r0.u_now, r1.u_now
            u0[..., hi0:hi0 + M] = u1[..., lo1:lo1 + M]
            u1[..., lo1 - M:lo1] = u0[..., hi0 - M:hi0]
        for t in tiles:
            t.bp.boundary_gpu = list(t.fp.boundary_gpu)
            t.bp.u_last_two = t.fp.last_two

    # DD forward sanity vs reference replay. The replay (re)writes the
    # reference boundary ring and fills ref.record (record_out is bound
    # only after the public run). Public run and replay are bit-identical
    # now that internal allocation opts into the psi double-buffer (see
    # the "public-vs-replay quirk" section of _dd_cuda/REPORT.md, which is
    # scaffolding no longer checked out -- read it at commit c122dc17).
    rr = ref.fwd_runner()
    with torch.no_grad():
        rr.run_to(NT)
    rec_tiles = torch.zeros_like(ref.record)
    for t in tiles:
        for j, gi in enumerate(t.rec_idxs):
            rec_tiles[:, gi] = t.record[:, j]
    assert torch.equal(rec_tiles, ref.record), "3D DD forward record differs"

    # reference backward (monolithic) on the replayed ring + determinism
    ref.zero_backward_state()
    ref.bp.cut_face_mask = 0
    ref.bp.bw_it_begin, ref.bp.bw_it_end = -1, 0
    ref.bp.adjoint_wavefields = ref.L_adj
    ref.bp.forward_wavefields = ref.recon
    ref.bwd_func(ref.bp)
    snap_a = ref.backward_snapshot()
    ref.zero_backward_state()
    ref.bwd_func(ref.bp)
    snap_b = ref.backward_snapshot()
    for key in snap_a:
        for i, (a, b) in enumerate(zip(snap_a[key], snap_b[key])):
            assert torch.equal(a, b), f"determinism baseline: {key}[{i}]"

    # ---------------- DD backward ----------------
    for t in tiles:
        t.zero_backward_state()
        t.bp.cut_face_mask = t.cut_face_mask
    b0, b1 = tiles[0].bwd_runner(), tiles[1].bwd_runner()
    lo0, hi0, lo1, hi1 = tiles[0].lo, tiles[0].hi, tiles[1].lo, tiles[1].hi
    with torch.no_grad():
        for it in range(NT - 1, -1, -1):
            b0.run_segment(it + 1, it)
            b1.run_segment(it + 1, it)
            if it == 0:
                break
            l0, l1 = b0.lambda_now, b1.lambda_now
            l0[..., hi0:hi0 + M] = l1[..., lo1:lo1 + M]
            l1[..., lo1 - M:lo1] = l0[..., hi0 - M:hi0]
            f0, f1 = b0.recon_u_now, b1.recon_u_now
            f0[..., hi0:hi0 + M] = f1[..., lo1:lo1 + M]
            f1[..., lo1 - M:lo1] = f0[..., hi0 - M:hi0]
    assert b0.k_adj == NT and b0.k_f == NT - 1

    # ---------------- assemble + compare ----------------
    t0, t1 = tiles
    own0 = slice(t0.lo, t0.lo + NXP)
    own1 = slice(t1.lo, t1.lo + NXP)
    g_ref = ref.gbufs[1]
    assert torch.equal(t0.gbufs[1][..., own0], g_ref[..., PAD:PAD + NXP]), \
        "tile0 grad_vp differs (bitwise)"
    assert torch.equal(t1.gbufs[1][..., own1], g_ref[..., PAD + NXP:PAD + 2 * NXP]), \
        "tile1 grad_vp differs (bitwise)"
    for k, name in enumerate(["source_illum", "receiver_illum"]):
        assert torch.equal(t0.ibufs[k][..., own0],
                           ref.ibufs[k][..., PAD:PAD + NXP]), \
            f"tile0 {name} differs (bitwise)"
        assert torch.equal(t1.ibufs[k][..., own1],
                           ref.ibufs[k][..., PAD + NXP:PAD + 2 * NXP]), \
            f"tile1 {name} differs (bitwise)"
    assert t1.owns_src and not t0.owns_src
    assert torch.equal(t1.gbufs[0], ref.gbufs[0]), \
        "grad_wavelet differs (bitwise)"
