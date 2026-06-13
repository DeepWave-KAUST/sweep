"""Local single-process y-cut two-tile backward reproducer (acoustic3d).

Mirrors test_dd_backward_two_tile_3d.py but splits along Y (py=2, px=1) so we
can reproduce + localize the y-cut gradient bug WITHOUT NCCL (manual y-halo
swap on one GPU). Prints the gradient rel error and the argmax location to
see whether the error concentrates at the y-cut face.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))
from sweep.equations import Acoustic3D  # noqa: E402
from sweep.parallel import MeshTopology  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import (  # noqa: E402
    SteppedBackwardRunner, SteppedBindingRunner,
    acoustic_adj_pairs, acoustic_psi_pairs,
)

DEV = torch.device("cuda")
NZ, NY, NX = 24, 20, 32
NT, DT, SO, ABCN = 60, 0.0015, 4, 10
M = SO // 2
PAD = ABCN + M
NYP = NY // 2
SRC = (NX // 2, NY // 2, NZ // 4)          # on the y-cut -> owned by tile1
REC_GY = [11, 14, 17]                       # ALL in tile1 -> tile0 has a dummy
REC_X, REC_Z = NX // 2, 2
import os
FIX = os.environ.get("FIX", "0") == "1"     # zero a no-receiver tile's record
Y_LO_BIT, Y_HI_BIT = 16, 32


def ricker(nt, dt, fm=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a**2) * np.exp(-(a**2))).astype(np.float32)


def vp_global():
    vp = 1800.0 + 600.0 * np.linspace(0, 1, NZ, dtype=np.float32)[:, None, None]
    vp = np.broadcast_to(vp, (NZ, NY, NX)).copy()
    vp[8:14, 6:14, 10:22] += 180.0
    return vp


def make_prop(shape, topo=None):
    eq = Acoustic3D(spatial_order=SO, device=DEV, backend="torch")
    kw = dict(backend="torch", impl="c", shape=shape, dev=DEV, dh=10.0, dt=DT,
              source_type=["h1"], receiver_type=["h1"], abcn=ABCN,
              free_surface=False, pml_type="cpmlr", nt=NT, B=1, use_ckpt=False,
              boundary_saving_config={"enabled": True, "storage": "gpu",
                                      "transfer_interval": 1, "pinned_memory": False})
    if topo is not None:
        kw["model_parallel"] = topo
    return PropTorch(eq, **kw)


def capture_both(prop):
    cap = {}
    impl = prop._backend_impl
    fo, bo = impl.forward_func, impl.backward_bs_func
    impl.forward_func = lambda p: (cap.__setitem__("fp", p) or cap.__setitem__("fr", fo(p)) or cap["fr"])
    impl.backward_bs_func = lambda p: (cap.__setitem__("bp", p) or bo(p))
    cap["fo"], cap["bo"] = fo, bo
    return cap


def run_public(prop, wav, src, rec, vp):
    m = [torch.tensor(vp, device=DEV, requires_grad=True)]
    syn = prop(wav, src, rec, models=m)
    (syn[0] if isinstance(syn, (tuple, list)) else syn).sum().backward()


class Tile:
    def __init__(self, cap):
        self.fp, self.bp = cap["fp"], cap["bp"]
        self.fo, self.bo = cap["fo"], cap["bo"]
        self.fr = cap["fr"][2]
        L = list(self.fp.wavefields) or [torch.zeros_like(self.fp.models[0]) for _ in range(12)]
        self.L_fwd = L
        self.record = torch.zeros_like(self.fr); self.fp.record_out = self.record
        self.L_adj = list(self.bp.adjoint_wavefields)
        self.recon = [torch.zeros_like(self.bp.models[0]) for _ in range(3)]
        self.gbufs = [torch.zeros_like(self.bp.forward_source)] + [torch.zeros_like(m) for m in self.bp.models]
        self.ibufs = [torch.zeros_like(self.bp.models[0]) for _ in range(2)]
        self.bp.grads_out = self.gbufs; self.bp.illum_out = self.ibufs

    def fwd_runner(self):
        for t in self.L_fwd:
            t.zero_()
        return SteppedBindingRunner(self.fo, self.fp, self.L_fwd, acoustic_psi_pairs(3))

    def zero_bwd(self):
        for t in self.L_adj + self.recon + self.gbufs + self.ibufs:
            t.zero_()

    def bwd_runner(self):
        return SteppedBackwardRunner(self.bo, self.bp, self.L_adj, self.recon, acoustic_adj_pairs(3))


def main():
    wav = ricker(NT, DT)
    src = np.array([[list(SRC)]], dtype=np.int32)
    rec = np.array([[[REC_X, gy, REC_Z] for gy in REC_GY]], dtype=np.int32)

    ref_prop = make_prop((NZ, NY, NX))
    rc = capture_both(ref_prop)
    run_public(ref_prop, wav, src, rec, vp_global())
    ref = Tile(rc)
    residual = ref.fr.clone()
    ref.bp.adjoint_source = residual

    # tiles split in Y
    vp = vp_global()
    tiles = []
    for yi in range(2):
        topo = MeshTopology(py=2, px=1, shot_groups=1, world_size=2, rank=yi)
        y0 = yi * NYP
        vp_tile = vp[:, y0:y0 + NYP, :].copy()
        owns = y0 <= SRC[1] < y0 + NYP
        t_src = (np.array([[[SRC[0], SRC[1] - y0, SRC[2]]]], dtype=np.int32) if owns
                 else np.array([[[1, 1, 1]]], dtype=np.int32))
        t_wav = wav if owns else np.zeros_like(wav)
        own = [gy for gy in REC_GY if y0 <= gy < y0 + NYP]
        idxs = [REC_GY.index(gy) for gy in own]
        if own:
            t_rec = np.array([[[REC_X, gy - y0, REC_Z] for gy in own]], dtype=np.int32)
        else:
            t_rec = np.array([[[1, 1, 1]]], dtype=np.int32)   # dummy (like DDPropagator)
        prop = make_prop((NZ, NYP, NX), topo=topo)
        cap = capture_both(prop)
        run_public(prop, t_wav, t_src, t_rec, vp_tile)
        t = Tile(cap)
        t.rec_idxs = idxs
        t.no_rec = (len(own) == 0)
        t.cut_face_mask = Y_HI_BIT if yi == 0 else Y_LO_BIT
        t.y_off = y0
        tiles.append(t)   # adjoint set AFTER forward (mimics DDPropagator)

    gm = ref.fp.models[0]
    lo, hi = PAD, PAD + NYP
    with torch.no_grad():
        for t in tiles:
            sl = gm[..., t.y_off:t.y_off + NYP + 2 * PAD, :]   # y dim = -2
            t.fp.models[0].copy_(sl); t.bp.models[0].copy_(sl)
        r0, r1 = tiles[0].fwd_runner(), tiles[1].fwd_runner()
        for it in range(NT):
            r0.run_to(it + 1); r1.run_to(it + 1)
            u0, u1 = r0.u_now, r1.u_now
            u0[..., hi:hi + M, :] = u1[..., lo:lo + M, :]
            u1[..., lo - M:lo, :] = u0[..., hi - M:hi, :]
        for t in tiles:
            t.bp.boundary_gpu = list(t.fp.boundary_gpu); t.bp.u_last_two = t.fp.last_two

    rr = ref.fwd_runner()
    with torch.no_grad():
        rr.run_to(NT)
    rec_t = torch.zeros_like(ref.record)
    for t in tiles:
        for j, gi in enumerate(t.rec_idxs):
            rec_t[:, gi] = t.record[:, j]
    print("forward record bitexact:", torch.equal(rec_t, ref.record),
          "max|d|", (rec_t - ref.record).abs().max().item())

    # mimic DDPropagator.gradient: adjoint = the tile's OWN forward record
    # (a no-receiver tile carries a dummy; FIX zeroes it so it injects nothing)
    for t in tiles:
        if FIX and t.no_rec:
            t.record.zero_()
        t.bp.adjoint_source = t.record
    print(f"FIX={FIX}")

    ref.zero_bwd(); ref.bp.cut_face_mask = 0
    ref.bp.bw_it_begin, ref.bp.bw_it_end = -1, 0
    ref.bp.adjoint_wavefields = ref.L_adj; ref.bp.forward_wavefields = ref.recon
    ref.bo(ref.bp)

    for t in tiles:
        t.zero_bwd(); t.bp.cut_face_mask = t.cut_face_mask
    b0, b1 = tiles[0].bwd_runner(), tiles[1].bwd_runner()
    with torch.no_grad():
        for it in range(NT - 1, -1, -1):
            b0.run_segment(it + 1, it); b1.run_segment(it + 1, it)
            if it == 0:
                break
            l0, l1 = b0.lambda_now, b1.lambda_now
            l0[..., hi:hi + M, :] = l1[..., lo:lo + M, :]
            l1[..., lo - M:lo, :] = l0[..., hi - M:hi, :]
            f0, f1 = b0.recon_u_now, b1.recon_u_now
            f0[..., hi:hi + M, :] = f1[..., lo:lo + M, :]
            f1[..., lo - M:lo, :] = f0[..., hi - M:hi, :]

    g_ref = ref.gbufs[1]
    for yi, t in enumerate(tiles):
        got = t.gbufs[1][..., lo:hi, :]                     # tile interior (y)
        want = g_ref[..., PAD + t.y_off: PAD + t.y_off + NYP, :]
        d = (got - want).abs()
        rel = d.max().item() / (want.abs().max().item() + 1e-30)
        print(f"tile{yi} grad_vp: bitexact={torch.equal(got, want)} "
              f"max|d|={d.max().item():.3e} rel={rel:.3e}")
        if d.max() > 0:
            am = torch.unravel_index(d.argmax(), d.shape)
            iy = am[-2].item()  # local y (interior coords 0..NYP-1)
            print(f"  argmax at local (z,y,x)=({am[-3].item()},{iy},{am[-1].item()}) "
                  f"NYP={NYP}; cut side = {'HI(y=NYP-1)' if t.cut_face_mask & Y_HI_BIT else 'LO(y=0)'}; "
                  f"dist_to_cut={NYP - 1 - iy if t.cut_face_mask & Y_HI_BIT else iy}")
            # error as a function of y-distance from the cut (sum over z,x)
            per_y = d.sum(dim=(-3, -1)).flatten()
            print("  err sum per local-y:", " ".join(f"{v:.2e}" for v in per_y.tolist()))


if __name__ == "__main__":
    main()
