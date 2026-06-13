"""Local single-process y-cut two-tile backward reproducer (elastic3d).

Mirrors test_dd_elastic_backward_two_tile.py 3D path but splits along Y
(py=2) with the half-step protocol exchanged in Y, to localize the elastic
y-cut gradient bug WITHOUT NCCL.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))
from sweep.equations import Elastic3D  # noqa: E402
from sweep.parallel import MeshTopology  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import (  # noqa: E402
    SteppedBackwardRunner, SteppedBindingRunner,
)

DEV = torch.device("cuda")
NZ, NY, NX = 24, 20, 32
NT, DT, SO, ABCN = 60, 0.0015, 4, 10
M = SO // 2
PAD = ABCN + M
NYP = NY // 2
WSCALE = 1.0e6
NV, NPHYS, NWF, NRECON = 3, 9, 36, 12
SRC = (NX // 2, NY // 2, NZ // 4)
REC_GY = list(range(2, NY - 2, 3))
REC_X, REC_Z = NX // 2, 2
Y_LO_BIT, Y_HI_BIT = 16, 32


def ricker(nt, dt, fm=10.0, delay=0.06, scale=1.0):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return (scale * (1.0 - 2.0 * a**2) * np.exp(-(a**2))).astype(np.float32)


def models_global():
    g = np.linspace(0, 1, NZ * NY * NX, dtype=np.float32).reshape(NZ, NY, NX)
    vp = 2200.0 + 400.0 * g; vs = 1200.0 + 200.0 * g; rho = 2000.0 + 100.0 * g
    vp[8:14, 6:14, 10:22] += 100.0; vs[8:14, 6:14, 10:22] += 50.0
    rho[8:14, 6:14, 10:22] += 25.0
    return [vp, vs, rho]


def make_prop(shape, topo=None):
    eq = Elastic3D(spatial_order=SO, device=DEV, backend="torch")
    kw = dict(backend="torch", impl="c", shape=shape, dev=DEV, dh=10.0, dt=DT,
              source_type=["sxx", "syy", "szz"], receiver_type=["vx", "vy", "vz"],
              abcn=ABCN, free_surface=False, pml_type="cpmls", nt=NT, B=1,
              use_ckpt=False, boundary_saving_config={"enabled": True, "storage": "gpu",
                                                      "transfer_interval": 1, "pinned_memory": False})
    if topo is not None:
        kw["model_parallel"] = topo
    return PropTorch(eq, **kw)


def capture_both(prop):
    cap = {}; impl = prop._backend_impl
    fo, bo = impl.forward_func, impl.backward_bs_func
    impl.forward_func = lambda p: (cap.__setitem__("fp", p) or cap.__setitem__("fr", fo(p)) or cap["fr"])
    impl.backward_bs_func = lambda p: (cap.__setitem__("bp", p) or bo(p))
    cap["fo"], cap["bo"] = fo, bo
    return cap


def run_public(prop, wav, src, rec, mods):
    m = [torch.tensor(a, device=DEV, requires_grad=True) for a in mods]
    syn = prop(wav, src, rec, models=m)
    (syn[0] if isinstance(syn, (tuple, list)) else syn).sum().backward()


class Tile:
    def __init__(self, cap):
        self.fp, self.bp = cap["fp"], cap["bp"]
        self.fo, self.bo = cap["fo"], cap["bo"]
        self.fr = cap["fr"][2]
        self.L_fwd = list(self.fp.wavefields) or [torch.zeros_like(self.fp.models[0]) for _ in range(NWF)]
        self.record = torch.zeros_like(self.fr); self.fp.record_out = self.record
        self.L_adj = list(self.bp.adjoint_wavefields)
        self.recon = [torch.zeros_like(self.bp.models[0]) for _ in range(NRECON)]
        self.gbufs = [torch.zeros_like(m) for m in self.bp.models]
        self.bp.grads_out = self.gbufs; self.bp.illum_out = []

    def fwd_runner(self):
        for t in self.L_fwd:
            t.zero_()
        return SteppedBindingRunner(self.fo, self.fp, self.L_fwd, psi_pairs=(), u_blocks=())

    def zero_bwd(self):
        for t in self.L_adj + self.recon + self.gbufs:
            t.zero_()

    def bwd_runner(self):
        return SteppedBackwardRunner(self.bo, self.bp, self.L_adj, self.recon,
                                     adj_pairs=(), adj_u_blocks=(), recon_u_blocks=())


def main():
    wav = ricker(NT, DT, scale=WSCALE)
    src = np.array([[list(SRC)]], dtype=np.int32)
    rec = np.array([[[REC_X, gy, REC_Z] for gy in REC_GY]], dtype=np.int32)
    mods = models_global()

    ref_prop = make_prop((NZ, NY, NX))
    rc = capture_both(ref_prop)
    run_public(ref_prop, wav, src, rec, mods)
    ref = Tile(rc)
    residual = ref.fr.clone()
    ref.bp.adjoint_source = residual

    tiles = []
    for yi in range(2):
        topo = MeshTopology(py=2, px=1, shot_groups=1, world_size=2, rank=yi)
        y0 = yi * NYP
        tmods = [a[:, y0:y0 + NYP, :].copy() for a in mods]
        owns = y0 <= SRC[1] < y0 + NYP
        t_src = (np.array([[[SRC[0], SRC[1] - y0, SRC[2]]]], dtype=np.int32) if owns
                 else np.array([[[1, 1, 1]]], dtype=np.int32))
        t_wav = wav if owns else np.zeros_like(wav)
        own = [gy for gy in REC_GY if y0 <= gy < y0 + NYP]
        idxs = [REC_GY.index(gy) for gy in own]
        t_rec = np.array([[[REC_X, gy - y0, REC_Z] for gy in own]], dtype=np.int32)
        prop = make_prop((NZ, NYP, NX), topo=topo)
        cap = capture_both(prop)
        run_public(prop, t_wav, t_src, t_rec, tmods)
        t = Tile(cap)
        t.rec_idxs = idxs
        t.cut_face_mask = Y_HI_BIT if yi == 0 else Y_LO_BIT
        t.y_off = y0
        t.fp.cut_face_mask = t.cut_face_mask           # elastic3d fwd is cut-aware
        adj = torch.zeros_like(t.bp.adjoint_source)
        for j, gi in enumerate(idxs):
            adj[:, :, j] = residual[:, :, gi]
        t.bp.adjoint_source = adj
        tiles.append(t)

    gms = [ref.fp.models[i] for i in range(3)]
    lo, hi = PAD, PAD + NYP

    def swap_y(L0, L1, slots):
        for f in slots:
            a, b = L0[f], L1[f]
            a[..., hi:hi + M, :] = b[..., lo:lo + M, :]
            b[..., lo - M:lo, :] = a[..., hi - M:hi, :]

    with torch.no_grad():
        for t in tiles:
            for i in range(3):
                sl = gms[i][..., t.y_off:t.y_off + NYP + 2 * PAD, :]
                t.fp.models[i].copy_(sl); t.bp.models[i].copy_(sl)
        r0, r1 = tiles[0].fwd_runner(), tiles[1].fwd_runner()
        for it in range(NT):
            r0.run_phase(it + 1, 1); r1.run_phase(it + 1, 1)
            swap_y(r0.L, r1.L, range(NV))
            r0.run_phase(it + 1, 2); r1.run_phase(it + 1, 2)
            swap_y(r0.L, r1.L, range(NV, NPHYS))
        for t in tiles:
            t.bp.boundary_gpu = list(t.fp.boundary_gpu); t.bp.u_last_two = t.fp.last_two

    rr = ref.fwd_runner()
    with torch.no_grad():
        rr.run_to(NT)
    rec_t = torch.zeros_like(ref.record)
    for t in tiles:
        for j, gi in enumerate(t.rec_idxs):
            rec_t[:, :, gi] = t.record[:, :, j]
    print("forward record bitexact:", torch.equal(rec_t, ref.record),
          "max|d|", (rec_t - ref.record).abs().max().item())

    ref.zero_bwd(); ref.bp.cut_face_mask = 0
    ref.bp.bw_it_begin, ref.bp.bw_it_end = -1, 0
    ref.bp.adjoint_wavefields = ref.L_adj; ref.bp.forward_wavefields = ref.recon
    ref.bo(ref.bp)

    for t in tiles:
        t.zero_bwd(); t.bp.cut_face_mask = t.cut_face_mask
    b0, b1 = tiles[0].bwd_runner(), tiles[1].bwd_runner()

    def swap_pairs(L0, L1, slots):
        for f in slots:
            a, b = L0[f], L1[f]
            a[..., hi:hi + M, :] = b[..., lo:lo + M, :]
            b[..., lo - M:lo, :] = a[..., hi - M:hi, :]

    with torch.no_grad():
        for it in range(NT - 1, 0, -1):
            b0.run_phase(it + 1, it, 1); b1.run_phase(it + 1, it, 1)
            swap_pairs(b0.L_adj, b1.L_adj, range(NV))
            swap_pairs(b0.L_recon, b1.L_recon, range(NV, NPHYS))
            b0.run_phase(it + 1, it, 2); b1.run_phase(it + 1, it, 2)
            swap_pairs(b0.L_adj, b1.L_adj, range(NV, NPHYS))
            swap_pairs(b0.L_recon, b1.L_recon, range(NV))

    names = ["grad_vp", "grad_vs", "grad_rho"]
    for yi, t in enumerate(tiles):
        for k in range(3):
            got = t.gbufs[k][..., lo:hi, :]
            want = ref.gbufs[k][..., PAD + t.y_off: PAD + t.y_off + NYP, :]
            d = (got - want).abs()
            rel = d.max().item() / (want.abs().max().item() + 1e-30)
            tag = "OK " if rel < 1e-5 else "BAD"
            print(f"[{tag}] tile{yi} {names[k]}: bitexact={torch.equal(got, want)} "
                  f"max|d|={d.max().item():.3e} rel={rel:.3e}")
            if rel >= 1e-5:
                per_y = d.sum(dim=(0, 1, 2, 4)).flatten()
                print("      err per local-y:", " ".join(f"{v:.2e}" for v in per_y.tolist()))


if __name__ == "__main__":
    main()
