"""Multi-GPU vs single-GPU SEM DD gradient comparison (torchrun launcher).

    torchrun --nproc_per_node=P test/sem/dd_grad_compare.py \
        [--case 2d|3d|fs] [--memory boundary] [--store bf16|int8]
        [--out FILE.npz]

Every rank owns one x strip (NCCL interface exchanges); the vp gradient is
assembled with one all_reduce (ownership makes rank contributions disjoint).
Rank 0 also runs the FUSED single-domain reference TWICE (its run-to-run
atomics spread is the comparison floor) and saves everything to ``--out``
for plotting: single-GPU grad, multi-GPU grad, the fused floor, cut-column
positions, and the records. The model carries an anomaly STRADDLING a cut
so the interface columns are load-bearing, and the receiver line includes
the cut columns themselves.

``--case 3d`` does the same for ``AcousticSEM3D``, where a cut is a whole
YZ PLANE: the anomaly straddles it in x and is centred in y, and receivers
sit on two y rows (one row would not exercise the plane's second axis).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np
import torch
import torch.distributed as dist

import sweep.sem as S
from sweep.propagator import BoundarySaving
from sweep.sem.dd import SEMModelParallel, StripTopology
from sweep.sem.equations import (AcousticElasticSEM2D,
                                 AcousticPotentialSEM2D, AcousticSEM3D)
from sweep.sem.prop_c import (_SEMAcousticCpmlC,
                              _SEMFluidSolidC)

NO, DH, F0 = 4, 25.0, 8.0
# the models; with the absorbing layer outside them (3 elements in 2-D, 2 in
# 3-D, none on the free top) the meshes keep the sizes the cuts are set for
NZ, NX = 37, 73                    # mesh 12 x 24 elements
NT, DT = 600, 5e-4
NZ3, NY3, NX3 = 25, 9, 33          # mesh 8 x 6 x 12 elements
NT3, DT3 = 400, 1.0e-3
# fluid-solid: wide enough that every strip keeps an interior the coupled
# reverse march has to rebuild (band ~57% at px=2, not ~100%)
NZF, NXF, NWF = 37, 73, 8          # mesh 12 x 24 elements
NTF, DTF, F0F = 300, 1.2e-3, 6.0


def model():
    z = torch.linspace(0, 1, NZ)[:, None].expand(NZ, NX)
    x = torch.linspace(0, 1, NX)[None, :].expand(NZ, NX)
    vp = 2000.0 + 800.0 * z
    for cx, cz, amp in ((0.5, 0.45, 250.0),    # straddles the px=2/4 cut
                        (0.22, 0.6, -200.0), (0.78, 0.35, 180.0)):
        vp = vp + amp * torch.exp(-(((x - cx) / 0.06) ** 2
                                    + ((z - cz) / 0.1) ** 2))
    return vp.contiguous().cuda()


def model3d():
    """3-D model whose anomaly STRADDLES the px=2 cut plane in x."""
    z = torch.linspace(0, 1, NZ3)[:, None, None]
    y = torch.linspace(0, 1, NY3)[None, :, None]
    x = torch.linspace(0, 1, NX3)[None, None, :]
    vp = 2500.0 + 600.0 * z.expand(NZ3, NY3, NX3).clone()
    for cx, cy, cz, amp in ((0.5, 0.5, 0.45, 250.0),     # straddles the cut
                            (0.25, 0.4, 0.6, -180.0)):
        vp = vp + amp * torch.exp(-(((x - cx) / 0.07) ** 2
                                    + ((y - cy) / 0.2) ** 2
                                    + ((z - cz) / 0.12) ** 2))
    return vp.contiguous().cuda()


def geometry3d(nxg, nyg, cuts):
    def gid(ix, iy, iz):
        return (iz * nyg + iy) * nxg + ix
    src = gid(nxg // 2 - 5, nyg // 2, 2 * NO)
    cols = sorted(set(list(range(4, nxg - 4, 3)) + cuts))
    rec = torch.tensor([gid(cx, cy, NO)
                        for cy in (nyg // 3, 2 * nyg // 3)
                        for cx in cols]).cuda()
    return src, rec


def fused3d_runner(prop, memory=None, store=None):
    # memory / store are ``prop``'s own (constructor options); the arguments
    # only keep the three runners' signature
    def run(vp, wav, src, rec):
        return SEMModelParallel(prop, StripTopology(1))(wav, src, rec, [vp])
    return run


def fs_models(prop, grad=False):
    m = prop.mesh
    fl = (m.z_e.flatten(1).mean(1) < NWF * 25.0).reshape(-1, 1, 1).expand(
        m.n_elements, m.ngll, m.ngll)
    ms = [torch.where(fl, 1500.0, 2300.0).float().cuda(),
          torch.where(fl, 0.0, 1250.0).float().cuda(),
          torch.where(fl, 1000.0, 2100.0).float().cuda()]
    return [t.requires_grad_(True) for t in ms] if grad else ms


def fs_wavelet():
    t = np.arange(NTF) * DTF
    a = (np.pi * F0F * (t - 1.3 / F0F)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32) * 1e6,
                        device="cuda")


def fused_fs_runner(prop, memory=None, store=None):
    def run(models, wav, src, rec):
        eq, _ = prop._resolve_model_batch(models)
        pats = []
        for j in range(3):
            v = torch.zeros(eq.mesh.n_global, dtype=torch.float32,
                            device=eq.mesh.ibool.device)
            if j == 0:
                v[int(src)] = 1.0
            pats.append(v)
        parts, mats = SEMModelParallel(prop, StripTopology(1))._parts(
            eq, tuple(pats), rec)
        c = dict(parts[0].c)
        c["bs"] = memory == "boundary"
        sc, sx, sz = _SEMFluidSolidC.apply(*mats, wav, c)
        return torch.stack([sc, sx, sz], -1).transpose(0, 1)
    return run


def run_case_fs(prop, runner):
    """The coupled path carries three models; the reported gradient is vp
    (the fluid/solid P velocity), which is what the maps plot."""
    models = fs_models(prop, grad=True)
    wav = fs_wavelet()
    eq, _ = prop._resolve_model_batch([m.detach() for m in models])
    nxg = eq.mesh.nex * NO + 1
    px = dist.get_world_size() if dist.is_initialized() else 1
    cuts = [(eq.mesh.nex // px) * r * NO for r in range(1, px)]
    src = 2 * NO * nxg + nxg // 2
    cols = sorted(set(list(range(4, nxg - 4, 6)) + cuts))
    rec = (torch.tensor(cols) + (NWF + 1) * nxg).cuda()
    out = runner(models, wav, src, rec)
    (out ** 2).sum().mul(1e6).backward()
    return out.detach(), models[0].grad.clone(), cuts


def run_case3d(prop, runner):
    vp = model3d().requires_grad_(True)
    wav = wavelet(NT3, DT3)
    eq, _ = prop._resolve_model_batch([vp.detach()])
    m = eq.mesh
    nxg, nyg = m.nex * NO + 1, m.ney * NO + 1
    px = dist.get_world_size() if dist.is_initialized() else 1
    cuts = [(m.nex // px) * r * NO for r in range(1, px)]
    src, rec = geometry3d(nxg, nyg, cuts)
    out = runner(vp, wav, src, rec)
    (out ** 2).sum().mul(1e6).backward()       # same un-normalised loss
    return out.detach(), vp.grad.clone(), cuts


def wavelet(nt=NT, dt=DT):
    t = np.arange(nt) * dt
    a = (np.pi * F0 * (t - 1.2 / F0)) ** 2
    return torch.tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32),
                        device="cuda")


def geometry(nxg, cuts):
    src = 2 * NO * nxg + (nxg // 2 - 3)
    cols = sorted(set(list(range(4, nxg - 4, 3)) + cuts))
    rec = (torch.tensor(cols) + 1 * nxg).cuda()
    return src, rec


def run_case(prop, runner):
    vp = model().requires_grad_(True)
    wav = wavelet()
    eq, _ = prop._resolve_model_batch([vp.detach()])
    nxg = eq.mesh.nex * NO + 1
    px = dist.get_world_size() if dist.is_initialized() else 1
    cuts = [(eq.mesh.nex // px) * r * NO for r in range(1, px)]
    src, rec = geometry(nxg, cuts)
    out = runner(vp, wav, src, rec)
    # UN-normalised loss: under dist each rank's record holds only its own
    # columns, so a per-rank normaliser would differ from the fused global
    # one and scale every gradient (caught on the first real 2-GPU run:
    # grad rel 0.9985 while forward matched). A plain sum is additive
    # across ranks and identical to the fused reference.
    (out ** 2).sum().mul(1e6).backward()       # 1e6: keep fp32 well scaled
    return out.detach(), vp.grad.clone(), cuts


def fused_runner(prop, memory=None, store=None):
    def run(vp, wav, src, rec):
        eq, _ = prop._resolve_model_batch([vp])
        pattern = torch.zeros(eq.mesh.n_global, dtype=torch.float32,
                              device=eq.mesh.ibool.device)
        pattern[int(src)] = 1.0
        d1 = SEMModelParallel(prop, StripTopology(1))
        parts, mats = d1._parts(eq, pattern, rec)
        c = dict(parts[0].c)
        c["bs"] = memory == "boundary"
        c["bs_kind"] = {None: 0, "bf16": 1, "int8": 2}[store]
        return _SEMAcousticCpmlC.apply(mats[0], mats[1], mats[2], wav,
                                       c).transpose(0, 1)
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="2d", choices=("2d", "3d", "fs"))
    ap.add_argument("--memory", default=None, choices=(None, "boundary"))
    # ring storage precision; graded DD-vs-FUSED at the SAME store, which is
    # the DD question and stays meaningful even where int8 itself is poor
    ap.add_argument("--store", default=None, choices=(None, "bf16", "int8"))
    ap.add_argument("--out", default="dd_grad_compare.npz")
    args = ap.parse_args()
    # the gradient memory is the propagator's (a constructor option), the
    # ring precision BoundarySaving's; the fs case never took --store
    store = None if args.case == "fs" else args.store
    if store and args.memory != "boundary":
        ap.error("--store is the boundary-saving ring's precision; it needs "
                 "--memory boundary")
    memory = (BoundarySaving(storage_dtype=store or "fp32")
              if args.memory == "boundary" else None)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    d3 = args.case == "3d"
    if args.case == "fs":
        prop = S.PropSEM(AcousticElasticSEM2D, shape=(NZF, NXF), dh=25.0,
                         dt=DTF, nt=NTF, n_order=NO, free_surface=("top",),
                         abcn=3 * NO, memory=memory,
                         f_dominant=F0F, dev="cuda", impl="c",
                         receiver_type=["p", "vx", "vz"])
    elif d3:
        prop = S.PropSEM(AcousticSEM3D, shape=(NZ3, NY3, NX3), dh=DH,
                         dt=DT3, nt=NT3, n_order=NO, free_surface=("top",),
                         abcn=2 * NO, memory=memory,
                         f_dominant=F0, dev="cuda", impl="c")
    else:
        prop = S.PropSEM(AcousticPotentialSEM2D, shape=(NZ, NX), dh=DH,
                         dt=DT, nt=NT, n_order=NO, free_surface=("top",),
                         abcn=3 * NO, memory=memory,
                         f_dominant=F0, dev="cuda", impl="c")
    case_run = (run_case_fs if args.case == "fs"
                else (run_case3d if d3 else run_case))
    fused = (fused_fs_runner if args.case == "fs"
             else (fused3d_runner if d3 else fused_runner))

    ddp = SEMModelParallel(prop, StripTopology(world, rank=rank))
    if args.case == "fs":
        rec_dd, g_dd, cuts = case_run(
            prop, lambda ms, wav, src, rec: ddp(wav, src, rec, ms))
    else:
        rec_dd, g_dd, cuts = case_run(
            prop, lambda vp, wav, src, rec: ddp(wav, src, rec, [vp]))
    dist.all_reduce(g_dd)                      # disjoint ownership -> SUM
    full_dd = ddp.gather_record(rec_dd)

    if rank == 0:
        rec_1, g_1, _ = case_run(prop, fused(prop, args.memory, args.store))
        rec_1b, g_1b, _ = case_run(prop, fused(prop, args.memory,
                                               args.store))        # floor
        gd, g1 = g_dd.cpu().numpy(), g_1.cpu().numpy()
        floor = float(np.abs(g_1b.cpu().numpy() - g1).max() / np.abs(g1).max())
        rel = float(np.abs(gd - g1).max() / np.abs(g1).max())
        rrec = float((full_dd - rec_1.cpu()).abs().max()
                     / rec_1.abs().max())
        np.savez(args.out, g_multi=gd, g_single=g1,
                 g_single_rerun=g_1b.cpu().numpy(),
                 rec_multi=full_dd.numpy(), rec_single=rec_1.cpu().numpy(),
                 cuts=np.asarray(cuts), no=NO, dh=DH, world=world,
                 floor=floor, case=args.case,
                 memory=args.memory or "save-all",
                 store=args.store or "fp32")
        print(f"[dd-grad] case={args.case} mem={args.memory or 'save-all'} "
              f"store={args.store or 'fp32'} "
              f"world={world}  fwd rel={rrec:.3e}  "
              f"grad rel={rel:.3e}  fused floor={floor:.3e}", flush=True)
        print(f"[dd-grad] saved {args.out}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
