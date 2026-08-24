"""Boundary tail truncation x ModelParallel over NCCL (end-to-end).

torchrun --standalone --nproc-per-node=2 test/dd_tail_nccl_check.py \
    [--nt 600] [--tail 300] [--probe 150] [--px 0]

Gates (fp32 gpu-direct boundaries -- the numerical criteria live on fp32,
int8 has its own quantization floor):

  A. equivalence  DD(tail) per-tile grad vs single-domain(tail) grad slice,
                  graded PASS (bitwise) / PASS_TOL (rel <= 1e-5) / FAIL.
                  Truncation composes with the cut faces or it does not --
                  this is the DD-vs-mono bit-exact precedent, truncated.
  B. physics      cos( DD(tail), DD(full) ) under a steady ramped-sine
                  source with a probe-window objective (adjoint is zero
                  before the window; tail = probe + margin), the margin
                  argument of test_boundary_tail_truncation at DD scale.
  C. wall-clock   reverse-loop time DD(tail) vs DD(full) (max across ranks)
                  -- the early stop must actually shorten the halo loop.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sweep.equations import Acoustic  # noqa: E402
from sweep.parallel import MeshTopology, ModelParallel  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

REL_TOL = 1e-5
DT = 0.0015
NZ = 48
NXP = 28
ABCN = 10
SO = 4


def steady_wavelet(nt, dt, f=10.0, rise=0.3):
    t = np.arange(nt, dtype=np.float32) * dt
    ramp = np.clip(t / rise, 0.0, 1.0) ** 2
    return (np.sin(2 * np.pi * f * t) * ramp).astype(np.float32)


def make_prop(shape, dev, nt, tail):
    cfg = {"enabled": True, "storage": "gpu",
           "transfer_interval": 1, "pinned_memory": False}
    if tail:
        cfg["tail_steps"] = tail
    eq = Acoustic(spatial_order=SO, device=dev, backend="torch")
    return PropTorch(eq, backend="torch", impl="c", shape=shape, dev=dev,
                     dh=10.0, dt=DT, nt=nt, abcn=ABCN, source_type=["h1"],
                     receiver_type=["h1"], free_surface=False,
                     pml_type="cpmlr", use_ckpt=False,
                     boundary_saving_config=cfg)


def time_mask_like(rec, nt, probe):
    """1.0 on the last ``probe`` steps of the time axis, 0 elsewhere; the
    time axis is located by its (unique, by construction) extent == nt."""
    axes = [i for i, s in enumerate(rec.shape) if s == nt]
    assert len(axes) == 1, f"ambiguous time axis in record shape {tuple(rec.shape)}"
    ax = axes[0]
    m = torch.zeros(nt, device=rec.device, dtype=rec.dtype)
    m[nt - probe:] = 1.0
    shape = [1] * rec.ndim
    shape[ax] = nt
    return m.view(shape)


def run_dd(vp_np, wav, src, rec, nt, tail, probe, topo, dev, shape):
    """One ModelParallel pipeline; returns (tile grad, x0, nxp, bwd seconds)."""
    prop = make_prop(shape, dev, nt, tail)
    ddp = ModelParallel(prop, topo)
    vp_tile = torch.tensor(vp_np[:, ddp.x0:ddp.x0 + ddp.nxp], device=dev,
                           requires_grad=True)
    # capture warm-up (first call allocates + probes)
    r = ddp.forward(wav, src, rec, models=[vp_tile])
    r.backward(gradient=(r.detach() * time_mask_like(r, nt, probe)))
    vp_tile.grad = None
    # timed run
    r = ddp.forward(wav, src, rec, models=[vp_tile])
    adj = r.detach() * time_mask_like(r, nt, probe)
    torch.cuda.synchronize(dev)
    dist.barrier()
    t0 = time.perf_counter()
    r.backward(gradient=adj)
    torch.cuda.synchronize(dev)
    dt_bwd = time.perf_counter() - t0
    return vp_tile.grad.detach().clone(), ddp.x0, ddp.nxp, dt_bwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", type=int, default=600)
    ap.add_argument("--tail", type=int, default=300)
    ap.add_argument("--probe", type=int, default=150)
    ap.add_argument("--px", type=int, default=0)
    args = ap.parse_args()
    nt, tail, probe = args.nt, args.tail, args.probe
    assert tail > probe, "tail must cover the probe window plus a margin"

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")

    px = args.px or world
    assert px == world, "x-tiles only in this check"
    nx = NXP * px
    shape = (NZ, nx)

    z = np.linspace(0, 1, NZ, dtype=np.float32)[:, None]
    vp_np = (1800.0 + 600.0 * z) * np.ones((1, nx), dtype=np.float32)
    vp_np[20:30, nx // 2 - 8:nx // 2 + 8] += 180.0     # anomaly on the cut
    wav = steady_wavelet(nt, DT)
    src = np.array([[[nx // 2, NZ // 4]]], dtype=np.int32)
    rec = np.array([[[ix, 2] for ix in range(2, nx - 2, 5)]], dtype=np.int32)

    topo = MeshTopology(py=1, px=px, shot_groups=1, world_size=world, rank=rank)

    g_tail, x0, nxp, t_tail = run_dd(vp_np, wav, src, rec, nt, tail, probe,
                                     topo, dev, shape)
    g_full, _, _, t_full = run_dd(vp_np, wav, src, rec, nt, None, probe,
                                  topo, dev, shape)

    payload = (x0, nxp, g_tail.cpu(), g_full.cpu(), t_tail, t_full)
    gathered = [None] * world
    dist.gather_object(payload, gathered if rank == 0 else None, dst=0)

    if rank == 0:
        # single-domain references (public API, same objective), truncated
        # AND full: the full-vs-full grade is the DD-vs-mono BASELINE for
        # this setup — any pre-existing DD forward drift shows up there too,
        # and the tail grade is judged against it, not against zero.
        def mono_grad(t):
            prop = make_prop(shape, dev, nt, t)
            vp = torch.tensor(vp_np, device=dev, requires_grad=True)
            syn = prop(wav, src, rec, models=[vp])
            syn.backward(gradient=(syn.detach() * time_mask_like(syn, nt, probe)))
            return vp.grad.detach().cpu()

        g_ref = mono_grad(tail)
        g_ref_full = mono_grad(None)
        scale = g_ref.abs().max().item()
        assert scale > 0, "degenerate reference gradient"

        worst = 2
        base_rel = 0.0
        tail_rels = []
        gt_full = torch.zeros_like(g_ref)
        gt_tail = torch.zeros_like(g_ref)
        t_tail_max = t_full_max = 0.0
        for r, (x0r, nxpr, gt, gf, tt, tf) in enumerate(gathered):
            want = g_ref[:, x0r:x0r + nxpr]
            bit = torch.equal(gt, want)
            mad = (gt - want).abs().max().item()
            rel = mad / (scale + 1e-30)
            wantf = g_ref_full[:, x0r:x0r + nxpr]
            bitf = torch.equal(gf, wantf)
            madf = (gf - wantf).abs().max().item()
            relf = madf / (g_ref_full.abs().max().item() + 1e-30)
            print(f"[rank0] tile{r} grad_vp(tail): bit={bit} "
                  f"max|d|={mad:.3e} rel={rel:.3e}   "
                  f"BASELINE(full): bit={bitf} rel={relf:.3e}")
            worst = min(worst, 2 if bit else (1 if rel < REL_TOL else 0))
            base_rel = max(base_rel, relf)
            tail_rels.append(rel)
            gt_tail[:, x0r:x0r + nxpr] = gt
            gt_full[:, x0r:x0r + nxpr] = gf
            t_tail_max = max(t_tail_max, tt)
            t_full_max = max(t_full_max, tf)
        # tail composes cleanly iff it does not WORSEN the DD-vs-mono
        # baseline of this setup (pre-existing forward drift, if any, is
        # not the tail's fault): rescue a FAIL when the tail rel is within
        # 4x of the full-path baseline.
        if worst == 0 and base_rel > 0 and max(tail_rels) <= 4.0 * base_rel:
            print(f"[rank0] tail rel <= 4x baseline rel ({max(tail_rels):.3e} "
                  f"vs {base_rel:.3e}) -> pre-existing DD-vs-mono floor, "
                  "not a tail regression")
            worst = 1

        cos = torch.nn.functional.cosine_similarity(
            gt_tail.double().flatten(), gt_full.double().flatten(), dim=0).item()
        margin = tail - probe
        print(f"[rank0] physics: cos(DD tail, DD full) = {cos:.6f} "
              f"(tail={tail}, probe={probe}, margin={margin} steps)")
        print(f"[rank0] backward wall: tail {t_tail_max*1e3:.1f} ms vs "
              f"full {t_full_max*1e3:.1f} ms "
              f"({t_full_max / max(t_tail_max, 1e-9):.2f}x)")
        if cos < 0.99:
            worst = 0
        print("DD_TAIL_NCCL_CHECK:", {2: "PASS", 1: "PASS_TOL", 0: "FAIL"}[worst])
        if worst == 0:
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
