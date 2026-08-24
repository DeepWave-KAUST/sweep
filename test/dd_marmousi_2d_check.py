"""2-D DD on a REAL model: Marmousi, x-cut, bit-exact against one domain.

The other 2-D DD checks run on synthetic layered boxes with smooth contrasts.
This one uses Marmousi, which is what actually stresses a domain cut: sharp
lateral velocity jumps that sit ON the cut, a dipping section whose stencil
reads across it every step, and a free surface.

The lateral extent is odd (1701 at downsample 8), so px=2 cannot divide it and
``pad_to_mesh`` is exercised on the way — the same physical-leaf pattern the
solver docs recommend, with the gradient landing back on the physical grid.

2-D DD is an x-cut only: the trailing axes are ``(nz, nx)`` and padding nz
would move the free surface, so py must be 1.

Launch::

    torchrun --standalone --nproc-per-node=2 test/dd_marmousi_2d_check.py
    torchrun --standalone --nproc-per-node=4 test/dd_marmousi_2d_check.py --px 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sweep.datasets import load                           # noqa: E402
from sweep.equations import Acoustic                      # noqa: E402
from sweep.parallel import MeshTopology, pad_to_mesh      # noqa: E402
from sweep.parallel.dd_propagator import ModelParallel    # noqa: E402
from sweep.propagator.torch import PropTorch              # noqa: E402


def ricker(nt, dt, fm, delay):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a ** 2) * np.exp(-(a ** 2))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", type=int, default=2, help="tiles along x")
    ap.add_argument("--downsample", type=int, default=8)
    ap.add_argument("--nt", type=int, default=1500)
    ap.add_argument("--dt", type=float, default=1.0e-3)
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--abcn", type=int, default=20)
    ap.add_argument("--fm", type=float, default=5.0)
    ap.add_argument("--fs", type=int, default=1)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")
    assert args.px == world, f"2-D DD is an x-cut: px must equal world size {world}"

    d = load("marmousi", variant="2d-acoustic")
    ds = args.downsample
    vp = np.ascontiguousarray(d["vp"][::ds, ::ds]).astype(np.float32)
    dh = float(d["dh"][0]) * ds
    nz, nx = vp.shape

    mesh = MeshTopology(py=1, px=args.px, shot_groups=1,
                        world_size=world, rank=rank)
    padded_shape = tuple(pad_to_mesh(torch.as_tensor(vp), mesh).shape)
    dx = padded_shape[-1] - nx

    # Sanity on the discretisation: the comparison is bit-exact either way (both
    # sides run the same physics), but a dispersive setup makes a bad reference.
    vmin, vmax = float(vp.min()), float(vp.max())
    ppw = vmin / (2.5 * args.fm) / dh          # at ~2.5x the peak frequency
    cfl = vmax * args.dt / dh

    wav = torch.as_tensor(ricker(args.nt, args.dt, args.fm, 1.5 / args.fm),
                          device=dev)
    # A shot right over the dipping section, receivers spanning the full line so
    # every cut has traffic across it.
    src = np.array([[[nx // 2, 4]]], dtype=np.int64)
    rec = np.array([[[ix, 2] for ix in range(4, nx - 4, 4)]], dtype=np.int64)

    def build():
        eq = Acoustic(spatial_order=args.so, device=dev, backend="torch")
        return PropTorch(eq, backend="torch", impl="c", shape=padded_shape,
                         dh=dh, dt=args.dt, nt=args.nt, abcn=args.abcn,
                         source_type=["h1"], receiver_type=["h1"], dev=dev,
                         free_surface=bool(args.fs))

    if rank == 0:
        print(f"Marmousi 2-D /{ds}: physical ({nz},{nx}) -> padded "
              f"{padded_shape} (dx={dx})  dh={dh:.1f} m  px={args.px}")
        print(f"  vp {vmin:.0f}-{vmax:.0f} m/s   nt={args.nt} dt={args.dt}  "
              f"ppw~{ppw:.1f}  CFL={cfl:.2f}   fs={bool(args.fs)}")

    # ---- single domain, physical leaf + pad in the closure ----------------
    vp_ref = torch.tensor(vp, device=dev, requires_grad=True)
    r_ref = build()(wav, src, rec, models=[pad_to_mesh(vp_ref, mesh)])
    (r_ref.double() ** 2).sum().backward()

    # ---- DD, same pattern -------------------------------------------------
    ddp = ModelParallel(build(), mesh)
    vp_dd = torch.tensor(vp, device=dev, requires_grad=True)
    r_dd = ddp(wav, src, rec, models=[pad_to_mesh(vp_dd, mesh)])
    (r_dd.double() ** 2).sum().backward()
    dist.all_reduce(vp_dd.grad)

    shape_ok = (tuple(vp_dd.grad.shape) == (nz, nx)
                and tuple(vp_ref.grad.shape) == (nz, nx))
    r_ref_d = r_ref.detach()                  # still on the graph otherwise
    live = float(r_ref_d.abs().max()) > 0.0   # a zero record compares equal to anything
    gbit = bool(torch.equal(vp_dd.grad, vp_ref.grad))
    rel = float((vp_dd.grad - vp_ref.grad).norm() / (vp_ref.grad.norm() + 1e-30))

    R = r_ref.detach().cpu().numpy().squeeze()
    D = r_dd.detach().cpu().numpy().squeeze()
    own = getattr(ddp, "_own_rec_idx", None)
    own = (np.arange(rec.shape[1]) if own is None
           else np.asarray(own, dtype=np.int64).ravel())
    if R.shape[0] != rec.shape[1]:
        R = R.T
    if D.shape[0] != len(own):
        D = D.T
    rbit = bool(np.array_equal(D, R[own]))

    # The cut columns are where a halo bug shows up first; make sure the
    # gradient is actually populated there rather than quietly zero.
    cuts = [i * (padded_shape[-1] // args.px) for i in range(1, args.px)]
    cut_alive = all(float(vp_ref.grad[:, min(c, nx - 1)].abs().max()) > 0.0
                    for c in cuts) if cuts else True

    ok = shape_ok and live and gbit and rbit and cut_alive
    if rank == 0:
        print(f"  reference record carries signal : {live} "
              f"(max |rec| = {float(r_ref_d.abs().max()):.3e})")
        print(f"  grad on the physical grid       : {shape_ok}")
        print(f"  grad bit-exact vs single domain : {gbit}   (rel={rel:.2e})")
        print(f"  records bit-exact               : {rbit}")
        print(f"  gradient alive on the cut cols  : {cut_alive}  (x={cuts})")
        print(f"-> {'PASS' if ok else 'FAIL'}")

    fail = torch.tensor([0 if ok else 1], device=dev)
    dist.all_reduce(fail)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(int(fail.item() > 0))


if __name__ == "__main__":
    main()
