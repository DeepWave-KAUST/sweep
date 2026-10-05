"""AcousticLSRTM (2-D) DD parity vs single domain -- the 2-D twin of
dd_corner_check_lsrtm3d.py.

LSRTM carries TWO coupled fields (background + scattered) in one wavefield list,
so domain decomposition rotates and halo-exchanges both: the forward ships both
``u_now``, the boundary-saving backward both adjoints (lambda_sc and the RWI
background adjoint lambda_bg) and both reconstructions.  Compares the record and
BOTH model gradients -- reflectivity ``mp`` and the RWI tomographic ``vp`` --
against one card, bit for bit.

2-D DD is an x-cut (px == world size), and the model is padded to the mesh on
both sides (``pad_to_mesh``), exactly like dd_marmousi_2d_check.py.  A dipping
reflectivity sheet crosses every cut; every comparison first asserts its signal
is non-zero, and the gradients must be alive ON the cut columns, where a halo or
reconstruction bug shows first.

Launch: torchrun --standalone --nproc-per-node=N test/dd_corner_check_lsrtm2d.py --px N
        (--px 1 runs the full DD machinery on one card)
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np, torch, torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]; SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))
from sweep.equations import AcousticLSRTM                    # noqa: E402
from sweep.parallel import MeshTopology, pad_to_mesh         # noqa: E402
from sweep.parallel.dd_propagator import ModelParallel       # noqa: E402
from sweep.propagator import BoundarySaving, PropTorch        # noqa: E402
DT, DH = 0.0015, 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", type=int, default=1)
    ap.add_argument("--nt", type=int, default=700); ap.add_argument("--abcn", type=int, default=20)
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--src", type=str, default="mid",
                    help="'mid' = nx//2 (on the 1x2 cut line); 'off' = inside the first tile")
    ap.add_argument("--vp-grad", type=int, default=1,
                    help="1: vp requires grad (RWI vp gradient); 0: mp only, the classic LSRTM layout")
    args = ap.parse_args()
    vpg = bool(args.vp_grad)
    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count()); torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")
    assert args.px == world, f"2-D DD is an x-cut: px must equal world size {world}"

    nz, nx, nt = 70, 121, args.nt          # nx deliberately not a multiple of px
    zc = np.arange(nz, dtype=np.float32)[:, None]
    vp = np.broadcast_to(2000.0 + 8.0 * zc, (nz, nx)).astype(np.float32).copy(); vp[45:] += 400.0
    mp = np.zeros((nz, nx), np.float32)
    for ix in range(nx):                   # dipping sheet: crosses every cut
        mp[30 + (ix * 12) // nx, ix] = 0.05
    mp += 0.01 * np.sin(np.arange(nx, dtype=np.float32) / 5.0)[None, :] * (zc > 15)
    # mp reaches the side and bottom edges: the padding carries it into the absorbing
    # band, where the background adjoint's coupling runs through its pml_field

    mesh = MeshTopology(py=1, px=args.px, shot_groups=1, world_size=world, rank=rank)
    padded_shape = tuple(pad_to_mesh(torch.as_tensor(vp), mesh).shape)
    sx = nx // 2 if args.src == "mid" else nx // 6
    src = np.array([[[sx, 4]]], dtype=np.int64)
    rec = np.array([[[ix, 2] for ix in range(3, nx - 3, 3)]], dtype=np.int64)
    t = np.arange(nt, dtype=np.float32) * DT - 0.08; a = (np.pi * 15.0 * t) ** 2
    wav = torch.as_tensor(((1 - 2 * a) * np.exp(-a)).astype(np.float32), device=dev)

    def build():
        eq = AcousticLSRTM(spatial_order=args.so, device=dev, backend="torch")
        return PropTorch(eq, backend="torch", impl="c", shape=padded_shape, dh=DH, dt=DT, nt=nt,
                         abcn=args.abcn, source_type=["h1"], receiver_type=["sh1"], dev=dev,
                         free_surface=False, pml_type="cpmlr", memory=BoundarySaving())

    vp_ref = torch.tensor(vp, device=dev, requires_grad=vpg)
    mp_ref = torch.tensor(mp, device=dev, requires_grad=True)
    r_ref = build()(wav, src, rec, models=[pad_to_mesh(vp_ref, mesh), pad_to_mesh(mp_ref, mesh)])
    (r_ref.double() ** 2).sum().backward()

    ddp = ModelParallel(build(), mesh)
    vp_dd = torch.tensor(vp, device=dev, requires_grad=vpg)
    mp_dd = torch.tensor(mp, device=dev, requires_grad=True)
    r_dd = ddp(wav, src, rec, models=[pad_to_mesh(vp_dd, mesh), pad_to_mesh(mp_dd, mesh)])
    (r_dd.double() ** 2).sum().backward()
    dist.all_reduce(mp_dd.grad)
    if vpg:
        dist.all_reduce(vp_dd.grad)

    R = r_ref.detach().cpu().numpy().squeeze(); D = r_dd.detach().cpu().numpy().squeeze()
    own = getattr(ddp, "_own_rec_idx", None)
    own = (np.arange(rec.shape[1]) if own is None else np.asarray(own, dtype=np.int64).ravel())
    if R.shape[0] != rec.shape[1]: R = R.T
    if D.shape[0] != len(own): D = D.T
    graded = (vp_ref, mp_ref) if vpg else (mp_ref,)
    shape_ok = tuple(mp_dd.grad.shape) == (nz, nx) and (not vpg or tuple(vp_dd.grad.shape) == (nz, nx))
    signal = (float(np.abs(R).max()) > 0 and all(float(g.grad.norm()) > 0 for g in graded))
    cuts = [i * (padded_shape[-1] // args.px) for i in range(1, args.px)]
    cut_alive = all(float(g.grad[:, min(c, nx - 1)].abs().max()) > 0
                    for g in graded for c in cuts) if cuts else True
    # mp only: vp's gradient must not even exist on either side
    vp_bit = (bool(torch.equal(vp_dd.grad, vp_ref.grad)) if vpg
              else vp_dd.grad is None and vp_ref.grad is None)
    bits = (bool(np.array_equal(D, R[own])), vp_bit, bool(torch.equal(mp_dd.grad, mp_ref.grad)))
    vp_rel = (float((vp_dd.grad - vp_ref.grad).norm() / (vp_ref.grad.norm() + 1e-30)) if vpg else 0.0)
    mp_rel = float((mp_dd.grad - mp_ref.grad).norm() / (mp_ref.grad.norm() + 1e-30))
    ok = shape_ok and signal and cut_alive and all(bits)
    if rank == 0:
        print(f"LSRTM2D px{args.px} nt{nt} src={args.src}(x{sx}) vp_grad={int(vpg)} padded={padded_shape}: "
              f"signal={signal} shape_ok={shape_ok} cut_alive={cut_alive}(x={cuts}) | "
              f"vp.grad rel={vp_rel:.2e} mp.grad rel={mp_rel:.2e} "
              f"[bit rec={bits[0]} vp={bits[1]} mp={bits[2]}] -> {'PASS' if ok else 'FAIL'}", flush=True)
    fail = torch.tensor([0 if ok else 1], device=dev); dist.all_reduce(fail)
    dist.barrier(); dist.destroy_process_group()
    sys.exit(int(fail.item() > 0))


if __name__ == "__main__":
    main()
