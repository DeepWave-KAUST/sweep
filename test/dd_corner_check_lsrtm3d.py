"""AcousticLSRTM3D variant of dd_corner_check: DD parity vs single domain.

LSRTM carries TWO coupled fields (background + scattered) in one wavefield list,
so domain decomposition has to rotate and halo-exchange both: the forward ships
both ``u_now``, the boundary-saving backward ships both adjoints (lambda_sc and
the RWI background adjoint lambda_bg) and both reconstructions.  This check
compares the record and BOTH model gradients -- the reflectivity ``mp`` and the
RWI tomographic ``vp`` (terms II+III+IV) -- against one card.

The scattered field is generated across the whole model and every cut face (a
dipping reflectivity sheet plus lateral variation), so a stale halo of any of
the four carried fields shows up in the record or in a gradient.  Every
comparison first asserts its signal is non-zero: 0 == 0 is not a pass.

Launch: torchrun --standalone --nproc-per-node=N test/dd_corner_check_lsrtm3d.py --py P --px X
        (N = P * X; --py 1 --px 1 is the single-card run of the full DD machinery)
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np, torch, torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]; SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))
from sweep.equations import AcousticLSRTM3D               # noqa: E402
from sweep.parallel import MeshTopology                   # noqa: E402
from sweep.parallel.dd_propagator import ModelParallel    # noqa: E402
from sweep.propagator import BoundarySaving, PropTorch     # noqa: E402
DT, DH = 0.0012, 15.0


def ricker(nt, dt, fm=8.0, delay=0.12):
    t = np.arange(nt, dtype=np.float32) * dt - delay; a = np.pi * fm * t
    return ((1.0 - 2.0 * a**2) * np.exp(-(a**2))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", type=int, default=1); ap.add_argument("--px", type=int, default=1)
    ap.add_argument("--nt", type=int, default=420); ap.add_argument("--abcn", type=int, default=12)
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--src", type=str, default="mid",
                    help="'mid' = (nx//2, ny//2): ON the x (and y) cut lines; "
                         "'off' = inside one tile, away from every cut")
    ap.add_argument("--diag", type=int, default=0, help="print where the gradient error lives")
    ap.add_argument("--vp-grad", type=int, default=1,
                    help="1: vp requires grad (RWI vp gradient); 0: mp only, the classic LSRTM layout")
    args = ap.parse_args()
    vpg = bool(args.vp_grad)
    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count()); torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}"); assert args.py * args.px == world, (args.py, args.px, world)

    nz, ny, nx, nt = 44, 48, 52, args.nt; gshape = (nz, ny, nx)
    zc = np.arange(nz, dtype=np.float32)[:, None, None]
    vp = np.broadcast_to(2000.0 + 12.0 * zc, gshape).astype(np.float32).copy()
    vp[30:] += 300.0
    # reflectivity: a dipping sheet that crosses every cut, plus lateral wiggle,
    # reaching every model edge, so the edge padding carries it into the absorbing
    # band -- where the background adjoint's coupling runs through its pml_field
    mp = np.zeros(gshape, np.float32)
    for ix in range(nx):
        iz = 22 + (ix * 6) // nx
        mp[iz, :, ix] = 0.05
    mp += 0.01 * np.sin(np.arange(ny, dtype=np.float32) / 3.0)[None, :, None] * (zc > 10)

    sx, sy = (nx // 2, ny // 2) if args.src == "mid" else (nx // 4 + 1, ny // 4 + 1)
    src = np.array([[[sx, sy, 4]]], dtype=np.int64)
    rec = np.array([[[ix, iy, 2] for iy in range(3, ny - 3, 3) for ix in range(3, nx - 3, 3)]], dtype=np.int64)
    nrec = rec.shape[1]
    wav = torch.as_tensor(ricker(nt, DT), device=dev)

    def build_prop():
        eq = AcousticLSRTM3D(spatial_order=args.so, device=dev, backend="torch")
        return PropTorch(eq, backend="torch", impl="c", shape=gshape, dh=DH, dt=DT, nt=nt,
                         abcn=args.abcn, source_type=["h1"], receiver_type=["sh1"], dev=dev,
                         free_surface=False, pml_type="cpmlr", memory=BoundarySaving())

    vp_ref = torch.tensor(vp, device=dev, requires_grad=vpg)
    mp_ref = torch.tensor(mp, device=dev, requires_grad=True)
    rec_ref = build_prop()(wav, src, rec, models=[vp_ref, mp_ref])
    (rec_ref.double() ** 2).sum().backward()

    mesh = MeshTopology(py=args.py, px=args.px, shot_groups=1, world_size=world, rank=rank)
    ddp = ModelParallel(build_prop(), mesh)
    vp_dd = torch.tensor(vp, device=dev, requires_grad=vpg)
    mp_dd = torch.tensor(mp, device=dev, requires_grad=True)
    rec_dd = ddp(wav, src, rec, models=[vp_dd, mp_dd])
    own = getattr(ddp, "_own_rec_idx", None)
    own = (np.arange(nrec) if own is None else np.asarray(own, dtype=np.int64).ravel())
    (rec_dd.double() ** 2).sum().backward()
    dist.all_reduce(mp_dd.grad)
    if vpg:
        dist.all_reduce(vp_dd.grad)

    R = rec_ref.detach().cpu().numpy().squeeze(); D = rec_dd.detach().cpu().numpy().squeeze()
    if R.shape[0] != nrec: R = R.T
    if D.shape[0] != len(own): D = D.T
    signal = (float(np.abs(R).max()) > 0 and float(mp_ref.grad.norm()) > 0
              and (not vpg or float(vp_ref.grad.norm()) > 0))
    rec_rel = float(np.abs(D - R[own]).max()) / (float(np.abs(R).max()) + 1e-30)
    vp_rel = (float((vp_dd.grad - vp_ref.grad).norm() / (vp_ref.grad.norm() + 1e-30)) if vpg else 0.0)
    mp_rel = float((mp_dd.grad - mp_ref.grad).norm() / (mp_ref.grad.norm() + 1e-30))
    # mp only: vp's gradient must not even exist on either side
    vp_bit = (bool(torch.equal(vp_dd.grad, vp_ref.grad)) if vpg
              else vp_dd.grad is None and vp_ref.grad is None)
    bits = (bool(np.array_equal(D, R[own])), vp_bit, bool(torch.equal(mp_dd.grad, mp_ref.grad)))
    TOL = 1e-5   # the DD gate holds acoustic to rel=0; a stale halo is orders above this
    ok = signal and rec_rel < TOL and vp_rel < TOL and mp_rel < TOL and vp_bit
    if rank == 0 and args.diag:
        pairs = (("vp", vp_dd.grad, vp_ref.grad),) if vpg else ()
        for lab, gd, gr in pairs + (("mp", mp_dd.grad, mp_ref.grad),):
            d = (gd - gr).abs().double().cpu().numpy(); r = gr.abs().double().cpu().numpy()
            xp = d.sum(axis=(0, 1)); xr = r.sum(axis=(0, 1)) + 1e-300
            yp = d.sum(axis=(0, 2)); zp = d.sum(axis=(1, 2))
            iz, iy, ix = np.unravel_index(int(d.argmax()), d.shape)
            print(f"  [diag {lab}] max|err| at (z,y,x)=({iz},{iy},{ix})  src=(x{sx},y{sy})  cut x={nx//2} y={ny//2}", flush=True)
            print(f"  [diag {lab}] err share by x-band: seam(+/-3)={xp[nx//2-3:nx//2+3].sum()/xp.sum():.2f} "
                  f"src(+/-3)={xp[max(sx-3,0):sx+3].sum()/xp.sum():.2f}  | rel err per x (every 4th): "
                  + " ".join(f"{v:.1e}" for v in (xp/xr)[::4]), flush=True)
            print(f"  [diag {lab}] err share by z: top 8 rows={zp[:8].sum()/zp.sum():.2f}  "
                  f"by y: seam(+/-3)={yp[ny//2-3:ny//2+3].sum()/yp.sum():.2f}", flush=True)
    if rank == 0:
        gvp = f"{vp_ref.grad.norm():.2e}" if vpg else "n/a"
        print(f"LSRTM3D py{args.py}xpx{args.px} nt{nt} src={args.src} vp_grad={int(vpg)}: signal={signal} "
              f"|rec|max={np.abs(R).max():.2e} |g_vp|={gvp} |g_mp|={mp_ref.grad.norm():.2e} | "
              f"rec rel={rec_rel:.2e} vp.grad rel={vp_rel:.2e} mp.grad rel={mp_rel:.2e} "
              f"[bit rec={bits[0]} vp={bits[1]} mp={bits[2]}] -> {'PASS' if ok else 'FAIL'}(tol={TOL:.0e})",
              flush=True)
    fail = torch.tensor([0 if ok else 1], device=dev); dist.all_reduce(fail)
    dist.barrier(); dist.destroy_process_group()
    sys.exit(int(fail.item() > 0))


if __name__ == "__main__":
    main()
