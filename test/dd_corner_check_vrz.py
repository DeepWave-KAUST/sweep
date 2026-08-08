"""VRZ variant of dd_corner_check: 2-axis DD parity vs single domain for AcousticVRZ3D.
Same loud-corner geometry. Tests whether the multi-axis halo fix (bit-exact for Acoustic3D)
also covers DD-VRZ. Reports rec + BOTH vp.grad and z.grad bit-exactness.
Launch: torchrun --standalone --nproc-per-node=N test/dd_corner_check_vrz.py --py P --px X
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
# DD backward forces the SPLIT gradient kernel; make the single-domain reference use
# split too (else split-vs-fused differ by fast-math FMA and never match). Post-fix the
# grad should collapse from rel~2 to the forward-reconstruction level (~1e-5), not bitwise.
os.environ.setdefault("SWEEP_VRZ_GRAD_SPLIT", "1")
import numpy as np, torch, torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]; SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))
from sweep.equations import AcousticVRZ3D                 # noqa: E402
from sweep.parallel import MeshTopology                  # noqa: E402
from sweep.parallel.dd_propagator import ModelParallel   # noqa: E402
from sweep.propagator.torch import PropTorch             # noqa: E402
DT, DH = 0.0012, 15.0

def ricker(nt, dt, fm=6.0, delay=0.2):
    t = np.arange(nt, dtype=np.float32) * dt - delay; a = np.pi * fm * t
    return ((1.0 - 2.0 * a**2) * np.exp(-(a**2))).astype(np.float32)

def gardner_z(vp):  # z = rho*vp/1000 (MRayl); water rho=1, sediment Gardner 0.31*vp^0.25
    rho = np.where(vp <= 1505.0, 1.0, 0.31 * np.clip(vp, 1.0, None) ** 0.25)
    return (rho * vp / 1000.0).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", type=int, default=2); ap.add_argument("--px", type=int, default=2)
    ap.add_argument("--fs", type=int, default=1); ap.add_argument("--nt", type=int, default=1200)
    ap.add_argument("--abcn", type=int, default=20); ap.add_argument("--so", type=int, default=4)
    args = ap.parse_args()
    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count()); torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}"); assert args.py * args.px == world
    nz, ny, nx, nt = 64, 96, 80, args.nt; gshape = (nz, ny, nx)
    zc = np.arange(nz, dtype=np.float32)[:, None, None]
    vp = 1800.0 + 14.0 * np.maximum(zc - 12.0, 0.0); vp = np.broadcast_to(vp, gshape).astype(np.float32).copy()
    vp[:12] = 1500.0; vp[44:50] += 600.0; vp = np.minimum(vp, 3050.0)
    zimp = gardner_z(vp)
    src = np.array([[[37, 27, 20], [60, 47, 20], [75, 70, 20]]], dtype=np.int64)
    wav = torch.as_tensor(ricker(nt, DT), device=dev)
    rec = np.array([[[ix, iy, 0] for iy in range(2, ny - 2, 2) for ix in range(2, nx - 2, 2)]], dtype=np.int64)
    nrec = rec.shape[1]
    def build_prop():
        eq = AcousticVRZ3D(spatial_order=args.so, device=dev, backend="torch")
        return PropTorch(eq, backend="torch", impl="c", shape=gshape, dh=DH, dt=DT, nt=nt, abcn=args.abcn,
                         source_type=["h1"], receiver_type=["h1"], dev=dev, free_surface=bool(args.fs),
                         pml_type="cpmlr")   # VRZ requires CPML (matches SolverSpec vrz3d + production config)
    vp_ref = torch.tensor(vp, device=dev, requires_grad=True); z_ref = torch.tensor(zimp, device=dev, requires_grad=True)
    rec_ref = build_prop()(wav, src, rec, models=[vp_ref, z_ref]); (rec_ref.double() ** 2).sum().backward()
    mesh = MeshTopology(py=args.py, px=args.px, shot_groups=1, world_size=world, rank=rank)
    ddp = ModelParallel(build_prop(), mesh)
    vp_dd = torch.tensor(vp, device=dev, requires_grad=True); z_dd = torch.tensor(zimp, device=dev, requires_grad=True)
    rec_dd = ddp(wav, src, rec, models=[vp_dd, z_dd])
    own = getattr(ddp, "_own_rec_idx", None)
    own = (np.arange(nrec) if own is None else np.asarray(own, dtype=np.int64).ravel())
    (rec_dd.double() ** 2).sum().backward(); dist.all_reduce(vp_dd.grad); dist.all_reduce(z_dd.grad)
    R = rec_ref.detach().cpu().numpy().squeeze(); D = rec_dd.detach().cpu().numpy().squeeze()
    if R.shape[0] != nrec: R = R.T
    if D.shape[0] != len(own): D = D.T
    rec_bit = bool(np.array_equal(D, R[own]))
    vpg_bit = bool(torch.equal(vp_dd.grad, vp_ref.grad)); zg_bit = bool(torch.equal(z_dd.grad, z_ref.grad))
    dmax = float(np.abs(D - R[own]).max()) / (float(np.abs(R).max()) + 1e-30)
    vgr = float((vp_dd.grad - vp_ref.grad).norm() / (vp_ref.grad.norm() + 1e-30))
    zgr = float((z_dd.grad - z_ref.grad).norm() / (z_ref.grad.norm() + 1e-30))
    if rank == 0:  # localize the vp-grad error: seam column vs global?
        _gd = vp_dd.grad.detach().cpu().numpy(); _gr = vp_ref.grad.detach().cpu().numpy()
        _df = np.abs(_gd - _gr); _nz, _ny, _nx = _df.shape
        _xp = _df.mean(axis=(0, 1)); _yp = _df.mean(axis=(0, 2))
        _xc = _nx // 2; _yc = _ny // 2
        print(f"[DDVRZ-LOC] shape={_df.shape} max={_df.max():.3e}@{np.unravel_index(int(_df.argmax()), _df.shape)} "
              f"| x-prof peak@ix={int(_xp.argmax())} xp[nx/2={_xc}]/med={_xp[_xc]/(np.median(_xp)+1e-30):.1f} "
              f"| y-prof peak@iy={int(_yp.argmax())} yp[ny/2={_yc}]/med={_yp[_yc]/(np.median(_yp)+1e-30):.1f}", flush=True)
    TOL = 1e-3   # post-fix grad rel should fall to ~forward level (~1e-5); rel~2 = the bug
    ok = (dmax < TOL) and (vgr < TOL) and (zgr < TOL)
    if rank == 0:
        print(f"VRZ py{args.py}xpx{args.px} nt{nt} fs{args.fs}: rec(max_rel={dmax:.2e}) "
              f"vpgrad(rel={vgr:.2e}) zgrad(rel={zgr:.2e}) [bit rec={rec_bit} vp={vpg_bit} z={zg_bit}] "
              f"-> {'PASS' if ok else 'FAIL'}(tol={TOL:.0e})", flush=True)
    fail = torch.tensor([0 if ok else 1], device=dev); dist.all_reduce(fail); dist.barrier(); dist.destroy_process_group()
    sys.exit(int(fail.item() > 0))

if __name__ == "__main__":
    main()
