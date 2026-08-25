"""LSRTM3D variant of dd_corner_check: DD-vs-single parity for AcousticLSRTM3D.

FORWARD ONLY.  ``acoustic_lsrtm3d``'s CUDA forward now honours the stepped range
(it_begin/it_end) and DD exchanges the halo of BOTH coupled fields (background +
scattered) after every step.  The backward is not stepped yet, so ``_run_adjoint``
raises by design; this check also asserts that refusal instead of comparing
gradients (a silently-wrong gradient is the failure mode DD guards against).

The scattered field is the interesting one: it is driven by the coupling
``mp * vp^2 * lap(bg)``, so a stale background halo on a cut face would leak into
the recorded scattered data even if bg itself were exchanged correctly.

Launch: torchrun --standalone --nproc-per-node=N test/dd_corner_check_lsrtm3d.py --py P --px X
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
from sweep.propagator.torch import PropTorch              # noqa: E402
DT, DH = 0.0012, 15.0


def ricker(nt, dt, fm=6.0, delay=0.2):
    t = np.arange(nt, dtype=np.float32) * dt - delay; a = np.pi * fm * t
    return ((1.0 - 2.0 * a**2) * np.exp(-(a**2))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", type=int, default=2); ap.add_argument("--px", type=int, default=2)
    ap.add_argument("--fs", type=int, default=0); ap.add_argument("--nt", type=int, default=600)
    ap.add_argument("--abcn", type=int, default=20); ap.add_argument("--so", type=int, default=4)
    args = ap.parse_args()
    dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count()); torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}"); assert args.py * args.px == world

    nz, ny, nx, nt = 64, 96, 80, args.nt; gshape = (nz, ny, nx)
    zc = np.arange(nz, dtype=np.float32)[:, None, None]
    vp = 1800.0 + 14.0 * np.maximum(zc - 12.0, 0.0)
    vp = np.broadcast_to(vp, gshape).astype(np.float32).copy()
    vp[:12] = 1500.0; vp[44:50] += 600.0; vp = np.minimum(vp, 3050.0)
    # reflectivity: a couple of layers + lateral variation, so the scattered field
    # is generated across the whole model (and across every cut face).
    mp = np.zeros(gshape, np.float32)
    mp[30, :, :] = 0.04
    mp[47, :, :] = -0.03
    mp += 0.01 * np.sin(np.arange(nx, dtype=np.float32) / 7.0)[None, None, :]

    src = np.array([[[37, 27, 20], [60, 47, 20], [75, 70, 20]]], dtype=np.int64)
    wav = torch.as_tensor(ricker(nt, DT), device=dev)
    rec = np.array([[[ix, iy, 0] for iy in range(2, ny - 2, 2)
                     for ix in range(2, nx - 2, 2)]], dtype=np.int64)
    nrec = rec.shape[1]

    def build_prop():
        eq = AcousticLSRTM3D(spatial_order=args.so, device=dev, backend="torch")
        return PropTorch(eq, backend="torch", impl="c", shape=gshape, dh=DH, dt=DT, nt=nt,
                         abcn=args.abcn, source_type=["h1"], receiver_type=["sh1"],
                         dev=dev, free_surface=bool(args.fs), pml_type="cpmlr")

    with torch.no_grad():
        vp_t = torch.tensor(vp, device=dev); mp_t = torch.tensor(mp, device=dev)
        rec_ref = build_prop()(wav, src, rec, models=[vp_t, mp_t])

        mesh = MeshTopology(py=args.py, px=args.px, shot_groups=1, world_size=world, rank=rank)
        ddp = ModelParallel(build_prop(), mesh)
        rec_dd = ddp(wav, src, rec, models=[vp_t.clone(), mp_t.clone()])

    own = getattr(ddp, "_own_rec_idx", None)
    own = (np.arange(nrec) if own is None else np.asarray(own, dtype=np.int64).ravel())
    R = rec_ref.detach().cpu().numpy().squeeze(); D = rec_dd.detach().cpu().numpy().squeeze()
    if R.shape[0] != nrec: R = R.T
    if D.shape[0] != len(own): D = D.T
    rec_bit = bool(np.array_equal(D, R[own]))
    dmax = float(np.abs(D - R[own]).max()) / (float(np.abs(R).max()) + 1e-30)

    # the backward must refuse (not silently run the whole record per stepped call)
    grad_guard = False
    try:
        ddp._run_adjoint(torch.zeros_like(rec_dd))
    except NotImplementedError:
        grad_guard = True
    except Exception:
        grad_guard = False

    TOL = 1e-6   # forward halo exchange should be bitwise; tolerate fp32 noise only
    ok = (dmax < TOL) and grad_guard
    if rank == 0:
        print(f"LSRTM3D py{args.py}xpx{args.px} nt{nt} fs{args.fs}: "
              f"rec(max_rel={dmax:.2e}, bit={rec_bit}) grad_guard={grad_guard} "
              f"-> {'PASS' if ok else 'FAIL'}(tol={TOL:.0e})", flush=True)
    fail = torch.tensor([0 if ok else 1], device=dev); dist.all_reduce(fail)
    dist.barrier(); dist.destroy_process_group()
    sys.exit(int(fail.item() > 0))


if __name__ == "__main__":
    main()
