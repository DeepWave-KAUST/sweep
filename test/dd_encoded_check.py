"""Encoded-supershot (source-encoding) correctness check for ModelParallel.

An encoded supershot = B point sources superposed into ONE wavefield, each with
its own signed wavelet (wavelet_super shape (B, nt)). This is the production
sweep-tasks path. DD must subset the per-source wavelet to each tile's owned
sources (dd_propagator._prepare_call). This test compares DD vs single-domain
PropTorch on the SAME encoded supershot: scalar misfit + GLOBAL model gradient
(no receiver-reassembly needed — the gradient of a replicated global leaf is
all_reduced to global shape).

torchrun --standalone --nproc-per-node=2 test/dd_encoded_check.py --px 2
torchrun --standalone --nproc-per-node=8 test/dd_encoded_check.py --px 4 --py 2
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sweep.equations import Acoustic3D  # noqa: E402
from sweep.parallel import MeshTopology, ModelParallel  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

DT = 0.0015


def ricker(nt, dt, fm=10.0, delay=0.09):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a**2) * np.exp(-(a**2))).astype(np.float32)


def cos_rel(a, b):
    a = a.double().flatten(); b = b.double().flatten()
    cos = (a @ b) / (a.norm() * b.norm() + 1e-300)
    rel = (a - b).norm() / (b.norm() + 1e-300)
    return float(cos), float(rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--px", type=int, default=2)
    ap.add_argument("--py", type=int, default=1)
    ap.add_argument("--nz", type=int, default=32)
    ap.add_argument("--ny", type=int, default=24)
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--nt", type=int, default=140)
    ap.add_argument("--B", type=int, default=8, help="encoded sources in the supershot")
    ap.add_argument("--abcn", type=int, default=20)
    ap.add_argument("--so", type=int, default=4)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li); dev = torch.device(f"cuda:{li}")
    px, py = args.px, args.py
    assert px * py == world, f"px*py={px*py} != world={world}"
    nz, ny, nx, nt, B = args.nz, args.ny, args.nx, args.nt, args.B
    gshape = (nz, ny, nx)

    # global smooth model + a box anomaly (same init on every rank)
    z = np.linspace(0, 1, nz, dtype=np.float32)[:, None, None]
    init_vp = np.broadcast_to(1800.0 + 600.0 * z, gshape).astype(np.float32).copy()
    init_vp[nz // 3:2 * nz // 3, ny // 3:2 * ny // 3, nx // 3:2 * nx // 3] += 250.0

    # encoded supershot: B sources spread across x (so an x-cut splits them),
    # each with a fixed +/-1 signed wavelet -> wavelet_super (B, nt).
    rng = np.random.RandomState(0)
    # spread sources in BOTH x and y, AVOIDING the decomposition cut lines
    # (px x-cuts at k*nx/px, py y-cut at ny/2) so no source/receiver sits on a
    # tile boundary — otherwise the halo reconstruction at the cut looks like a
    # numeric failure (geometry artifact, not a DD bug).
    xs = np.linspace(3, nx - 4, B).astype(np.int64)
    ys = np.where(np.arange(B) % 2 == 0, ny // 4, 3 * ny // 4).astype(np.int64)
    src = np.array([[[int(xs[i]), int(ys[i]), 3] for i in range(B)]], dtype=np.int64)  # (1,B,3)
    signs = rng.choice([-1.0, 1.0], size=B).astype(np.float32)
    base = ricker(nt, DT)
    wav_enc = (base[None, :] * signs[:, None]).astype(np.float32)            # (B, nt)
    wav_shared = base.copy()                                                 # (nt,)
    rec = np.array([[[ix, iy, 2]
                     for iy in range(3, ny - 2, 5)
                     for ix in range(2, nx - 2, 4)]], dtype=np.int64)        # (1, nrec, 3)

    def build_prop():
        eq = Acoustic3D(spatial_order=args.so, device=dev, backend="torch")
        return PropTorch(eq, backend="torch", impl="c", shape=gshape, dh=10.0, dt=DT,
                         nt=nt, abcn=args.abcn, source_type=["h1"], receiver_type=["h1"],
                         dev=dev, free_surface=False)

    results = {}
    for tag, wav in (("shared(nt,)", wav_shared), ("encoded(B,nt)", wav_enc)):
        # ---- single-domain reference (full model on this GPU) ----
        vp_ref = torch.tensor(init_vp, device=dev, requires_grad=True)
        prop_ref = build_prop()
        wav_t = torch.as_tensor(wav, device=dev)
        rec_ref = prop_ref(wav_t, src, rec, models=[vp_ref])
        loss_ref = (rec_ref.double() ** 2).sum()
        loss_ref.backward()
        g_ref = vp_ref.grad.detach().clone()

        # ---- DD ----
        mesh = MeshTopology(py=py, px=px, shot_groups=1, world_size=world, rank=rank)
        ddp = ModelParallel(build_prop(), mesh)
        vp_dd = torch.tensor(init_vp, device=dev, requires_grad=True)
        rec_dd = ddp(wav_t, src, rec, models=[vp_dd])
        loss_dd = (rec_dd.double() ** 2).sum()
        dist.all_reduce(loss_dd)
        loss_dd.backward()
        dist.all_reduce(vp_dd.grad)
        g_dd = vp_dd.grad.detach().clone()

        cos, rel = cos_rel(g_dd, g_ref)
        lref, ldd = float(loss_ref), float(loss_dd)
        lrel = abs(ldd - lref) / (abs(lref) + 1e-300)
        results[tag] = (lref, ldd, lrel, cos, rel,
                        bool(torch.equal(g_dd, g_ref)))

    if rank == 0:
        print(f"\n=== encoded-supershot DD check  world={world} px={px} py={py} "
              f"global={gshape} B={B} nt={nt} ===", flush=True)
        for tag, (lref, ldd, lrel, cos, rel, bit) in results.items():
            ok = "PASS" if (cos > 0.9999 and rel < 1e-4 and lrel < 1e-4) else "FAIL"
            print(f"  [{tag:14s}] loss ref={lref:.6e} dd={ldd:.6e} rel={lrel:.2e} | "
                  f"grad cos={cos:.8f} rel_l2={rel:.2e} bit_exact={bit} -> {ok}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
