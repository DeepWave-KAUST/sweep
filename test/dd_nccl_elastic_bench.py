"""Elastic DD weak-scaling benchmark (forward, production half-step loop).

torchrun --nproc-per-node=<P> test/dd_nccl_elastic_bench.py \
    --ndim 2 --nz 2048 --nxp 2048 --nt 400 [--so 4] [--abcn 20] [--fast]
torchrun --nproc-per-node=<P> test/dd_nccl_elastic_bench.py \
    --ndim 3 --nz 192 --ny 192 --nxp 192 --nt 150 [--fast]

Weak scaling: every rank owns an (nz x nxp) [2-D] / (nz x ny x nxp) [3-D]
tile regardless of P; the global model is nz x (nxp * P). Ideal: wall time
independent of P, equal to the single-GPU run of one tile.

Elastic uses the E1 HALF-STEP exchange protocol, which splits by PHYSICS
(velocity / stress), not by spatial region: each phase computes the full
tile, so the velocity-halo and stress-halo exchanges sit on the critical
path (phase 2 reads the exchanged velocity halo; the next step's phase 1
reads the exchanged stress halo). v1 is therefore SERIAL — there is no
SPECFEM-style compute/comm overlap (that would need cut-strip-only elastic
kernels, not implemented). Per step it exchanges NV velocity + (NPHYS-NV)
stress fields M-wide: 5 fields/step in 2-D, 9 in 3-D — substantially more
than the acoustic 1 field/step, so the achievable efficiency is bounded by
this fixed per-step comm volume.

Reports per_step in the DD_BENCH format consumed by summarize_scale.py.
Run px=1 (no --fast) for the baseline, px=N --exchange-mode none for the
no-comm timing floor, and px=N --fast for the real serial number; the
weak-scaling efficiency is none/serial in the same job.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sweep.equations import Elastic, Elastic3D  # noqa: E402
from sweep.parallel import MeshTopology, ModelParallelMesh, exchange_halos  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import SteppedBindingRunner  # noqa: E402

NWF = {2: 15, 3: 36}
NPHYS = {2: 5, 3: 9}
NV = {2: 2, 3: 3}
X_LO_BIT, X_HI_BIT = 1, 2

SO = 4
M = SO // 2


def ricker(nt, dt, fm=10.0, delay=0.06, scale=1.0e6):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return (scale * (1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def capture(prop):
    cap = {}
    impl = prop._backend_impl
    orig = impl.forward_func

    def wrapper(params):
        out = orig(params)
        cap["params"] = params
        cap["raw_out"] = out
        return out

    impl.forward_func = wrapper
    cap["func"] = orig
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndim", type=int, default=2, choices=(2, 3))
    ap.add_argument("--nz", type=int, default=2048)
    ap.add_argument("--ny", type=int, default=192)
    ap.add_argument("--nxp", type=int, default=2048, help="x cells PER RANK")
    ap.add_argument("--nt", type=int, default=400)
    ap.add_argument("--abcn", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.0004)
    ap.add_argument("--so", type=int, default=4, choices=(2, 4, 6, 8))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--free-surface", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="use FastHaloSet instead of exchange_halos")
    ap.add_argument("--exchange-mode", default=None,
                    choices=("none", "full"),
                    help="none = skip exchange (timing floor, WRONG numerics); "
                         "full = FastHaloSet")
    args = ap.parse_args()

    global SO, M
    SO = args.so
    M = SO // 2

    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dev_idx = local_rank % max(1, torch.cuda.device_count())  # oversubscribe-safe
    torch.cuda.set_device(dev_idx)
    dev = torch.device(f"cuda:{dev_idx}")

    ndim, nz, ny, nxp, nt = args.ndim, args.nz, args.ny, args.nxp, args.nt
    abcn, dt, fs = args.abcn, args.dt, args.free_surface
    pad = abcn + M

    topo = MeshTopology(py=1, px=world, shot_groups=1, world_size=world, rank=rank)
    mesh = ModelParallelMesh(grid=(1, world)) if world > 1 else None

    if ndim == 2:
        shape = (nz, nxp)
        src = np.array([[[nxp // 2, nz // 4]]], dtype=np.int32)
        rec = np.array([[[nxp // 2, 2]]], dtype=np.int32)
        eq_cls = Elastic
        stype, rtype = ["sxx", "szz"], ["vx", "vz"]
    else:
        shape = (nz, ny, nxp)
        src = np.array([[[nxp // 2, ny // 2, nz // 4]]], dtype=np.int32)
        rec = np.array([[[nxp // 2, ny // 2, 2]]], dtype=np.int32)
        eq_cls = Elastic3D
        stype, rtype = ["sxx", "syy", "szz"], ["vx", "vy", "vz"]

    vp = np.full(shape, 3000.0, dtype=np.float32)
    vs = np.full(shape, 1730.0, dtype=np.float32)
    rho = np.full(shape, 2200.0, dtype=np.float32)
    wavelet = ricker(nt, dt)

    eq = eq_cls(spatial_order=SO, device=dev, backend="torch")
    prop = PropTorch(
        eq, backend="torch", impl="c", shape=shape, dev=dev, dh=10.0, dt=dt,
        source_type=stype, receiver_type=rtype, abcn=abcn, free_surface=fs,
        pml_type="cpmls", nt=nt, B=1, use_ckpt=False,
        boundary_saving_config={"enabled": False}, model_parallel=topo,
    )
    cap = capture(prop)
    with torch.no_grad():
        prop(wavelet, src, rec,
             models=[torch.tensor(vp, device=dev), torch.tensor(vs, device=dev),
                     torch.tensor(rho, device=dev)])
    p, func = cap["params"], cap["func"]
    L = list(p.wavefields)
    if not L:
        L = [torch.zeros_like(p.models[0]) for _ in range(NWF[ndim])]
    p.record_out = torch.zeros_like(cap["raw_out"][2])

    cut_mask = ((X_LO_BIT if rank > 0 else 0)
                | (X_HI_BIT if rank < world - 1 else 0))
    if ndim == 3 and world > 1:
        p.cut_face_mask = cut_mask  # elastic3d forward is cut-aware

    lo, hi = pad, pad + nxp
    mode = args.exchange_mode or ("full" if args.fast else "std")
    fast_set = None
    if mode == "full" and world > 1:
        from sweep.parallel.fast_halo import FastHaloSet
        fast_set = FastHaloSet(mesh, M, ("x",))

    def do_exchange(slots):
        if mode == "none" or world == 1:
            return
        for f in slots:
            view = L[f][..., lo - M: hi + M]
            if mode == "full":
                fast_set.exchange(view)
            else:
                exchange_halos([view], mesh, M, ("x",))

    def one_run():
        for t in L:
            t.zero_()
        runner = SteppedBindingRunner(func, p, L, psi_pairs=(), u_blocks=())
        if world > 1:
            dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for it in range(nt):
                runner.run_phase(it + 1, 1)            # velocity
                do_exchange(range(NV[ndim]))
                runner.run_phase(it + 1, 2)            # stress + tail
                do_exchange(range(NV[ndim], NPHYS[ndim]))
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
        return time.perf_counter() - t0

    # warm-up >= 1.6 s (GPU clock ramp; see lesson_gpu_clock_ramp_bench)
    tw = time.perf_counter()
    while time.perf_counter() - tw < 1.6:
        if nt <= 50:
            one_run()
        else:
            break
    one_run()  # one full warm-up run

    times = [one_run() for _ in range(args.repeats)]
    best = min(times)
    cv = float(np.std(times) / np.mean(times))
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    if rank == 0:
        per_step = best / nt * 1e3
        tag = f"{mode}-so{SO}-ab{abcn}{'-fs' if fs else ''}"
        print(
            f"DD_BENCH ndim={ndim} tile={shape} px={world} nt={nt} ex={tag} "
            f"best={best:.3f}s per_step={per_step:.3f}ms cv={cv:.3f} "
            f"peak_mem={peak_gb:.2f}GB times={['%.3f' % t for t in times]}"
        )

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
