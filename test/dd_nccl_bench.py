"""DD scaling benchmark — production-mode loop (no per-step host sync).

torchrun --nproc-per-node=<P> test/dd_nccl_bench.py --nxp 4096 --nz 4096 --nt 500
torchrun --nproc-per-node=<P> test/dd_nccl_bench.py --ndim 3 --nxp 320 --ny 320 --nz 320 --nt 200

Weak scaling: every rank owns a tile of (nz x nxp) [2-D] regardless of P;
the global model is nz x (nxp * P). Ideal: wall time independent of P and
equal to the single-GPU run of one tile (launch with --nproc-per-node=1).

The step loop queues the C++ segment and the NCCL halo exchange without
host synchronisation; only the final step syncs. Per-step host work is the
wavefield-list rotation (sub-microsecond, measured on KW60443).
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

from sweep.equations import Acoustic, Acoustic3D  # noqa: E402
from sweep.parallel import MeshTopology, ModelParallelMesh, exchange_halos  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import (  # noqa: E402
    SteppedBindingRunner,
    acoustic_psi_pairs,
)

SO = 4
M = SO // 2


def ricker(nt, dt, fm=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


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
    ap.add_argument("--ndim", type=int, default=2)
    ap.add_argument("--nz", type=int, default=4096)
    ap.add_argument("--ny", type=int, default=320)   # 3-D only
    ap.add_argument("--nxp", type=int, default=4096, help="x cells PER RANK")
    ap.add_argument("--nt", type=int, default=500)
    ap.add_argument("--abcn", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.0005)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--fast", action="store_true",
                    help="use FastHaloSet instead of exchange_halos")
    ap.add_argument("--exchange-mode", default=None,
                    choices=("none", "stage", "full"),
                    help="diagnostic: none = skip exchange entirely (timing "
                         "upper bound, WRONG numerics); stage = staging "
                         "copies only, no NCCL; full = normal")
    ap.add_argument("--overlap", action="store_true",
                    help="SPECFEM-style comm/compute overlap: phase-1 cut "
                         "strips -> NCCL halo exchange on a comm stream "
                         "concurrent with the phase-2 interior stencil. "
                         "Requires the fast/full exchange path and world>1.")
    ap.add_argument("--so", type=int, default=4, choices=(2, 4, 6, 8),
                    help="spatial order; halo width M = so/2 follows")
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
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    ndim, nz, nxp, nt, abcn, dt = (
        args.ndim, args.nz, args.nxp, args.nt, args.abcn, args.dt
    )
    pad = abcn + M

    topo = MeshTopology(py=1, px=world, shot_groups=1, world_size=world, rank=rank)
    mesh = ModelParallelMesh(grid=(1, world)) if world > 1 else None

    if ndim == 2:
        shape = (nz, nxp)
        vp = np.full(shape, 2500.0, dtype=np.float32)
        src = np.array([[[nxp // 2, nz // 4]]], dtype=np.int32)
        rec = np.array([[[nxp // 2, 2]]], dtype=np.int32)
        eq_cls = Acoustic
    else:
        shape = (nz, args.ny, nxp)
        vp = np.full(shape, 2500.0, dtype=np.float32)
        src = np.array([[[nxp // 2, args.ny // 2, nz // 4]]], dtype=np.int32)
        rec = np.array([[[nxp // 2, args.ny // 2, 2]]], dtype=np.int32)
        eq_cls = Acoustic3D

    wavelet = ricker(nt, dt)
    equation = eq_cls(spatial_order=SO, device=dev, backend="torch")
    prop = PropTorch(
        equation, backend="torch", impl="c", shape=shape, dev=dev, dh=10.0,
        dt=dt, source_type=["h1"], receiver_type=["h1"], abcn=abcn,
        free_surface=False, pml_type="cpmlr", nt=nt, B=1, use_ckpt=False,
        boundary_saving_config={"enabled": False},
        model_parallel=topo,
    )
    cap = capture(prop)
    with torch.no_grad():
        prop(wavelet, src, rec, models=[torch.tensor(vp, device=dev)])
    p, func = cap["params"], cap["func"]
    L = list(p.wavefields)
    if not L:
        L = [torch.zeros_like(p.models[0]) for _ in range(9 if ndim == 2 else 12)]
    p.record_out = torch.zeros_like(cap["raw_out"][2])
    psi_pairs = acoustic_psi_pairs(ndim)

    lo = pad
    hi = pad + nxp

    mode = args.exchange_mode or ("full" if (args.fast or args.overlap) else "std")
    fast_set = None
    if mode in ("full", "stage") and world > 1:
        from sweep.parallel.fast_halo import FastHaloSet
        fast_set = FastHaloSet(mesh, M, ("x",))

    if args.overlap:
        assert world > 1, "--overlap needs world > 1 (no cuts otherwise)"
        assert mode == "full", "--overlap requires the full NCCL exchange path"
        # Phase-1 strips carry no source contribution when exchanged, so
        # the source must sit >= M away from every cut.
        src_x = int(src[0, 0, 0])
        assert M <= src_x < nxp - M, "overlap: source within M of a cut"
        cut_mask = 0
        if mesh.neighbour_rank("x", -1) is not None:
            cut_mask |= 1
        if mesh.neighbour_rank("x", +1) is not None:
            cut_mask |= 2
        p.cut_face_mask = cut_mask

    def do_exchange(view):
        if mode == "none":
            return
        if mode == "stage":
            ex = fast_set._cache.get(view.data_ptr())
            if ex is None:
                fast_set.exchange(view)  # builds cache (one real exchange)
                return
            for sbuf, sview in ex._send_views:
                sbuf.copy_(sview)
            for rview, rbuf in ex._recv_views:
                rview.copy_(rbuf)
            return
        if mode == "full":
            fast_set.exchange(view)
        else:
            exchange_halos([view], mesh, M, ("x",))

    comm = torch.cuda.Stream() if args.overlap else None
    comm_evt = torch.cuda.Event() if args.overlap else None

    def one_run_serial():
        for t in L:
            t.zero_()
        runner = SteppedBindingRunner(func, p, L, psi_pairs)
        if world > 1:
            dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for it in range(nt):
                runner.run_to(it + 1)
                if world > 1:
                    do_exchange(runner.u_now[..., lo - M: hi + M])
        torch.cuda.synchronize()
        if world > 1:
            dist.barrier()
        return time.perf_counter() - t0

    def one_run_overlap():
        for t in L:
            t.zero_()
        runner = SteppedBindingRunner(func, p, L, psi_pairs)
        dist.barrier()
        torch.cuda.synchronize()
        compute = torch.cuda.current_stream()
        t0 = time.perf_counter()
        with torch.no_grad():
            for it in range(nt):
                runner.run_phase(it + 1, 1)          # cut boundary strips
                comm_evt.record()
                with torch.cuda.stream(comm):
                    comm.wait_event(comm_evt)
                    # u_next strips -> NCCL + staging on the comm stream;
                    # after phase-2's swap this tensor IS u_now, so the
                    # halo lands where the next step's stencil reads it.
                    fast_set.exchange(runner.u_next[..., lo - M: hi + M])
                runner.run_phase(it + 1, 2)          # interior + tail
                compute.wait_stream(comm)            # halo ready for next phase 1
        torch.cuda.synchronize()
        dist.barrier()
        return time.perf_counter() - t0

    one_run = one_run_overlap if args.overlap else one_run_serial

    # warm-up >= 1.5 s
    tw = time.perf_counter()
    while time.perf_counter() - tw < 1.6:
        one_run() if nt <= 50 else None
        if nt > 50:
            break
    one_run()  # one full warm-up run

    times = [one_run() for _ in range(args.repeats)]
    best = min(times)
    cv = float(np.std(times) / np.mean(times))
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    if rank == 0:
        per_step = best / nt * 1e3
        tag = f"{mode}{'+overlap' if args.overlap else ''}-so{SO}-ab{abcn}"
        print(
            f"DD_BENCH ndim={ndim} tile={shape} px={world} nt={nt} ex={tag} "
            f"best={best:.3f}s per_step={per_step:.3f}ms cv={cv:.3f} "
            f"peak_mem={peak_gb:.2f}GB times={['%.3f' % t for t in times]}"
        )

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
