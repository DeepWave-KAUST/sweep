"""NCCL multi-GPU DD forward vs single-domain reference (M3).

Launch:  torchrun --nproc-per-node=<P> test/dd_nccl_check.py [--abcn 8]

Each rank owns one x-tile of the global model and advances the stepped
impl='c' forward one step at a time, exchanging u_now halos through
``sweep.parallel.exchange_halos`` (NCCL batch_isend_irecv). Rank 0 also
runs the full single-domain model on its own GPU and asserts bitwise
equality of records and final wavefields (same-model GPUs => IEEE
deterministic kernels => bit equality is the correct bar).

Also prints a coarse per-step timing split (compute vs exchange) as a
first scaling data point for the efficiency milestone.
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


def ricker(nt, dt, fm=10.0, delay=0.06, scale=1.0):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return (scale * (1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def make_prop(shape, abcn, nt, dt, dev, topo=None):
    eq_cls = Acoustic if len(shape) == 2 else Acoustic3D
    equation = eq_cls(spatial_order=SO, device=dev, backend="torch")
    kwargs = dict(
        backend="torch", impl="c", shape=shape, dev=dev, dh=10.0, dt=dt,
        source_type=["h1"], receiver_type=["h1"], abcn=abcn,
        free_surface=False, pml_type="cpmlr", nt=nt, B=1,
        use_ckpt=False, boundary_saving_config={"enabled": False},
    )
    if topo is not None:
        kwargs["model_parallel"] = topo
    return PropTorch(equation, **kwargs)


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


def make_runner(prop, wavelet, sources, receivers, vp_np, dev):
    cap = capture(prop)
    models = [torch.tensor(vp_np, device=dev)]
    with torch.no_grad():
        prop(wavelet, sources, receivers, models=models)
    p, func = cap["params"], cap["func"]
    ndim = vp_np.ndim
    L = list(p.wavefields)
    if not L:
        L = [torch.zeros_like(p.models[0]) for _ in range(9 if ndim == 2 else 12)]
    for t in L:
        t.zero_()
    record = torch.zeros_like(cap["raw_out"][2])
    p.record_out = record
    return SteppedBindingRunner(func, p, L, acoustic_psi_pairs(ndim)), record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nz", type=int, default=48)
    ap.add_argument("--nx", type=int, default=None,
                    help="global nx (default 28*px)")
    ap.add_argument("--nt", type=int, default=120)
    ap.add_argument("--ndim", type=int, default=2, choices=(2, 3))
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--abcn", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.0015)
    ap.add_argument("--fast", action="store_true",
                    help="use FastHaloSet instead of exchange_halos")
    ap.add_argument("--so", type=int, default=4, choices=(2, 4, 6, 8),
                    help="spatial order; halo width M = so/2 follows")
    args = ap.parse_args()

    global SO, M
    SO = args.so
    M = SO // 2

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    px = world
    nz = args.nz
    nx = args.nx or 28 * px
    nxp = nx // px
    nt, abcn, dt = args.nt, args.abcn, args.dt
    pad = abcn + M

    ndim, ny = args.ndim, args.ny
    if ndim == 2:
        vp = 1800.0 + 600.0 * np.linspace(0, 1, nz, dtype=np.float32)[:, None]
        vp = np.broadcast_to(vp, (nz, nx)).copy()
        vp[nz // 3: nz // 2, nx // 3: 2 * nx // 3] += 180.0
        src_g = (nx // 2, nz // 4)
    else:
        vp = 1800.0 + 600.0 * np.linspace(0, 1, nz, dtype=np.float32)[:, None, None]
        vp = np.broadcast_to(vp, (nz, ny, nx)).copy()
        vp[nz // 3: nz // 2, ny // 4: 3 * ny // 4, nx // 3: 2 * nx // 3] += 180.0
        src_g = (nx // 2, ny // 2, nz // 4)
    wavelet = ricker(nt, dt)
    rec_gx = list(range(2, nx - 2, 6))
    rec_z = 2

    topo = MeshTopology(py=1, px=px, shot_groups=1, world_size=world, rank=rank)
    mesh = ModelParallelMesh(grid=(1, px))

    x0 = rank * nxp
    vp_tile = vp[..., x0:x0 + nxp].copy()
    owns_src = x0 <= src_g[0] < x0 + nxp
    if ndim == 2:
        t_src = (np.array([[[src_g[0] - x0, src_g[1]]]], dtype=np.int32)
                 if owns_src else np.array([[[1, 1]]], dtype=np.int32))
        rec_pt = lambda gx: [gx - x0, rec_z]  # noqa: E731
        dummy_rec = [1, 1]
    else:
        t_src = (np.array([[[src_g[0] - x0, src_g[1], src_g[2]]]], dtype=np.int32)
                 if owns_src else np.array([[[1, 1, 1]]], dtype=np.int32))
        rec_pt = lambda gx: [gx - x0, ny // 2, rec_z]  # noqa: E731
        dummy_rec = [1, 1, 1]
    t_wavelet = wavelet if owns_src else np.zeros_like(wavelet)
    own_rec = [gx for gx in rec_gx if x0 <= gx < x0 + nxp]
    own_idx = [rec_gx.index(gx) for gx in own_rec]
    t_rec = (np.array([[rec_pt(gx) for gx in own_rec]], dtype=np.int32)
             if own_rec else np.array([[dummy_rec]], dtype=np.int32))

    shape_tile = (nz, nxp) if ndim == 2 else (nz, ny, nxp)
    prop = make_prop(shape_tile, abcn, nt, dt, dev, topo=topo)
    runner, record = make_runner(prop, t_wavelet, t_src, t_rec, vp_tile, dev)

    fast_set = None
    if args.fast:
        from sweep.parallel.fast_halo import FastHaloSet
        fast_set = FastHaloSet(mesh, M, ("x",))

    lo, hi = pad, pad + nxp
    torch.cuda.synchronize()
    dist.barrier()
    t_comp = t_comm = 0.0
    t0 = time.perf_counter()
    with torch.no_grad():
        for it in range(nt):
            ta = time.perf_counter()
            runner.run_to(it + 1)
            torch.cuda.synchronize()
            tb = time.perf_counter()
            view = runner.u_now[..., lo - M: hi + M]
            if fast_set is not None:
                fast_set.exchange(view)
            else:
                exchange_halos([view], mesh, M, ("x",))
            torch.cuda.synchronize()
            tc = time.perf_counter()
            t_comp += tb - ta
            t_comm += tc - tb
    total = time.perf_counter() - t0

    # ---- gather records + final physical tiles on rank 0 ----
    rec_cpu = record.cpu()
    u_cpu = runner.u_now[..., lo:hi].cpu()
    gathered = [None] * world
    dist.gather_object((own_idx, rec_cpu, u_cpu), gathered if rank == 0 else None, dst=0)

    if rank == 0:
        ref_src = np.array([[list(src_g)]], dtype=np.int32)
        if ndim == 2:
            ref_rec = np.array([[[gx, rec_z] for gx in rec_gx]], dtype=np.int32)
            shape_full = (nz, nx)
        else:
            ref_rec = np.array([[[gx, ny // 2, rec_z] for gx in rec_gx]],
                               dtype=np.int32)
            shape_full = (nz, ny, nx)
        prop_full = make_prop(shape_full, abcn, nt, dt, dev)
        runner_full, record_full = make_runner(
            prop_full, wavelet, ref_src, ref_rec, vp, dev
        )
        with torch.no_grad():
            runner_full.run_to(nt)
        ref_rec_cpu = record_full.cpu()
        ref_u = runner_full.u_now.cpu()

        # Cross-GPU runs may differ at ulp level even for identical inputs
        # (see dd_cross_gpu_diag.py); grade bit / tol / fail accordingly.
        REL_TOL = 1e-5
        u_scale = ref_u.abs().max().item() + 1e-30
        rec_all = torch.zeros_like(ref_rec_cpu)
        n_bit, ok = 0, True
        for r, (idxs, rc, uc) in enumerate(gathered):
            for j, gi in enumerate(idxs):
                rec_all[:, gi] = rc[:, j]
            ref_tile = ref_u[..., pad + r * nxp: pad + r * nxp + nxp]
            bit = torch.equal(uc, ref_tile)
            mad = (uc - ref_tile).abs().max().item()
            print(f"[rank0] tile {r}: u_now bitexact={bit} max|diff|={mad:.3e} "
                  f"rel={mad / u_scale:.3e}")
            n_bit += bit
            ok &= bit or (mad / u_scale < REL_TOL)
        bit_rec = torch.equal(rec_all, ref_rec_cpu)
        mad_rec = (rec_all - ref_rec_cpu).abs().max().item()
        rec_scale = ref_rec_cpu.abs().max().item() + 1e-30
        print(f"[rank0] record: bitexact={bit_rec} max|diff|={mad_rec:.3e} "
              f"rel={mad_rec / rec_scale:.3e}")
        ok &= bit_rec or (mad_rec / rec_scale < REL_TOL)
        all_bit = bit_rec and n_bit == world
        grid_str = f"{nz}x{nx}" if ndim == 2 else f"{nz}x{ny}x{nx}"
        print(f"[rank0] timing: total={total:.3f}s compute={t_comp:.3f}s "
              f"exchange={t_comm:.3f}s ({100 * t_comm / total:.1f}%) "
              f"nt={nt} grid={grid_str} px={px}")
        print("DD_NCCL_CHECK:",
              "PASS" if all_bit else ("PASS_TOL" if ok else "FAIL"))
        if not ok:
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
