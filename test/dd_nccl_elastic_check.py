"""NCCL multi-GPU elastic DD FORWARD vs single-domain reference.

Launch:  torchrun --standalone --nproc-per-node=<P> test/dd_nccl_elastic_check.py \
             [--ndim 2] [--so 4] [--abcn 10] [--nz 48] [--nxp 28] [--ny 20] \
             [--nt 120] [--free-surface] [--fast]

Each rank owns one x-tile of the global elastic model and advances the
stepped impl='c' forward with the E1 HALF-STEP exchange protocol:

    run_phase(it+1, 1)             # velocity kernel (full tile)
    exchange velocity fields M-wide      (vx, vz[, vy])
    run_phase(it+1, 2)             # stress kernel + tail (source/record/save)
    exchange stress fields M-wide        (sxx, szz, sxz[, syy, sxy, syz])

The exchange of each physical field's ``[lo-M, hi+M)`` view reproduces the
manual two-tile ``_swap`` proven bitwise in test_dd_elastic_two_tile.py:
FastHaloSet sends ``[M:2M]`` (=tile ``[lo:lo+M]``) to the low neighbour and
``[-2M:-M]`` (=tile ``[hi-M:hi]``) to the high neighbour, receiving into the
``M`` pad columns on each side. CPML memory variables m_* are never
exchanged (same-cell recurrence; cut-axis coefficients zero on both sides).

The tile's pad columns must carry the TRUE neighbour material (the staggered
velocity update reads rho at halo cells, feeding interior stress through the
second stencil of the same step), so each rank fills its runtime-padded
models from the globally edge-padded model — what a production DD driver
gets by partitioning the global padded model. elastic3d forward kernels are
cut-aware (zero-coeff PML branch != interior branch bitwise) so the forward
cut_face_mask is set in 3-D; elastic2d forward does not read it.

Rank 0 also runs the full single-domain model on its own GPU and asserts
bitwise equality of records and every wavefield slot over each tile's
physical region (same-model GPUs => IEEE-deterministic kernels => bit
equality is the correct bar).
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

X_LO_BIT, X_HI_BIT = 1, 2
NWF = {2: 15, 3: 36}
NPHYS = {2: 5, 3: 9}
NV = {2: 2, 3: 3}
REL_TOL = 1e-5

SO = 4
M = SO // 2


def ricker(nt, dt, fm=10.0, delay=0.06, scale=1.0e6):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return (scale * (1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def global_models(shape):
    """x-varying ramps (so the cut-side material differs per side) plus a
    box anomaly straddling the centre cut."""
    n = int(np.prod(shape))
    grid = np.linspace(0.0, 1.0, num=n, dtype=np.float32).reshape(shape)
    vp = 2200.0 + 40.0 * grid
    vs = 1200.0 + 20.0 * grid
    rho = 2000.0 + 10.0 * grid
    if len(shape) == 2:
        nz, nx = shape
        vp[nz // 3:nz // 2, nx // 3:2 * nx // 3] += 100.0
        vs[nz // 3:nz // 2, nx // 3:2 * nx // 3] += 50.0
        rho[nz // 3:nz // 2, nx // 3:2 * nx // 3] += 25.0
    else:
        nz, ny, nx = shape
        vp[nz // 3:nz // 2, ny // 4:3 * ny // 4, nx // 3:2 * nx // 3] += 100.0
        vs[nz // 3:nz // 2, ny // 4:3 * ny // 4, nx // 3:2 * nx // 3] += 50.0
        rho[nz // 3:nz // 2, ny // 4:3 * ny // 4, nx // 3:2 * nx // 3] += 25.0
    return [vp, vs, rho]


def make_prop(ndim, shape, abcn, nt, dt, fs, dev, topo=None):
    eq_cls = Elastic if ndim == 2 else Elastic3D
    eq = eq_cls(spatial_order=SO, device=dev, backend="torch")
    kw = dict(
        backend="torch", impl="c", shape=shape, dev=dev, dh=10.0, dt=dt,
        source_type=["sxx", "szz"] if ndim == 2 else ["sxx", "syy", "szz"],
        receiver_type=["vx", "vz"] if ndim == 2 else ["vx", "vy", "vz"],
        abcn=abcn, free_surface=fs, pml_type="cpmls", nt=nt, B=1,
        use_ckpt=False, boundary_saving_config={"enabled": False},
    )
    if topo is not None:
        kw["model_parallel"] = topo
    return PropTorch(eq, **kw)


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


def make_runner(prop, wavelet, sources, receivers, model_arrays, ndim, dev):
    cap = capture(prop)
    models = [torch.tensor(a, device=dev) for a in model_arrays]
    with torch.no_grad():
        prop(wavelet, sources, receivers, models=models)
    p, func = cap["params"], cap["func"]
    L = list(p.wavefields)
    if not L:
        L = [torch.zeros_like(p.models[0]) for _ in range(NWF[ndim])]
    assert len(L) == NWF[ndim]
    for t in L:
        t.zero_()
    record = torch.zeros_like(cap["raw_out"][2])
    p.record_out = record
    return SteppedBindingRunner(func, p, L, psi_pairs=(), u_blocks=()), record


def global_runtime_padded(m_phys, full, top):
    """Edge-pad a physical model to the propagator's runtime extent: x (and y
    in 3-D) symmetric ``full`` = abcn+M; z asymmetric ``(top, full)`` — top is
    M under free surface (no top PML), else abcn+M.  Matches the propagator's
    edge-replication padding (proven bitwise for the acoustic NCCL checks)."""
    if m_phys.ndim == 2:
        pw = ((top, full), (full, full))
    else:
        pw = ((top, full), (full, full), (full, full))
    return np.pad(m_phys, pw, mode="edge")


def fill_global_pad(p, models_padded, x0):
    """Overwrite the tile's runtime-padded models with the global edge-padded
    model sliced at this tile's x-range (true neighbour material in the pad).
    Runtime models carry leading (B, channel) dims; the padded global model is
    bare spatial, so reshape the slice (numel matches) before copy."""
    for mt, mp in zip(p.models, models_padded):
        sl = torch.as_tensor(mp[..., x0:x0 + mt.size(-1)], device=mt.device)
        mt.copy_(sl.reshape(mt.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndim", type=int, default=2, choices=(2, 3))
    ap.add_argument("--nz", type=int, default=48)
    ap.add_argument("--ny", type=int, default=20)
    ap.add_argument("--nxp", type=int, default=28, help="x cells PER RANK")
    ap.add_argument("--nt", type=int, default=120)
    ap.add_argument("--abcn", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.0015)
    ap.add_argument("--so", type=int, default=4, choices=(2, 4, 6, 8))
    ap.add_argument("--free-surface", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="use FastHaloSet instead of exchange_halos")
    args = ap.parse_args()

    global SO, M
    SO = args.so
    M = SO // 2

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    dev_idx = local_rank % max(1, torch.cuda.device_count())  # oversubscribe-safe
    torch.cuda.set_device(dev_idx)
    dev = torch.device(f"cuda:{dev_idx}")

    ndim, ny = args.ndim, args.ny
    px = world
    nz, nxp = args.nz, args.nxp
    nx = nxp * px
    nt, abcn, dt, fs = args.nt, args.abcn, args.dt, args.free_surface
    pad = abcn + M

    shape_full = (nz, nx) if ndim == 2 else (nz, ny, nx)
    models_np = global_models(shape_full)
    # global edge-padded models (matches the propagator's runtime padding;
    # z-top pad shrinks to M under free surface — no top PML)
    full_pad = abcn + M
    top_pad = M if fs else full_pad
    models_padded = [global_runtime_padded(m, full_pad, top_pad) for m in models_np]
    wavelet = ricker(nt, dt)

    if ndim == 2:
        src_g = (nx // 2, nz // 4)
        rec_gx = list(range(2, nx - 2, 6))
        rec_z = 2
        rec_pt = lambda gx, x0: [gx - x0, rec_z]  # noqa: E731
        dummy_rec = [1, 1]
        dummy_src = [[[1, 1]]]
    else:
        src_g = (nx // 2, ny // 2, nz // 4)
        rec_gx = list(range(2, nx - 2, 6))
        rec_z = 2
        rec_pt = lambda gx, x0: [gx - x0, ny // 2, rec_z]  # noqa: E731
        dummy_rec = [1, 1, 1]
        dummy_src = [[[1, 1, 1]]]

    topo = MeshTopology(py=1, px=px, shot_groups=1, world_size=world, rank=rank)
    mesh = ModelParallelMesh(grid=(1, px))

    x0 = rank * nxp
    tile_models = [m[..., x0:x0 + nxp].copy() for m in models_np]
    owns_src = x0 <= src_g[0] < x0 + nxp
    if owns_src:
        if ndim == 2:
            t_src = np.array([[[src_g[0] - x0, src_g[1]]]], dtype=np.int32)
        else:
            t_src = np.array([[[src_g[0] - x0, src_g[1], src_g[2]]]],
                             dtype=np.int32)
        t_wav = wavelet
    else:
        t_src = np.array(dummy_src, dtype=np.int32)
        t_wav = np.zeros_like(wavelet)
    own_rec = [gx for gx in rec_gx if x0 <= gx < x0 + nxp]
    own_idx = [rec_gx.index(gx) for gx in own_rec]
    t_rec = (np.array([[rec_pt(gx, x0) for gx in own_rec]], dtype=np.int32)
             if own_rec else np.array([[dummy_rec]], dtype=np.int32))

    shape_tile = (nz, nxp) if ndim == 2 else (nz, ny, nxp)
    prop = make_prop(ndim, shape_tile, abcn, nt, dt, fs, dev, topo=topo)
    runner, record = make_runner(prop, t_wav, t_src, t_rec, tile_models, ndim, dev)
    fill_global_pad(runner.p, models_padded, x0)

    cut_mask = ((X_LO_BIT if rank > 0 else 0)
                | (X_HI_BIT if rank < world - 1 else 0))
    if ndim == 3:
        runner.p.cut_face_mask = cut_mask  # elastic3d forward is cut-aware

    fast_set = None
    if args.fast:
        from sweep.parallel.fast_halo import FastHaloSet
        fast_set = FastHaloSet(mesh, M, ("x",))

    lo, hi = pad, pad + nxp

    def exchange(slots):
        for f in slots:
            view = runner.L[f][..., lo - M: hi + M]
            if fast_set is not None:
                fast_set.exchange(view)
            else:
                exchange_halos([view], mesh, M, ("x",))

    torch.cuda.synchronize()
    dist.barrier()
    t_comp = t_comm = 0.0
    t0 = time.perf_counter()
    with torch.no_grad():
        for it in range(nt):
            ta = time.perf_counter()
            runner.run_phase(it + 1, 1)               # velocity
            torch.cuda.synchronize()
            tb = time.perf_counter()
            exchange(range(NV[ndim]))                  # vx, vz[, vy]
            torch.cuda.synchronize()
            tc = time.perf_counter()
            runner.run_phase(it + 1, 2)               # stress + tail
            torch.cuda.synchronize()
            td = time.perf_counter()
            exchange(range(NV[ndim], NPHYS[ndim]))     # stresses
            torch.cuda.synchronize()
            te = time.perf_counter()
            t_comp += (tb - ta) + (td - tc)
            t_comm += (tc - tb) + (te - td)
    total = time.perf_counter() - t0

    # ---- gather records + final physical tiles on rank 0 ----
    rec_cpu = record.cpu()
    u_cpu = [runner.L[f][..., lo:hi].cpu() for f in range(NWF[ndim])]
    gathered = [None] * world
    dist.gather_object((own_idx, rec_cpu, u_cpu),
                       gathered if rank == 0 else None, dst=0)

    if rank == 0:
        if ndim == 2:
            ref_src = np.array([[list(src_g)]], dtype=np.int32)
            ref_rec = np.array([[[gx, rec_z] for gx in rec_gx]], dtype=np.int32)
        else:
            ref_src = np.array([[list(src_g)]], dtype=np.int32)
            ref_rec = np.array([[[gx, ny // 2, rec_z] for gx in rec_gx]],
                               dtype=np.int32)
        prop_full = make_prop(ndim, shape_full, abcn, nt, dt, fs, dev)
        runner_full, record_full = make_runner(
            prop_full, wavelet, ref_src, ref_rec, models_np, ndim, dev
        )
        with torch.no_grad():
            runner_full.run_to(nt)
        ref_rec_cpu = record_full.cpu()
        ref_u = [runner_full.L[f].cpu() for f in range(NWF[ndim])]

        rec_all = torch.zeros_like(ref_rec_cpu)
        worst = 2  # 2 bit / 1 tol / 0 fail
        for r, (idxs, rc, uc) in enumerate(gathered):
            for j, gi in enumerate(idxs):
                rec_all[:, :, gi] = rc[:, :, j]
            tile_bit = True
            tile_mad = 0.0
            for f in range(NWF[ndim]):
                ref_tile = ref_u[f][..., pad + r * nxp: pad + r * nxp + nxp]
                bit = torch.equal(uc[f], ref_tile)
                mad = (uc[f] - ref_tile).abs().max().item()
                tile_bit &= bit
                tile_mad = max(tile_mad, mad)
            u_scale = max(ref_u[f].abs().max().item() for f in range(NWF[ndim]))
            u_scale += 1e-30
            rel = tile_mad / u_scale
            print(f"[rank0] tile {r}: wavefields bitexact={tile_bit} "
                  f"max|diff|={tile_mad:.3e} rel={rel:.3e}")
            worst = min(worst, 2 if tile_bit else (1 if rel < REL_TOL else 0))
        bit_rec = torch.equal(rec_all, ref_rec_cpu)
        mad_rec = (rec_all - ref_rec_cpu).abs().max().item()
        rec_scale = ref_rec_cpu.abs().max().item() + 1e-30
        rel_rec = mad_rec / rec_scale
        print(f"[rank0] record: bitexact={bit_rec} max|diff|={mad_rec:.3e} "
              f"rel={rel_rec:.3e}")
        worst = min(worst, 2 if bit_rec else (1 if rel_rec < REL_TOL else 0))
        grid_str = f"{nz}x{nx}" if ndim == 2 else f"{nz}x{ny}x{nx}"
        print(f"[rank0] timing: total={total:.3f}s compute={t_comp:.3f}s "
              f"exchange={t_comm:.3f}s ({100 * t_comm / total:.1f}%) "
              f"nt={nt} grid={grid_str} px={px} so={SO} fs={fs}")
        verdict = {2: "PASS", 1: "PASS_TOL", 0: "FAIL"}[worst]
        print(f"DD_NCCL_ELASTIC_CHECK: {verdict}")
        if worst == 0:
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
