"""Multi-GPU NCCL DD BACKWARD vs single-domain reference (end-to-end).

torchrun --standalone --nproc-per-node=<P> test/dd_nccl_backward_check.py

Each rank owns one x-tile: DD forward (boundary saving on, per-step u_now
NCCL halo exchange) then DD backward (per-step descending segments with
cut_face_mask, per-step lambda + recon u_now NCCL exchanges — the protocol
proven bitwise on one GPU in test_dd_backward_two_tile.py). Rank 0 runs
the single-domain reference forward+backward on its own GPU and compares
the assembled grad_vp / grad_wavelet / illuminations.

Grading: PASS (bitwise) / PASS_TOL (rel <= 1e-5) / FAIL.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(SRC_ROOT), str(REPO_ROOT / "test")):
    if p not in sys.path:
        sys.path.insert(0, p)

from sweep.equations import Acoustic  # noqa: E402
from sweep.parallel import MeshTopology, ModelParallelMesh  # noqa: E402
from sweep.parallel.fast_halo import FastHaloSet  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from test_dd_backward_two_tile import TileState, capture_both  # noqa: E402

NZ = 48
NT = 120
DT = 0.0015
SO = 4
M = SO // 2
ABCN = 10
PAD = ABCN + M
SRC_GZ = NZ // 4
REC_GZ = 2
X_LO_BIT, X_HI_BIT = 1, 2
REL_TOL = 1e-5


def ricker(nt, dt, fm=10.0, delay=0.06):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return ((1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def make_prop(shape, dev, topo=None):
    eq = Acoustic(spatial_order=SO, device=dev, backend="torch")
    kw = dict(
        backend="torch", impl="c", shape=shape, dev=dev, dh=10.0, dt=DT,
        source_type=["h1"], receiver_type=["h1"], abcn=ABCN,
        free_surface=False, pml_type="cpmlr", nt=NT, B=1, use_ckpt=False,
        boundary_saving_config={"enabled": True, "storage": "gpu",
                                "transfer_interval": 1,
                                "pinned_memory": False},
    )
    if topo is not None:
        kw["model_parallel"] = topo
    return PropTorch(eq, **kw)


def run_public_once(prop, wavelet, sources, receivers, vp_np, dev):
    models = [torch.tensor(vp_np, device=dev, requires_grad=True)]
    rec = prop(wavelet, sources, receivers, models=models)
    rec.pow(2).mean().backward()


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")

    px = world
    nxp = 28
    nx = nxp * px
    src_gx = nx // 2 if px > 1 else nxp // 2  # on a cut for even px
    rec_gx = list(range(2, nx - 2, 6))

    vp = 1800.0 + 600.0 * np.linspace(0, 1, NZ, dtype=np.float32)[:, None]
    vp = np.broadcast_to(vp, (NZ, nx)).copy()
    vp[20:30, nx // 3: 2 * nx // 3] += 180.0
    vp_padded = np.pad(vp, PAD, mode="edge")  # global run-grid model
    wavelet = ricker(NT, DT)

    topo = MeshTopology(py=1, px=px, shot_groups=1, world_size=world, rank=rank)
    mesh = ModelParallelMesh(grid=(1, px)) if world > 1 else None

    # ---------------- tile setup (capture via one public run) ----------------
    x0 = rank * nxp
    vp_tile = vp[:, x0:x0 + nxp].copy()
    owns_src = x0 <= src_gx < x0 + nxp
    t_src = (np.array([[[src_gx - x0, SRC_GZ]]], dtype=np.int32)
             if owns_src else np.array([[[1, 1]]], dtype=np.int32))
    t_wav = wavelet if owns_src else np.zeros_like(wavelet)
    own_rec = [gx for gx in rec_gx if x0 <= gx < x0 + nxp]
    own_idx = [rec_gx.index(gx) for gx in own_rec]
    t_rec = np.array([[[gx - x0, REC_GZ] for gx in own_rec]], dtype=np.int32)

    prop = make_prop((NZ, nxp), dev, topo=topo)
    cap = capture_both(prop)
    run_public_once(prop, t_wav, t_src, t_rec, vp_tile, dev)
    tile = TileState(cap)
    if world > 1:
        tile.cut_face_mask = ((X_LO_BIT if rank > 0 else 0)
                              | (X_HI_BIT if rank < world - 1 else 0))
    else:
        tile.cut_face_mask = 0

    # Physical-region start in THIS tile's runtime buffer = x_lo PML width +
    # halo M.  The P-series compact pad gives cut (neighbour-facing) faces 0
    # PML width, so an interior tile starts at M, not the symmetric PAD=abcn+M
    # — read the real per-rank pad off the prop instead of assuming the edge
    # tile's value (the old ``lo = PAD`` mis-indexed every non-edge tile).
    xlo_pml = prop._backend_impl.padding[0]
    lo, hi = xlo_pml + M, xlo_pml + M + nxp

    # spec sharp edge: the fused adjoint reads vp at stencil taps — the
    # tile model's pad must carry the GLOBAL model, not edge replication.
    # vp_padded is the global model symmetric-padded by PAD per side, so
    # global physical column g lives at vp_padded[:, PAD + g].  The tile
    # buffer's physical region begins at ``lo`` (= xlo_pml + M), so tile
    # runtime ix=0 maps to global padded column (PAD + x0) - lo.  The copy
    # must fill the tile buffer's REAL runtime width (asymmetric under the
    # compact pad), read off the actual buffer .shape rather than nxp+2*PAD.
    w = tile.fp.models[0].shape[-1]
    start = (PAD + x0) - lo
    gsl = torch.tensor(
        vp_padded[:, start:start + w], device=dev
    )
    tile.fp.models[0].copy_(gsl)
    tile.bp.models[0].copy_(gsl)

    # ---------------- reference (rank 0) + shared residual ----------------
    if rank == 0:
        ref_src = np.array([[[src_gx, SRC_GZ]]], dtype=np.int32)
        ref_rec = np.array([[[gx, REC_GZ] for gx in rec_gx]], dtype=np.int32)
        ref_prop = make_prop((NZ, nx), dev)
        ref_cap = capture_both(ref_prop)
        run_public_once(ref_prop, wavelet, ref_src, ref_rec, vp, dev)
        ref = TileState(ref_cap)
        rr = ref.fwd_runner()
        with torch.no_grad():
            rr.run_to(NT)
        residual = ref.record.clone()           # (1, nrec, NT) raw layout
        ref.bp.boundary_gpu = list(ref.fp.boundary_gpu)
        ref.bp.u_last_two = ref.fp.last_two
    else:
        residual = torch.zeros((1, len(rec_gx), NT), device=dev)
    if world > 1:
        dist.broadcast(residual, src=0)

    adj = torch.zeros_like(tile.bp.adjoint_source)
    for j, gi in enumerate(own_idx):
        adj[:, j] = residual[:, gi]
    tile.bp.adjoint_source = adj

    # ---------------- DD forward (NCCL u_now exchange) ----------------
    # lo/hi are the cut-aware tile interior bounds computed above (lo = real
    # x_lo PML + M); they drive the halo views and the gather crops below.
    # The acoustic forward in_pml split is cut-aware (phys_x0 = cut? M : abcn+M),
    # so the forward params need this tile's cut_face_mask too — else the
    # asymmetric (compact-pad) interior tile takes the wrong PML branch (P6).
    tile.fp.cut_face_mask = tile.cut_face_mask
    fwd_halo = FastHaloSet(mesh, M, ("x",)) if world > 1 else None
    fr = tile.fwd_runner()
    with torch.no_grad():
        for it in range(NT):
            fr.run_to(it + 1)
            if world > 1:
                fwd_halo.exchange(fr.u_now[..., lo - M: hi + M])
    tile.bp.boundary_gpu = list(tile.fp.boundary_gpu)
    tile.bp.u_last_two = tile.fp.last_two

    # ---------------- DD backward (NCCL lambda + recon exchanges) ----------
    tile.zero_backward_state()
    tile.bp.cut_face_mask = tile.cut_face_mask
    bwd_halo = FastHaloSet(mesh, M, ("x",)) if world > 1 else None
    br = tile.bwd_runner()
    with torch.no_grad():
        for it in range(NT - 1, -1, -1):
            br.run_segment(it + 1, it)
            if it == 0:
                break  # adjoint-only tail consumes no halos
            if world > 1:
                bwd_halo.exchange(br.lambda_now[..., lo - M: hi + M])
                bwd_halo.exchange(br.recon_u_now[..., lo - M: hi + M])
    assert br.k_adj == NT and br.k_f == NT - 1

    # ---------------- gather + compare on rank 0 ----------------
    payload = (
        own_idx,
        tile.gbufs[0].cpu(),                      # grad_wavelet (1, nsrc, nt)
        tile.gbufs[1][..., lo:hi].cpu(),          # grad_vp owned slice
        [b[..., lo:hi].cpu() for b in tile.ibufs],
        bool(owns_src),
    )
    gathered = [None] * world
    if world > 1:
        dist.gather_object(payload, gathered if rank == 0 else None, dst=0)
    else:
        gathered = [payload]

    if rank == 0:
        with torch.no_grad():
            bb = ref.bwd_runner()
            ref.zero_backward_state()
            ref.bp.adjoint_source = residual
            bb.run_segment(NT, 0)                  # monolithic reference

        def grade(name, got, want):
            bit = torch.equal(got, want)
            mad = (got - want).abs().max().item()
            scale = want.abs().max().item() + 1e-30
            rel = mad / scale
            print(f"[rank0] {name}: bitexact={bit} max|d|={mad:.3e} rel={rel:.3e}")
            return 2 if bit else (1 if rel < REL_TOL else 0)

        worst = 2
        for r, (idxs, gw, gvp, ill, has_src) in enumerate(gathered):
            want_vp = ref.gbufs[1][..., PAD + r * nxp: PAD + r * nxp + nxp].cpu()
            worst = min(worst, grade(f"tile{r} grad_vp", gvp, want_vp))
            for k, name in enumerate(("src_illum", "rec_illum")):
                want_i = ref.ibufs[k][..., PAD + r * nxp: PAD + r * nxp + nxp].cpu()
                worst = min(worst, grade(f"tile{r} {name}", ill[k], want_i))
            if has_src:
                worst = min(worst, grade(f"tile{r} grad_wavelet", gw,
                                         ref.gbufs[0].cpu()))
        verdict = {2: "PASS", 1: "PASS_TOL", 0: "FAIL"}[worst]
        print(f"DD_NCCL_BACKWARD_CHECK: {verdict}")
        if worst == 0:
            sys.exit(1)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
