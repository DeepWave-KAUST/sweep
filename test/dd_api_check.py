"""DDPropagator (one-call multi-GPU DD API) vs single-domain reference.

torchrun --standalone --nproc-per-node=<P> test/dd_api_check.py \
    [--family acoustic|elastic] [--ndim 2|3] [--so 4] [--abcn 10] [--free-surface]

Each rank drives the global problem through DDPropagator (which auto-slices the
tile, fills the model halo, remaps src/rec, and runs the per-step halo loop).
Rank 0 also builds the full single-domain reference (stepped forward record +
monolithic backward_bs grads_out, cropped to the physical interior) and
compares the assembled record and each tile's model gradient.

Grading: PASS (bitwise) / PASS_TOL (rel <= 1e-5) / FAIL.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sweep.equations import Acoustic, Acoustic3D, Elastic, Elastic3D  # noqa: E402
from sweep.parallel import DDPropagator, MeshTopology  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402
from sweep.propagator._stepped import (  # noqa: E402
    SteppedBackwardRunner, SteppedBindingRunner,
    acoustic_adj_pairs, acoustic_psi_pairs,
)

REL_TOL = 1e-5
DT = 0.0015


def ricker(nt, dt, fm=10.0, delay=0.06, scale=1.0):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    arg = np.pi * fm * t
    return (scale * (1.0 - 2.0 * arg**2) * np.exp(-(arg**2))).astype(np.float32)


def eq_cls(family, ndim):
    return {("acoustic", 2): Acoustic, ("acoustic", 3): Acoustic3D,
            ("elastic", 2): Elastic, ("elastic", 3): Elastic3D}[(family, ndim)]


def types(family, ndim):
    if family == "acoustic":
        return ["h1"], ["h1"]
    s = ["sxx", "szz"] if ndim == 2 else ["sxx", "syy", "szz"]
    r = ["vx", "vz"] if ndim == 2 else ["vx", "vy", "vz"]
    return s, r


def reference(family, ndim, shape, so, abcn, fs, nt, dev, models_np, src, rec, wav):
    """Single-domain stepped forward record + monolithic grads_out (raw,
    runtime), plus interior-crop bounds."""
    st, rt = types(family, ndim)
    pml = "cpmls" if family == "elastic" else "cpmlr"
    prop = PropTorch(eq_cls(family, ndim)(spatial_order=so, device=dev, backend="torch"),
                     backend="torch", impl="c", shape=shape, dev=dev, dh=10.0, dt=DT,
                     source_type=st, receiver_type=rt, abcn=abcn, free_surface=fs,
                     pml_type=pml, nt=nt, B=1, use_ckpt=False,
                     boundary_saving_config={"enabled": True, "storage": "gpu",
                                             "transfer_interval": 1, "pinned_memory": False})
    impl = prop._backend_impl
    cap = {}
    fo, bo = impl.forward_func, impl.backward_bs_func
    impl.forward_func = lambda p: (cap.__setitem__("fp", p) or cap.__setitem__("fr", fo(p)) or cap["fr"])
    impl.backward_bs_func = lambda p: (cap.__setitem__("bp", p) or bo(p))
    models = [torch.tensor(m, device=dev, requires_grad=True) for m in models_np]
    syn = prop(wav, src, rec, models=models)
    (syn[0] if isinstance(syn, (tuple, list)) else syn).sum().backward()
    impl.forward_func, impl.backward_bs_func = fo, bo

    fp, bp = cap["fp"], cap["bp"]
    nwf = len(fp.wavefields) or len(list(bp.adjoint_wavefields))
    Lf = list(fp.wavefields) or [torch.zeros_like(fp.models[0]) for _ in range(nwf)]
    record = torch.zeros_like(cap["fr"][2]); fp.record_out = record
    for t in Lf:
        t.zero_()
    if family == "acoustic":
        r = SteppedBindingRunner(fo, fp, Lf, acoustic_psi_pairs(ndim))
    else:
        r = SteppedBindingRunner(fo, fp, Lf, psi_pairs=(), u_blocks=())
    with torch.no_grad():
        r.run_to(nt)

    # monolithic backward_bs with grads_out (zero the captured adjoint list —
    # it carries stale values from the capture's public backward)
    L_adj = list(bp.adjoint_wavefields) or [torch.zeros_like(bp.models[0]) for _ in range(nwf)]
    for t in L_adj:
        t.zero_()
    nrecon = 3 if family == "acoustic" else (7 if ndim == 2 else 12)
    recon = [torch.zeros_like(bp.models[0]) for _ in range(nrecon)]
    if family == "acoustic":
        gbufs = [torch.zeros_like(bp.forward_source)] + [torch.zeros_like(m) for m in bp.models]
        illum = [torch.zeros_like(bp.models[0]), torch.zeros_like(bp.models[0])]
    else:
        gbufs = [torch.zeros_like(m) for m in bp.models]
        illum = []
    bp.grads_out = gbufs; bp.illum_out = illum
    bp.adjoint_source = record.clone()
    bp.boundary_gpu = list(fp.boundary_gpu); bp.u_last_two = fp.last_two
    bp.adjoint_wavefields = L_adj; bp.forward_wavefields = recon
    bp.bw_it_begin, bp.bw_it_end, bp.step_phase, bp.cut_face_mask = -1, 0, 0, 0
    with torch.no_grad():
        bo(bp)
    model_g = gbufs[1:] if family == "acoustic" else gbufs
    return record, model_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="acoustic", choices=("acoustic", "elastic"))
    ap.add_argument("--ndim", type=int, default=2, choices=(2, 3))
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--abcn", type=int, default=10)
    ap.add_argument("--nt", type=int, default=60)
    ap.add_argument("--px", type=int, default=0, help="x tiles (default: world)")
    ap.add_argument("--py", type=int, default=1, help="y tiles (3-D only, 2x2 = px=py=2)")
    ap.add_argument("--free-surface", action="store_true")
    args = ap.parse_args()
    fam, ndim, so, abcn, nt, fs = (args.family, args.ndim, args.so, args.abcn,
                                   args.nt, args.free_surface)
    M = so // 2
    pad = abcn + M

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li); dev = torch.device(f"cuda:{li}")

    px = args.px or world
    py = args.py
    assert px * py == world, f"px*py={px * py} != world={world}"
    nz, ny = 48, 20
    nxp = 28
    nx = nxp * px
    shape = (nz, nx) if ndim == 2 else (nz, ny, nx)
    scale = 1.0e6 if fam == "elastic" else 1.0
    n = int(np.prod(shape))
    grid = np.linspace(0, 1, n, dtype=np.float32).reshape(shape)
    if fam == "acoustic":
        models_np = [1800.0 + 600.0 * grid]
    else:
        models_np = [2200.0 + 400.0 * grid, 1200.0 + 200.0 * grid, 2000.0 + 100.0 * grid]
    wav = ricker(nt, DT, scale=scale)
    if ndim == 2:
        src = np.array([[[nx // 2, nz // 4]]], dtype=np.int32)
        rec = np.array([[[ix, 2] for ix in range(2, nx - 2, 5)]], dtype=np.int32)
    else:
        src = np.array([[[nx // 2, ny // 2, nz // 4]]], dtype=np.int32)
        rec = np.array([[[ix, ny // 2, 2] for ix in range(2, nx - 2, 5)]], dtype=np.int32)

    topo = MeshTopology(py=py, px=px, shot_groups=1, world_size=world, rank=rank)
    st, rt = types(fam, ndim)
    ddp = DDPropagator(eq_cls(fam, ndim)(spatial_order=so, device=dev, backend="torch"),
                       shape, dh=10.0, dt=DT, nt=nt, abcn=abcn, spatial_order=so,
                       source_type=st, receiver_type=rt, model_parallel=topo, dev=dev,
                       free_surface=fs)
    rec_tile = ddp.forward(wav, src, rec, models=models_np)   # pass GLOBAL, auto-slice
    grads_tile = ddp.gradient(rec_tile)                       # adjoint = tile record
    full_rec = ddp.gather_record(rec_tile)

    y0 = getattr(ddp, "y0", 0); nyp = getattr(ddp, "nyp", ny)
    payload = (ddp._own_rec_idx, ddp.x0, ddp.nxp, y0, nyp, [g.cpu() for g in grads_tile])
    gathered = [None] * world
    dist.gather_object(payload, gathered if rank == 0 else None, dst=0)

    if rank == 0:
        ref_rec, ref_g = reference(fam, ndim, shape, so, abcn, fs, nt, dev,
                                   models_np, src, rec, wav)
        ref_rec = ref_rec.cpu()
        ztop = M if fs else pad

        # grade rel against the GLOBAL reference scale, not the per-tile max:
        # far-field tiles have gradient ~1e-20, and a per-tile denominator makes
        # rel blow up on machine-zero diffs (the classic near-zero metric trap).
        def grade(name, got, want, scale):
            bit = torch.equal(got, want)
            mad = (got - want).abs().max().item()
            rel = mad / (scale + 1e-30)
            print(f"[rank0] {name}: bit={bit} max|d|={mad:.3e} rel={rel:.3e}")
            return 2 if bit else (1 if rel < REL_TOL else 0)

        rec_scale = ref_rec.abs().max().item()
        g_scale = [g.abs().max().item() for g in ref_g]
        worst = grade("record", full_rec.cpu(), ref_rec, rec_scale)
        for r, (idx, x0, nxp_r, y0, nyp_r, g_list) in enumerate(gathered):
            for k, g in enumerate(g_list):
                if ndim == 2:
                    want = ref_g[k][..., ztop:ztop + nz, pad + x0: pad + x0 + nxp_r]
                else:
                    want = ref_g[k][..., ztop:ztop + nz, pad + y0: pad + y0 + nyp_r,
                                    pad + x0: pad + x0 + nxp_r]
                worst = min(worst, grade(f"tile{r} grad[{k}]", g, want.cpu(), g_scale[k]))
        print(f"[rank0] family={fam} ndim={ndim} grid={shape} px={px} py={py} "
              f"so={so} fs={fs} nt={nt}")
        print("DD_API_CHECK:", {2: "PASS", 1: "PASS_TOL", 0: "FAIL"}[worst])
        if worst == 0:
            sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
