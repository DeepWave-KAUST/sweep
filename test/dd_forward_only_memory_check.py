"""A forward-only DD caller must not pay for the adjoint machinery.

``no_grad`` means ``no_grad``: a caller who never asks for a gradient — the
"generate observed data" loop that opens every FWI script — should not be
holding the adjoint wavefields, the reconstruction/gradient/illumination
buffers, or (the big one) the boundary-saving ring, which is sized by ``nt``
and dominates a real DD run.

Checks, on a real 2-tile cut:

* a first call with no grad-requiring model works and leaves NO adjoint-side
  allocation: ``bp`` unset, and the adjoint buffer set empty;
* the boundary ring is likewise absent on such an instance;
* asking for a gradient afterwards still works and is BIT-EXACT against an
  instance that was gradient-capable from its first call — deferring the
  adjoint capture must not change a single bit;
* the forward-only instance holds strictly less GPU memory than the
  gradient-capable one.

The third check is the one that keeps the optimisation honest: it is easy to
save memory by capturing something degraded.

Launch::

    torchrun --standalone --nproc-per-node=2 test/dd_forward_only_memory_check.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sweep.equations import Acoustic3D                    # noqa: E402
from sweep.parallel import MeshTopology                   # noqa: E402
from sweep.parallel.dd_propagator import ModelParallel     # noqa: E402
from sweep.propagator.torch import PropTorch              # noqa: E402

DT, DH = 0.0012, 15.0
MiB = 2 ** 20


def ricker(nt, dt, fm=6.0, delay=0.2):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a ** 2) * np.exp(-(a ** 2))).astype(np.float32)


def nbytes(x):
    if torch.is_tensor(x):
        return x.numel() * x.element_size()
    if isinstance(x, (list, tuple)):
        return sum(nbytes(i) for i in x)
    return 0


def adjoint_bytes(ddp):
    """Everything only ``_run_adjoint`` ever reads, plus the C-side set."""
    total = sum(nbytes(getattr(ddp, a, None))
                for a in ("L_adj", "recon", "gbufs", "illum",
                          "coupling", "adj_coeffs"))
    bp = getattr(ddp, "bp", None)
    if bp is not None:
        total += nbytes(getattr(bp, "adjoint_wavefields", None))
    return total


def ring_bytes(ddp):
    """The nt-scaled boundary ring — the dominant term in a real DD run."""
    fp = getattr(ddp, "fp", None)
    if fp is None:
        return 0
    return sum(nbytes(getattr(fp, a, None))
               for a in ("boundary_gpu", "boundary_gpu_full", "last_two"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py", type=int, default=1)
    ap.add_argument("--px", type=int, default=2)
    ap.add_argument("--nt", type=int, default=400)
    ap.add_argument("--abcn", type=int, default=20)
    ap.add_argument("--so", type=int, default=4)
    ap.add_argument("--nz", type=int, default=64)
    ap.add_argument("--ny", type=int, default=64)
    ap.add_argument("--nx", type=int, default=96)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")
    assert args.py * args.px == world, f"py*px must equal world size {world}"
    log = print if rank == 0 else (lambda *a, **k: None)

    nz, ny, nx, nt = args.nz, args.ny, args.nx, args.nt
    shape = (nz, ny, nx)
    z = np.arange(nz, dtype=np.float32)[:, None, None]
    vp = np.broadcast_to(1800.0 + 12.0 * z, shape).astype(np.float32).copy()
    vp[30:36] += 400.0

    wav = torch.as_tensor(ricker(nt, DT), device=dev)
    src = np.array([[[nx // 2, ny // 2, 6]]], dtype=np.int64)
    rec = np.array([[[ix, ny // 2, 2] for ix in range(4, nx - 4, 2)]],
                   dtype=np.int64)

    def build():
        eq = Acoustic3D(spatial_order=args.so, device=dev, backend="torch")
        prop = PropTorch(eq, backend="torch", impl="c", shape=shape, dh=DH,
                         dt=DT, nt=nt, abcn=args.abcn, source_type=["h1"],
                         receiver_type=["h1"], dev=dev, free_surface=True)
        mesh = MeshTopology(py=args.py, px=args.px, shot_groups=1,
                            world_size=world, rank=rank)
        return ModelParallel(prop, mesh)

    # ---- reference: gradient-capable from its very first call -------------
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    ref = build()
    v_ref = torch.tensor(vp, device=dev, requires_grad=True)
    r_ref = ref(wav, src, rec, models=[v_ref]).detach().clone()
    (r_ref.double() ** 2).sum()          # keep r_ref alive, no backward yet
    v_b = torch.tensor(vp, device=dev, requires_grad=True)
    (ref(wav, src, rec, models=[v_b]).double() ** 2).sum().backward()
    torch.cuda.synchronize()
    ref_held = torch.cuda.memory_allocated() - base

    # ---- forward-only: no model requires grad, caller wants no gradient ---
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    base2 = torch.cuda.memory_allocated()
    fwd = build()
    with torch.no_grad():
        r_fwd = fwd(wav, src, rec, models=[torch.tensor(vp, device=dev)]).clone()
    torch.cuda.synchronize()
    fwd_held = torch.cuda.memory_allocated() - base2

    adj_b, ring_b = adjoint_bytes(fwd), ring_bytes(fwd)
    had_bp = getattr(fwd, "bp", None) is not None   # snapshot: promotion sets it
    live = float(r_ref.abs().max()) > 0.0        # comparisons must not be vacuous
    rec_bit = bool(torch.equal(r_fwd, r_ref))
    no_adjoint = (getattr(fwd, "bp", None) is None) and adj_b == 0
    no_ring = ring_b == 0
    lighter = fwd_held < ref_held

    # ---- promote: ask the forward-only instance for a gradient ------------
    v_p = torch.tensor(vp, device=dev, requires_grad=True)
    (fwd(wav, src, rec, models=[v_p]).double() ** 2).sum().backward()
    grad_bit = bool(torch.equal(v_p.grad, v_b.grad))

    # ---- ...and against a genuine SINGLE-DOMAIN run -----------------------
    # The DD-vs-DD comparison above only proves the promoted instance matches
    # another ModelParallel. Every existing DD-vs-single-domain check takes the
    # grad-from-the-first-call path, so without this the promotion path would
    # rest on a transitive argument (promoted == grad-first, grad-first ==
    # single) chained across two different test setups rather than measured.
    # Each rank's DD gradient is a partial over its owned receivers, so
    # all_reduce assembles the global gradient the single domain computes.
    eq_s = Acoustic3D(spatial_order=args.so, device=dev, backend="torch")
    prop_single = PropTorch(eq_s, backend="torch", impl="c", shape=shape, dh=DH,
                            dt=DT, nt=nt, abcn=args.abcn, source_type=["h1"],
                            receiver_type=["h1"], dev=dev, free_surface=True)
    v_s = torch.tensor(vp, device=dev, requires_grad=True)
    (prop_single(wav, src, rec, models=[v_s]).double() ** 2).sum().backward()

    g_dd = v_p.grad.clone()
    dist.all_reduce(g_dd)
    vs_single = bool(torch.equal(g_dd, v_s.grad))

    ok = (live and rec_bit and no_adjoint and no_ring and lighter and grad_bit
          and vs_single)
    if rank == 0:
        print(f"mesh py{args.py}xpx{args.px}  shape={shape}  nt={nt}  so={args.so}")
        print(f"reference record carries signal:      {live}"
              f"  (max |rec| = {float(r_ref.abs().max()):.3e})")
        print(f"forward-only record bit-exact:        {rec_bit}")
        print(f"no adjoint buffers after fwd-only:    {no_adjoint}"
              f"  (bp={had_bp}, adjoint={adj_b / MiB:.1f} MiB)")
        print(f"no boundary ring after fwd-only:      {no_ring}"
              f"  (ring={ring_b / MiB:.1f} MiB)")
        print(f"forward-only holds less memory:       {lighter}"
              f"  ({fwd_held / MiB:.1f} vs {ref_held / MiB:.1f} MiB)")
        print(f"gradient after promotion bit-exact:   {grad_bit}")
        print(f"promoted DD == SINGLE DOMAIN:         {vs_single}")
        print(f"-> {'PASS' if ok else 'FAIL'}")

    fail = torch.tensor([0 if ok else 1], device=dev)
    dist.all_reduce(fail)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(int(fail.item() > 0))


if __name__ == "__main__":
    main()
