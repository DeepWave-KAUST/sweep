"""The FIRST call into ModelParallel must survive a caller's ``torch.no_grad()``.

Generating observed data is the textbook first call — no gradient wanted — and
users write it as ``with torch.no_grad(): ddp(...)``. That call triggers the
one-time ``_capture``, which probes the solver with a throwaway
forward+backward to grab the C param objects. ``backward()`` needs grad MODE,
which the caller just switched off, so before the fix this died with

    RuntimeError: element 0 of tensors does not require grad and does not have
    a grad_fn

(The autograd entry point ``_DDForward.forward`` already had the same guard —
``autograd.Function.forward`` also runs with grad disabled — so only the
non-autograd path was exposed.)

Checks, on a real 2-tile cut:

* a first call made inside ``no_grad`` succeeds;
* its record is BIT-EXACT against a fresh ``ModelParallel`` whose first call was
  grad-enabled — the fix must not perturb what gets captured;
* the gradient still works afterwards on the no_grad-captured instance, and is
  bit-exact against the reference instance.

Launch::

    torchrun --standalone --nproc-per-node=2 test/dd_nograd_first_call_check.py
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

DT = 0.0012
DH = 15.0


def ricker(nt, dt, fm=6.0, delay=0.2):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a ** 2) * np.exp(-(a ** 2))).astype(np.float32)


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

    # ---- reference: first call in the ordinary (grad-enabled) way ----------
    ref = build()
    vp_ref = torch.tensor(vp, device=dev)
    r_ref = ref(wav, src, rec, models=[vp_ref]).detach().clone()

    # ---- the case under test: FIRST EVER call inside no_grad --------------
    ddp = build()
    crashed = None
    try:
        with torch.no_grad():
            vp_ng = torch.tensor(vp, device=dev)
            r_ng = ddp(wav, src, rec, models=[vp_ng]).detach().clone()
    except RuntimeError as e:                       # what the bug looked like
        crashed, r_ng = e, None

    first_ok = crashed is None
    rec_bit = bool(first_ok and torch.equal(r_ng, r_ref))
    # torch.equal on two all-zero tensors is vacuously True, which would turn
    # every bit-exact assertion here into "the call did not raise". Demand real
    # signal in the reference record so the comparisons mean something.
    live = float(r_ref.abs().max()) > 0.0

    # ---- the no_grad-captured instance must still do gradients ------------
    # Compared BOTH ways on purpose: against another ModelParallel (same mesh,
    # same partial-gradient convention) and against a genuine single-domain
    # PropTorch. The DD-vs-DD half alone would leave the whole no_grad-first
    # path resting on a transitive argument, since every pre-existing
    # DD-vs-single-domain check enters with a grad-carrying model on call one.
    grad_ok = single_ok = False
    if first_ok:
        v_a = torch.tensor(vp, device=dev, requires_grad=True)
        (ddp(wav, src, rec, models=[v_a]).double() ** 2).sum().backward()
        v_b = torch.tensor(vp, device=dev, requires_grad=True)
        (ref(wav, src, rec, models=[v_b]).double() ** 2).sum().backward()
        grad_ok = bool(torch.equal(v_a.grad, v_b.grad))

        eq_s = Acoustic3D(spatial_order=args.so, device=dev, backend="torch")
        prop_s = PropTorch(eq_s, backend="torch", impl="c", shape=shape, dh=DH,
                           dt=DT, nt=nt, abcn=args.abcn, source_type=["h1"],
                           receiver_type=["h1"], dev=dev, free_surface=True)
        v_s = torch.tensor(vp, device=dev, requires_grad=True)
        (prop_s(wav, src, rec, models=[v_s]).double() ** 2).sum().backward()
        g_a = v_a.grad.clone()
        dist.all_reduce(g_a)          # per-rank partials -> the global gradient
        single_ok = bool(torch.equal(g_a, v_s.grad))

    # ---- inference_mode is NOT liftable: demand a legible error -----------
    # enable_grad() cannot re-enable autograd inside inference_mode, and the
    # captured params would be inference tensors that reject _set_models' later
    # in-place writes. The capture must say that itself, up front.
    inf_ok = False
    try:
        with torch.inference_mode():
            build()(wav, src, rec, models=[torch.tensor(vp, device=dev)])
    except RuntimeError as e:
        inf_ok = "inference_mode" in str(e)
        if rank == 0 and not inf_ok:
            print(f"  inference_mode raised, but unhelpfully: {str(e)[:120]}")

    ok = first_ok and rec_bit and grad_ok and inf_ok and live and single_ok
    if rank == 0:
        print(f"mesh py{args.py}xpx{args.px}  shape={shape}  nt={nt}  so={args.so}")
        print(f"reference record carries signal:          {live}"
              f"  (max |rec| = {float(r_ref.abs().max()):.3e})")
        print(f"first call under no_grad: {first_ok}"
              f"{'' if first_ok else f'  <-- {type(crashed).__name__}: {crashed}'}")
        print(f"record bit-exact vs grad-enabled capture: {rec_bit}")
        print(f"gradient still bit-exact afterwards:      {grad_ok}")
        print(f"no_grad-first DD == SINGLE DOMAIN:        {single_ok}")
        print(f"inference_mode gives a legible error:     {inf_ok}")
        print(f"-> {'PASS' if ok else 'FAIL'}")

    fail = torch.tensor([0 if ok else 1], device=dev)
    dist.all_reduce(fail)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(int(fail.item() > 0))


if __name__ == "__main__":
    main()
