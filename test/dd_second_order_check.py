"""ModelParallel refuses create_graph=True and forward-mode AD with a clear error
(a dual model that does not require grad took the plain stepped forward and
came back with its tangent dropped), and a plain gradient still runs.
sweep.autodiff takes its products by differences: with each rank passing its
own tile (and a direction ten times weaker on one tile), every rank must take
the same step, so the tiles of H h and the traces of J h match the
single-domain products.

torchrun --standalone --nproc-per-node=2 test/dd_second_order_check.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import sweep  # noqa: E402
from sweep.autodiff import hessian_vector_product, jacobian_vector_product  # noqa: E402
from sweep.equations import Acoustic                  # noqa: E402
from sweep.parallel import MeshTopology, ModelParallel  # noqa: E402
from sweep.propagator.torch import PropTorch  # noqa: E402

DT = 0.0015


def ricker(nt, dt, fm=10.0, delay=0.08):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a ** 2) * np.exp(-(a ** 2))).astype(np.float32)


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")

    so, abcn, nt = 4, 10, 80
    nz, nxp = 48, 28
    nx = nxp * world
    grid = np.linspace(0, 1, nz * nx, dtype=np.float32).reshape(nz, nx)
    true_g = 1800.0 + 600.0 * grid
    init_g = np.broadcast_to(1800.0 + 600.0 * np.linspace(0, 1, nz, np.float32)[:, None], (nz, nx)).copy()
    wav = ricker(nt, DT)
    src = np.array([[[nx // 2, nz // 4]]], dtype=np.int32)
    rec = np.array([[[ix, 2] for ix in range(2, nx - 2, 4)]], dtype=np.int32)

    topo = MeshTopology(py=1, px=world, shot_groups=1, world_size=world, rank=rank)
    prop = PropTorch(Acoustic(spatial_order=so, device=dev, backend="torch"),
                     backend="torch", impl="c", shape=(nz, nx), dh=12.0, dt=DT, nt=nt,
                     abcn=abcn, source_type=["h1"], receiver_type=["h1"], dev=dev)
    ddp = ModelParallel(prop, topo)
    x0 = ddp.x0
    obs = ddp.forward(wav, src, rec, models=[true_g]).clone()

    results = {}
    # (1) create_graph=True is refused in the first backward, naming what works
    vp = torch.tensor(init_g[:, x0:x0 + nxp], device=dev, requires_grad=True)
    syn = ddp.forward(wav, src, rec, models=[vp])
    try:
        torch.autograd.grad(0.5 * (syn - obs).pow(2).sum(), vp, create_graph=True)
        results["create_graph"] = "NOT REFUSED"
    except RuntimeError as e:
        msg = str(e)
        results["create_graph"] = "refused" if ("ModelParallel" in msg and "twice-differentiable" in msg) else f"other: {msg[:200]}"

    # (2) a plain gradient still runs after that
    vp = torch.tensor(init_g[:, x0:x0 + nxp], device=dev, requires_grad=True)
    (0.5 * (ddp.forward(wav, src, rec, models=[vp]) - obs).pow(2).sum()).backward()
    results["plain_grad_nonzero"] = bool(vp.grad.abs().max().item() > 0)

    # (3) forward-mode AD is refused with a clear error
    try:
        with fwAD.dual_level():
            vpd = fwAD.make_dual(torch.tensor(init_g[:, x0:x0 + nxp], device=dev), torch.ones(nz, nxp, device=dev))
            ddp.forward(wav, src, rec, models=[vpd])
        results["forward_ad"] = "NOT REFUSED"
    except Exception as e:
        msg = str(e)
        results["forward_ad"] = "refused" if "forward-mode" in msg else f"other {type(e).__name__}: {msg[:200]}"

    # (4) products by differences from per-tile models match the single-domain ones
    single = PropTorch(Acoustic(spatial_order=so, device=dev, backend="torch"),
                       backend="torch", impl="c", shape=(nz, nx), dh=12.0, dt=DT, nt=nt,
                       abcn=abcn, source_type=["h1"], receiver_type=["h1"], dev=dev)
    gen = torch.Generator().manual_seed(0)
    h = torch.randn(nz, nx, generator=gen)
    h[[0, -1], :] = 0.0
    h[:, [0, -1]] = 0.0
    h[:, nxp:] *= 0.1                                  # every tile but the first: a weaker direction
    h = h.to(dev)
    vp0 = torch.tensor(init_g, dtype=torch.float32, device=dev)
    obs_all = single(wav, src, rec, models=[torch.tensor(true_g, dtype=torch.float32, device=dev)]).detach()
    (hv_ref,) = hessian_vector_product(single, wav, src, rec, [vp0], [h],
                                       lambda d: 0.5 * ((d - obs_all) ** 2).sum(), route="difference")
    _, jh_ref = jacobian_vector_product(single, wav, src, rec, [vp0], [h], route="difference")
    tile = lambda a: a[:, x0:x0 + nxp].contiguous()
    (hv,) = hessian_vector_product(ddp, wav, src, rec, [tile(vp0)], [tile(h)],
                                   lambda d: 0.5 * ((d - obs) ** 2).sum())
    _, jh = jacobian_vector_product(ddp, wav, src, rec, [tile(vp0)], [tile(h)])
    own = list(ddp.own_receiver_indices)
    rel = lambda a, b: float((a - b).norm() / b.norm())
    errs = torch.tensor([rel(hv, tile(hv_ref)), rel(jh, jh_ref[:, :, own]) if own else 0.0], device=dev)
    dist.all_reduce(errs, op=dist.ReduceOp.MAX)
    results["hv_tile_rel"], results["jh_trace_rel"] = (float(x) for x in errs)

    ok = (results["create_graph"] == "refused" and results["plain_grad_nonzero"]
          and results["forward_ad"] == "refused"
          and results["hv_tile_rel"] < 1e-2 and results["jh_trace_rel"] < 1e-3)
    if rank == 0:
        print(f"[rank0] {results}")
        print("DD_SECOND_ORDER_CHECK:", "PASS" if ok else "FAIL")
    dist.destroy_process_group()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
