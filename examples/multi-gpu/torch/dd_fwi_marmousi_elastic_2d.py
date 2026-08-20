"""Elastic FWI on Marmousi with the model split across GPUs -- and on one.

The elastic sibling of ``dd_fwi_marmousi_2d.py``, aimed at the path the
acoustic run cannot touch: a BODY-FORCE source (vz) with velocity receivers,
one shot sitting exactly on a tile cut, boundary saving on. vs and rho are
derived from vp (Poisson + Gardner) so the example needs no extra data; the
whole model is solid and there is no free surface -- deriving vs in a water
column makes a slow solid that drags the global points-per-wavelength under
sampling, and the free-surface axis is already covered by the DD test matrix.

fm=4 Hz at dh=10 m keeps ppw(vs_min) ~ 5.9; dt comes from the elastic CFL.

Run the pair (``--check`` diffs the second run against the first)::

    torchrun --standalone --nproc-per-node=2 dd_fwi_marmousi_elastic_2d.py \\
        --px 2 --tag dd2 --outdir out
    python dd_fwi_marmousi_elastic_2d.py --px 1 --pad-px 2 --tag single \\
        --outdir out --check dd2
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
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sweep.datasets import load                          # noqa: E402
from sweep.equations import Elastic                      # noqa: E402
from sweep.parallel import MeshTopology, ModelParallel, pad_to_mesh  # noqa: E402
from sweep.propagator.torch import PropTorch             # noqa: E402


def ricker(nt, dt, fm, delay, scale):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return (scale * (1.0 - 2.0 * a ** 2) * np.exp(-(a ** 2))).astype(np.float32)


def smooth2d(x, sigma):
    if sigma <= 0:
        return x
    r = max(1, int(3 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device,
                                       dtype=x.dtype) / sigma) ** 2)
    k = k / k.sum()
    out = x[None, None]
    out = F.conv2d(F.pad(out, (0, 0, r, r), mode="replicate"),
                   k.view(1, 1, -1, 1))
    out = F.conv2d(F.pad(out, (r, r, 0, 0), mode="replicate"),
                   k.view(1, 1, 1, -1))
    return out[0, 0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--px", type=int, default=1)
    p.add_argument("--pad-px", type=int, default=None)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--nshot", type=int, default=8)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--cfl", type=float, default=0.35)
    p.add_argument("--so", type=int, default=4)
    p.add_argument("--abcn", type=int, default=20)
    p.add_argument("--fm", type=float, default=4.0)
    p.add_argument("--downsample", type=int, default=8)
    p.add_argument("--source-type", type=str, default="vz",
                   help="body force by default -- the path the acoustic "
                        "example cannot reach")
    p.add_argument("--lr-vp", type=float, default=10.0)
    p.add_argument("--lr-vs", type=float, default=6.0)
    p.add_argument("--init-smooth", type=float, default=15.0)
    p.add_argument("--grad-smooth", type=float, default=2.0)
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--check", type=str, default=None)
    p.add_argument("--outdir", type=str, default=".")
    a = p.parse_args()

    use_dist = "RANK" in os.environ
    if use_dist:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    li = int(os.environ.get("LOCAL_RANK", 0)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")
    assert a.px == world, f"--px {a.px} but world size is {world}"
    pad_px = a.px if a.pad_px is None else a.pad_px
    assert pad_px % a.px == 0
    log = (lambda *x, **k: print(*x, flush=True, **k)) if rank == 0 \
        else (lambda *x, **k: None)
    t0 = time.time()

    # ---- models: vp from Marmousi, vs Poisson, rho Gardner ----------------
    d = load("marmousi", variant="2d-acoustic")
    ds = a.downsample
    vp_np = np.ascontiguousarray(d["vp"][::ds, ::ds]).astype(np.float32)
    dh = float(d["dh"][0]) * ds
    nz, nx = vp_np.shape
    vp_true = torch.as_tensor(vp_np, device=dev)
    vs_true = vp_true / 1.732
    rho_true = 310.0 * vp_true.pow(0.25)
    vpr = (float(vp_true.min()), float(vp_true.max()))
    vsr = (float(vs_true.min()), float(vs_true.max()))
    dt = a.cfl * dh / vpr[1]
    nt = int(round(a.seconds / dt))
    ppw = vsr[0] / (2.5 * a.fm) / dh
    assert ppw >= 5.0, f"ppw(vs_min)={ppw:.2f} < 5 -- lower --fm or --downsample"

    vp_init = smooth2d(vp_true, a.init_smooth)
    vs_init = smooth2d(vs_true, a.init_smooth)

    inv_shape = tuple(pad_to_mesh(vp_true, px=pad_px).shape)
    log(f"[{a.tag}] Marmousi elastic 2-D /{ds}: physical ({nz},{nx}) -> "
        f"{inv_shape}  dh={dh:.1f} m  px={a.px}")
    log(f"[{a.tag}] nt={nt} ({nt * dt:.2f} s)  dt={dt * 1e6:.0f} us  "
        f"CFL={vpr[1] * dt / dh:.2f}  ppw(vs_min)={ppw:.1f}  "
        f"src={a.source_type}  shots={a.nshot}  iters={a.iters}")

    # ---- acquisition: one shot ON every cut, the rest spread --------------
    wav = torch.as_tensor(ricker(nt, dt, a.fm, 1.5 / a.fm, 1e4), device=dev)
    cuts = [i * (inv_shape[-1] // pad_px) for i in range(1, pad_px)]
    sx = sorted(set(np.linspace(80, nx - 80, a.nshot).astype(int).tolist()
                    + [c for c in cuts if 0 < c < nx]))[:a.nshot]
    sz, rz = 8, 6
    rec = np.array([[[ix, rz] for ix in range(6, nx - 6, 6)]], dtype=np.int64)
    log(f"[{a.tag}] shot x = {sx}   (cuts at {cuts})")

    def build():
        eq = Elastic(spatial_order=a.so, device=dev, backend="torch")
        return PropTorch(eq, backend="torch", impl="c", shape=inv_shape,
                         dev=dev, dh=dh, dt=dt, nt=nt, abcn=a.abcn,
                         source_type=[a.source_type],
                         receiver_type=["vx", "vz"],
                         free_surface=False, pml_type="cpmls", B=1,
                         use_ckpt=False,
                         boundary_saving_config={"enabled": True,
                                                 "storage": "gpu",
                                                 "transfer_interval": 1,
                                                 "pinned_memory": False})

    if a.px > 1:
        mesh = MeshTopology(py=1, px=a.px, shot_groups=1, world_size=world,
                            rank=rank)
        prop = ModelParallel(build(), mesh)
    else:
        prop = build()

    def padded(*models):
        return [pad_to_mesh(m, px=pad_px) for m in models]

    # ---- observed data through the true model -----------------------------
    # models=None after the first shot reuses the already padded/exchanged
    # model (DD only — plain PropTorch gives models=None another meaning).
    shots = []
    reuse = isinstance(prop, ModelParallel)
    with torch.no_grad():
        mt = padded(vp_true, vs_true, rho_true)
        for i, ix in enumerate(sx):
            src = np.array([[[int(ix), sz]]], dtype=np.int64)
            m = None if (reuse and i > 0) else mt
            shots.append((src, prop(wav, src, rec, models=m).clone()))
    log(f"[{a.tag}] obs ready ({time.time() - t0:.0f} s)")

    if a.px > 1:
        own = np.asarray(prop.own_receiver_indices, dtype=np.int64).ravel()
        cnt = torch.zeros(rec.shape[1], device=dev)
        cnt[torch.as_tensor(own, device=dev)] += 1.0
        dist.all_reduce(cnt)
        assert bool((cnt == 1).all()), "receiver ownership is not a partition"

    # ---- FWI: vp and vs on physical-grid leaves, rho known ----------------
    vp = vp_init.clone().requires_grad_(True)
    vs = vs_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [vp], "lr": a.lr_vp},
                            {"params": [vs], "lr": a.lr_vs}])
    hist = {"loss": [], "rmse_vp": [], "rmse_vs": [], "sec": []}
    for it in range(a.iters):
        t1 = time.time()
        opt.zero_grad(set_to_none=True)
        tot = torch.zeros((), device=dev, dtype=torch.float64)
        for src, obs in shots:
            syn = prop(wav, src, rec, models=padded(vp, vs, rho_true))
            j = 0.5 * (syn - obs).double().pow(2).sum()
            j.backward()
            tot = tot + j.detach()
        if world > 1:
            dist.all_reduce(tot)
            dist.all_reduce(vp.grad)
            dist.all_reduce(vs.grad)
        if a.grad_smooth > 0:
            vp.grad = smooth2d(vp.grad, a.grad_smooth)
            vs.grad = smooth2d(vs.grad, a.grad_smooth)
        opt.step()
        with torch.no_grad():
            vp.clamp_(*vpr)
            vs.clamp_(*vsr)
            rv = float((vp - vp_true).pow(2).mean().sqrt())
            rs = float((vs - vs_true).pow(2).mean().sqrt())
        hist["loss"].append(float(tot))
        hist["rmse_vp"].append(rv)
        hist["rmse_vs"].append(rs)
        hist["sec"].append(time.time() - t1)
        log(f"[{a.tag}] it {it:3d}  J = {float(tot):.16e}  "
            f"RMSE vp {rv:7.2f}  vs {rs:7.2f}  {time.time() - t1:5.1f} s")

    drop = 1.0 - hist["loss"][-1] / hist["loss"][0]
    conv = drop > 0.5
    log(f"[{a.tag}] J {hist['loss'][0]:.4e} -> {hist['loss'][-1]:.4e} "
        f"({100 * drop:.1f} % lower)  "
        f"RMSE vp {float((vp_init - vp_true).pow(2).mean().sqrt()):.2f} -> "
        f"{hist['rmse_vp'][-1]:.2f}   vs "
        f"{float((vs_init - vs_true).pow(2).mean().sqrt()):.2f} -> "
        f"{hist['rmse_vs'][-1]:.2f}")
    log(f"[{a.tag}] {'CONVERGED' if conv else 'NOT CONVERGED'} "
        f"(misfit drop {100 * drop:.1f} %, threshold 50 %)   "
        f"wall {time.time() - t0:.0f} s")

    if rank == 0:
        out = Path(a.outdir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"ehist_{a.tag}.npz", **{k: np.array(v) for k, v in
                                                hist.items()},
                 inv_shape=np.array(inv_shape), px=a.px)
        np.save(out / f"evp_final_{a.tag}.npy", vp.detach().cpu().numpy())
        np.save(out / f"evs_final_{a.tag}.npy", vs.detach().cpu().numpy())
        if a.check:
            try:
                h = np.load(out / f"ehist_{a.check}.npz")
                ovp = np.load(out / f"evp_final_{a.check}.npy")
                ovs = np.load(out / f"evs_final_{a.check}.npy")
                n = min(len(hist["loss"]), len(h["loss"]))
                same = int((np.array(hist["loss"][:n]) == h["loss"][:n]).sum())
                dvp = float(np.abs(vp.detach().cpu().numpy() - ovp).max())
                dvs = float(np.abs(vs.detach().cpu().numpy() - ovs).max())
                ident = (np.array_equal(vp.detach().cpu().numpy(), ovp)
                         and np.array_equal(vs.detach().cpu().numpy(), ovs))
                log(f"[{a.tag}] vs {a.check}: misfit identical {same}/{n}, "
                    f"max|dvp|={dvp:.3e} max|dvs|={dvs:.3e}, "
                    f"models identical = {ident}")
                conv = conv and ident
            except FileNotFoundError:
                log(f"[{a.tag}] --check {a.check}: nothing to compare yet")

    if use_dist:
        dist.barrier()
        dist.destroy_process_group()
    sys.exit(0 if conv or rank != 0 else 1)


if __name__ == "__main__":
    main()
