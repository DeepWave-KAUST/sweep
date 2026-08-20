"""2-D FWI on Marmousi with the model split across GPUs -- and the same run on one.

This is a full inversion, not a one-gradient check: 20 shots, Adam, a loss
history. The point of running it twice is that ``--px 4`` (four tiles, four
GPUs) and ``--px 1`` (no domain decomposition at all, no ``torch.distributed``)
must produce the SAME loss curve and the SAME model, iteration for iteration.
A single-gradient parity test cannot see an error that only compounds; a
60-iteration curve can.

For that comparison to mean anything both sides must solve the same physics.
``Nx`` (1701 at downsample 8) is not divisible by 4, so the DD run pads it to
1704 -- a slightly longer domain, which is a slightly different problem. Hence
``--pad-px``, which is decoupled from ``--px``: the single-GPU run pads to the
same 1704 without splitting anything. Leave ``--pad-px`` at its default and you
measure something different but also worth knowing -- what the pad itself does
to an inversion (``--obs-pad-px`` keeps the observed data fixed while the
modelling grid changes, which is the field-data situation).

The model is the optimisation variable on the PHYSICAL grid; ``pad_to_mesh``
is applied inside the loss closure, so every run -- padded or not, split or not
-- optimises the same (nz, 1701) array and the results compare elementwise.

Run the pair (``--check`` diffs the second run against the first)::

    torchrun --standalone --nproc-per-node=4 dd_fwi_marmousi_2d.py \\
        --px 4 --tag dd4 --outdir out
    python dd_fwi_marmousi_2d.py --px 1 --pad-px 4 --tag single \\
        --outdir out --check dd4

and, if you want the pad's own cost rather than DD's, a third run on the
unpadded grid against the same observed data::

    python dd_fwi_marmousi_2d.py --px 1 --pad-px 1 --obs-pad-px 4 \\
        --tag nopad --outdir out --check dd4

Each run writes ``dd_fwi_marmousi_2d_<tag>.png`` (model, error, curves) plus
``hist_<tag>.npz`` / ``vp_final_<tag>.npy`` for your own comparisons.
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
from sweep.equations import Acoustic                     # noqa: E402
from sweep.parallel import MeshTopology, ModelParallel, pad_to_mesh  # noqa: E402
from sweep.propagator.torch import PropTorch             # noqa: E402


def ricker(nt, dt, fm, delay):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a ** 2) * np.exp(-(a ** 2))).astype(np.float32)


def smooth2d(x, sigma):
    """Separable Gaussian blur on (nz, nx), edge-replicating."""
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


def build_prop(shape, dev, a):
    eq = Acoustic(spatial_order=a.so, device=dev, backend="torch")
    return PropTorch(eq, backend="torch", impl="c", shape=shape, dev=dev,
                     dh=a.dh, dt=a.dt, nt=a.nt, abcn=a.abcn,
                     source_type=["h1"], receiver_type=["h1"],
                     free_surface=True, pml_type="cpmlr", B=1, use_ckpt=False,
                     boundary_saving_config={"enabled": True, "storage": "gpu",
                                             "transfer_interval": 1,
                                             "pinned_memory": False,
                                             "storage_dtype": a.boundary_dtype})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--px", type=int, default=1,
                   help="model tiles along x; 1 = plain single GPU, no DD")
    p.add_argument("--pad-px", type=int, default=None,
                   help="pad x up to a multiple of this (default: --px). "
                        "Equal on both sides = same physics = comparable.")
    p.add_argument("--obs-pad-px", type=int, default=None,
                   help="grid the observed data is modelled on "
                        "(default: --pad-px)")
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--nshot", type=int, default=20)
    # dt follows the grid: halving --downsample halves dh and so must halve dt.
    # Ask for a record LENGTH and let the CFL number pick dt, so the same
    # command line stays stable (and stable-in-the-CFL-sense) at any
    # resolution. --dt / --nt override if you want them fixed.
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--cfl", type=float, default=0.47,
                   help="vmax*dt/dh; 0.47 is what the 4th-order runs use")
    p.add_argument("--nt", type=int, default=None)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--so", type=int, default=4)
    p.add_argument("--abcn", type=int, default=20)
    p.add_argument("--fm", type=float, default=5.0)
    p.add_argument("--downsample", type=int, default=8)   # -> 10 m, 8.2 ppw
    p.add_argument("--lr", type=float, default=12.0, help="Adam step, m/s")
    p.add_argument("--init-smooth", type=float, default=20.0)
    p.add_argument("--grad-smooth", type=float, default=2.0)
    p.add_argument("--boundary-dtype", type=str, default="fp32",
                   choices=("fp32", "fp16", "bf16", "int8"),
                   help="compress the saved boundary ring. fp32 keeps DD "
                        "and single-GPU bit-identical; the lossy modes do "
                        "not, because a tile quantises different blocks "
                        "than the whole domain does.")
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--check", type=str, default=None,
                   help="tag of an earlier run in --outdir to compare against")
    p.add_argument("--outdir", type=str, default=".")
    a = p.parse_args()

    # torchrun sets RANK; without it this is an ordinary single-process run and
    # torch.distributed is never touched -- the honest "one card" baseline.
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
    obs_pad_px = pad_px if a.obs_pad_px is None else a.obs_pad_px
    assert pad_px % a.px == 0, "--pad-px must be a multiple of --px"
    log = print if rank == 0 else (lambda *x, **k: None)
    t0 = time.time()

    # ---- models ------------------------------------------------------------
    d = load("marmousi", variant="2d-acoustic")
    ds = a.downsample
    vp_np = np.ascontiguousarray(d["vp"][::ds, ::ds]).astype(np.float32)
    a.dh = float(d["dh"][0]) * ds
    nz, nx = vp_np.shape
    vp_true = torch.as_tensor(vp_np, device=dev)
    water = float(vp_np[0, 0])
    wb = int((vp_np != water).argmax(axis=0).min())       # shallowest sea floor
    vp_init = smooth2d(vp_true, a.init_smooth)
    vp_init[:wb] = water                                  # the water is known
    vmin, vmax = float(vp_true.min()), float(vp_true.max())
    if a.dt is None:
        a.dt = a.cfl * a.dh / vmax
    if a.nt is None:
        a.nt = int(round(a.seconds / a.dt))

    inv_shape = tuple(pad_to_mesh(vp_true, px=pad_px).shape)
    obs_shape = tuple(pad_to_mesh(vp_true, px=obs_pad_px).shape)
    log(f"[{a.tag}] Marmousi 2-D /{ds}: physical ({nz},{nx})  dh={a.dh:.1f} m  "
        f"sea floor row {wb} ({wb * a.dh:.0f} m)")
    log(f"[{a.tag}] inversion grid {inv_shape}   obs grid {obs_shape}   "
        f"px={a.px} ({world} GPU{'s' if world > 1 else ''})  "
        f"pad_px={pad_px}  obs_pad_px={obs_pad_px}")
    log(f"[{a.tag}] nt={a.nt} ({a.nt * a.dt:.1f} s)  dt={a.dt * 1e6:.1f} us  "
        f"CFL={vmax * a.dt / a.dh:.2f}  order={a.so}  fm={a.fm} Hz  "
        f"ppw={vmin / (2.5 * a.fm) / a.dh:.1f}  shots={a.nshot}  "
        f"iters={a.iters}  lr={a.lr} m/s")

    # ---- acquisition -------------------------------------------------------
    wav = torch.as_tensor(ricker(a.nt, a.dt, a.fm, 1.5 / a.fm), device=dev)
    sz, rz = 4, 6
    sx = np.linspace(60, nx - 60, a.nshot).round().astype(np.int64)
    rec = np.array([[[ix, rz] for ix in range(6, nx - 6, 4)]], dtype=np.int64)

    if a.px > 1:
        mesh = MeshTopology(py=1, px=a.px, shot_groups=1, world_size=world,
                            rank=rank)
        prop = ModelParallel(build_prop(inv_shape, dev, a), mesh)
    else:
        prop = build_prop(inv_shape, dev, a)
    obs_prop = prop if obs_shape == inv_shape else build_prop(obs_shape, dev, a)
    if obs_prop is not prop:
        assert a.px == 1, "a differing --obs-pad-px is a single-GPU convenience"

    # ---- observed data through the true model ------------------------------
    # no_grad keeps the DD capture forward-only: no adjoint wavefields and no
    # boundary ring while the observed data is generated. It is promoted on the
    # first backward below.
    shots = []
    # models=None on every shot after the first reuses the model the first
    # call already edge-padded and halo-exchanged — the model never changes
    # inside this loop, so re-running the NCCL model-halo collective per shot
    # would be pure waste.  DD only: the plain single-GPU PropTorch gives
    # models=None a different meaning (the equation's own parameters).
    reuse = isinstance(obs_prop, ModelParallel)
    with torch.no_grad():
        vt = pad_to_mesh(vp_true, px=obs_pad_px)
        for i, ix in enumerate(sx):
            src = np.array([[[int(ix), sz]]], dtype=np.int64)
            m = None if (reuse and i > 0) else [vt]
            shots.append((src, obs_prop(wav, src, rec, models=m).clone()))
    if obs_prop is not prop:
        del obs_prop
        torch.cuda.empty_cache()
    log(f"[{a.tag}] obs: {a.nshot} shots x {rec.shape[1]} receivers "
        f"({time.time() - t0:.0f} s)")

    # Each rank returns only the receivers inside its own tile, so the summed
    # misfit equals the single-GPU one ONLY if ownership is a true partition.
    if a.px > 1:
        own = prop.own_receiver_indices
        assert own, "no receiver ownership map to check"
        idx = torch.as_tensor(np.asarray(own, dtype=np.int64).ravel(), device=dev)
        cnt = torch.zeros(rec.shape[1], device=dev)
        cnt[idx] += 1.0
        dist.all_reduce(cnt)
        assert bool((cnt == 1).all()), "receiver ownership is not a partition"
        log(f"[{a.tag}] receiver ownership is a partition of "
            f"{rec.shape[1]} traces")

    # ---- FWI ---------------------------------------------------------------
    vp = vp_init.clone().requires_grad_(True)             # PHYSICAL-grid leaf
    opt = torch.optim.Adam([vp], lr=a.lr)
    hist = {"loss": [], "rmse": [], "gmax": [], "sec": []}
    for it in range(a.iters):
        t1 = time.time()
        opt.zero_grad(set_to_none=True)
        tot = torch.zeros((), device=dev, dtype=torch.float64)
        for src, obs in shots:
            syn = prop(wav, src, rec, models=[pad_to_mesh(vp, px=pad_px)])
            # Accumulate the misfit in float64. Under DD each rank sums only
            # its own receivers, so the reduction order differs from the
            # single-GPU sum; in fp32 that alone shows up at ~1e-7 relative
            # and would bury any real divergence between the two runs.
            j = 0.5 * (syn - obs).double().pow(2).sum()
            j.backward()
            tot = tot + j.detach()
        if world > 1:
            dist.all_reduce(tot)                          # [DD] global misfit
            dist.all_reduce(vp.grad)                      # [DD] global gradient
        g = vp.grad
        g[:wb] = 0.0                                      # the water is known
        if a.grad_smooth > 0:
            g = smooth2d(g, a.grad_smooth)
            g[:wb] = 0.0
            vp.grad = g
        gmax = float(g.abs().max())
        opt.step()
        with torch.no_grad():
            vp.clamp_(vmin, vmax)
            vp[:wb] = water
            rmse = float((vp - vp_true)[wb:].pow(2).mean().sqrt())
        hist["loss"].append(float(tot))
        hist["rmse"].append(rmse)
        hist["gmax"].append(gmax)
        hist["sec"].append(time.time() - t1)
        log(f"[{a.tag}] it {it:3d}  J = {float(tot):.16e}  "
            f"RMSE = {rmse:8.3f} m/s  |g|max = {gmax:.6e}  "
            f"{time.time() - t1:5.1f} s")

    rmse0 = float((vp_init - vp_true)[wb:].pow(2).mean().sqrt())
    log(f"[{a.tag}] === done ===")
    log(f"[{a.tag}] J   {hist['loss'][0]:.6e} -> {hist['loss'][-1]:.6e} "
        f"({100 * (1 - hist['loss'][-1] / hist['loss'][0]):.1f} % lower)")
    log(f"[{a.tag}] RMSE below the sea floor  {rmse0:.2f} -> "
        f"{hist['rmse'][-1]:.2f} m/s")
    log(f"[{a.tag}] peak memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} "
        f"GiB per GPU x {world}")
    log(f"[{a.tag}] wall {time.time() - t0:.0f} s "
        f"({np.mean(hist['sec']):.1f} s / iteration)")

    if rank == 0:
        out = Path(a.outdir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"hist_{a.tag}.npz",
                 loss=np.array(hist["loss"], dtype=np.float64),
                 rmse=np.array(hist["rmse"], dtype=np.float64),
                 gmax=np.array(hist["gmax"], dtype=np.float64),
                 sec=np.array(hist["sec"], dtype=np.float64),
                 inv_shape=np.array(inv_shape), obs_shape=np.array(obs_shape),
                 px=a.px, pad_px=pad_px, obs_pad_px=obs_pad_px, world=world,
                 wb=wb, dh=a.dh, rmse0=rmse0)
        np.save(out / f"vp_final_{a.tag}.npy", vp.detach().cpu().numpy())
        np.save(out / f"vp_init_{a.tag}.npy", vp_init.cpu().numpy())
        log(f"[{a.tag}] wrote {out / f'hist_{a.tag}.npz'} and "
            f"{out / f'vp_final_{a.tag}.npy'}")
        _plot(out, a.tag, vp_true, vp_init, vp.detach(), hist,
              int(inv_shape[-1]), a.px, wb, a.dh, log)
        if a.check:
            _check(out, a.tag, a.check, vp.detach().cpu().numpy(),
                   np.array(hist["loss"]), vp_true.cpu().numpy(),
                   wb, a.dh, log)

    if use_dist:
        dist.barrier()
        dist.destroy_process_group()


def _check(out, tag, other, vp_new, loss, vp_true, wb, dh, log):
    """Compare this run against an earlier one written to the same directory."""
    try:
        h = np.load(out / f"hist_{other}.npz")
        v = np.load(out / f"vp_final_{other}.npy")
    except FileNotFoundError:
        log(f"[{tag}] --check {other}: nothing to compare against yet")
        return
    n = min(len(loss), len(h["loss"]))
    same = int((loss[:n] == h["loss"][:n]).sum())
    dv = np.abs(vp_new - v).max()
    log(f"[{tag}] vs {other}: misfit identical on {same}/{n} iterations, "
        f"max |dvp| = {dv:.3e} m/s, models identical = "
        f"{np.array_equal(vp_new, v)}")
    mine = tuple(np.load(out / f"hist_{tag}.npz")["inv_shape"])
    log(f"[{tag}] grids: this {mine}   {other} {tuple(h['inv_shape'])}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except Exception as e:
        log(f"[{tag}] (comparison plot skipped: {e})")
        return
    nz, nx = vp_true.shape
    km = dict(extent=[0, nx * dh / 1e3, nz * dh / 1e3, 0], aspect="auto",
              origin="upper")
    # Draw the OTHER run's tile cuts: when a single-GPU run is checked
    # against a 4-tile one, the cuts under test are the 4-tile run's.
    px, padx = int(h["px"]), int(h["inv_shape"][-1])
    fig, ax = plt.subplots(1, 3, figsize=(18, 3.6), constrained_layout=True)
    lo, hi = float(vp_true.min()), float(vp_true.max())
    im = ax[0].imshow(vp_new, cmap="jet", vmin=lo, vmax=hi, **km)
    ax[0].set_title(f"FWI {tag}", fontsize=10)
    plt.colorbar(im, ax=ax[0], shrink=0.9, label="vp (m/s)")

    d = vp_new - v
    m = np.abs(d).max()
    # A fixed tiny range, so an exactly-zero difference reads as a blank panel
    # rather than as noise stretched to fill the colour bar.
    lim = max(m, 1e-3)
    im = ax[1].imshow(d, cmap="seismic", norm=TwoSlopeNorm(
        vmin=-lim, vcenter=0.0, vmax=lim), **km)
    ax[1].set_title(f"{tag} - {other}   max |dvp| = {m:.3g} m/s", fontsize=10)
    plt.colorbar(im, ax=ax[1], shrink=0.9)
    for a_ in ax[:2]:
        a_.set_xlabel("x (km)")
        a_.set_ylabel("z (km)")
        a_.axhline(wb * dh / 1e3, color="w", ls=":", lw=0.8, alpha=0.7)
        for i in range(1, px):
            a_.axvline(i * (padx // px) * dh / 1e3, color="k", ls="--",
                       lw=0.9, alpha=0.7)

    rel = np.abs(loss[:n] - h["loss"][:n]) / np.abs(loss[:n])
    ax[2].semilogy(np.maximum(rel, 1e-18), lw=1.5, color="k")
    ax[2].axhline(2.2e-16, color="r", ls="--", lw=0.9, label="1 ulp (fp64)")
    ax[2].set_xlabel("iteration")
    ax[2].set_ylabel(f"|J_{tag} - J_{other}| / J_{tag}")
    ax[2].set_title(f"misfit gap: identical on {same}/{n} iterations",
                    fontsize=10)
    ax[2].grid(alpha=0.3)
    ax[2].legend(fontsize=9)
    fig.suptitle(f"Marmousi 2-D FWI, {tag} vs {other}, {n} iterations   "
                 f"(dashed = tile cuts)")
    f = out / f"dd_fwi_marmousi_2d_{tag}_vs_{other}.png"
    fig.savefig(f, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"[{tag}] saved {f}")


def _plot(out, tag, vp_true, vp_init, vp_new, hist, padx, px, wb, dh, log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except Exception as e:                          # plotting is optional
        log(f"[{tag}] (plot skipped: {e})")
        return
    T, I, N = (x.cpu().numpy() for x in (vp_true, vp_init, vp_new))
    nz, nx = T.shape
    km = dict(extent=[0, nx * dh / 1e3, nz * dh / 1e3, 0], aspect="auto",
              origin="upper")
    cuts = [i * (padx // px) * dh / 1e3 for i in range(1, px)]

    fig, ax = plt.subplots(3, 2, figsize=(15, 11), constrained_layout=True)
    lo, hi = float(T.min()), float(T.max())
    for a_, img, ttl in ((ax[0, 0], T, "true vp"),
                         (ax[0, 1], I, "start (smoothed)"),
                         (ax[1, 0], N, f"FWI result ({tag})")):
        im = a_.imshow(img, cmap="jet", vmin=lo, vmax=hi, **km)
        a_.set_title(ttl, fontsize=10)
        plt.colorbar(im, ax=a_, shrink=0.85, label="vp (m/s)")
    err = N - T
    p = np.percentile(err, [0.5, 99.5])
    im = ax[1, 1].imshow(err, cmap="seismic", norm=TwoSlopeNorm(
        vmin=min(p[0], -1.0), vcenter=0.0, vmax=max(p[1], 1.0)), **km)
    ax[1, 1].set_title("FWI - true (m/s)", fontsize=10)
    plt.colorbar(im, ax=ax[1, 1], shrink=0.85)
    for a_ in ax[:2].ravel():
        a_.set_xlabel("x (km)")
        a_.set_ylabel("z (km)")
        a_.axhline(wb * dh / 1e3, color="w", ls=":", lw=0.8, alpha=0.7)
        for c in cuts:
            a_.axvline(c, color="k", ls="--", lw=0.9, alpha=0.7)

    ax[2, 0].semilogy(hist["loss"], lw=1.6)
    ax[2, 0].set_ylabel("misfit J")
    ax[2, 1].plot(hist["rmse"], lw=1.6)
    ax[2, 1].set_ylabel("RMSE below sea floor (m/s)")
    for a_ in ax[2]:
        a_.set_xlabel("iteration")
        a_.grid(alpha=0.3)
    fig.suptitle(f"Marmousi 2-D FWI, {px} tile{'s' if px > 1 else ''}, "
                 f"{len(hist['loss'])} iterations   (dashed = tile cuts)")
    fig.savefig(out / f"dd_fwi_marmousi_2d_{tag}.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    log(f"[{tag}] saved {out / f'dd_fwi_marmousi_2d_{tag}.png'}")


if __name__ == "__main__":
    main()
