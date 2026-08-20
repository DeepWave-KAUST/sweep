"""One full FWI model update on Overthrust 3-D, with the model split across GPUs.

This is the whole iteration, end to end: observed data from the true model,
a smoothed starting model, a multi-shot gradient, one preconditioned
steepest-descent step with a backtracking line search, and a re-evaluated
misfit that says whether the update actually helped.

The DD-specific part is three lines — a ``MeshTopology``, ``ModelParallel``,
and two ``all_reduce`` calls (global misfit, global gradient). Everything else
is ordinary single-GPU FWI code.

The model is the OPTIMISATION VARIABLE on the physical (unpadded) grid;
:func:`sweep.parallel.pad_to_mesh` is applied inside the closure, so the
gradient, the update and the saved arrays all live on the physical grid and
never inherit the tile padding. See ``test/dd_pad_grad_check.py``.

Run (4 GPUs, 2x2 tiles)::

    torchrun --standalone --nproc-per-node=4 \\
        examples/multi-gpu/torch/dd_fwi_overthrust_update.py --py 2 --px 2

Physics defaults match notebook 25 (order-8, 5 Hz Ricker, 3.0 s): above
3.5 points per wavelength at vmin, so there is no visible grid dispersion.
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
from sweep.equations import Acoustic3D                   # noqa: E402
from sweep.parallel import MeshTopology, ModelParallel, pad_to_mesh  # noqa: E402
from sweep.propagator.torch import PropTorch             # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def ricker(nt, dt, fm, delay):
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = np.pi * fm * t
    return ((1.0 - 2.0 * a ** 2) * np.exp(-(a ** 2))).astype(np.float32)


def smooth3d(x, sigma):
    """Separable Gaussian blur on (nz, ny, nx), edge-replicating. Used both to
    build the starting model and to precondition the gradient."""
    sz, sy, sx = (sigma, sigma, sigma) if np.isscalar(sigma) else sigma
    out = x[None, None]
    for axis, s in enumerate((sz, sy, sx)):
        if s <= 0:
            continue
        r = max(1, int(3 * s))
        k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=x.device,
                                           dtype=x.dtype) / s) ** 2)
        k = k / k.sum()
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = k.numel()
        pad = [0, 0, 0, 0, 0, 0]                 # (x0,x1, y0,y1, z0,z1)
        pad[2 * (2 - axis)] = pad[2 * (2 - axis) + 1] = r
        out = F.conv3d(F.pad(out, pad, mode="replicate"), k.view(shape))
    return out[0, 0]


def build_prop(shape, dev, a):
    eq = Acoustic3D(spatial_order=a.so, device=dev, backend="torch")
    return PropTorch(eq, backend="torch", impl="c", shape=shape, dev=dev,
                     dh=a.dh, dt=a.dt, nt=a.nt, abcn=a.abcn,
                     source_type=["h1"], receiver_type=["h1"],
                     free_surface=True, pml_type="cpmlr", B=1, use_ckpt=False,
                     boundary_saving_config={"enabled": True, "storage": "gpu",
                                             "transfer_interval": 1,
                                             "pinned_memory": False})


def misfit(ddp, wav, shots, rec, vp_phys, mesh, grad=False):
    """Sum 0.5*||syn - obs||^2 over shots. With ``grad=True`` the graph is kept
    so a single backward accumulates the multi-shot gradient into ``vp_phys``."""
    total = torch.zeros((), device=vp_phys.device, dtype=torch.float64)
    for src, obs in shots:
        vp_padded = pad_to_mesh(vp_phys, mesh)           # differentiable pad
        if grad:
            syn = ddp(wav, src, rec, models=[vp_padded])
            j = 0.5 * (syn - obs).pow(2).sum().double()
            j.backward()                                 # accumulate this shot
        else:
            with torch.no_grad():
                syn = ddp(wav, src, rec, models=[vp_padded])
                j = 0.5 * (syn - obs).pow(2).sum().double()
        total = total + j.detach()
    dist.all_reduce(total)                               # [DD] global misfit
    return float(total.item())


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--py", type=int, default=2, help="tiles along y")
    p.add_argument("--px", type=int, default=2, help="tiles along x")
    p.add_argument("--nshot", type=int, default=5)
    p.add_argument("--nt", type=int, default=1500)       # 3.0 s
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--so", type=int, default=8)
    p.add_argument("--abcn", type=int, default=20)
    p.add_argument("--fm", type=float, default=5.0)
    p.add_argument("--delay", type=float, default=0.25)
    p.add_argument("--dh", type=float, default=50.0)     # overwritten by dataset
    p.add_argument("--init-smooth", type=float, default=6.0,
                   help="Gaussian sigma (cells) for the starting model")
    p.add_argument("--grad-smooth", type=float, default=2.0,
                   help="Gaussian sigma (cells) applied to the gradient")
    p.add_argument("--mask-top", type=int, default=8,
                   help="zero the gradient in the top N cells (source region)")
    p.add_argument("--max-update", type=float, default=120.0,
                   help="m/s: the step is scaled so this is the largest change")
    p.add_argument("--ls-trials", type=int, default=3,
                   help="backtracking trials (step halved each time)")
    p.add_argument("--outdir", type=str,
                   default=str(Path(__file__).resolve().parent))
    a = p.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    li = int(os.environ.get("LOCAL_RANK", rank)) % max(1, torch.cuda.device_count())
    torch.cuda.set_device(li)
    dev = torch.device(f"cuda:{li}")
    assert a.py * a.px == world, f"py*px must equal world size {world}"
    log = print if rank == 0 else (lambda *x, **k: None)
    t0 = time.time()

    # ---- 1. models: true from the dataset, start = heavily smoothed ---------
    d = load("overthrust", variant="3d-acoustic", downsample=2)
    vp_true_np = np.ascontiguousarray(d["vp"]).astype("float32")
    a.dh = float(d["dh"][0])
    nz, ny, nx = vp_true_np.shape
    vp_true = torch.as_tensor(vp_true_np, device=dev)
    vp_init = smooth3d(vp_true, a.init_smooth)

    mesh = MeshTopology(py=a.py, px=a.px, shot_groups=1, world_size=world,
                        rank=rank)
    padded_shape = tuple(pad_to_mesh(vp_true, mesh).shape)
    log(f"[dd-fwi] physical {(nz, ny, nx)} -> padded {padded_shape} "
        f"(mesh py={a.py} px={a.px}, {world} GPUs)")
    log(f"[dd-fwi] dh={a.dh} m  nt={a.nt} ({a.nt * a.dt:.1f} s)  order={a.so}  "
        f"fm={a.fm} Hz  shots={a.nshot}")

    # ---- 2. acquisition: shots along x at mid-crossline, fixed spread -------
    wav = torch.as_tensor(ricker(a.nt, a.dt, a.fm, a.delay), device=dev)
    sx = np.linspace(nx // 5, 4 * nx // 5, a.nshot).astype(np.int64)
    rec = np.array([[[ix, ny // 2, 4] for ix in range(6, nx - 6, 2)]],
                   dtype=np.int64)
    prop = build_prop(padded_shape, dev, a)
    ddp = ModelParallel(prop, mesh)

    # ---- 3. observed data through the TRUE model ---------------------------
    # torch.no_grad() here is not cosmetic: it keeps this instance's capture
    # forward-only, so generating the observed data allocates neither the
    # adjoint wavefields nor the nt-scaled boundary ring. The capture is
    # promoted automatically at step 4, the first time a gradient is asked for.
    shots = []
    vp_true_padded = pad_to_mesh(vp_true, mesh)
    with torch.no_grad():
        for i, ix in enumerate(sx):
            src = np.array([[[int(ix), ny // 2, 4]]], dtype=np.int64)
            # models=None after the first shot reuses the model the first call
            # already padded and halo-exchanged (it never changes in this loop).
            m = [vp_true_padded] if i == 0 else None
            shots.append((src, ddp(wav, src, rec, models=m)))
    log(f"[dd-fwi] obs ready: {a.nshot} shots x {rec.shape[1]} receivers "
        f"({time.time() - t0:.0f} s)")

    # ---- 4. gradient of the multi-shot misfit at the starting model ---------
    vp = vp_init.clone().requires_grad_(True)            # PHYSICAL-grid leaf
    j0 = misfit(ddp, wav, shots, rec, vp, mesh, grad=True)
    dist.all_reduce(vp.grad)                             # [DD] global gradient
    g = vp.grad.detach().clone()
    assert tuple(g.shape) == (nz, ny, nx), "gradient must be on the physical grid"

    # ---- 5. precondition, then one step with a backtracking line search -----
    g[:a.mask_top] = 0.0                                 # drop the source spike
    if a.grad_smooth > 0:
        g = smooth3d(g, a.grad_smooth)
    step = a.max_update / (g.abs().max().item() + 1e-30)
    log(f"[dd-fwi] J0 = {j0:.6e}   |grad|max = {g.abs().max().item():.3e}"
        f"   ({time.time() - t0:.0f} s)")

    vp_new, j1, used = None, j0, 0.0
    for k in range(a.ls_trials):
        trial = (vp_init - step * g).clamp_(vp_true.min(), vp_true.max())
        jt = misfit(ddp, wav, shots, rec, trial, mesh)
        log(f"[dd-fwi]   trial {k}: max|dvp| = {a.max_update / 2 ** k:6.1f} m/s"
            f"   J = {jt:.6e}   ({'accept' if jt < j0 else 'reject'})")
        if jt < j0:
            vp_new, j1, used = trial, jt, a.max_update / 2 ** k
            break
        step *= 0.5
    if vp_new is None:                                   # nothing helped
        log("[dd-fwi] line search failed -- no step reduced the misfit")
        dist.destroy_process_group()
        sys.exit(1)

    # ---- 6. report: did the MODEL get closer to the truth, not just J? ------
    # A whole-volume RMSE is a bad scoreboard for ONE iteration: a handful of
    # shots only illuminates a fraction of the model, and the untouched rest
    # dilutes the change to nothing. Report both, but read the second one.
    dvp = (vp_new - vp_init).abs()
    lit = dvp > 0.05 * dvp.max()                         # where the step landed

    def rms(x, sel=None):
        e = (x - vp_true) if sel is None else (x - vp_true)[sel]
        return float(e.pow(2).mean().sqrt().item())

    log(f"\n[dd-fwi] === one update ===")
    log(f"[dd-fwi] misfit      J0 = {j0:.6e} -> J1 = {j1:.6e} "
        f"({100 * (1 - j1 / j0):.1f} % lower)")
    log(f"[dd-fwi] model RMSE     {rms(vp_init):.2f} -> {rms(vp_new):.2f} m/s "
        f"(whole volume -- mostly unilluminated)")
    log(f"[dd-fwi] RMSE in step   {rms(vp_init, lit):.2f} -> {rms(vp_new, lit):.2f} "
        f"m/s over {100 * lit.float().mean().item():.1f} % of cells")
    log(f"[dd-fwi] applied step   max |dvp| = {used:.1f} m/s")
    log(f"[dd-fwi] peak memory    {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} "
        f"GiB per tile ({world} tiles)")
    log(f"[dd-fwi] wall time      {time.time() - t0:.0f} s")

    if rank == 0:
        out = Path(a.outdir)
        np.save(out / "dd_update_init.npy", vp_init.cpu().numpy())
        np.save(out / "dd_update_new.npy", vp_new.cpu().numpy())
        np.save(out / "dd_update_grad.npy", g.cpu().numpy())
        _plot(out, vp_true, vp_init, vp_new, g, mesh, (nz, ny, nx), j0, j1, log)

    dist.barrier()
    dist.destroy_process_group()


def _plot(out, vp_true, vp_init, vp_new, g, mesh, shape, j0, j1, log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except Exception as e:                                # plotting is optional
        log(f"[dd-fwi] (plot skipped: {e})")
        return
    nz, ny, nx = shape
    iy = ny // 2
    T, I, N = (x[:, iy, :].cpu().numpy() for x in (vp_true, vp_init, vp_new))
    G, D = g[:, iy, :].cpu().numpy(), (vp_new - vp_init)[:, iy, :].cpu().numpy()
    cuts = [i * (nx // mesh.px) for i in range(1, mesh.px)]

    def signed(ax, img, ttl):
        lo, hi = np.percentile(img, [0.5, 99.5])          # asymmetric, 0 stays white
        m = np.abs(img).max() or 1.0
        im = ax.imshow(img, cmap="seismic", aspect="auto", origin="upper",
                       norm=TwoSlopeNorm(vmin=min(lo, -1e-3 * m), vcenter=0.0,
                                         vmax=max(hi, 1e-3 * m)))
        ax.set_title(ttl, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.85)

    fig, ax = plt.subplots(2, 3, figsize=(16, 7), constrained_layout=True)
    lo, hi = float(vp_true.min()), float(vp_true.max())
    for k, (img, ttl) in enumerate(((T, "true vp"), (I, "start (smoothed)"),
                                    (N, "after 1 update"))):
        im = ax[0, k].imshow(img, cmap="jet", vmin=lo, vmax=hi, aspect="auto",
                             origin="upper")
        ax[0, k].set_title(f"{ttl}   inline y={iy}", fontsize=10)
        plt.colorbar(im, ax=ax[0, k], shrink=0.85, label="vp (m/s)")
    signed(ax[1, 0], G, "gradient (preconditioned)")
    signed(ax[1, 1], D, "update  vp_new - vp_start (m/s)")
    signed(ax[1, 2], N - T, "error after update  vp_new - vp_true (m/s)")
    for a_ in ax.ravel():
        a_.set_xlabel("x"); a_.set_ylabel("z")
        for c in cuts:
            a_.axvline(c, color="k", ls="--", lw=0.8, alpha=0.6)
    fig.suptitle(f"DD FWI, one model update on Overthrust 3-D   "
                 f"J {j0:.3e} -> {j1:.3e}   ({mesh.py}x{mesh.px} tiles)")
    fig.savefig(out / "dd_fwi_overthrust_update.png", dpi=200,
                bbox_inches="tight")
    log(f"[dd-fwi] saved {out / 'dd_fwi_overthrust_update.png'}")


if __name__ == "__main__":
    main()
